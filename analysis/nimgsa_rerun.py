"""
nimgsa_rerun.py -- NIMGSA (Jin et al. 2022, jin2022nimgsa) rerun with its own model code, under
its published protocol and under progressively corrected ones.

NIMGSA's code (github.com/zhanglabNKU/NIMGSA, commit 5bf10a9; no licence, so it is cloned
outside this repo and only its models.py is imported, unmodified) reveals what the paper does
not say (read 2026-11-14):

  * the five folds are over miRNA ROWS: each fold zeroes the whole row of 20% of the miRNAs
    in the input matrix and trains on the rest (cold-start miRNAs), so held-out associations
    are removed from the input;
  * BUT show_auc() scores the whole 495 x 383 matrix against the full m-d.txt, so 80% of the
    scored rows are training rows the model was given and learns to reconstruct;
  * m-m.txt is fixed for all folds and 73.5% of its entries equal the GIP kernel of the FULL
    association matrix, so a held-out miRNA's neighbours in the similarity graph are chosen
    from its true associations (d-d.txt: 17%; diseases are never held out);
  * negatives are every zero of the matrix (uniform).

This script reproduces fivefoldcv.py exactly (same RNG order: np.random.seed / torch.manual_seed,
then np.random.shuffle of the rows, then one model per fold) and adds, per fold:

  similarity  'published'  m-m / d-d as shipped
              'refit'      entries equal to GIP(full Y) are replaced by GIP of the fold's
                           training matrix A0 (approximation: the non-GIP part, miRNA
                           functional similarity, is itself derived from HMDD associations and
                           cannot be recomputed from the files, so a residual leak remains)
  evaluation  'full'       their show_auc: every cell, against the full matrix
              'test_rows'  only the held-out rows, all their zeros as negatives (uniform)
              'test_rows_matched'  held-out rows, one degree-matched negative per positive:
                           same miRNA, a disease in the same training-degree bin, exact sampler
                           with nearest-bin fallback (training/splits.py, as the manuscript)

and model-free scores on the same cells:
  disease_degree  training-column degree of the disease (the floor for a zeroed row)
  echo            A0 itself plus a degree tie-breaker: the input handed back unchanged; under
                  'full' this shows how much a model gets for reproducing its own input
  sim_propagation m-m @ A0 (+ degree tie-breaker): a kNN-style vote of similar miRNAs, with the
                  similarity variant of the row it is reported under

Usage:
  python analysis/nimgsa_rerun.py --nimgsa-dir PATH [--seeds 1 2 3 4 5] [--epochs 300]
Writes results/comparison/nimgsa_rerun.json (suffix _smoke when --epochs < 300).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.splits import degree_bins, sample_degree_matched_negatives_exact  # noqa: E402

# Their hyper-parameters (fivefoldcv.py argparse defaults, as published).
HIDDEN, LR, WD, ALPHA = 64, 0.01, 1e-7, 0.5


def normalized(w: np.ndarray) -> np.ndarray:  # fivefoldcv.py::normalized, verbatim logic
    deg = np.diag(np.sum(w, axis=0))
    with np.errstate(divide="ignore"):
        degpow = np.power(deg, -0.5)
    degpow[np.isinf(degpow)] = 0
    return degpow @ w @ degpow


def gip(a: np.ndarray) -> np.ndarray:
    """Gaussian interaction-profile kernel over the rows of a (gamma' = 1)."""
    sq = (a ** 2).sum(1)
    tot = sq.sum()
    g = a.shape[0] / tot if tot > 0 else 0.0
    d = sq[:, None] + sq[None, :] - 2 * a @ a.T
    return np.exp(-g * np.maximum(d, 0))


def auroc_aupr_full(ymat: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """fivefoldcv.py / utils.py::show_auc: every cell against the full matrix."""
    fpr, tpr, _ = roc_curve(y.flatten(), ymat.flatten())
    p, r, _ = precision_recall_curve(y.flatten(), ymat.flatten())
    return float(auc(fpr, tpr)), float(auc(r, p))


class GNNp(nn.Module):  # fivefoldcv.py::GNNp, with the graphs passed in instead of globals
    def __init__(self, LP, n_m, n_d, gm, gd):
        super().__init__()
        self.gnnpl = LP(HIDDEN, n_d)
        self.gnnpd = LP(HIDDEN, n_m)
        self.gm, self.gd = gm, gd

    def forward(self, y0):
        yl, zl = self.gnnpl(self.gm, y0)
        yd, zd = self.gnnpd(self.gd, y0.t())
        return yl, zl, yd, zd


def train_fold(LP, a0: torch.Tensor, gm, gd, epochs: int) -> np.ndarray:
    """fivefoldcv.py::train + trainres, same loss and same final forward pass."""
    net = GNNp(LP, a0.shape[0], a0.shape[1], gm, gd).to(a0.device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    for _ in range(epochs):
        net.train()
        yl, zl, yd, zd = net(a0)
        lpl = F.binary_cross_entropy(yl, a0)
        lpd = F.binary_cross_entropy(yd, a0.t())
        value = ALPHA * yl + (1 - ALPHA) * yd.t()
        att = torch.softmax(torch.mm(zl, zd.t()) / math.sqrt(HIDDEN), dim=-1) * value
        loss = ALPHA * lpl + (1 - ALPHA) * lpd + F.mse_loss(att, a0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            net(a0)  # their loop runs an eval forward each epoch; kept for RNG parity
    net.eval()
    with torch.no_grad():
        yl, _, yd, _ = net(a0)
    return (ALPHA * yl + (1 - ALPHA) * yd.t()).cpu().numpy()


def evaluate(score: np.ndarray, y: np.ndarray, test_rows: np.ndarray, matched_neg: np.ndarray):
    """AUROC of one score matrix under the three evaluations."""
    full = float(roc_auc_score(y.flatten(), score.flatten()))
    sub_y, sub_s = y[test_rows].flatten(), score[test_rows].flatten()
    test = float(roc_auc_score(sub_y, sub_s))
    pr, pc = np.nonzero(y[test_rows])
    pos = score[test_rows[pr], pc]
    neg = score[matched_neg[0], matched_neg[1]]
    matched = float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))
    return {"full": full, "test_rows": test, "test_rows_matched": matched}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nimgsa-dir", required=True, type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(args.nimgsa_dir))
    from models import LP  # their module, unmodified

    dev = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    nd = args.nimgsa_dir
    y = np.loadtxt(nd / "m-d.txt", delimiter=",")
    mm = np.loadtxt(nd / "m-m.txt", delimiter=",")
    dd = np.loadtxt(nd / "d-d.txt", delimiter=",")
    ours = np.loadtxt(ROOT / "data/raw/hmdd_survey/nimgsa/matrix.csv", delimiter=",")
    assert np.array_equal(y, ours), "m-d.txt differs from data/raw/hmdd_survey/nimgsa/matrix.csv"
    mm_gip_mask = np.abs(mm - gip(y)) < 1e-4
    dd_gip_mask = np.abs(dd - gip(y.T)) < 1e-4
    n_m, n_d = y.shape
    all_pos = torch.from_numpy(np.argwhere(y > 0).T.copy()).long()

    out = {"meta": {"nimgsa_commit": "5bf10a9", "epochs": args.epochs, "seeds": args.seeds,
                    "device": str(dev), "torch": torch.__version__,
                    "mm_gip_fraction": float(mm_gip_mask.mean()),
                    "dd_gip_fraction": float(dd_gip_mask.mean())},
           "runs": []}
    t0 = time.time()
    for seed in args.seeds:
        # fivefoldcv.py: set_seed(seed) at start, then shuffle inside fivefoldcv()
        np.random.seed(seed)
        torch.manual_seed(seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed(seed)
        idx = np.arange(n_m)
        np.random.shuffle(idx)
        fold_scores = {"published": [], "refit": []}
        folds = []
        for i in range(5):
            test_rows = np.sort(idx[i * n_m // 5:(i + 1) * n_m // 5])
            a0 = y.copy()
            a0[test_rows, :] = 0
            # degree-matched negatives for this fold: same miRNA, disease in the same
            # training-degree bin, never a known association (exact, nearest-bin fallback).
            # Own generator, so the torch global RNG (model init, dropout) is untouched.
            col_deg = torch.from_numpy(a0.sum(0)).long()
            bins = degree_bins(col_deg)
            pr, pc = np.nonzero(y[test_rows])
            scored = torch.from_numpy(np.vstack([test_rows[pr], pc])).long()
            neg, n_unif, n_nb = sample_degree_matched_negatives_exact(
                scored, all_pos, bins, n_d, torch.Generator().manual_seed(1000 * seed + i))
            folds.append((test_rows, a0, neg.numpy(), col_deg.numpy(), {
                "seed": seed, "fold": i + 1, "n_test_rows": int(len(test_rows)),
                "n_test_pos": int(len(pr)), "matched_uniform_fallback": n_unif,
                "matched_neighbour_fallback": n_nb}))
        # 'published' folds first, in fivefoldcv.py's order, so the torch RNG stream (model
        # init, dropout) matches theirs fold by fold; 'refit' then reseeds separately.
        for sim in ("published", "refit"):
            if sim == "refit":
                torch.manual_seed(10_000 + seed)
                if dev.type == "cuda":
                    torch.cuda.manual_seed(10_000 + seed)
            for test_rows, a0, neg, col_deg, rec in folds:
                tie = col_deg[None, :] / (col_deg.max() + 1.0) * 1e-3
                if sim == "published":
                    mm_s, dd_s = mm, dd
                else:
                    mm_s = np.where(mm_gip_mask, gip(a0), mm)
                    dd_s = np.where(dd_gip_mask, gip(a0.T), dd)
                gm = torch.from_numpy(normalized(mm_s)).float().to(dev)
                gd = torch.from_numpy(normalized(dd_s)).float().to(dev)
                a0t = torch.from_numpy(a0).float().to(dev)
                score = train_fold(LP, a0t, gm, gd, args.epochs)
                fold_scores[sim].append(score)
                rec[f"model_{sim}"] = evaluate(score, y, test_rows, neg)
                rec[f"model_{sim}"]["full_aupr"] = auroc_aupr_full(score, y)[1]
                rec[f"sim_propagation_{sim}"] = evaluate(mm_s @ a0 + tie, y, test_rows, neg)
                if sim == "published":
                    rec["disease_degree"] = evaluate(
                        np.repeat(col_deg[None, :].astype(float), n_m, 0), y, test_rows, neg)
                    rec["echo"] = evaluate(a0 + tie, y, test_rows, neg)
                else:
                    out["runs"].append(rec)
                    print(f"seed {seed} fold {rec['fold']}: published model "
                          f"full {rec['model_published']['full']:.4f} "
                          f"test {rec['model_published']['test_rows']:.4f} "
                          f"matched {rec['model_published']['test_rows_matched']:.4f} | refit "
                          f"test {rec['model_refit']['test_rows']:.4f} "
                          f"matched {rec['model_refit']['test_rows_matched']:.4f} | "
                          f"echo full {rec['echo']['full']:.4f} | fallback "
                          f"u{rec['matched_uniform_fallback']}/n{rec['matched_neighbour_fallback']} "
                          f"[{time.time() - t0:.0f}s]", flush=True)
        # their "Max result": best fold's matrix (by full AUROC), rescaled, scored on full Y
        fulls = [r["model_published"]["full"] for r in out["runs"] if r["seed"] == seed]
        best = fold_scores["published"][int(np.argmax(fulls))]
        best = (best - best.min()) / best.max()
        out.setdefault("max_result", {})[str(seed)] = auroc_aupr_full(best, y)[0]

    # summary: mean over folds per seed (their headline "Final result"), then over seeds
    keys = [("model_published", e) for e in ("full", "test_rows", "test_rows_matched")] + \
           [("model_refit", e) for e in ("full", "test_rows", "test_rows_matched")] + \
           [(m, e) for m in ("disease_degree", "echo", "sim_propagation_published",
                             "sim_propagation_refit")
            for e in ("full", "test_rows", "test_rows_matched")]
    summ = {}
    for m, e in keys:
        per_seed = [st.mean(r[m][e] for r in out["runs"] if r["seed"] == s) for s in args.seeds]
        summ[f"{m}.{e}"] = {"mean": st.mean(per_seed),
                            "sd": st.stdev(per_seed) if len(per_seed) > 1 else 0.0,
                            "per_seed": per_seed}
    out["summary"] = summ
    print("\n=== summary (mean over folds, then over seeds) ===")
    for k, v in summ.items():
        print(f"  {k:42} {v['mean']:.4f} +- {v['sd']:.4f}")
    print("  their 'Max result' per seed:", out["max_result"])
    tag = "" if args.epochs >= 300 else "_smoke"
    path = ROOT / f"results/comparison/nimgsa_rerun{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print("Wrote", path)


if __name__ == "__main__":
    main()
