"""
ngcn_rerun.py -- NGCN's own released code (cao2024ngcn, drug-target), as released and corrected step by step.

NGCN (github.com/Junyue28/NGCN, NGCN.rar + data.rar; no licence, kept outside the repo) reports AUROC
0.910 on Luo et al.'s DTINet data with 1:1 negatives (its Table 1). What the released main.py does:

  * split_interaction_file: uniform negatives (ratio x #positives) drawn from every unknown pair, then
    80/20 train_test_split of positives and negatives separately (one split per seed, not 10-fold CV);
  * EvaluateModule.forward: sigmoid(drug_feat @ D_W @ P_W @ protein_feat^T) on PRECOMPUTED feature files
    (feature/drug_vector_d200.txt, protein_vector_d400.txt); the adjacency arguments are unused, so the
    spectral GCN feature update the paper describes is not in the released model;
  * the test AUROC is computed every epoch for 3000 epochs and printed every 5; there is no validation set.

The same 708 x 1512 matrix as HGDTI: 1,088 proteins have no interaction, so a uniform negative usually
sits on a protein that a degree score already rules out.

Steps (per seed, NGCN's own split function and model class, imported unmodified):
  1 released      full training mask, 3000 epochs: test AUROC at the last epoch and at the best epoch
                  (what reading the printed test curve gives)
  2 val_selected  1/8 of the training positives and negatives held out as validation (removed from the
                  loss mask); epoch chosen by validation AUROC; scored on their test set (uniform negatives)
  3 matched       the step-2 model at its validation epoch, scored on the same test positives against
                  degree-matched negatives (exact sampler, protein degree from the fitting positives)
  4 dtinet_feat   steps 2-3 with DTINet's own released features (Luo et al.; run_DCA.m builds them from the
                  drug-drug, drug-disease, drug-side-effect, drug-similarity and protein-protein,
                  protein-disease, protein-similarity networks only, no drug-target edge), to check that
                  NGCN's undocumented feature files carry no drug-target information
protocheck 0.2.0 audits steps 2-3 (visible edges = the fitting positives; the model's input holds no
drug-target edge, so encoder_edges = the fitting positives as well).

Usage: python analysis/ngcn_rerun.py --repo PATH_TO_NGCN --dtinet PATH_TO_DTINet --ratio 1 [--seeds ...]
Writes results/comparison/ngcn_rerun_r<ratio>.json (suffix _smoke when --epochs < 3000).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from protocheck.audit import audit  # noqa: E402
from training.splits import degree_bins, sample_degree_matched_negatives_exact  # noqa: E402


def load_their_modules(repo: Path):
    """main.py imports tensorboard_logger at module level; it only logs, so a stub stands in."""
    tb = types.ModuleType("tensorboard_logger")
    tb.Logger = lambda *a, **k: types.SimpleNamespace(log_value=lambda *a, **k: None)
    sys.modules.setdefault("tensorboard_logger", tb)
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("ngcn_main", repo / "main.py")
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)  # defines split_interaction_file, evaluate; __main__ does not run
    from model.model import EvaluateModule
    from utils.SSELoss import MyLoss
    return main, EvaluateModule, MyLoss


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def train_curve(EvaluateModule, MyLoss, feats, Y, mask, evals, epochs, seed, dev):
    """Their training loop (Adam lr 1e-3, weight decay 0.1, SSE on the masked matrix); returns the AUROC
    of every epoch on each evaluation set {name: (idx (k,2), labels)}."""
    np.random.seed(seed)          # their D_weight / P_weight are drawn with np.random
    torch.manual_seed(seed)
    dfe, pfe, dd, pp = feats
    model = EvaluateModule(dfe.shape[0], dfe.shape[1], 200, pfe.shape[0], pfe.shape[1], 400, 1000).to(dev)
    crit = MyLoss().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.1)
    T = lambda a: torch.from_numpy(a).float().to(dev)  # noqa: E731
    dfe_t, pfe_t, dd_t, pp_t, Y_t, M_t = T(dfe), T(pfe), T(dd), T(pp), T(Y), T(mask)
    curves = {k: [] for k in evals}
    for _ in range(epochs):
        out = model(dd_t, dfe_t, pp_t, pfe_t)
        loss = crit(out * M_t, Y_t * M_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        o = out.detach().cpu().numpy()
        for k, (idx, lab) in evals.items():
            curves[k].append(auc(lab, o[idx[:, 0], idx[:, 1]]))
    return curves


def audit_step(fit_pos, te_pos, fit_neg, te_neg, model_auc, n_d, n_p):
    rep = audit(n_d, n_p, fit_pos.T, te_pos.T, encoder_edges=fit_pos.T, train_neg=fit_neg.T,
                eval_neg=te_neg.T, model_auroc=model_auc)
    out = {}
    for f in rep.findings:
        ev = f.evidence or {}
        out[f.check] = {"status": f.status, "message": f.message}
        if f.check == "model_free_baseline":
            out[f.check].update(floor=ev.get("reference_floor"), heuristic=ev.get("reference_heuristic"),
                                margin=ev.get("margin"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--dtinet", required=True, type=Path)
    ap.add_argument("--ratio", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=3000)
    a = ap.parse_args()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main_mod, EvaluateModule, MyLoss = load_their_modules(a.repo)
    data = a.repo / "data"
    Y = np.loadtxt(data / "mat_drug_protein.txt")
    n_d, n_p = Y.shape
    dd, pp = np.loadtxt(data / "mat_drug_drug.txt"), np.loadtxt(data / "mat_protein_protein.txt")
    feat_sets = {
        "ngcn": (np.loadtxt(a.repo / "feature/drug_vector_d200.txt"), np.loadtxt(a.repo / "feature/protein_vector_d400.txt")),
        "dtinet": (np.loadtxt(a.dtinet / "feature/drug_vector_d100.txt"), np.loadtxt(a.dtinet / "feature/protein_vector_d400.txt")),
    }
    all_pos = torch.from_numpy(np.argwhere(Y == 1).T.copy()).long()
    out = {"meta": {"ratio": a.ratio, "seeds": a.seeds, "epochs": a.epochs, "device": str(dev),
                    "n_dti": int(Y.sum()), "proteins_without_dti": int((Y.sum(0) == 0).sum())}, "runs": []}
    t0 = time.time()
    for seed in a.seeds:
        train_P, test_idx, test_y = main_mod.split_interaction_file(ratio=a.ratio, seed=seed, data_dir=str(data))
        tr_idx = np.argwhere(train_P == 1)
        tr_lab = Y[tr_idx[:, 0], tr_idx[:, 1]]
        tr_pos, tr_neg = tr_idx[tr_lab == 1], tr_idx[tr_lab == 0]
        te_pos, te_neg = test_idx[test_y == 1], test_idx[test_y == 0]
        rng = np.random.default_rng(10_000 + seed)
        va_p = rng.choice(len(tr_pos), len(tr_pos) // 8, replace=False)
        va_n = rng.choice(len(tr_neg), len(tr_neg) // 8, replace=False)
        fit_pos, fit_neg = np.delete(tr_pos, va_p, 0), np.delete(tr_neg, va_n, 0)
        va_idx = np.r_[tr_pos[va_p], tr_neg[va_n]]
        va_y = np.r_[np.ones(len(va_p)), np.zeros(len(va_n))]
        fit_mask = np.zeros_like(Y)
        fit_mask[fit_pos[:, 0], fit_pos[:, 1]] = 1
        fit_mask[fit_neg[:, 0], fit_neg[:, 1]] = 1
        col_deg = torch.from_numpy(np.bincount(fit_pos[:, 1], minlength=n_p)).long()
        mneg, n_unif, n_nb = sample_degree_matched_negatives_exact(
            torch.from_numpy(te_pos.T.copy()).long(), all_pos, degree_bins(col_deg), n_p,
            torch.Generator().manual_seed(1000 * seed))
        mneg = mneg.numpy().T
        m_idx = np.r_[te_pos, mneg]
        m_y = np.r_[np.ones(len(te_pos)), np.zeros(len(mneg))]
        rec = {"seed": seed, "n_test_pos": int(len(te_pos)), "n_test_neg": int(len(te_neg)),
               "matched_uniform_fallback": n_unif, "matched_neighbour_fallback": n_nb}

        f = feat_sets["ngcn"]
        c1 = train_curve(EvaluateModule, MyLoss, (f[0], f[1], dd, pp), Y, train_P,
                         {"test": (test_idx, test_y)}, a.epochs, seed, dev)["test"]
        rec["released"] = {"last": c1[-1], "best": max(c1), "best_epoch": int(np.argmax(c1)) + 1}
        for name, (dfe, pfe) in feat_sets.items():
            c = train_curve(EvaluateModule, MyLoss, (dfe, pfe, dd, pp), Y, fit_mask,
                            {"val": (va_idx, va_y), "test": (test_idx, test_y), "matched": (m_idx, m_y)},
                            a.epochs, seed, dev)
            e = int(np.argmax(c["val"]))
            r = {"epoch": e + 1, "val": c["val"][e], "uniform": c["test"][e], "matched": c["matched"][e],
                 "uniform_best_test_epoch": max(c["test"])}
            r["audit_uniform"] = audit_step(fit_pos, te_pos, fit_neg, te_neg, r["uniform"], n_d, n_p)
            r["audit_matched"] = audit_step(fit_pos, te_pos, fit_neg, mneg, r["matched"], n_d, n_p)
            rec[f"val_selected_{name}"] = r
        out["runs"].append(rec)
        g, d = rec["val_selected_ngcn"], rec["val_selected_dtinet"]
        print(f"seed {seed}: released last {c1[-1]:.4f} best {max(c1):.4f} (ep {rec['released']['best_epoch']}) | "
              f"val-selected uniform {g['uniform']:.4f} (floor {g['audit_uniform']['model_free_baseline']['floor']:.4f}) "
              f"matched {g['matched']:.4f} (floor {g['audit_matched']['model_free_baseline']['floor']:.4f}) | "
              f"DTINet features {d['uniform']:.4f} / {d['matched']:.4f} [{time.time() - t0:.0f}s]", flush=True)

    def m(get):
        v = [get(r) for r in out["runs"]]
        return {"mean": st.mean(v), "sd": st.stdev(v) if len(v) > 1 else 0.0}
    out["summary"] = {
        "released_last": m(lambda r: r["released"]["last"]),
        "released_best": m(lambda r: r["released"]["best"]),
        **{f"{k}_{t}": m(lambda r, k=k, t=t: r[f"val_selected_{k}"][t])
           for k in feat_sets for t in ("uniform", "matched")},
        **{f"floor_{t}": m(lambda r, t=t: r["val_selected_ngcn"][f"audit_{t}"]["model_free_baseline"]["floor"])
           for t in ("uniform", "matched")},
    }
    print("\n=== summary ===\n", json.dumps(out["summary"], indent=1))
    tag = f"_r{a.ratio}" + ("" if a.epochs >= 3000 else "_smoke")
    path = ROOT / f"results/comparison/ngcn_rerun{tag}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print("Wrote", path)


if __name__ == "__main__":
    main()
