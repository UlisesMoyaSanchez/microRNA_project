"""
higldp_rerun.py -- HiGLDP's own released code (wang2026higldp, lncRNA-disease), as released and corrected.

HiGLDP (github.com/St0neWang/HiGLDP @ 9808ba8 = Zenodo 10.5281/zenodo.18522355, no licence, kept outside the repo)
reports AUROC 0.9696 by fivefold CV. It classifies lncRNA-disease *pairs* as nodes (22,622: 11,311 known
associations and 11,311 negatives). What its released code does:

  * negatives: a fixed set of 11,311 unknown pairs whose mean lncRNA and disease degrees (5.27, 56.2) match a
    uniform draw over all unknown pairs (5.41, 56.8), against 29.2 and 185.7 for the positives; the text gives
    only the 1:1 ratio;
  * the structural graph (5fold/edgeX.txt) joins every two *training* pair-nodes that share a lncRNA or a
    disease, labels unused; test pair-nodes are isolated in it; the association feature graph (afg1) is a
    cosine 1-NN graph over all 22,622 pair features;
  * the pair features concatenate a lncRNA and a disease vector (RWR over similarity networks, then a denoising
    autoencoder); one of the four lncRNA networks is the Gaussian interaction-profile kernel of the *full*
    lncRNA-disease matrix (similarity_generate.m: GIP(mat_Lnc_DO)), test associations included;
  * main.py trains 200 full-batch epochs, scores the test nodes every epoch, and reports the maximum test AUROC
    over the epochs (and, separately, the maximum test AUPR); 122 pairs are never tested.

Steps (analysis/higldp_rerun_run.py, env higldp_py310):
  released / ported / foldfeat / validation -- see that file; scored sets:
    theirs            test positives vs their test negatives
    matched_disease   (validation model) negatives keeping each test positive's lncRNA, disease drawn from the
                      same training-degree bin (exact sampler); matched_lncrna the reverse; known positives excluded
  epochs: max over test (their report), last (200), and, for validation, the epoch with the best validation AUROC.
protocheck 0.2.0 audits each scored set in both orientations; encoder input = the associations the GIP kernel
saw (all for released/ported, training for foldfeat/validation).

Usage (driven by training/slurm_higldp_rerun.sh):
  python analysis/higldp_rerun.py --prep --repo R --work W
  python analysis/higldp_rerun.py --report --repo R --work W [--smoke]
Writes results/comparison/higldp_rerun.json (suffix _smoke with --smoke).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from protocheck.audit import audit  # noqa: E402
from training.splits import degree_bins, sample_degree_matched_negatives_exact  # noqa: E402

N_L, N_D, N_PAIRS = 1821, 185, 22622
FOLDS = [1, 2, 3, 4, 5]
STEPS = ["released", "ported", "foldfeat", "validation"]


def load(repo: Path):
    index = np.loadtxt(repo / "Dataset/data/index.txt", dtype=np.int64)
    labels = np.loadtxt(repo / "Dataset/data/label.txt", dtype=np.int64)
    return index, labels


def prep(repo: Path, work: Path) -> None:
    index, labels = load(repo)
    allpos = index[labels == 1][:, 1:]
    T = lambda x: torch.from_numpy(np.ascontiguousarray(x.T)).long()  # noqa: E731
    for f in FOLDS:
        tr = np.loadtxt(repo / f"5fold/train{f}.txt", dtype=np.int64)
        te = np.loadtxt(repo / f"5fold/test{f}.txt", dtype=np.int64)
        rng = np.random.default_rng(f)
        val = np.sort(rng.choice(tr, size=len(tr) // 8, replace=False))
        fit = np.setdiff1d(tr, val)
        fit_pos = index[fit][labels[fit] == 1][:, 1:]
        te_pos = index[te][labels[te] == 1][:, 1:]
        g = torch.Generator().manual_seed(7 + f)
        deg_l = torch.from_numpy(np.bincount(fit_pos[:, 0], minlength=N_L)).long()
        deg_d = torch.from_numpy(np.bincount(fit_pos[:, 1], minlength=N_D)).long()
        md = sample_degree_matched_negatives_exact(T(te_pos), T(allpos), degree_bins(deg_d), N_D, g)[0].numpy().T
        ml = sample_degree_matched_negatives_exact(T(te_pos[:, ::-1]), T(allpos[:, ::-1]), degree_bins(deg_l), N_L,
                                                   g)[0].numpy().T[:, ::-1]
        np.savez(work / f"higldp_fold{f}_split.npz", train=tr, test=te, val=val,
                 extra_pairs=np.ascontiguousarray(np.r_[md, ml]), n_md=len(md), n_ml=len(ml))
        print(f"fold {f}: train {len(tr)} (val {len(val)}), test {len(te)}, matched {len(md)}+{len(ml)}")


def audit_both(tr, te, enc, neg, auc_v):
    out = {}
    for name, (rows, cols, fn) in {"lncrna_x_disease": (N_L, N_D, lambda a: a.T),
                                   "disease_x_lncrna": (N_D, N_L, lambda a: a[:, ::-1].T)}.items():
        rep = audit(rows, cols, fn(tr), fn(te), encoder_edges=fn(enc), eval_neg=fn(neg), model_auroc=auc_v)
        o = {}
        for fi in rep.findings:
            ev = fi.evidence or {}
            o[fi.check] = {"status": fi.status, "message": fi.message}
            if fi.check == "model_free_baseline":
                o[fi.check].update(floor=ev.get("reference_floor"), heuristic=ev.get("reference_heuristic"),
                                   margin=ev.get("margin"))
        out[name] = o
    best = max(out, key=lambda k: out[k]["model_free_baseline"]["floor"])
    out["floor"] = out[best]["model_free_baseline"]["floor"]
    out["floor_heuristic"] = f"{out[best]['model_free_baseline']['heuristic']} ({best})"
    return out


def aucs(scores, ypos, yneg):
    """per-epoch AUROC of scores[:, ypos] (positives) against scores[:, yneg]."""
    y = np.r_[np.ones(len(ypos)), np.zeros(len(yneg))]
    return np.array([roc_auc_score(y, np.r_[s[ypos], s[yneg]]) for s in scores])


def report(repo: Path, work: Path, smoke: bool) -> None:
    index, labels = load(repo)
    allpos = index[labels == 1][:, 1:]
    res = {"meta": {"folds": FOLDS, "steps": STEPS}, "runs": []}
    for f in FOLDS:
        S = np.load(work / f"higldp_fold{f}_split.npz")
        tr, te, val, extra = S["train"], S["test"], S["val"], S["extra_pairs"]
        n_md = int(S["n_md"])
        te_pos, te_neg = index[te][labels[te] == 1][:, 1:], index[te][labels[te] == 0][:, 1:]
        seeds = sorted({int(p.stem.split("_s")[-1]) for p in work.glob(f"higldp_fold{f}_*_s*.npz")})
        for seed in seeds:
            rec = {"fold": f, "seed": seed}
            for step in STEPS:
                p = work / f"higldp_fold{f}_{step}_s{seed}.npz"
                if not p.exists():
                    continue
                Z = np.load(p)
                sc, nodes = Z["scores"], Z["eval_nodes"]
                pos_i = np.flatnonzero(np.isin(nodes, te) & (labels[np.minimum(nodes, N_PAIRS - 1)] == 1)
                                       & (nodes < N_PAIRS))
                neg_i = np.flatnonzero(np.isin(nodes, te) & (labels[np.minimum(nodes, N_PAIRS - 1)] == 0)
                                       & (nodes < N_PAIRS))
                fit = np.setdiff1d(tr, val) if step == "validation" else tr
                fit_pos = index[fit][labels[fit] == 1][:, 1:]
                enc = allpos if step in ("released", "ported") else fit_pos
                curve = aucs(sc, pos_i, neg_i)
                r = {"max_test": float(curve.max()), "max_test_epoch": int(curve.argmax()),
                     "last": float(curve[-1]), "epochs": len(curve)}
                sel = len(curve) - 1
                if step == "validation":
                    vpos = np.flatnonzero(np.isin(nodes, val) & (labels[np.minimum(nodes, N_PAIRS - 1)] == 1))
                    vneg = np.flatnonzero(np.isin(nodes, val) & (labels[np.minimum(nodes, N_PAIRS - 1)] == 0))
                    vcurve = aucs(sc, vpos, vneg)
                    sel = int(vcurve.argmax())
                    r.update(selected=float(curve[sel]), selected_epoch=sel, val_auc=float(vcurve[sel]))
                    xi = np.flatnonzero(nodes >= N_PAIRS)
                    for name, sl, negs in (("matched_disease", xi[:n_md], extra[:n_md]),
                                           ("matched_lncrna", xi[n_md:], extra[n_md:])):
                        y = np.r_[np.ones(len(pos_i)), np.zeros(len(sl))]
                        v = float(roc_auc_score(y, np.r_[sc[sel, pos_i], sc[sel, sl]]))
                        rec[f"{step}_{name}"] = {"auc": v, "audit": audit_both(fit_pos, te_pos, enc, negs, v)}
                r["audit"] = audit_both(fit_pos, te_pos, enc, te_neg, float(curve[sel]))
                rec[step] = r
            res["runs"].append(rec)
            print(f"fold {f} seed {seed}: " + " | ".join(
                f"{k} {v.get('max_test', v.get('auc')):.4f}"
                + (f"/last {v['last']:.4f}" if "last" in v else "")
                + (f"/sel {v['selected']:.4f}" if "selected" in v else "")
                + f" floor {v['audit']['floor']:.4f}"
                for k, v in rec.items() if isinstance(v, dict)), flush=True)

    def mm(vals):
        return {"mean": st.mean(vals), "sd": st.stdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)}
    summ = {}
    for step in STEPS:
        rs = [r[step] for r in res["runs"] if step in r]
        if not rs:
            continue
        summ[step] = {k: mm([r[k] for r in rs]) for k in ("max_test", "last") }
        summ[step]["floor"] = mm([r["audit"]["floor"] for r in rs])
        if step == "validation":
            summ[step]["selected"] = mm([r["selected"] for r in rs])
    for k in ("validation_matched_disease", "validation_matched_lncrna"):
        rs = [r[k] for r in res["runs"] if k in r]
        if rs:
            summ[k] = {"auc": mm([r["auc"] for r in rs]), "floor": mm([r["audit"]["floor"] for r in rs])}
    res["summary"] = summ
    print(json.dumps(summ, indent=1))
    path = ROOT / f"results/comparison/higldp_rerun{'_smoke' if smoke else ''}.json"
    path.write_text(json.dumps(res, indent=2, default=float))
    print("Wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.prep:
        prep(a.repo, a.work)
    if a.report:
        report(a.repo, a.work, a.smoke)
