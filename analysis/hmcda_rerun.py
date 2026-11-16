"""
hmcda_rerun.py -- HMCDA's own released code (liang2023hmcda, circRNA-disease), as released and corrected.

HMCDA (github.com/shiyangl/HMCDA @ 4d9e8bb, no licence, kept outside the repo; a MAGNN adaptation) reports
AUROC 0.9135 by "fivefold cross-validation" with five training negatives per positive. What its released
code does (preprocess_HMCDA.ipynb, run.HMCDA.py, data/raw/HMCDA/gendata.py):

  * circrna_disease.dat has 2,160 rows but 1,992 distinct pairs (168 duplicate rows); splits index rows, so a
    pair can sit in training and test at once: in the one released split (example5.npz) 56 of 432 test
    positives are also training rows;
  * the metapath graph (c-d-c, c-m-c, d-c-d, d-m-d, d-d) is built from the training rows only, so held-out
    pairs are absent except through those duplicates; the d-d metapath never matches (its rows carry no node
    offset) and is empty;
  * the "fivefold CV" is five 70/10/20 rotations of a shuffled index list (gendata.py, unseeded, drops row
    2159); only example5.npz is released and the notebook hard-codes it;
  * negatives: the exclusion loop assumes the positives are sorted; they are not, so almost every positive
    stays in both candidate pools (about 3 expected among 648 validation/test negatives); validation and test
    negatives uniform 1:1; training draws one negative per positive per batch (the text says five);
  * early stopping on validation loss (patience 5), the best-validation checkpoint scored on test.

Steps (analysis/hmcda_rerun_mg.py runs their code in a Python 3.6 / PyTorch 1.2 / DGL 0.3.1 environment):
  released          their split, rows, negatives and loop
  dedup             distinct pairs; a pair held out for test (then validation) is removed from training;
                    validation/test negatives drawn from true non-positives; training candidates = all but the
                    training positives (their intent)
  matched_disease / matched_circrna   the dedup model against negatives that keep the circRNA (disease) and draw
                    a disease (circRNA) of matching training degree; known positives excluded
Splits: ex5 (released; seeds 1-5) and rot1..rot5 (their gendata rotations under random.seed(0); seed 1).
protocheck 0.2.0 audits each scored set in both orientations; encoder input = the training circRNA-disease pairs.

Usage (driven by training/slurm_hmcda_rerun.sh):
  python analysis/hmcda_rerun.py --prep --split S --work DIR          (after hmcda_rerun_mg.py --dump)
  python analysis/hmcda_rerun.py --report --work DIR
Writes results/comparison/hmcda_rerun.json (suffix _smoke with --smoke).
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

N_C, N_D = 1556, 243
SPLITS = {"ex5": [1, 2, 3, 4, 5], **{f"rot{r}": [1] for r in range(1, 6)}}


def prep(split: str, work: Path) -> None:
    D = np.load(work / f"hmcda_{split}_data.npz")
    tr, te, allpos = D["dedup_train_pos"], D["dedup_test_pos"], D["all_pos"]
    g = torch.Generator().manual_seed(7)
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a.T)).long()  # noqa: E731
    deg_c = torch.from_numpy(np.bincount(tr[:, 0], minlength=N_C)).long()
    deg_d = torch.from_numpy(np.bincount(tr[:, 1], minlength=N_D)).long()
    md = sample_degree_matched_negatives_exact(T(te), T(allpos), degree_bins(deg_d), N_D, g)[0].numpy().T
    mc = sample_degree_matched_negatives_exact(T(te[:, ::-1]), T(allpos[:, ::-1]), degree_bins(deg_c), N_C,
                                               g)[0].numpy().T[:, ::-1]
    np.savez(work / f"hmcda_{split}_matched.npz", disease=np.ascontiguousarray(md), circrna=np.ascontiguousarray(mc))
    print("prepared matched negatives", split, md.shape, mc.shape)


def audit_both(tr, te, enc, neg, auc_v):
    out = {}
    for name, (rows, cols, f) in {"circrna_x_disease": (N_C, N_D, lambda a: a.T),
                                  "disease_x_circrna": (N_D, N_C, lambda a: a[:, ::-1].T)}.items():
        rep = audit(rows, cols, f(tr), f(te), encoder_edges=f(enc), eval_neg=f(neg), model_auroc=auc_v)
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


STEPS = [("released", "theirs"), ("dedup", "theirs"), ("dedup", "matched_disease"), ("dedup", "matched_circrna")]


def report(work: Path, smoke: bool) -> None:
    res = {"meta": {"splits": SPLITS}, "runs": []}
    for split, seeds in SPLITS.items():
        D = np.load(work / f"hmcda_{split}_data.npz")
        M = np.load(work / f"hmcda_{split}_matched.npz")
        for seed in seeds:
            rec = {"split": split, "seed": seed}
            for variant, nset in STEPS:
                f = work / f"hmcda_{split}_{variant}_s{seed}_scores.npz"
                if not f.exists():
                    continue
                sc = np.load(f)
                p, n = sc["pos"], sc[nset]
                v = float(roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n]))
                tr = np.unique(D[f"{variant}_train_pos"], axis=0)
                te = np.unique(D[f"{variant}_test_pos"], axis=0)
                neg = D[f"{variant}_test_neg"] if nset == "theirs" else M[nset.split("_")[1]]
                rec[f"{variant}_{nset}"] = {"auc": v, "epochs": int(sc["epochs"]),
                                            "audit": audit_both(tr, te, tr, neg, v)}
            if split == "ex5":
                rec["test_rows_also_training"] = int(np.isin(
                    D["released_test_pos"][:, 0] * N_D + D["released_test_pos"][:, 1],
                    D["released_train_pos"][:, 0] * N_D + D["released_train_pos"][:, 1]).sum())
            res["runs"].append(rec)
            print(f"{split} seed {seed}: " + " | ".join(
                f"{k} {rec[k]['auc']:.4f}/{rec[k]['audit']['floor']:.4f}" for k in rec if isinstance(rec[k], dict)),
                flush=True)

    def mm(runs, get):
        v = [get(r) for r in runs]
        return {"mean": st.mean(v), "sd": st.stdev(v) if len(v) > 1 else 0.0, "n": len(v)}
    res["summary"] = {}
    for grp, runs in (("ex5_seeds", [r for r in res["runs"] if r["split"] == "ex5"]),
                      ("rotations", [r for r in res["runs"] if r["split"].startswith("rot")])):
        keys = [f"{v}_{n}" for v, n in STEPS if all(f"{v}_{n}" in r for r in runs) and runs]
        res["summary"][grp] = {**{k: mm(runs, lambda r, k=k: r[k]["auc"]) for k in keys},
                               **{f"floor_{k}": mm(runs, lambda r, k=k: r[k]["audit"]["floor"]) for k in keys}}
    print(json.dumps(res["summary"], indent=1))
    path = ROOT / f"results/comparison/hmcda_rerun{'_smoke' if smoke else ''}.json"
    path.write_text(json.dumps(res, indent=2, default=float))
    print("Wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--split")
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.prep:
        prep(a.split, a.work)
    if a.report:
        report(a.work, a.smoke)
