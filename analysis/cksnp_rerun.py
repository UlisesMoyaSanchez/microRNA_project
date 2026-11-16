"""
cksnp_rerun.py -- CKSNP-GNN's own released code (li2022cksnpgnn, miRNA-disease), as released and corrected.

CKSNP-GNN (code.zip linked from the article's supplement, public.aibiochem.net; GAEMDA's code with CKSNAP
miRNA features; no licence, kept outside the repo) reports AUROC 0.9371 by 5-fold CV on HMDD v3.2 (901 x 877,
16,427 associations). What the released code does:

  * sample(): 16,427 uniform negatives from all unknown pairs (fixed random_state=2008), pooled with the
    positives and split by KFold(5, shuffle) over pairs;
  * build_graph(): EVERY sampled pair, positive or negative, is an edge (both directions); training propagates
    over the training pairs, the test score over the full graph, so test positives and test negatives alike
    are edges at test time;
  * node features: miRNA = CKSNAP5-Pearson sequence similarity (dense, no GIP); disease = MeSH semantic
    similarity where non-zero (3.9% of entries), otherwise a Gaussian interaction-profile (GIP) kernel computed
    once (data/rbf.py) from the full association matrix, test associations included;
  * 100 epochs x 10 full-batch Adam steps; the last epoch is reported; each test pair is scored in both
    edge directions (test_eid holds both).

Steps (per seed and fold; cksnp_rerun_mx.py runs their code in a Python 3.7 / mxnet 1.6 environment):
  released            their features, scored over their full graph
  released_gtrain     their features, scored over the training-pair graph
  foldgip             GIP recomputed from the fold's training positives, full graph
  foldgip_gtrain      ... over the training-pair graph
  + matched_disease / matched_mirna   the foldgip_gtrain model against negatives that keep the miRNA (disease)
                      and draw a disease (miRNA) of matching training degree; pairs that are positives or
                      training negatives are excluded
protocheck 0.2.0 audits each scored set in both orientations; encoder input = the edges the model propagates over.

Usage (driven by training/slurm_cksnp_rerun.sh):
  python analysis/cksnp_rerun.py --prep --seed S --work DIR     (after cksnp_rerun_mx.py --dump)
  python analysis/cksnp_rerun.py --report --seeds S1 S2 ... --work DIR
Writes results/comparison/cksnp_rerun.json (suffix _smoke with --smoke).
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

N_M, N_D = 901, 877


def loc(p):
    """their (miRNA id, disease id), 1-based -> (k, 2) local (miRNA, disease)."""
    return np.c_[p[:, 0] - 1, p[:, 1] - 1].astype(np.int64)


def prep(seed: int, work: Path) -> None:
    f = np.load(work / f"cksnp_s{seed}_folds.npz")
    S = f["samples"]
    allpos = loc(S[S[:, 2] == 1])
    out = {}
    g = torch.Generator().manual_seed(1000 * seed)
    for i in range(5):
        tr = S[f[f"train_{i}"]]
        te = S[f[f"test_{i}"]]
        trp, trn, tep = loc(tr[tr[:, 2] == 1]), loc(tr[tr[:, 2] == 0]), loc(te[te[:, 2] == 1])
        excl = np.r_[allpos, trn]
        deg_m = torch.from_numpy(np.bincount(trp[:, 0], minlength=N_M)).long()
        deg_d = torch.from_numpy(np.bincount(trp[:, 1], minlength=N_D)).long()
        T = lambda a: torch.from_numpy(np.ascontiguousarray(a.T)).long()  # noqa: E731
        md = sample_degree_matched_negatives_exact(T(tep), T(excl), degree_bins(deg_d), N_D, g)[0].numpy().T
        mm = sample_degree_matched_negatives_exact(T(tep[:, ::-1]), T(excl[:, ::-1]), degree_bins(deg_m), N_M,
                                                   g)[0].numpy().T[:, ::-1]
        out[f"disease_{i}"] = md + 1          # back to their 1-based ids
        out[f"mirna_{i}"] = np.ascontiguousarray(mm) + 1
    np.savez(work / f"cksnp_s{seed}_matched.npz", **out)
    print("prepared matched negatives, seed", seed)


def audit_both(tr, te, trn, enc, neg, auc_v):
    out = {}
    for name, (rows, cols, f) in {"mirna_x_disease": (N_M, N_D, lambda a: a.T),
                                  "disease_x_mirna": (N_D, N_M, lambda a: a[:, ::-1].T)}.items():
        rep = audit(rows, cols, f(tr), f(te), encoder_edges=f(enc), train_neg=f(trn), eval_neg=f(neg),
                    model_auroc=auc_v)
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


STEPS = [("released", "g", "theirs"), ("released", "gtrain", "theirs"), ("foldgip", "g", "theirs"),
         ("foldgip", "gtrain", "theirs"), ("foldgip", "gtrain", "matched_disease"),
         ("foldgip", "gtrain", "matched_mirna")]


def report(seeds, work: Path, smoke: bool) -> None:
    res = {"meta": {"seeds": seeds, "n_mirna": N_M, "n_disease": N_D}, "runs": []}
    for seed in seeds:
        f = np.load(work / f"cksnp_s{seed}_folds.npz")
        m = np.load(work / f"cksnp_s{seed}_matched.npz")
        sc = np.load(work / f"cksnp_s{seed}_scores.npz")
        S = f["samples"]
        for i in range(5):
            tr, te = S[f[f"train_{i}"]], S[f[f"test_{i}"]]
            trp, trn, tep = loc(tr[tr[:, 2] == 1]), loc(tr[tr[:, 2] == 0]), loc(te[te[:, 2] == 1])
            ten = loc(te[te[:, 2] == 0])
            negs = {"theirs": ten, "matched_disease": loc(m[f"disease_{i}"]), "matched_mirna": loc(m[f"mirna_{i}"])}
            rec = {"seed": seed, "fold": i + 1, "their_path_released": float(sc[f"released_their_path_auc_{i}"]),
                   "their_path_foldgip": float(sc[f"foldgip_their_path_auc_{i}"])}
            for feat, gname, nset in STEPS:
                p, n = sc[f"{feat}_{gname}_pos_{i}"], sc[f"{feat}_{gname}_{nset}_{i}"]
                v = float(roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n]))
                enc = np.r_[trp, trn] if gname == "gtrain" else np.r_[trp, trn, tep, ten]
                key = f"{feat}_{gname}_{nset}"
                rec[key] = {"auc": v, "audit": audit_both(trp, tep, trn, enc, negs[nset], v)}
            res["runs"].append(rec)
            print(f"seed {seed} fold {i + 1}: " + " | ".join(
                f"{k} {rec[k]['auc']:.4f}/{rec[k]['audit']['floor']:.4f}" for k in rec if isinstance(rec[k], dict)),
                flush=True)

    def mm(get):
        v = [get(r) for r in res["runs"]]
        return {"mean": st.mean(v), "sd": st.stdev(v) if len(v) > 1 else 0.0}
    keys = ["_".join(s) for s in STEPS]
    res["summary"] = {"their_path_released": mm(lambda r: r["their_path_released"]),
                      **{k: mm(lambda r, k=k: r[k]["auc"]) for k in keys},
                      **{f"floor_{k}": mm(lambda r, k=k: r[k]["audit"]["floor"]) for k in keys}}
    print(json.dumps(res["summary"], indent=1))
    path = ROOT / f"results/comparison/cksnp_rerun{'_smoke' if smoke else ''}.json"
    path.write_text(json.dumps(res, indent=2, default=float))
    print("Wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.prep:
        prep(a.seed, a.work)
    if a.report:
        report(a.seeds, a.work, a.smoke)
