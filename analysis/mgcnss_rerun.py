"""
mgcnss_rerun.py -- MGCNSS's own released code (tian2024mgcnss, miRNA-disease), as released and corrected step by
step.

MGCNSS (github.com/15136943622/MGCNSS @ f8a87e6, no licence, kept outside the repo) reports AUROC 0.9874 by
5-fold CV on the canonical HMDD matrix (495 x 383, 5,430 associations), with negatives chosen by its own
distance-based selector. What its released code does (Link_Prediction.py, src/Model.py,
src/link_prediction_evaluate.py; src/Utils.py ships only as bytecode, and its get_model, read with Python 3.6's
dis, is `MHGCN(nfeat, nhid, out, dropout, stdv, layer)` -- provided here as a module):

  * the network the GCN propagates over is result.mat's raw_association, all 5,430 associations, test included;
    its disease->miRNA block is built as fliplr(flipud(o_ass.T)), which reverses both axes;
  * of the similarity layers it fuses, A[1] (miRNA) and B[0] (disease) are exactly the Gaussian interaction-profile
    kernel of the full association matrix (max |diff| 0); the node features (Y.csv, a random walk with restart)
    carry a non-zero disease->miRNA block, so they too were computed with association edges;
  * the fold files store positives 1-based and negatives 0-based; the code reads both as 1-based, so a negative's
    miRNA i is scored as miRNA i-1 (miRNA 0: the last disease's row in training, skipped in evaluation) and its
    disease j as node j+494 (disease 0: miRNA 494);
  * full-batch training on the fold's training pairs (Adam lr 5e-4, wd 5e-4, re-created every epoch; README:
    2,000 epochs, embedding 256, 2 layers), test AUROC every epoch, and the epoch with the best test AUROC
    reported (predict_model).

Steps (per fold of the five released folds and seed):
  1 released     as above; test-selected epoch (and the last)
  2 index        negatives read with their real (0-based) indices
  3 removed      + every test positive removed from the propagated network
  4 foldsim      + A[1], B[0] recomputed (their GIP formula) from the training network; the features'
                 disease->miRNA block zeroed (the RWR that made them is not released)
  5 validation   + one eighth of the training pairs held out (their positives also removed from the network and
                 the kernels); epoch chosen by validation AUROC
  6 uniform      the step-5 model on uniform negatives over every unknown pair (the paper's 0.9437 is its
                 selector applied to training only)
  7/8 matched    the step-5 model on degree-matched negatives (miRNA kept and disease from the same training-degree
                 bin, and the reverse; exact sampler)
protocheck 0.2.0 audits each scored set in both orientations; encoder input = the propagated network.

Usage: python analysis/mgcnss_rerun.py --repo PATH [--seeds 1 2 3] [--epochs 2000]
Writes results/comparison/mgcnss_rerun.json (suffix _smoke when --epochs < 2000).
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics as st
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from protocheck.audit import audit  # noqa: E402
from training.splits import degree_bins, sample_degree_matched_negatives_exact  # noqa: E402

N_M, N_D = 495, 383
FOLDS = [("train7", "test7_1"), ("train_71", "test71"), ("train_72", "test72"), ("train_73", "test73"),
         ("train_74", "test74")]


def gip(X):
    g = 1.0 / (np.sum(X ** 2) / X.shape[0])
    sq = (X ** 2).sum(1)
    return np.exp(-g * (sq[:, None] + sq[None] - 2 * X @ X.T))


def read(path):
    a = np.loadtxt(path)
    return a[a[:, 2] == 1][:, :2].astype(int), a[a[:, 2] == 0][:, :2].astype(int)


def load_model_cls(repo):
    sys.path.insert(0, repo)
    argv, sys.argv = sys.argv, [sys.argv[0]]   # their src.args parses sys.argv at import
    from src.Model import MHGCN
    sys.argv = argv
    utils = types.ModuleType("src.Utils")       # get_model, as its bytecode reads

    def get_model(model_opt, nfeat, A, B, o_ass, nhid, out, dropout, cuda, stdv, layer):
        if model_opt != "MGCNSS":
            raise NotImplementedError("model:{} is not implemented!".format(model_opt))
        return MHGCN(nfeat=nfeat, nhid=nhid, out=out, dropout=dropout, stdv=stdv, layer=layer)
    utils.get_model = get_model
    sys.modules["src.Utils"] = utils
    return get_model


def audit_both(tr, te, enc, neg, auc_v):
    out = {}
    for name, (rows, cols, f) in {"mirna_x_disease": (N_M, N_D, lambda a: a.T),
                                  "disease_x_mirna": (N_D, N_M, lambda a: a[:, ::-1].T)}.items():
        rep = audit(rows, cols, f(tr), f(te), encoder_edges=f(enc), eval_neg=f(neg), model_auroc=auc_v)
        o = {}
        for fi in rep.findings:
            ev = fi.evidence or {}
            o[fi.check] = {"status": fi.status}
            if fi.check == "model_free_baseline":
                o[fi.check].update(floor=ev.get("reference_floor"), heuristic=ev.get("reference_heuristic"),
                                   margin=ev.get("margin"))
        out[name] = o
    best = max(out, key=lambda k: out[k]["model_free_baseline"]["floor"])
    out["floor"] = out[best]["model_free_baseline"]["floor"]
    out["floor_heuristic"] = f"{out[best]['model_free_baseline']['heuristic']} ({best})"
    return out


def run(repo, get_model, mat, fold, seed, step, epochs, dev):
    """One training. Pairs are in 0-based (miRNA, disease) unless step == released (file coordinates, read as
    their code reads them)."""
    tr_name, te_name = FOLDS[fold]
    D = Path(repo) / "data/train"
    trp_f, trn_f = read(D / f"{tr_name}.txt")
    tep_f, ten_f = read(D / f"{te_name}.txt")
    R = mat["raw_association"].astype(float)
    trp, tep = trp_f - 1, tep_f - 1                     # positives are 1-based
    trn, ten = trn_f, ten_f                             # negatives are 0-based
    A, B, feat = mat["A"].astype(float), mat["B"].astype(float), mat["feature"].astype(float)
    rng = np.random.RandomState(seed)
    val = None
    if step == "validation":
        pi, ni = rng.permutation(len(trp)), rng.permutation(len(trn))
        kp, kn = len(trp) // 8, len(trn) // 8
        val = (trp[pi[:kp]], trn[ni[:kn]])
        trp, trn = trp[pi[kp:]], trn[ni[kn:]]
    if step in ("removed", "foldsim", "validation"):
        R = R.copy()
        R[tep[:, 0], tep[:, 1]] = 0
        if val is not None:
            R[val[0][:, 0], val[0][:, 1]] = 0
    if step in ("foldsim", "validation"):
        A, B, feat = A.copy(), B.copy(), feat.copy()
        A[1], B[0] = gip(R), gip(R.T)
        feat[N_M:, :N_M] = 0
    # their row lookup: released reads negatives as 1-based
    if step == "released":
        rows = lambda p, neg: (p[:, 0] - 1, p[:, 1] + 494) if neg else (p[:, 0], p[:, 1] + N_M)  # noqa: E731
        trn_use, ten_use = trn_f, ten_f
    else:
        rows = lambda p, neg: (p[:, 0], p[:, 1] + N_M)  # noqa: E731
        trn_use, ten_use = trn, ten
    torch.manual_seed(seed * 100 + fold)
    np.random.seed(seed * 100 + fold)
    model = get_model("MGCNSS", 878, A, B, R, 384, 256, 0, False, stdv=1 / 72, layer=2).to(dev)
    feat_t = torch.tensor(feat)

    def emb():
        return model(feat_t, A, B, R, 2)

    def pair_scores(E, p, neg):
        r1, r2 = rows(p, neg)
        keep = r1 >= 0 if (step == "released" and neg) else np.ones(len(p), bool)   # their dict lookup skips '-1'
        E = E.detach().cpu().numpy()
        return (E[r1[keep]] * E[r2[keep]]).sum(1)

    def auc(E, p, n, nflag=True):
        sp, sn = pair_scores(E, p, False), pair_scores(E, n, nflag)
        return float(roc_auc_score(np.r_[np.ones(len(sp)), np.zeros(len(sn))], np.r_[sp, sn]))

    tp1, tp2 = rows(trp, False)          # positives: their m-1, d+494 on 1-based = the right rows
    tn1, tn2 = rows(trn_use, True)
    curve, best, best_state = [], (-1, -1.0), None
    for ep in range(epochs):
        opt = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0005)   # re-created, as theirs
        E = emb()
        pos_out = (E[torch.as_tensor(tp1)] * E[torch.as_tensor(tp2)]).sum(1)
        neg_out = -(E[torch.as_tensor(tn1)] * E[torch.as_tensor(tn2)]).sum(1)
        loss = -torch.mean(F.logsigmoid(pos_out)) - torch.mean(F.logsigmoid(neg_out))
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            E = emb()
        t_auc = auc(E, tep, ten_use)
        sel = auc(E, val[0], val[1]) if val is not None else t_auc
        curve.append(t_auc)
        if sel > best[1]:
            best = (ep, sel)
            best_state = copy.deepcopy(model.state_dict()) if val is not None else None
    out = {"selected_epoch": best[0], "selected": curve[best[0]], "last": curve[-1], "best_test": max(curve)}
    if val is not None:
        model.load_state_dict(best_state)
        with torch.no_grad():
            E = emb()
        out["E"] = E.detach().cpu().numpy()
        out["R"] = R
    out["train_pos"], out["test_pos"], out["test_neg"] = trp, tep, ten
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    get_model = load_model_cls(a.repo)
    mat = loadmat(str(Path(a.repo) / "data/result.mat"))
    R0 = mat["raw_association"]
    assert R0.sum() == 5430
    assert np.abs(mat["A"][1] - gip(R0.astype(float))).max() < 1e-9 and np.abs(
        mat["B"][0] - gip(R0.T.astype(float))).max() < 1e-9, "A[1]/B[0] are not GIP of the full matrix"
    allpos = np.argwhere(R0 > 0)
    res = {"meta": {"seeds": a.seeds, "epochs": a.epochs, "folds": a.folds}, "runs": []}
    t0 = time.time()
    for seed in a.seeds:
        for fold in a.folds:
            rec = {"seed": seed, "fold": fold + 1}
            for step in ("released", "index", "removed", "foldsim", "validation"):
                o = run(a.repo, get_model, mat, fold, seed, step, a.epochs, dev)
                enc = np.argwhere(o["R"] > 0) if "R" in o else (
                    allpos if step in ("released", "index") else None)
                if enc is None:
                    Rr = R0.copy()
                    Rr[o["test_pos"][:, 0], o["test_pos"][:, 1]] = 0
                    enc = np.argwhere(Rr > 0)
                rec[step] = {k: o[k] for k in ("selected_epoch", "selected", "last", "best_test")}
                rec[step]["audit"] = audit_both(o["train_pos"], o["test_pos"], enc, o["test_neg"], o["selected"])
                if step == "validation":
                    E, tep, trp = o["E"], o["test_pos"], o["train_pos"]
                    g = torch.Generator().manual_seed(1000 * seed + fold)
                    # uniform over every unknown pair
                    rs = np.random.RandomState(1000 * seed + fold)
                    un = []
                    while len(un) < len(tep):
                        m, d = rs.randint(N_M), rs.randint(N_D)
                        if R0[m, d] == 0:
                            un.append((m, d))
                    un = np.array(un)
                    T = lambda x: torch.from_numpy(np.ascontiguousarray(x.T)).long()  # noqa: E731
                    deg_m = torch.from_numpy(np.bincount(trp[:, 0], minlength=N_M)).long()
                    deg_d = torch.from_numpy(np.bincount(trp[:, 1], minlength=N_D)).long()
                    md = sample_degree_matched_negatives_exact(T(tep), T(allpos), degree_bins(deg_d), N_D, g)[0]
                    mm = sample_degree_matched_negatives_exact(T(tep[:, ::-1]), T(allpos[:, ::-1]), degree_bins(deg_m),
                                                               N_M, g)[0].numpy().T[:, ::-1]
                    sets = {"uniform": un, "matched_disease": md.numpy().T, "matched_mirna": np.ascontiguousarray(mm)}
                    sc = lambda p: (E[p[:, 0]] * E[N_M + p[:, 1]]).sum(1)  # noqa: E731
                    for nm, neg in sets.items():
                        v = float(roc_auc_score(np.r_[np.ones(len(tep)), np.zeros(len(neg))], np.r_[sc(tep), sc(neg)]))
                        rec[nm] = {"auc": v, "audit": audit_both(trp, tep, enc, neg, v)}
                print(f"seed {seed} fold {fold + 1} {step}: selected {rec[step]['selected']:.4f} "
                      f"(epoch {rec[step]['selected_epoch']}), last {rec[step]['last']:.4f}, "
                      f"floor {rec[step]['audit']['floor']:.4f} [{time.time() - t0:.0f}s]", flush=True)
            print(f"seed {seed} fold {fold + 1}: " + " | ".join(
                f"{k} {rec[k]['auc']:.4f}/{rec[k]['audit']['floor']:.4f}" for k in ("uniform", "matched_disease",
                                                                                  "matched_mirna")), flush=True)
            res["runs"].append(rec)

    def mm_(get):
        v = [get(r) for r in res["runs"]]
        return {"mean": st.mean(v), "sd": st.stdev(v) if len(v) > 1 else 0.0}
    S = {}
    for step in ("released", "index", "removed", "foldsim", "validation"):
        for k in ("selected", "last"):
            S[f"{step}_{k}"] = mm_(lambda r, s=step, k=k: r[s][k])
        S[f"{step}_floor"] = mm_(lambda r, s=step: r[s]["audit"]["floor"])
    for nm in ("uniform", "matched_disease", "matched_mirna"):
        S[nm] = mm_(lambda r, n=nm: r[n]["auc"])
        S[f"{nm}_floor"] = mm_(lambda r, n=nm: r[n]["audit"]["floor"])
    res["summary"] = S
    print(json.dumps(S, indent=1))
    path = ROOT / f"results/comparison/mgcnss_rerun{'_smoke' if a.epochs < 2000 else ''}.json"
    path.write_text(json.dumps(res, indent=2, default=float))
    print("Wrote", path)


if __name__ == "__main__":
    main()
