"""
hlgnn_rerun.py -- HLGNN-MDA's own released code (yu2022hlgnnmda, miRNA-disease), as released, then scored
against degree-matched negatives.

HLGNN-MDA (github.com/LiangYu-Xidian/HLGNN-MDA @ e3ac824; no licence, kept outside the repo) reports AUROC
0.9309 (hop 4) / 0.9297 (hop 1) from tenfold cross-validation on HMDD v2.0 (495 x 383, 5,430 associations).
What the released Python/Main.py does (README command: --bi-network --first-kind-node-num 0 --hop 1):

  * sample_neg: one random 90/10 split of the positives (no CV script is released), and uniform negatives
    drawn from every unknown pair, 1:1, oriented (disease, miRNA) while positives are (miRNA, disease);
  * A[test_pos] = 0 in both directions: the test links are masked before any subgraph is extracted, and
    subgraph_extraction_labeling also drops the target link of every training subgraph (SEAL);
  * topology only: DRNL labels on hop-h enclosing subgraphs, DGCNN, no similarity network by default;
  * 50 epochs, Adam lr 1e-4, batch 50; the test AUROC is printed every epoch and the last epoch is reported.
  * Main.py seeds `random`/numpy with the DGCNN parser's seed (always 1), so --seed changes only torch.

Steps (per seed; their sample_neg, subgraph extraction, Classifier and loop_dataset, imported unmodified):
  theirs            their split and test negatives, last epoch (also the best test epoch, for reference)
  theirs_flipped    the same negatives scored as (miRNA, disease), to check that pair order carries no signal
  matched_disease   the same test positives against negatives that keep the miRNA and draw a disease of
                    matching degree (exact sampler; degree from the training positives)
  matched_mirna     keep the disease, draw a miRNA of matching degree
With --train-neg matched, the training negatives are disease-degree matched to the training positives
instead of uniform (the model's best chance against matched test negatives).
protocheck 0.2.0 audits every set in both orientations (encoder input = training positives) and keeps the
higher floor.

Compatibility shims (no change in computation): gensim stubbed (only node2vec embeddings use it, off by
default); networkx 3 renamed from_scipy_sparse_matrix; torch 2 removed ctx.saved_variables.

Usage: python analysis/hlgnn_rerun.py --repo PATH_TO_HLGNN-MDA [--hop 1] [--seeds ...] [--train-neg uniform]
Writes results/comparison/hlgnn_rerun_h<hop>_<train-neg>.json (suffix _smoke when --epochs < 50).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
import time
import types
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as ssp
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from protocheck.audit import audit  # noqa: E402
from training.splits import degree_bins, sample_degree_matched_negatives_exact  # noqa: E402


def load_their_modules(repo: Path):
    import networkx as nx
    if not hasattr(nx, "from_scipy_sparse_matrix"):
        nx.from_scipy_sparse_matrix = nx.from_scipy_sparse_array
    if "gensim" not in sys.modules:
        g = types.ModuleType("gensim")
        gm = types.ModuleType("gensim.models")
        gm.Word2Vec = None
        g.models = gm
        sys.modules["gensim"], sys.modules["gensim.models"] = g, gm
    argv, sys.argv = sys.argv, [sys.argv[0]]          # their parsers read sys.argv at import
    sys.path[:0] = [str(repo / "pytorch_DGCNN"), str(repo / "pytorch_DGCNN/lib"), str(repo / "Python")]
    import main as dg                                   # pytorch_DGCNN/main.py
    import pytorch_util as pu
    import util_functions as U
    sys.argv = argv

    @staticmethod
    def backward(ctx, grad_output):                     # their MySpMM.backward with saved_tensors
        sp_mat, dense_mat = ctx.saved_tensors
        g2 = torch.mm(sp_mat.t(), grad_output) if ctx.needs_input_grad[1] else None
        return None, g2
    pu.MySpMM.backward = backward
    return dg, U


def to_local(pairs, n_m):
    """their (row, col) node pairs in the bipartite graph -> (k, 2) local (miRNA, disease)."""
    r, c = np.asarray(pairs[0]), np.asarray(pairs[1])
    m = np.where(r < n_m, r, c)
    d = np.where(r < n_m, c, r) - n_m
    return np.c_[m, d]


def audit_both(tr, te, trn, neg, auc_v, n_m, n_d):
    out = {}
    for name, (rows, cols, f) in {"mirna_x_disease": (n_m, n_d, lambda a: a.T),
                                  "disease_x_mirna": (n_d, n_m, lambda a: a[:, ::-1].T)}.items():
        rep = audit(rows, cols, f(tr), f(te), encoder_edges=f(tr), train_neg=f(trn), eval_neg=f(neg),
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


def matched(pos_local, allp_local, deg, n_cols, g, side):
    """side 'disease': keep miRNA, match disease degree; 'mirna': keep disease, match miRNA degree."""
    if side == "disease":
        return sample_degree_matched_negatives_exact(torch.from_numpy(pos_local.T.copy()).long(),
                                                     torch.from_numpy(allp_local.T.copy()).long(),
                                                     degree_bins(deg), n_cols, g)[0].numpy().T
    flip = lambda a: a[:, ::-1].T.copy()  # noqa: E731
    return sample_degree_matched_negatives_exact(torch.from_numpy(flip(pos_local)).long(),
                                                 torch.from_numpy(flip(allp_local)).long(),
                                                 degree_bins(deg), n_cols, g)[0].numpy().T[:, ::-1].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--hop", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--train-neg", choices=["uniform", "matched"], default="uniform")
    a = ap.parse_args()
    dg, U = load_their_modules(a.repo)
    ca = dg.cmd_args

    net0 = sio.loadmat(a.repo / "Python/data/miRNA_disease_5430.mat")["net"].toarray()
    n_m, n_d = net0.shape
    # their bi-network branch, no similarity network (README command)
    net = ssp.csc_matrix(np.vstack((np.hstack((np.zeros((n_m, n_m)), net0)),
                                    np.hstack((net0.T, np.zeros((n_d, n_d)))))))
    allp = np.argwhere(net0 == 1)
    out = {"meta": {"hop": a.hop, "seeds": a.seeds, "epochs": a.epochs, "train_neg": a.train_neg,
                    "n_mirna": n_m, "n_disease": n_d, "n_assoc": int(net0.sum()),
                    "diseases_deg_le1": int((net0.sum(0) <= 1).sum()),
                    "mirnas_deg_le1": int((net0.sum(1) <= 1).sum())}, "runs": []}
    t0 = time.time()
    for seed in a.seeds:
        random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
        train_pos, train_neg, test_pos, test_neg = U.sample_neg(net, 0.1, max_train_num=100000,
                                                                bi_network=True, first_kind_node_num=n_m)
        A = net.copy()
        A[test_pos[0], test_pos[1]] = 0
        A[test_pos[1], test_pos[0]] = 0
        A.eliminate_zeros()
        tr, te = to_local(train_pos, n_m), to_local(test_pos, n_m)
        assert not set(map(tuple, te)) & set(map(tuple, tr))
        deg_m = torch.from_numpy(np.bincount(tr[:, 0], minlength=n_m)).long()
        deg_d = torch.from_numpy(np.bincount(tr[:, 1], minlength=n_d)).long()
        g = torch.Generator().manual_seed(1000 * seed)
        if a.train_neg == "matched":
            trn = matched(tr, allp, deg_d, n_d, g, "disease")
            train_neg = (trn[:, 0], trn[:, 1] + n_m)
        trn = to_local(train_neg, n_m)
        te_neg_theirs = to_local(test_neg, n_m)
        neg_sets = {"theirs": (test_neg, te_neg_theirs),
                    "theirs_flipped": ((te_neg_theirs[:, 0], te_neg_theirs[:, 1] + n_m), te_neg_theirs)}
        for side, deg, nc in (("disease", deg_d, n_d), ("mirna", deg_m, n_m)):
            mn = matched(te, allp, deg, nc, g, side)
            neg_sets[f"matched_{side}"] = ((mn[:, 0], mn[:, 1] + n_m), mn)

        train_graphs, test_graphs, max_n_label = U.links2subgraphs(A, train_pos, train_neg, test_pos, test_neg,
                                                                   a.hop, None, None)
        pos_graphs = test_graphs[:len(test_pos[0])]
        eval_graphs = {"theirs": test_graphs}
        for k in ("theirs_flipped", "matched_disease", "matched_mirna"):
            ng, ml = U.selfDefine_links2subgraphs(A, neg_sets[k][0], 0, a.hop, None, None)
            max_n_label = max(max_n_label, ml)
            eval_graphs[k] = pos_graphs + ng

        # their DGCNN configuration (Main.py)
        ca.gm, ca.latent_dim, ca.hidden, ca.out_dim, ca.dropout = "DGCNN", [32, 32, 32, 1], 128, 0, True
        ca.num_class, ca.mode, ca.learning_rate, ca.batch_size, ca.printAUC = 2, "cpu", 1e-4, 50, True
        ca.feat_dim, ca.attr_dim = max_n_label + 1, 0
        sizes = sorted(gr.num_nodes for gr in train_graphs + test_graphs)
        ca.sortpooling_k = max(10, sizes[int(math.ceil(0.6 * len(sizes))) - 1])
        clf = dg.Classifier()
        opt = torch.optim.Adam(clf.parameters(), lr=ca.learning_rate)
        idx = list(range(len(train_graphs)))
        curves = {k: [] for k in eval_graphs}
        scores = {}
        for ep in range(a.epochs):
            random.shuffle(idx)
            clf.train()
            dg.loop_dataset(train_graphs, clf, idx, optimizer=opt)
            clf.eval()
            with torch.no_grad():
                for k, gl in eval_graphs.items():
                    _, (s, y) = dg.loop_dataset(gl, clf, list(range(len(gl))), retain_result=True)
                    curves[k].append(float(roc_auc_score(y, s)))
                    scores[k] = (s, y)
            print(f"  seed {seed} ep {ep + 1}: " + " ".join(f"{k} {v[-1]:.4f}" for k, v in curves.items()),
                  flush=True)
        rec = {"seed": seed, "n_train_pos": int(len(tr)), "n_test_pos": int(len(te)),
               "sortpooling_k": int(ca.sortpooling_k)}
        for k, (s, y) in scores.items():
            v = curves[k][-1]
            rec[k] = {"auc": v, "ap": float(average_precision_score(y, s)), "best_epoch_auc": max(curves[k]),
                      "audit": audit_both(tr, te, trn, neg_sets[k][1], v, n_m, n_d)}
        out["runs"].append(rec)
        print(f"seed {seed} [{time.time() - t0:.0f}s]: " + " | ".join(
            f"{k} {rec[k]['auc']:.4f} (floor {rec[k]['audit']['floor']:.4f} {rec[k]['audit']['floor_heuristic']})"
            for k in scores), flush=True)

    def m(get):
        v = [get(r) for r in out["runs"]]
        return {"mean": st.mean(v), "sd": st.stdev(v) if len(v) > 1 else 0.0}
    keys = ("theirs", "theirs_flipped", "matched_disease", "matched_mirna")
    out["summary"] = {**{k: m(lambda r, k=k: r[k]["auc"]) for k in keys},
                      **{f"floor_{k}": m(lambda r, k=k: r[k]["audit"]["floor"]) for k in keys},
                      "theirs_best_epoch": m(lambda r: r["theirs"]["best_epoch_auc"])}
    print("\n=== summary ===\n", json.dumps(out["summary"], indent=1))
    tag = f"_h{a.hop}_{a.train_neg}" + ("" if a.epochs >= 50 else "_smoke")
    path = ROOT / f"results/comparison/hlgnn_rerun{tag}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print("Wrote", path)


if __name__ == "__main__":
    main()
