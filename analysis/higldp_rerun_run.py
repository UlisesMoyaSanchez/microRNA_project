"""
higldp_rerun_run.py -- trains HiGLDP's own released model on one fold / seed / step (runs in the higldp_py310 env:
Python 3.10, PyTorch 1.13.1, PyG 2.3.1, TensorFlow 2.10, as its README pins). Driven by
training/slurm_higldp_rerun.sh; analysis/higldp_rerun.py prepares the splits and reports (see its docstring).

Their MATLAB preprocessing (Dataset/matlab: similarity_generate.m, GIP.m, RWR.m, merge.m, run_merge.m) is ported
line by line to numpy (no MATLAB on the cluster); their DenoisingAutoencoder/DAE.py, data_concat.py logic,
dataprocess.py (association feature graph), function.py, model.py and main.py's loop are used as released.

Steps:
  released    their shipped features (Lnc_dim600, DO_dim400), afg1 and fold files; their loop
  ported      the same pipeline recomputed through our port (full association matrix, as theirs)
  foldfeat    the lncRNA GIP kernel computed from the fold's training associations only (test positives zeroed)
  validation  foldfeat with one eighth of the training nodes held out (their positives also zeroed); extra,
              unlabelled nodes for the degree-matched negatives are added to the graph so they can be scored
Every epoch's scores on the evaluation nodes are saved; the epoch is chosen at report time.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

N_L, N_D, N_PAIRS = 1821, 185, 22622


# ---------------------------------------------------------------- port of Dataset/matlab
def jaccard_sim(M):  # similarity_generate.m: Sim = 1 - pdist(M,'jaccard'); + eye; NaN -> 0
    M = (M != 0).astype(np.float64)
    inter = M @ M.T
    cnt = M.sum(1)
    union = cnt[:, None] + cnt[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        S = inter / union
    np.fill_diagonal(S, 0.0)  # squareform leaves a zero diagonal
    S = S + np.eye(M.shape[0])
    S[np.isnan(S)] = 0.0
    return S


def gip_sim(A):  # GIP.m
    A = A.astype(np.float64)
    rm = 1.0 / ((A ** 2).sum() / A.shape[0])
    sq = (A ** 2).sum(1)
    d = np.maximum(sq[:, None] + sq[None, :] - 2 * A @ A.T, 0.0)
    return np.exp(-rm * d)


def rwr(A, maxiter=20, restart_prob=0.5):  # RWR.m
    n = A.shape[0]
    A = A + np.diag((A.sum(0) == 0).astype(float))
    P = A / A.sum(0, keepdims=True)
    Q = np.eye(n)
    for _ in range(maxiter):
        Q_new = (1 - restart_prob) * P @ Q + restart_prob * np.eye(n)
        delta = np.linalg.norm(Q - Q_new, "fro")
        Q = Q_new
        if delta < 1e-6:
            break
    return Q


def merge(nets):  # merge.m
    Q = np.concatenate([rwr(N) for N in nets], axis=1)
    alpha = 1.0 / Q.shape[0]
    return np.log(Q + alpha) - np.log(alpha)


def features(repo: Path, LD: np.ndarray, seed: int):
    d = repo / "Dataset" / "data"
    ld = lambda f: np.loadtxt(d / f)  # noqa: E731
    lnc_nets = [jaccard_sim(ld("mat_Lnc_mRNA.txt")), jaccard_sim(ld("mat_Lnc_Protein.txt")),
                jaccard_sim(ld("mat_Lnc_RBP.txt")), gip_sim(LD)]
    dis_nets = [ld("Similarity_Matrix_DO.txt"), jaccard_sim(ld("mat_DO_circRNA.txt")),
                jaccard_sim(ld("mat_DO_Metabolite.txt"))]
    lnc_vec, do_vec = merge(lnc_nets), merge(dis_nets)
    sys.path.insert(0, str(repo / "Dataset" / "DenoisingAutoencoder"))
    import tensorflow._api.v2.compat.v1 as tf  # noqa: E402
    tf.set_random_seed(seed)
    np.random.seed(seed)
    from DAE import DAE  # noqa: E402  (their runDAE.py arguments)
    lnc = np.asarray(DAE(lnc_vec, lnc_vec.shape[1], 20, 16, 1, 600, [600]))
    dis = np.asarray(DAE(do_vec, do_vec.shape[1], 20, 32, 1, 400, [400]))
    return lnc, dis


def concat(index, lnc, dis):  # data_concat.py
    f = np.zeros((len(index), lnc.shape[1] + dis.shape[1]))
    f[:, :lnc.shape[1]] = lnc[index[:, 1]]
    f[:, lnc.shape[1]:] = dis[index[:, 2]]
    return f


def afg(feat, k=1):  # dataprocess.py: cosine k-NN, keep start < end
    from sklearn.metrics.pairwise import cosine_similarity as cos
    dist = cos(feat)
    out = []
    for i in range(dist.shape[0]):
        for vv in np.argpartition(dist[i, :], -(k + 1))[-(k + 1):]:
            if vv != i and i < vv:
                out.append((i, vv))
    return np.array(out, dtype=np.int64)


def struct_edges(index, train_nodes):
    """their edgeX.txt: every pair of training nodes that share a lncRNA or a disease (labels unused)."""
    tr = np.sort(train_nodes)
    out = []
    for col in (1, 2):
        key = index[tr, col]
        order = np.argsort(key, kind="stable")
        s, k = tr[order], key[order]
        b = np.flatnonzero(np.r_[True, k[1:] != k[:-1], True])
        for a, z in zip(b[:-1], b[1:]):
            g = s[a:z]
            if len(g) > 1:
                i, j = np.triu_indices(len(g), 1)
                out.append(np.stack([g[i], g[j]], 1))
    return np.concatenate(out)


# ---------------------------------------------------------------- their training (main.py), per fold
def train(repo: Path, cfg, n, feat, sedges, fedges, labels, idx_train, eval_nodes, seed, epochs, out):
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    sys.path.insert(0, str(repo))
    from function import normalize, sparse_mx_to_torch_sparse_tensor  # their helpers
    from model import HiGLDP

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def adj(e):  # function.load_graph
        a = sp.coo_matrix((np.ones(e.shape[0]), (e[:, 0], e[:, 1])), shape=(n, n), dtype=np.float32)
        a = a + a.T.multiply(a.T > a) - a.multiply(a.T > a)
        return sparse_mx_to_torch_sparse_tensor(normalize(a + sp.eye(a.shape[0])))

    sadj, fadj = adj(sedges), adj(fedges)
    # function.get_adj walks the rows of the coalesced sparse tensor and lists (row, col) in order: that is
    # exactly its coalesced index set (asserted on a small graph in --check-adj)
    asadj, afadj = sadj.coalesce().indices(), fadj.coalesce().indices()
    features = torch.FloatTensor(feat.astype(np.float32))
    labels_t = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    model = HiGLDP(nfeat=cfg["fdim"], nhid1=cfg["nhid1"], nhid2=cfg["nhid2"], nclass=2, n=n,
                   dropout=cfg["dropout"]).to(dev)
    features, sadj, fadj, asadj, afadj = (t.to(dev) for t in (features, sadj, fadj, asadj, afadj))
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scores = np.zeros((epochs, len(eval_nodes)), dtype=np.float32)
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, sadj, fadj, asadj, afadj).cpu()
        loss = F.nll_loss(output[idx_train], labels_t[idx_train])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            o = model(features, sadj, fadj, asadj, afadj).cpu()
        scores[ep] = o[eval_nodes, 1].numpy()  # function.score uses output[:, 1] (a log-probability)
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"  epoch {ep} loss {loss.item():.4f}", flush=True)
    np.savez_compressed(out, scores=scores, eval_nodes=eval_nodes)


def check_adj(repo: Path):
    import torch
    sys.path.insert(0, str(repo))
    from function import get_adj, normalize, sparse_mx_to_torch_sparse_tensor
    rng = np.random.default_rng(0)
    e = rng.integers(0, 300, size=(2000, 2))
    a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(300, 300), dtype=np.float32)
    a = a + a.T.multiply(a.T > a) - a.multiply(a.T > a)
    t = sparse_mx_to_torch_sparse_tensor(normalize(a + sp.eye(300)))
    assert torch.equal(get_adj(t), t.coalesce().indices()), "get_adj differs from coalesced indices"
    print("get_adj == coalesced indices: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--step", choices=["released", "ported", "foldfeat", "validation"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--check-adj", action="store_true")
    a = ap.parse_args()
    if a.check_adj:
        check_adj(a.repo)
        return
    t0 = time.time()
    R = a.repo
    cfg = dict(lr=1e-3, weight_decay=1e-3, nhid1=512, nhid2=256, dropout=0.3, fdim=1000)  # config.txt
    for line in (R / "config.txt").read_text().splitlines():  # confirm the released values
        if "=" in line:
            k, v = (s.strip() for s in line.split("=", 1))
            if k in cfg:
                assert float(v) == float(cfg[k]), (k, v)
    S = np.load(a.work / f"higldp_fold{a.fold}_split.npz")
    index = np.loadtxt(R / "Dataset/data/index.txt", dtype=np.int64)
    labels = np.loadtxt(R / "Dataset/data/label.txt", dtype=np.int64)
    LD = np.loadtxt(R / "Dataset/data/mat_Lnc_DO.txt")
    train_nodes, test_nodes = S["train"], S["test"]
    out = a.work / f"higldp_fold{a.fold}_{a.step}_s{a.seed}.npz"

    if a.step == "released":
        lnc = np.loadtxt(R / "Dataset/Lnc_dim600.txt")
        dis = np.loadtxt(R / "Dataset/DO_dim400.txt")
        feat = concat(index, lnc, dis)
        fed = np.loadtxt(R / "Dataset/afg/afg1.txt", dtype=np.int64)
        sed = np.loadtxt(R / f"5fold/edge{a.fold}.txt", dtype=np.int64)
        tr = np.loadtxt(R / f"5fold/train{a.fold}.txt", dtype=np.int64)
        assert np.array_equal(np.sort(tr), np.sort(train_nodes))
        mine = struct_edges(index, train_nodes)
        assert len(mine) == len(sed) and set(map(tuple, np.sort(mine, 1))) == set(map(tuple, np.sort(sed, 1)))
        train(R, cfg, N_PAIRS, feat, sed, fed, labels, train_nodes, test_nodes, a.seed, a.epochs, out)
    else:
        LDt = LD.copy()
        fit = train_nodes
        if a.step in ("foldfeat", "validation"):
            hide = test_nodes if a.step == "foldfeat" else np.r_[test_nodes, S["val"]]
            h = index[hide][labels[hide] == 1]
            LDt[h[:, 1], h[:, 2]] = 0
        if a.step == "validation":
            fit = np.setdiff1d(train_nodes, S["val"])
            extra = S["extra_pairs"]  # (k, 2) lncRNA, disease: degree-matched negatives, unlabelled nodes
            index_x = np.r_[index, np.c_[np.arange(N_PAIRS, N_PAIRS + len(extra)), extra]]
            labels_x = np.r_[labels, np.zeros(len(extra), dtype=np.int64)]
            eval_nodes = np.r_[test_nodes, S["val"], np.arange(N_PAIRS, N_PAIRS + len(extra))]
        else:
            index_x, labels_x, eval_nodes = index, labels, test_nodes
        lnc, dis = features(R, LDt, a.seed)
        feat = concat(index_x, lnc, dis)
        fed = afg(feat)
        sed = struct_edges(index_x, fit)
        print(f"fold {a.fold} {a.step}: GIP input {int(LDt.sum())} associations, {len(fed)} afg edges, "
              f"{len(sed)} struct edges", flush=True)
        train(R, cfg, len(index_x), feat, sed, fed, labels_x, fit, eval_nodes, a.seed, a.epochs, out)
    print(f"fold {a.fold} {a.step} seed {a.seed} done [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    main()
