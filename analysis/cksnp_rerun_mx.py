"""
cksnp_rerun_mx.py -- CKSNP-GNN's own released code (li2022cksnpgnn), run in its own environment
(Python 3.7, mxnet 1.6.0, DGL 0.4.3post1, DGLBACKEND=mxnet). Driven by analysis/cksnp_rerun.py; see its
docstring for the steps.

  --dump   their sample() and KFold split for each seed -> <work>/cksnp_s<seed>_folds.npz
  --train  per fold, their GraphEncoder/BilinearDecoder trained with their loop (100 epochs x 10 Adam steps,
           lr 1e-3, wd 1e-3, dropout 0.5, embedding 256, 2 layers, CKSNAP5-Pearson miRNA similarity), twice:
             released  their feature files (disease GIP kernel of the full association matrix)
             foldgip   GIP recomputed (their rbf.py formula) from the fold's training positives only
           and scored (both pair orientations, as their test_eid holds both edge directions) on:
             graph g (every sampled pair is an edge, test included: as released) or g_train (training pairs only),
             against their test negatives and the degree-matched negatives in <work>/cksnp_s<seed>_matched.npz
           -> <work>/cksnp_s<seed>_scores.npz
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import ndarray as nd, gluon
from mxnet.gluon import loss as gloss
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics.pairwise import rbf_kernel

MSSM = "miRNA_MSSM/miRNA_CKSNAP5_pearson.csv"


def gip(X):
    """their data/rbf.py: gamma = 1 / mean squared row norm, sklearn rbf_kernel."""
    return rbf_kernel(X, gamma=1.0 / (np.sum(X ** 2) / X.shape[0]))


def setup(repo, seed, ctx):
    os.chdir(repo)
    sys.path.insert(0, repo)
    import dgl
    dgl.load_backend("mxnet")
    from utils import build_graph, sample, load_data
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)
    g, dmap, mmap = build_graph("data", MSSM, random_seed=seed, ctx=ctx)
    samples = sample("data", random_seed=seed)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    folds = [(tr, te) for tr, te in kf.split(samples[:, 2])]
    return g, samples, folds, load_data


def dump(repo, seed, work):
    ctx = mx.cpu(0)
    g, samples, folds, _ = setup(repo, seed, ctx)
    out = {"samples": samples.astype(np.int64)}
    for i, (tr, te) in enumerate(folds):
        out["train_%d" % i], out["test_%d" % i] = tr, te
    np.savez(os.path.join(work, "cksnp_s%d_folds.npz" % seed), **out)
    print("dumped seed", seed, samples.shape, flush=True)


def train(repo, seed, work, epochs):
    ctx = mx.cpu(0)
    g, samples, folds, load_data = setup(repo, seed, ctx)
    from model import GNNMDA, GraphEncoder, BilinearDecoder
    chk = np.load(os.path.join(work, "cksnp_s%d_folds.npz" % seed))
    assert (chk["samples"] == samples).all(), "split differs from the dumped one"
    matched = np.load(os.path.join(work, "cksnp_s%d_matched.npz" % seed))
    ID, IM = load_data("data", MSSM)
    nd_, nm_ = ID.shape[0], IM.shape[0]
    D_SSM = pd.read_csv("data/D_SSM.csv", header=0, index_col=0).values
    M_SSM = pd.read_csv(MSSM, header=0, index_col=0).values
    adj = pd.read_csv("data/miRNA_Disease_AdjMatrix.csv", header=0, index_col=0).values.astype(float)
    pos = samples[samples[:, 2] == 1]
    A2 = np.zeros_like(adj)
    A2[pos[:, 0] - 1, pos[:, 1] - 1] = 1
    assert (A2 == adj).all(), "adjacency rows/cols are not miRNA/disease ids - 1"
    assert np.allclose(np.where(D_SSM == 0, gip(adj.T), D_SSM), ID, atol=1e-5), "released disease features != SSM|GIP(full)"

    samples_df = pd.DataFrame(samples, columns=["miRNA", "disease", "label"])
    sdv = [int(d) - 1 for d in samples[:, 1]]
    smv = [int(m) - 1 + nd_ for m in samples[:, 0]]
    cross_entropy = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    out = {}
    t0 = time.time()
    for i, (tr, te) in enumerate(folds):
        samples_df["train"], samples_df["test"] = 0, 0
        samples_df["train"].iloc[tr] = 1
        samples_df["test"].iloc[te] = 1
        edge_data = {"train": nd.from_numpy(samples_df["train"].values.astype("int32")).copyto(ctx),
                     "test": nd.from_numpy(samples_df["test"].values.astype("int32")).copyto(ctx)}
        g.edges[sdv, smv].data.update(edge_data)
        g.edges[smv, sdv].data.update(edge_data)
        trp = samples[tr][samples[tr][:, 2] == 1]
        Xf = np.zeros_like(adj)
        Xf[trp[:, 0] - 1, trp[:, 1] - 1] = 1
        feats = {"released": (ID, IM),
                 "foldgip": (np.where(D_SSM == 0, gip(Xf.T), D_SSM), np.where(M_SSM == 0, gip(Xf), M_SSM))}
        te_s = samples[te]
        sets = {"theirs": te_s[te_s[:, 2] == 0][:, :2],
                "matched_disease": matched["disease_%d" % i], "matched_mirna": matched["mirna_%d" % i]}
        te_pos = te_s[te_s[:, 2] == 1][:, :2]
        for fname, (IDf, IMf) in feats.items():
            d_data = nd.zeros(shape=(g.number_of_nodes(), nd_), dtype="float32", ctx=ctx)
            d_data[:nd_, :] = nd.array(IDf)
            m_data = nd.zeros(shape=(g.number_of_nodes(), nm_), dtype="float32", ctx=ctx)
            m_data[nd_:nd_ + nm_, :] = nd.array(IMf)
            g.ndata["d_features"], g.ndata["m_features"] = d_data, m_data
            train_eid = g.filter_edges(lambda edges: edges.data["train"]).astype("int64")
            g_train = g.edge_subgraph(train_eid, preserve_nodes=True)
            g_train.copy_from_parent()
            rating_train = g_train.edata["rating"]
            src_train, dst_train = g_train.all_edges()
            mx.random.seed(seed * 100 + i)
            model = GNNMDA(GraphEncoder(embedding_size=256, n_layers=2, G=g_train, aggregator="GraphSAGE",
                                        dropout=0.5, slope=0.2, ctx=ctx), BilinearDecoder(feature_size=256))
            model.collect_params().initialize(init=mx.init.Xavier(magnitude=np.sqrt(2.0)), ctx=ctx)
            trainer = gluon.Trainer(model.collect_params(), "adam", {"learning_rate": 1e-3, "wd": 1e-3})
            for _ in range(epochs):
                for _ in range(10):
                    with mx.autograd.record():
                        loss = cross_entropy(model(g_train, src_train, dst_train), rating_train).mean()
                        loss.backward()
                    trainer.step(1)
            # their reported number: test_eid over g (both edge directions), last epoch
            test_eid = g.filter_edges(lambda edges: edges.data["test"]).astype("int64")
            s_t, d_t = g.find_edges(test_eid)
            h = model.encoder(g)
            sc = model.decoder(h[s_t], h[d_t]).asnumpy()
            out["%s_their_path_auc_%d" % (fname, i)] = metrics.roc_auc_score(
                g.edges[test_eid].data["rating"].asnumpy(), sc)
            for gname, G in (("g", g), ("gtrain", g_train)):
                h = model.encoder(G)

                def both(pairs):
                    d = nd.array(pairs[:, 1] - 1, dtype="int64")
                    m = nd.array(pairs[:, 0] - 1 + nd_, dtype="int64")
                    return np.r_[model.decoder(h[d], h[m]).asnumpy(), model.decoder(h[m], h[d]).asnumpy()]
                out["%s_%s_pos_%d" % (fname, gname, i)] = both(te_pos)
                for sname, neg in sets.items():
                    out["%s_%s_%s_%d" % (fname, gname, sname, i)] = both(neg)
            print("seed %d fold %d %s: their path %.4f [%.0fs]" % (
                seed, i + 1, fname, out["%s_their_path_auc_%d" % (fname, i)], time.time() - t0), flush=True)
    np.savez(os.path.join(work, "cksnp_s%d_scores.npz" % seed), **out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--dump", action="store_true")
    a = ap.parse_args()
    a.work = os.path.abspath(a.work)
    if a.dump:
        dump(a.repo, a.seed, a.work)
    else:
        train(a.repo, a.seed, a.work, a.epochs)
