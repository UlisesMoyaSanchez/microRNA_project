"""
hmcda_rerun_mg.py -- HMCDA's own released code (liang2023hmcda), run in its own environment (Python 3.6,
PyTorch 1.2.0 CPU, DGL 0.3.1, as its README pins). Driven by analysis/hmcda_rerun.py; see its docstring for
the steps. Their model (model.MAGNN_lp) and minibatch parser (utils.tools.parse_minibatch_HMCDA) run
unmodified; their notebook's preprocessing (metapath lists from the training associations) and their
training loop (run.HMCDA.py) are ported line for line, made callable on any split.

  --dump   per split: positives and negatives, as released and de-duplicated -> <work>/hmcda_<split>_data.npz
  --train  per split, variant and seed: metapaths from the variant's training positives, their training loop
           (1 sampled negative per positive per batch, early stopping on validation loss, patience 5), the
           best-validation checkpoint scored on the test positives, their test negatives and the degree-matched
           negatives in <work>/hmcda_<split>_matched.npz -> <work>/hmcda_<split>_<variant>_s<seed>_scores.npz
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

N_C, N_D, N_M = 1556, 243, 840
ETYPES = [[[0, 1], [2, 3]], [[1, 0], [4, 5], [None]]]
USE_MASKS = [[True, False], [True, False, False]]
NO_MASKS = [[False] * 2, [False] * 3]


def raw(repo):
    r = os.path.join(repo, "data/raw/HMCDA")
    rd = lambda f, n: pd.read_csv(os.path.join(r, f), encoding="utf-8", delimiter="\t", names=n)  # noqa: E731
    return (rd("circrna_disease.dat", ["circrnaID", "diseaseID"]), rd("circrna_mirna.dat", ["circrnaID", "mirnaID"]),
            rd("mirna_disease.dat", ["mirnaID", "diseaseID"]), rd("disease_disease.dat", ["diseaseID", "idiseaseID"]))


def split_idx(repo, name):
    """ex5 = the one released split; rot1..rot5 = their gendata.py rotations under random.seed(0)."""
    if name == "ex5":
        z = np.load(os.path.join(repo, "data/raw/HMCDA/example5.npz"))
        return z["train_idx"], z["val_idx"], z["test_idx"]
    datalen = 2160
    arr = list(range(0, 2159))           # their range, which drops row 2159
    random.seed(0)
    random.shuffle(arr)
    s1, s2, s3, s4, s5, s6, s7, s8 = [int(datalen * f) for f in (0.7, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)]
    rot = {"rot1": (arr[:s1], arr[s1:s2], arr[s2:]),
           "rot2": (arr[:s4] + arr[s2:], arr[s4:s3], arr[s3:s2]),
           "rot3": (arr[:s6] + arr[s3:], arr[s6:s5], arr[s5:s3]),
           "rot4": (arr[:s8] + arr[s5:], arr[s8:s7], arr[s7:s5]),
           "rot5": (arr[s6:], arr[s7:s6], arr[:s7])}[name]
    return tuple(np.array(a) for a in rot)


def their_negatives(cd, train_idx, n_val, n_test):
    """Their notebook, verbatim logic: exclusion assumes the positives are sorted (they are not)."""
    np.random.seed(453289)
    neg = []
    counter = 0
    for i in range(N_C):
        for j in range(N_D):
            if counter < len(cd) and i == cd[counter, 0] and j == cd[counter, 1]:
                counter += 1
            else:
                neg.append([i, j])
    neg = np.array(neg)
    idx = np.random.choice(len(neg), n_val + n_test, replace=False)
    val_neg, test_neg = neg[sorted(idx[:n_val])], neg[sorted(idx[n_val:])]
    tr = cd[train_idx]
    trn = []
    counter = 0
    for i in range(N_C):
        for j in range(N_D):
            if counter < len(tr) and i == tr[counter, 0] and j == tr[counter, 1]:
                counter += 1
            else:
                trn.append([i, j])
    return np.array(trn), val_neg, test_neg


def key(p):
    return p[:, 0].astype(np.int64) * N_D + p[:, 1]


def dump(repo, split, work):
    cdf = raw(repo)[0]
    cd = cdf[["circrnaID", "diseaseID"]].to_numpy() - 1
    tr_i, va_i, te_i = split_idx(repo, split)
    out = {}
    trn, van, ten = their_negatives(cd, tr_i, len(va_i), len(te_i))
    out.update(released_train_pos=cd[tr_i], released_val_pos=cd[va_i], released_test_pos=cd[te_i],
               released_train_neg=trn, released_val_neg=van, released_test_neg=ten)
    # de-duplicated: unique pairs; a pair held out for test (then validation) never enters training
    uniq = lambda a: np.unique(a, axis=0)  # noqa: E731
    te = uniq(cd[te_i])
    va = uniq(cd[va_i])
    va = va[~np.isin(key(va), key(te))]
    tr = uniq(cd[tr_i])
    tr = tr[~np.isin(key(tr), np.r_[key(te), key(va)])]
    allk = key(uniq(cd))
    grid = np.array([[i, j] for i in range(N_C) for j in range(N_D)])
    rng = np.random.RandomState(453289)
    nonpos = grid[~np.isin(key(grid), allk)]
    idx = rng.choice(len(nonpos), len(va) + len(te), replace=False)
    out.update(dedup_train_pos=tr, dedup_val_pos=va, dedup_test_pos=te,
               dedup_train_neg=grid[~np.isin(key(grid), key(tr))],   # their intent: all but training positives
               dedup_val_neg=nonpos[np.sort(idx[:len(va)])], dedup_test_neg=nonpos[np.sort(idx[len(va):])],
               all_pos=uniq(cd))
    np.savez(os.path.join(work, "hmcda_%s_data.npz" % split), **out)
    dup = np.isin(key(cd[te_i]), key(cd[tr_i])).sum()
    print("dumped", split, "rows", len(cd), "unique", len(allk), "test positives also training rows:", dup,
          "of", len(te_i), "| dedup train/val/test", len(tr), len(va), len(te), flush=True)


def metapaths(repo, train_pos):
    """Their notebook's metapath construction (0-1-0, 0-2-0, 1-0-1, 1-2-1, 1-1) from the given training
    circRNA-disease pairs; returns what utils.data.load_HMCDA_data returns."""
    _, cm, md, dd = raw(repo)
    dim = N_C + N_M + N_D
    adjM = np.zeros((dim, dim), dtype=int)
    for c, d in train_pos:
        adjM[c, N_C + d] += 1
        adjM[N_C + d, c] += 1
    for _, row in cm.iterrows():
        cid, mid = row["circrnaID"] - 1, N_C + N_D + row["mirnaID"] - 1
        adjM[cid, mid] += 1
        adjM[mid, cid] += 1
    for _, row in md.iterrows():
        mid, did = N_C + N_D + row["mirnaID"] - 1, N_C + row["diseaseID"] - 1
        adjM[mid, did] += 1
        adjM[did, mid] += 1
    for _, row in dd.iterrows():
        adjM[N_C + row["diseaseID"] - 1, N_C + row["idiseaseID"] - 1] += 1
    type_mask = np.zeros(dim, dtype=int)
    type_mask[N_C:N_C + N_D] = 1
    type_mask[N_C + N_D:] = 2
    c_d = {i: adjM[i, N_C:N_C + N_D].nonzero()[0] for i in range(N_C)}
    d_c = {i: adjM[N_C + i, :N_C].nonzero()[0] for i in range(N_D)}
    m_c = {i: adjM[N_C + N_D + i, :N_C].nonzero()[0] for i in range(N_M)}
    m_d = {i: adjM[N_C + N_D + i, N_C:N_C + N_D].nonzero()[0] for i in range(N_M)}

    def srt(a, cols):
        a = np.array(a).reshape(-1, 3) if len(a) else np.zeros((0, 3), dtype=int)
        return a[np.lexsort([a[:, c] for c in cols[::-1]])] if len(a) else a
    c_d_c = [(c1, d, c2) for d, l in d_c.items() for c1 in l for c2 in l]
    c_d_c = np.array(c_d_c); c_d_c[:, 1] += N_C; c_d_c = srt(c_d_c, [0, 2, 1])
    c_m_c = [(c1, m, c2) for m, l in m_c.items() for c1 in l for c2 in l]
    c_m_c = np.array(c_m_c); c_m_c[:, 1] += N_C + N_D; c_m_c = srt(c_m_c, [0, 2, 1])
    d_c_d = [(d1, c, d2) for c, l in c_d.items() for d1 in l for d2 in l]
    d_c_d = np.array(d_c_d); d_c_d[:, [0, 2]] += N_C; d_c_d = srt(d_c_d, [0, 2, 1])
    d_m_d = [(d1, m, d2) for m, l in m_d.items() for d1 in l for d2 in l]
    d_m_d = np.array(d_m_d); d_m_d[:, 1] += N_C + N_D; d_m_d[:, [0, 2]] += N_C; d_m_d = srt(d_m_d, [0, 2, 1])
    d_d = dd.to_numpy(dtype=np.int32) - 1
    d_d = d_d[np.lexsort([d_d[:, 1], d_d[:, 0]])]     # their 1-1 rows carry no offset: never matched below
    adj, idx = [[], []], [[], []]
    for i, (mps, targets, off) in enumerate(((("c_d_c", c_d_c), ("c_m_c", c_m_c)), range(N_C), 0),
                                            ((("d_c_d", d_c_d), ("d_m_d", d_m_d), ("d_d", d_d)), range(N_D), N_C))):
        for _, arr in mps:
            lines, mapping = [], {}
            left = right = 0
            for t in targets:
                while right < len(arr) and arr[right, 0] == t + off:
                    right += 1
                mapping[t] = arr[left:right, ::-1]
                nb = arr[left:right, -1] - off
                lines.append(("%d " % t + " ".join(map(str, nb))).strip() if len(nb) else "%d" % t)
                left = right
            adj[i].append(lines)
            idx[i].append(mapping)
    return adj, idx, type_mask


def train(repo, split, variant, seed, work, epochs, patience):
    sys.path.insert(0, repo)
    os.chdir(repo)
    from model import MAGNN_lp
    from utils.tools import index_generator, parse_minibatch_HMCDA
    from utils.pytorchtools import EarlyStopping
    D = np.load(os.path.join(work, "hmcda_%s_data.npz" % split))
    Mt = np.load(os.path.join(work, "hmcda_%s_matched.npz" % split))
    g = lambda k: D["%s_%s" % (variant, k)]  # noqa: E731
    trp, vap, tep, trn, van, ten = (g(k) for k in ("train_pos", "val_pos", "test_pos", "train_neg", "val_neg",
                                                    "test_neg"))
    adjl, idxl, type_mask = metapaths(repo, trp)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    feats, in_dims = [], []
    for i in range(3):
        dim = (type_mask == i).sum()
        in_dims.append(dim)
        ind = torch.LongTensor(np.vstack((np.arange(dim), np.arange(dim))))
        feats.append(torch.sparse.FloatTensor(ind, torch.FloatTensor(np.ones(dim)), torch.Size([dim, dim])))
    net = MAGNN_lp([2, 2], 6, ETYPES, in_dims, 256, 256, 8, 128, "RotatE0", 0.01)
    opt = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
    ck = os.path.join(work, "ckpt_hmcda_%s_%s_s%d.pt" % (split, variant, seed))
    es = EarlyStopping(patience=patience, verbose=False, save_path=ck)

    def embed(pairs, masks):
        gl, il, ml = parse_minibatch_HMCDA(adjl, idxl, pairs, device, 100, masks, N_C)
        [ec, ed], _ = net((gl, feats, type_mask, il, ml))
        return torch.bmm(ec.view(-1, 1, ec.shape[1]), ed.view(-1, ed.shape[1], 1)).flatten()

    tr_gen = index_generator(batch_size=64, num_data=len(trp))
    va_gen = index_generator(batch_size=64, num_data=len(vap), shuffle=False)
    t0 = time.time()
    for epoch in range(epochs):
        net.train()
        for _ in range(tr_gen.num_iterations()):
            b = tr_gen.next(); b.sort()
            nb = np.random.choice(len(trn), len(b)); nb.sort()
            pos = embed(trp[b].tolist(), USE_MASKS)
            neg = -embed(trn[nb].tolist(), NO_MASKS)
            loss = -torch.mean(F.logsigmoid(pos) + F.logsigmoid(neg))
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        vl = []
        with torch.no_grad():
            for _ in range(va_gen.num_iterations()):
                b = va_gen.next()
                vl.append(-torch.mean(F.logsigmoid(embed(vap[b].tolist(), NO_MASKS)) +
                                      F.logsigmoid(-embed(van[b].tolist(), NO_MASKS))))
        vl = torch.mean(torch.tensor(vl))
        es(vl, net)
        print("%s %s seed %d epoch %d val_loss %.4f [%.0fs]" % (split, variant, seed, epoch, vl.item(),
                                                                time.time() - t0), flush=True)
        if es.early_stop:
            break
    net.load_state_dict(torch.load(ck))
    net.eval()

    def score(pairs):
        out = []
        with torch.no_grad():
            for s in range(0, len(pairs), 64):
                out.append(torch.sigmoid(embed(pairs[s:s + 64].tolist(), NO_MASKS)).numpy())
        return np.concatenate(out)
    np.random.seed(10_000 + seed)
    res = {"pos": score(tep), "theirs": score(ten), "epochs": epoch + 1}
    if variant == "dedup":
        res["matched_disease"] = score(Mt["disease"])
        res["matched_circrna"] = score(Mt["circrna"])
    auc = roc_auc_score(np.r_[np.ones(len(res["pos"])), np.zeros(len(res["theirs"]))], np.r_[res["pos"], res["theirs"]])
    np.savez(os.path.join(work, "hmcda_%s_%s_s%d_scores.npz" % (split, variant, seed)), **res)
    print("%s %s seed %d: test AUROC vs their negatives %.4f (epochs %d)" % (split, variant, seed, auc, epoch + 1),
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--variant", choices=["released", "dedup"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--dump", action="store_true")
    a = ap.parse_args()
    a.work = os.path.abspath(a.work)
    if a.dump:
        dump(a.repo, a.split, a.work)
    else:
        train(a.repo, a.split, a.variant, a.seed, a.work, a.epochs, a.patience)
