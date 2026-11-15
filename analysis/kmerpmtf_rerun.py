"""
kmerpmtf_rerun.py -- kmerPMTF's own released code (zhang2024kmerpmtf, plant miRNA-target), as released and
corrected step by step.

kmerPMTF (github.com/Weihankk/kmerPMTF @ c4f9a3e, no licence; data: figshare 24916023, CC BY 4.0) reports
AUROC 0.84 / 0.91 / 0.80 / 0.82 on Arabidopsis, rice, tomato and peach (its Table 2, 5-fold CV). What the
released step3_train.py does, per fold:

  * positives: every miRNA-gene pair in PmiREN; negatives: as many unlabeled pairs, drawn only among genes
    and miRNAs that have at least one positive (the paper says the test set holds "all the remaining
    negative samples");
  * training graph = every edge (gene-gene and miRNA-miRNA k-mer cosine >= 0.4, plus miRNA-gene positives)
    minus the fold's test pairs; TEST graph = every edge minus the fold's TRAINING pairs, so the held-out
    positives are edges of the graph the model is evaluated on;
  * model: GCNConv(cached=True) -> GCNConv(cached=True) -> ChebConv(K) -> MLP on the pair's embeddings.
    The two GCNConv layers cache the normalised training graph on their first call and reuse it at test
    time, so only the ChebConv layer sees the test graph, and with K = 1 (the paper's "limiting K = 1" and
    the code's default) ChebConv does not propagate at all. The README's example uses K = 2.
  * 3000 epochs, no validation, no epoch selection.

The preprocessing (their step1/step2) is reimplemented vectorised with the same outputs: their k-mer counter
does one pandas .loc per k-mer and would take days per genome. Seed matching keeps their str.count.

Steps, per species, seed and K in {1, 2}:
  released     their test graph at test time (cached layers as released)
  train_graph  the training graph at test time
and each model is scored on three negative sets for the same held-out positives:
  theirs       the fold's 1:1 negatives among genes and miRNAs with positives (as released)
  paper_text   every remaining unlabeled pair, as the paper describes (sub-sampled to 100x positives)
  matched      degree-matched negatives (exact sampler, gene degree from the training positives)
protocheck 0.2.0 audits each (encoder edges = the miRNA-gene edges the propagating layer sees at test).

Usage: python analysis/kmerpmtf_rerun.py --repo PATH --data PATH --species Prunus_persica [--seeds ...]
Writes results/comparison/kmerpmtf_rerun_<species>.json (suffix _smoke when --epochs < 3000).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from protocheck.audit import audit  # noqa: E402
from training.splits import degree_bins, sample_degree_matched_negatives_exact  # noqa: E402

FILES = {  # their README / data layout
    "Arabidopsis_thaliana": "Athaliana_167_TAIR10.transcript_primaryTranscriptOnly.fa",
    "Oryza_sativa": "Osativa_323_v7.0.transcript_primaryTranscriptOnly.fa",
    "Solanum_lycopersicum": "Solanum_lycopersicum.SL3.0.cdna.all.fa",
    "Prunus_persica": "Prunus_persica_v2.0.a1.primaryTrs.fa",
}
REPORTED = {"Arabidopsis_thaliana": 0.84, "Oryza_sativa": 0.91, "Solanum_lycopersicum": 0.80, "Prunus_persica": 0.82}
LUT = np.full(256, -1, np.int64)
for _i, _c in enumerate("ATCG"):   # their kmer_list is product('ATCG'); only ACGT k-mers are counted
    LUT[ord(_c)] = _i


def kmer_counts(seqs, k=7):
    w4 = 4 ** np.arange(k - 1, -1, -1)
    out = np.zeros((len(seqs), 4 ** k), np.float32)
    for r, s in enumerate(seqs):
        a = LUT[np.frombuffer(s.encode(), np.uint8)]
        if len(a) < k:
            continue
        w = np.lib.stride_tricks.sliding_window_view(a, k)
        w = w[(w >= 0).all(1)]
        out[r] = np.bincount(w @ w4, minlength=4 ** k)
    return out


def cosine(X):
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    return torch.from_numpy(Xn)


def thresholded_edges(X, thr, offset, dev):
    """(i, j, value) for every ordered pair with cosine >= thr (diagonal included, as their dense matrix)."""
    Xt = cosine(X).to(dev)
    rows, cols, vals = [], [], []
    for i in range(0, len(Xt), 2048):
        C = Xt[i:i + 2048] @ Xt.T
        r, c = torch.nonzero(C >= thr, as_tuple=True)
        rows.append((r + i + offset).cpu()), cols.append((c + offset).cpu()), vals.append(C[r, c].cpu())
    return torch.cat(rows).numpy(), torch.cat(cols).numpy(), torch.cat(vals).numpy()


def seed_features(mi_com, mi_rc, genes, match_length=15):
    """utils.stat_miRNA_in_gene, same seed order and the same str.count, without the per-cell .loc."""
    seed_list, seen = [], set()
    for name in mi_com:
        c, rc = mi_com[name], mi_rc[name]
        for j in range(len(c) - match_length + 1):
            for sd in (c[j:j + match_length], rc[j:j + match_length]):
                if sd not in seen:
                    seen.add(sd), seed_list.append(sd)
    col = {s: i for i, s in enumerate(seed_list)}
    G = np.zeros((len(genes), len(seed_list)), np.float32)
    for r, g in enumerate(genes.values()):
        present = {g[j:j + match_length] for j in range(len(g) - match_length + 1)} & seen
        for sd in present:
            G[r, col[sd]] = g.count(sd)
    M = np.zeros((len(mi_com), len(seed_list)), np.float32)
    for r, name in enumerate(mi_com):
        c, rc = mi_com[name], mi_rc[name]
        for sd, ci in col.items():
            if sd in c or sd in rc:
                M[r, ci] = c.count(sd) + rc.count(sd)
    return np.vstack([G, M])


def prepare(repo, data, species, work, dev):
    cache = work / f"{species}_prep.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    sys.path.insert(0, str(repo))
    import utils as U
    from model.DeepNeuralNetworks import DeepNeuralNetworks
    d = data / species
    mi = U.read_fasta_to_dict(str(d / "PmiREN" / f"{species}_mature.fa"))
    tr = U.read_fasta_to_dict(str(d / "Genome" / FILES[species]))
    mi_com, mi_rc = U.complement_fasta_dict(mi), U.reverse_complement_fasta_dict(mi)
    mi_kmer = kmer_counts(list(mi_com.values())) + kmer_counts(list(mi_rc.values()))
    tr_kmer = kmer_counts(list(tr.values()))
    genes, mirnas = list(tr), list(mi)
    n_g = len(genes)
    gg = thresholded_edges(tr_kmer, 0.4, 0, dev)
    mm = thresholded_edges(mi_kmer, 0.4, n_g, dev)
    tgt = pd.read_csv(d / "PmiREN" / f"{species}_targetGene.txt", sep="\t", header=0)
    gi, mj = {g: i for i, g in enumerate(genes)}, {m: j for j, m in enumerate(mirnas)}
    keep = tgt["target gene ID"].isin(gi) & tgt["miRNA locus ID"].isin(mj)
    pos = np.unique(np.c_[tgt.loc[keep, "target gene ID"].map(gi), tgt.loc[keep, "miRNA locus ID"].map(mj)], axis=0)
    attr = seed_features(mi_com, mi_rc, tr)
    # step2: their autoencoder, unchanged hyper-parameters (D 256, lr 1e-3, 500 epochs, MSE sum)
    torch.manual_seed(0)
    x = torch.from_numpy(attr).to(dev)
    ae = DeepNeuralNetworks(x.shape[1], 256).to(dev)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    crit = torch.nn.MSELoss(reduction="sum")
    for _ in range(500):
        opt.zero_grad()
        crit(ae(x)[1], x).backward()
        opt.step()
    with torch.no_grad():
        enc = ae(x)[0].cpu().numpy()
    out = dict(gg=np.stack(gg[:2]), gg_w=gg[2], mm=np.stack(mm[:2]), mm_w=mm[2], pos=pos, enc=enc,
               n_g=np.array(n_g), n_m=np.array(len(mirnas)), n_targets_in_file=np.array(len(tgt)))
    work.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def their_pairs(pos, n_g, n_m, seed):
    """step3_train.py's positive / negative lists: G_M.melt order (miRNA-major), shuffle, 1:1 sample."""
    random.seed(seed)
    np.random.seed(seed)
    Y = np.zeros((n_g, n_m), bool)
    Y[pos[:, 0], pos[:, 1]] = True
    gg, mm = np.meshgrid(np.arange(n_g), np.arange(n_m), indexing="xy")   # melt: column (miRNA) outer
    flat_g, flat_m = gg.ravel(), mm.ravel()
    lab = Y[flat_g, flat_m]
    P = np.c_[flat_g[lab], flat_m[lab]]
    pg, pm = np.unique(P[:, 0]), np.unique(P[:, 1])
    nmask = (~lab) & np.isin(flat_g, pg) & np.isin(flat_m, pm)
    N = np.c_[flat_g[nmask], flat_m[nmask]]
    ip, ineg = list(range(len(P))), list(range(len(N)))
    random.shuffle(ip), random.shuffle(ineg)
    P = P[ip]
    N = pd.DataFrame(N[ineg]).sample(len(P)).values   # pandas .sample -> numpy global RNG, as theirs
    return P, N, Y


def build_edges(prep, gm_pairs, n_g):
    """Their gcn edge list: every non-zero entry of the symmetric heterogeneous matrix (both orders), then
    concatenated with its reverse, so each edge enters twice per direction."""
    gm = np.asarray(gm_pairs).reshape(-1, 2)
    f = np.r_[prep["gg"][0], prep["mm"][0], gm[:, 0], gm[:, 1] + n_g]
    t = np.r_[prep["gg"][1], prep["mm"][1], gm[:, 1] + n_g, gm[:, 0]]
    w = np.r_[prep["gg_w"], prep["mm_w"], np.ones(2 * len(gm), np.float32)]
    ei = torch.from_numpy(np.c_[np.r_[f, t], np.r_[t, f]].T.copy()).long()
    return ei, torch.from_numpy(np.r_[w, w]).float().view(-1, 1)


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def audit_step(tr_pos_mg, te_pos_mg, enc_mg, tr_neg_mg, ev_neg_mg, model_auc, n_m, n_g):
    rep = audit(n_m, n_g, tr_pos_mg.T, te_pos_mg.T, encoder_edges=enc_mg.T, train_neg=tr_neg_mg.T,
                eval_neg=ev_neg_mg.T, model_auroc=model_auc)
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
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--species", required=True, choices=list(FILES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[123, 1, 2])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--work", type=Path, default=ROOT / "results/kmerpmtf_work")
    a = ap.parse_args()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    prep = prepare(a.repo, a.data, a.species, a.work, dev)
    n_g, n_m = int(prep["n_g"]), int(prep["n_m"])
    print(f"{a.species}: {n_g} genes, {n_m} miRNAs, {len(prep['pos'])} positives "
          f"({int(prep['n_targets_in_file'])} rows in the target file), {prep['gg'].shape[1]} gene-gene and "
          f"{prep['mm'].shape[1]} miRNA-miRNA entries >= 0.4 [{time.time() - t0:.0f}s]", flush=True)
    sys.path.insert(0, str(a.repo))
    from model.ChebConvConvolutional import ChebConvConvolutional
    from model.MLP import MLP
    x = torch.from_numpy(prep["enc"]).float().to(dev)
    all_pos_mg = torch.from_numpy(prep["pos"][:, ::-1].T.copy()).long()   # (miRNA, gene)
    out = {"meta": {"species": a.species, "reported_auroc": REPORTED[a.species], "seeds": a.seeds, "ks": a.ks,
                    "epochs": a.epochs, "n_genes": n_g, "n_mirnas": n_m, "n_pos": int(len(prep["pos"]))},
           "runs": []}
    from sklearn.model_selection import KFold
    for seed in a.seeds:
        P, N, Y = their_pairs(prep["pos"], n_g, n_m, seed)
        unl = np.argwhere(~Y)   # (gene, miRNA): the paper's "all the remaining negative samples"
        for k, (tr_i, te_i) in enumerate(list(KFold(n_splits=a.folds).split(P))):
            trP, teP, trN, teN = P[tr_i], P[te_i], N[tr_i], N[te_i]
            tr_set = {tuple(p) for p in np.r_[trP, trN].tolist()}
            te_set = {tuple(p) for p in np.r_[teP, teN].tolist()}
            all_set = [tuple(p) for p in prep["pos"].tolist()]
            gm_train = [p for p in all_set if p not in te_set]   # = their gcn_edges_train (G-M part)
            gm_test = [p for p in all_set if p not in tr_set]    # = their gcn_edges_test (G-M part)
            ei_tr, w_tr = build_edges(prep, gm_train, n_g)
            ei_te, w_te = build_edges(prep, gm_test, n_g)
            col_deg = torch.from_numpy(np.bincount(trP[:, 0], minlength=n_g)).long()
            mneg = sample_degree_matched_negatives_exact(
                torch.from_numpy(teP[:, ::-1].T.copy()).long(), all_pos_mg, degree_bins(col_deg), n_g,
                torch.Generator().manual_seed(1000 * seed + k))[0].numpy().T[:, ::-1]   # back to (gene, miRNA)
            rng = np.random.default_rng(100 * seed + k)
            pool = unl[~np.isin(unl[:, 0] * n_m + unl[:, 1], trN[:, 0] * n_m + trN[:, 1])]
            ptxt = pool[rng.choice(len(pool), size=min(len(pool), 100 * len(teP)), replace=False)]
            negsets = {"theirs": teN, "paper_text": ptxt, "matched": mneg}
            for K in a.ks:
                seed_all(seed)
                gm = ChebConvConvolutional(node_attr=x, ChebConvK=K, dim_l1=2048, dim_l2=1024, dim_output=512).to(dev)
                mlp = MLP(dim_input=1024, dim_l1=256, dim_l2=64).to(dev)
                opt = torch.optim.Adam([{"params": gm.parameters(), "lr": 1e-4},
                                        {"params": mlp.parameters(), "lr": 1e-4}])
                crit = torch.nn.BCELoss()
                tr_pairs = np.r_[trP, trN]
                tr_lab = torch.tensor(np.r_[np.ones(len(trP)), np.zeros(len(trN))], dtype=torch.float32, device=dev)
                fi, ti = torch.from_numpy(tr_pairs[:, 0]).to(dev), torch.from_numpy(tr_pairs[:, 1] + n_g).to(dev)
                e_tr, ww_tr = ei_tr.to(dev), w_tr.to(dev)
                gm.train(), mlp.train()
                for _ in range(a.epochs):
                    opt.zero_grad()
                    h = gm(x, e_tr, ww_tr)
                    loss = crit(mlp(torch.cat((h[fi], h[ti]), 1)).squeeze(), tr_lab)
                    loss.backward()
                    opt.step()
                gm.eval(), mlp.eval()
                rec = {"seed": seed, "fold": k + 1, "K": K, "n_test_pos": int(len(teP))}
                for gname, (ei, w) in {"released": (ei_te, w_te), "train_graph": (ei_tr, w_tr)}.items():
                    with torch.no_grad():
                        h = gm(x, ei.to(dev), w.to(dev))   # GCNConv layers reuse their cached training graph
                    rec[gname] = {}
                    enc_gm = np.array(gm_test if (gname == "released" and K >= 2) else gm_train)
                    for nname, neg in negsets.items():
                        pairs = np.r_[teP, neg]
                        y = np.r_[np.ones(len(teP)), np.zeros(len(neg))]
                        with torch.no_grad():
                            s = torch.cat([mlp(torch.cat((h[torch.from_numpy(pairs[i:i + 500000, 0]).to(dev)],
                                                          h[torch.from_numpy(pairs[i:i + 500000, 1] + n_g).to(dev)]), 1)).view(-1)
                                           for i in range(0, len(pairs), 500000)]).cpu().numpy()
                        v = auc(y, s)
                        rec[gname][nname] = {"auc": v, "audit": audit_step(
                            trP[:, ::-1], teP[:, ::-1], enc_gm[:, ::-1], trN[:, ::-1], neg[:, ::-1], v, n_m, n_g)}
                out["runs"].append(rec)
                r = rec
                fl = lambda g, n: r[g][n]["audit"]["model_free_baseline"]["floor"]  # noqa: E731
                print(f"seed {seed} fold {k + 1} K={K}: released theirs {r['released']['theirs']['auc']:.4f} "
                      f"(floor {fl('released', 'theirs'):.4f}) | train graph theirs {r['train_graph']['theirs']['auc']:.4f} "
                      f"paper-text {r['train_graph']['paper_text']['auc']:.4f} (floor {fl('train_graph', 'paper_text'):.4f}) "
                      f"matched {r['train_graph']['matched']['auc']:.4f} (floor {fl('train_graph', 'matched'):.4f}) "
                      f"[{time.time() - t0:.0f}s]", flush=True)

    summ = {}
    for K in a.ks:
        for g in ("released", "train_graph"):
            for n in ("theirs", "paper_text", "matched"):
                per_seed = [st.mean(r[g][n]["auc"] for r in out["runs"] if r["seed"] == s and r["K"] == K) for s in a.seeds]
                fl_seed = [st.mean(r[g][n]["audit"]["model_free_baseline"]["floor"] for r in out["runs"]
                                   if r["seed"] == s and r["K"] == K) for s in a.seeds]
                summ[f"K{K}_{g}_{n}"] = {"auc": st.mean(per_seed), "sd": st.stdev(per_seed) if len(per_seed) > 1 else 0.0,
                                         "floor": st.mean(fl_seed)}
    out["summary"] = summ
    print("\n=== summary ===\n", json.dumps(summ, indent=1))
    tag = "" if a.epochs >= 3000 else "_smoke"
    path = ROOT / f"results/comparison/kmerpmtf_rerun_{a.species}{tag}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print("Wrote", path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    main()
