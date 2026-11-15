"""
hgdti_negatives_arm.py -- does HGDTI's "reliable" negative sampling make a better MODEL, or an easier TEST?

HGDTI's Fig. 8 reports AUROC 0.987 with its reliable negatives against 0.953 with random sampling and
concludes that "the importance of negative sampling technology is self-evident". In its code the
negatives are sampled once, before the cross-validation split (sample_links, then StratifiedKFold), so
switching the sampler changes the TEST negatives as well as the training ones: the figure compares two
exams, not two models. Without any model, HGDTI's own similarity score separates the positives from its
reliable negatives at AUROC 0.937 and from uniform ones at 0.824 (BITACORA 2026-11-14).

This completes the 2 x 3 design that answers the question with HGDTI's own code:

  training negatives   reliable (analysis/hgdti_rerun.py, steps 2-4)  |  random (this script, arm B)
  test negatives       the fold's reliable ones, uniform, degree-matched (the same three sets in both arms)

  arm B   same folds, same training/validation/test positives as hgdti_rerun.py (rebuilt in the same RNG
          order); training and validation negatives drawn uniformly from all unknown pairs (as many as
          the reliable arm has, never one of the fold's test negatives); every pass on the training
          adjacency; epoch by validation AUPR (hgdti_rerun.train_with_train_graph)
  arm C   Fig. 8's random-sampling bar as the released code runs it: 10 uniform negatives per positive
          drawn once, then their StratifiedKFold / train_test_split / train_and_evaluate unchanged
          (test adjacency from the test labels)

Usage: python analysis/hgdti_negatives_arm.py --repo PATH --seeds 10 [--epochs 1000]
Writes results/comparison/hgdti_negatives_arm_s<seeds>.json (suffix _smoke for short runs).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from hgdti_rerun import (  # noqa: E402
    P_N, auc, load_their_module, pairs, run_model, train_with_train_graph,
)
from training.splits import degree_bins, sample_degree_matched_negatives_exact  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[10])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=10)
    a_cli = ap.parse_args()
    argv0 = sys.argv[:]
    cuda = torch.cuda.is_available()
    mod = load_their_module(a_cli.repo, cuda)
    sys.argv = argv0
    mod.args.epochs = a_cli.epochs
    from sklearn.model_selection import StratifiedKFold, train_test_split

    Y = np.loadtxt(a_cli.repo / "data" / "mat_drug_protein.txt")
    all_pos = torch.from_numpy(np.argwhere(Y == 1).T.copy()).long()
    unknown = np.argwhere(Y == 0)
    out = {"meta": {"epochs": a_cli.epochs, "seeds": a_cli.seeds, "cuda": cuda}, "arm_B": [], "arm_C": []}
    t0 = time.time()
    for seed in a_cli.seeds:
        # ---- rebuild hgdti_rerun.py's folds exactly (same RNG order, nothing else drawn in between)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        mo = mod.model_class(mod.args)
        wp, wn, ns, wpt, wnt, nst = mo.sample_neg_links(0.5)
        data_set, _ = mo.sample_links(wp, wn, ns, wpt, wnt, nst)
        kf = StratifiedKFold(n_splits=10, shuffle=True)
        folds = []
        for tr_i, te_i in kf.split(data_set[:, :2], data_set[:, 2]):
            DTItrain, DTItest = data_set[tr_i], data_set[te_i]
            DTItrain, DTIvalid = train_test_split(DTItrain, test_size=0.1)
            folds.append((DTItrain, DTIvalid, DTItest))

        # ---- arm B: random training negatives, evaluated on the reliable arm's three test sets
        for k, (DTItrain, DTIvalid, DTItest) in enumerate(folds[:a_cli.folds]):
            te_pos = pairs(DTItest[DTItest[:, 2] == 1])
            # the same three test sets as hgdti_rerun.py (same generators)
            used = set(map(tuple, np.vstack([DTItrain[:, :2], DTIvalid[:, :2]]).tolist()))
            cand = unknown[[tuple(c) not in used for c in unknown.tolist()]]
            rng = np.random.default_rng(100 * seed + k)
            un = cand[rng.choice(len(cand), size=10 * te_pos.shape[1], replace=False)]
            tr_pos_rel = pairs(DTItrain[DTItrain[:, 2] == 1])
            col_deg = torch.from_numpy(np.bincount(tr_pos_rel[1], minlength=P_N)).long()
            mneg = sample_degree_matched_negatives_exact(
                torch.from_numpy(te_pos).long(), all_pos, degree_bins(col_deg), P_N,
                torch.Generator().manual_seed(1000 * seed + k))[0].numpy()
            pos_rows = np.c_[te_pos.T, np.ones(te_pos.shape[1], int)]
            tests = {"reliable": DTItest,
                     "uniform": np.vstack([pos_rows, np.c_[un, np.zeros(len(un), int)]]).astype(int),
                     "matched": np.vstack([pos_rows, np.c_[mneg.T, np.zeros(mneg.shape[1], int)]]).astype(int)}
            # random training / validation negatives, disjoint from every test negative of this fold
            forbid = set(map(tuple, np.vstack([DTItest[:, :2], un, mneg.T]).tolist()))
            pool = unknown[[tuple(c) not in forbid for c in unknown.tolist()]]
            n_tr = int((DTItrain[:, 2] == 0).sum())
            n_va = int((DTIvalid[:, 2] == 0).sum())
            rng_b = np.random.default_rng(5000 + 100 * seed + k)
            pick = pool[rng_b.choice(len(pool), size=n_tr + n_va, replace=False)]
            trainB = np.vstack([DTItrain[DTItrain[:, 2] == 1], np.c_[pick[:n_tr], np.zeros(n_tr, int)]]).astype(int)
            validB = np.vstack([DTIvalid[DTIvalid[:, 2] == 1], np.c_[pick[n_tr:], np.zeros(n_va, int)]]).astype(int)
            model, adj_tr, _ = train_with_train_graph(mod, mo, trainB, validB, a_cli.epochs)
            rec = {"seed": seed, "fold": k + 1}
            for name, T in tests.items():
                rec[name] = auc(run_model(mod, mo, model, T, adj_tr), T[:, 2])
            out["arm_B"].append(rec)
            print(f"seed {seed} fold {k + 1} arm B (random-trained): reliable-test {rec['reliable']:.4f} | "
                  f"uniform-test {rec['uniform']:.4f} | matched-test {rec['matched']:.4f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)

        # ---- arm C: Fig. 8's random-sampling bar, the released pipeline with uniform negatives
        rng_c = np.random.default_rng(9000 + seed)
        negC = unknown[rng_c.choice(len(unknown), size=10 * len(wp), replace=False)]
        dataC = np.vstack([np.c_[np.asarray(wp), np.ones(len(wp), int)],
                           np.c_[negC, np.zeros(len(negC), int)]]).astype(int)
        kfC = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        for k, (tr_i, te_i) in enumerate(list(kfC.split(dataC[:, :2], dataC[:, 2]))[:a_cli.folds]):
            DTItrain, DTItest = dataC[tr_i], dataC[te_i]
            DTItrain, DTIvalid = train_test_split(DTItrain, test_size=0.1, random_state=seed * 100 + k)
            aucC, auprC = mo.train_and_evaluate(DTItrain=DTItrain, DTIvalid=DTIvalid, DTItest=DTItest,
                                                num_step=k + 1)
            out["arm_C"].append({"seed": seed, "fold": k + 1, "auc": float(aucC), "aupr": float(auprC)})
            print(f"seed {seed} fold {k + 1} arm C (Fig. 8 random bar, released pipeline): "
                  f"AUROC {aucC:.4f} AUPR {auprC:.4f} [{time.time() - t0:.0f}s]", flush=True)

    summ = {"arm_B": {}, "arm_C": {}}
    for name in ("reliable", "uniform", "matched"):
        v = [st.mean(r[name] for r in out["arm_B"] if r["seed"] == s) for s in a_cli.seeds]
        summ["arm_B"][name] = {"mean": st.mean(v), "per_seed": v}
    for m in ("auc", "aupr"):
        v = [st.mean(r[m] for r in out["arm_C"] if r["seed"] == s) for s in a_cli.seeds]
        summ["arm_C"][m] = {"mean": st.mean(v), "per_seed": v}
    out["summary"] = summ
    print("\n=== summary ===\n", json.dumps(summ, indent=1))
    tag = "_s" + "-".join(map(str, a_cli.seeds)) + ("" if (a_cli.epochs >= 1000 and a_cli.folds >= 10) else "_smoke")
    path = ROOT / f"results/comparison/hgdti_negatives_arm{tag}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print("Wrote", path)


if __name__ == "__main__":
    main()
