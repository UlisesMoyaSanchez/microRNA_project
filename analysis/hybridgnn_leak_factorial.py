"""
hybridgnn_leak_factorial.py -- which of HybridGNN's three leaks inflates its AUROC, and by how much,
independently of the order in which they are removed.

analysis/hybridgnn_rerun.py removed the three full-matrix leaks of HybridGNN's released code together
(step 3) and then changed the negatives (step 4), so its attribution depends on that order. Here each
leak is a factor with two levels, built from the FULL association matrix (as released) or from the
fold's fitting positives only:

  G  graph    the message-passing edge_index (their prepare_gcn_data edges)
  S  kernel   the GIP similarity whose PCA is the node feature (their compute_similarity + PCA)
  M  mdmf     the matrix-factorisation feature (their MDMF, trained on A with the similarities as
              regularisers; the 'clean' level uses the fold's A and the fold's similarities)

All 2^3 = 8 combinations are trained per fold, with the epoch selected on a validation split (same
carve-out as hybridgnn_rerun.py, so selection is not a factor), and each model is scored on uniform
test negatives (as released) and on degree-matched ones (N, the fourth factor). The summary gives
every cell's mean AUROC and the Shapley value of each of the four factors on it: the average, over
all 24 orders of removing them, of the AUROC lost when that factor is removed.

Usage:
  python analysis/hybridgnn_leak_factorial.py --repo PATH --dataset hmdd_v4|canonical [--seeds ...]
Writes results/comparison/hybridgnn_leak_factorial_<dataset>.json (suffix _smoke when --epochs < 200).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from hybridgnn_rerun import (  # noqa: E402
    EPOCHS, PARAMS, auc_on, flat_to_pairs, load_matrix, load_their_module, logits_all, new_model,
    set_globals, train_select_on_val,
)
from training.splits import degree_bins, sample_degree_matched_negatives_exact  # noqa: E402

FACTORS = ("G", "S", "M", "N")  # graph, kernel, mdmf leaks; uniform negatives


def features(mod, mat_graph, sim_s, mat_m, sim_m, n_m, n_d):
    """Their pipeline with each input chosen separately: edges from mat_graph, PCA of sim_s,
    MDMF on (mat_m, sim_m)."""
    data, _, _ = mod.prepare_gcn_data(mat_graph, sim_s[0], sim_s[1], feature_dim=PARAMS["pca_dim"])
    data.n_mirnas = n_m
    mdmf = mod.MDMF(num_mirnas=n_m, num_diseases=n_d, latent_dim=PARAMS["latent_dim"],
                    lambda_reg=PARAMS["lambda_reg"])
    U, V = mod.train_mdmf(mdmf, torch.tensor(mat_m.values, dtype=torch.float),
                          torch.tensor(sim_m[0].values, dtype=torch.float),
                          torch.tensor(sim_m[1].values, dtype=torch.float))
    data.x = torch.cat([data.x, torch.cat([U, V], dim=0).to(mod.device)], dim=1)
    return data


def shapley(cell: dict) -> dict:
    """cell[(g,s,m,n)] = AUROC, 1 = leaky level. Shapley value of each factor's leak."""
    out = {}
    for i, f in enumerate(FACTORS):
        tot = 0.0
        for order in itertools.permutations(range(4)):
            state = [1, 1, 1, 1]
            for j in order:
                if j == i:
                    before = cell[tuple(state)]
                    state[j] = 0
                    tot += before - cell[tuple(state)]
                    break
                state[j] = 0
        out[f] = tot / math.factorial(4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--dataset", required=True, choices=["hmdd_v4", "canonical"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    mod = load_their_module(args.repo)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mat = load_matrix(mod, args.dataset, args.repo)
    n_m, n_d = mat.shape
    Y = mat.values.astype(np.int64)
    all_pos_t = torch.from_numpy(flat_to_pairs(np.flatnonzero(Y.ravel() == 1), n_d)).long()
    print(f"{args.dataset}: {n_m} x {n_d}, device {dev}", flush=True)

    out = {"meta": {"repo_commit": "f441278", "dataset": args.dataset, "params": PARAMS,
                    "epochs": args.epochs, "seeds": args.seeds, "factors": {
                        "G": "message-passing graph from the full matrix",
                        "S": "GIP similarity (PCA node feature) from the full matrix",
                        "M": "MDMF feature trained on the full matrix",
                        "N": "uniform test negatives (0 = degree-matched)"}},
           "runs": []}
    t0 = time.time()
    for seed in args.seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        set_globals(mod, seed=seed, device=dev, matrix=mat, n_mirnas=n_m, n_diseases=n_d,
                    epochs=args.epochs, patience=20, output_folder="unused")
        sim_full = mod.compute_similarity(mat)
        folds = mod.get_cv_folds(mat, split_mode="random", n_folds=5)
        for k, (tr_mask, tr_lab, te_mask, te_lab) in enumerate(folds[:args.folds]):
            tr_lab, te_lab = tr_lab.numpy(), te_lab.numpy()
            tr_pos, tr_neg = tr_mask[tr_lab == 1], tr_mask[tr_lab == 0]
            te_pos = te_mask[te_lab == 1]
            rng = np.random.default_rng(10_000 + 100 * seed + k)  # = hybridgnn_rerun.py
            va_pos = rng.choice(tr_pos, size=len(tr_pos) // 8, replace=False)
            va_neg = rng.choice(tr_neg, size=len(va_pos), replace=False)
            fit_pos, fit_neg = np.setdiff1d(tr_pos, va_pos), np.setdiff1d(tr_neg, va_neg)
            fit_idx = np.concatenate([fit_pos, fit_neg])
            fit_lab = np.r_[np.ones(len(fit_pos)), np.zeros(len(fit_neg))]
            va_idx = np.concatenate([va_pos, va_neg])
            va_lab = np.r_[np.ones(len(va_pos)), np.zeros(len(va_neg))]

            Ytr = Y.copy().ravel()
            Ytr[np.concatenate([te_pos, va_pos])] = 0
            mat_tr = pd.DataFrame(Ytr.reshape(n_m, n_d), index=mat.index, columns=mat.columns)
            mat_tr.index.name, mat_tr.columns.name = "miRNA", "disease"
            sim_tr = mod.compute_similarity(mat_tr)

            col_deg = torch.from_numpy(Ytr.reshape(n_m, n_d).sum(0)).long()
            neg, _, _ = sample_degree_matched_negatives_exact(
                torch.from_numpy(flat_to_pairs(te_pos, n_d)).long(), all_pos_t, degree_bins(col_deg),
                n_d, torch.Generator().manual_seed(1000 * seed + k))  # = hybridgnn_rerun.py
            neg_flat = (neg[0] * n_d + neg[1]).numpy()
            m_idx = np.concatenate([te_pos, neg_flat])
            m_lab = np.r_[np.ones(len(te_pos)), np.zeros(len(neg_flat))]

            rec = {"seed": seed, "fold": k + 1, "cells": {}}
            for g, s_, m in itertools.product((1, 0), repeat=3):
                data = features(mod, mat if g else mat_tr, sim_full if s_ else sim_tr,
                                mat if m else mat_tr, sim_full if m else sim_tr, n_m, n_d)
                model = new_model(mod, data)
                train_select_on_val(mod, model, data, fit_idx, fit_lab, va_idx, va_lab, args.epochs, n_m)
                lg = logits_all(model, data, n_m)
                rec["cells"][f"{g}{s_}{m}1"] = auc_on(lg, te_mask, te_lab)
                rec["cells"][f"{g}{s_}{m}0"] = auc_on(lg, m_idx, m_lab)
            out["runs"].append(rec)
            c = rec["cells"]
            print(f"seed {seed} fold {k + 1}: all leaks {c['1111']:.4f} | none {c['0001']:.4f} | "
                  f"G only {c['1001']:.4f} S only {c['0101']:.4f} M only {c['0011']:.4f} | "
                  f"matched, none {c['0000']:.4f} [{time.time() - t0:.0f}s]", flush=True)

    keys = sorted(out["runs"][0]["cells"], reverse=True)
    mean_cell = {}
    for key in keys:
        per_seed = [st.mean(r["cells"][key] for r in out["runs"] if r["seed"] == s) for s in args.seeds]
        mean_cell[key] = {"mean": st.mean(per_seed), "sd": st.stdev(per_seed) if len(per_seed) > 1 else 0.0}
    sh_seed = []
    for s in args.seeds:
        cell = {tuple(int(ch) for ch in key): st.mean(r["cells"][key] for r in out["runs"] if r["seed"] == s)
                for key in keys}
        sh_seed.append(shapley(cell))
    out["summary"] = {"cells": mean_cell,
                      "shapley": {f: {"mean": st.mean(x[f] for x in sh_seed),
                                      "sd": st.stdev(x[f] for x in sh_seed) if len(sh_seed) > 1 else 0.0}
                                  for f in FACTORS},
                      "total": mean_cell["1111"]["mean"] - mean_cell["0000"]["mean"]}
    print("\n=== cells (G S M N; 1 = leaky / uniform) ===")
    for key in keys:
        print(f"  {key}  {mean_cell[key]['mean']:.4f} +- {mean_cell[key]['sd']:.4f}")
    print("=== Shapley share of the total inflation "
          f"({out['summary']['total']:.4f}) ===")
    for f in FACTORS:
        v = out["summary"]["shapley"][f]
        print(f"  {f}  {v['mean']:+.4f} +- {v['sd']:.4f}")
    tag = "" if args.epochs >= EPOCHS else "_smoke"
    path = ROOT / f"results/comparison/hybridgnn_leak_factorial_{args.dataset}{tag}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print("Wrote", path)


if __name__ == "__main__":
    main()
