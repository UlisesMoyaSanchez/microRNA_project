"""
verify_graph_construction.py — Independent audit that a built heterograph's edges are
what the config and docstrings claim, recomputed from the processed AnnData WITHOUT
reusing build_heterograph's edge builders.

Motivation: the two graph-construction bugs this project is about (co-expression on the
first-1000 genes instead of the most variable; the cell→gene threshold applied to z-scaled
data instead of log-norm) passed every green log. Rebuilding on a "fixed" graph earns no
trust until the new artifact is checked the same way. Reusing the build code to check the
build would hide exactly the bug class we care about, so this recomputes independently:
Pearson via numpy.corrcoef (not the build's manual unit-norming), the threshold count
directly from adata.raw.

For each edge type, three checks:
  EXACT      — recompute the edge set's size independently; it must equal the graph's count.
  COUNTERFACTUAL — compute what the PRE-FIX (buggy) construction would have produced and
               confirm it DIFFERS, so we know the fix actually changed the artifact.
  SPOT       — sample emitted edges and confirm each satisfies the stated predicate.

Read-only. CPU only. Exits non-zero if any EXACT/SPOT/membership check fails.

Usage:
  python analysis/verify_graph_construction.py --config configs/config_v3fixed_edgesplit.yaml
"""

from __future__ import annotations

import os
import sys
import pickle
import argparse
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import anndata as ad

REL_FWD = ("miRNA", "regulates", "gene")
REL_EXPR = ("cell", "expresses", "gene")
REL_COEXPR = ("gene", "coexpressed_with", "gene")


def _raw_hvg_matrix(adata: ad.AnnData):
    """adata.raw log-norm expression, subset to the HVG columns in HVG order so column j
    is gene j in gene2idx — the same view build_gene_features/build_expression_edges use,
    but assembled here independently from adata.raw."""
    if adata.raw is None or adata.raw.X is None:
        raise SystemExit("adata.raw is absent — cannot verify the log-norm threshold.")
    raw_var = list(adata.raw.var_names)
    hvg = list(adata.var_names)
    missing = [g for g in hvg if g not in set(raw_var)]
    if missing:
        raise SystemExit(f"{len(missing)} HVG genes absent from adata.raw (e.g. {missing[:3]}) "
                         "— the col_idx remap would be misaligned.")
    col_idx = [raw_var.index(g) for g in hvg]
    X = adata.raw.X[:, col_idx]
    return X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)


def check_expresses(graph, adata, threshold: float) -> bool:
    print("\n" + "=" * 76)
    print(f"cell→gene 'expresses' edges — threshold {threshold} on LOG-NORM expression")
    print("=" * 76)
    emitted = int(graph[REL_EXPR].edge_index.shape[1])

    raw = _raw_hvg_matrix(adata)
    # EXACT: entries of the log-norm matrix strictly above threshold (zeros are structural
    # in log-norm and sit below 0.10, so counting stored nnz > thr is the whole edge set).
    raw_count = int((raw.data > threshold).sum())

    # COUNTERFACTUAL: the pre-fix bug thresholded the z-scaled adata.X. Reproduce that count
    # so we can see the fix moved the artifact off it.
    Xs = adata.X
    Xs = Xs.toarray() if sp.issparse(Xs) else np.asarray(Xs)
    scaled_count = int((Xs > threshold).sum())

    ok_exact = (raw_count == emitted)
    ok_cf = (scaled_count != emitted)
    print(f"  graph emitted            : {emitted:,}")
    print(f"  independent log-norm >thr: {raw_count:,}   [{'MATCH' if ok_exact else 'MISMATCH'}]")
    print(f"  counterfactual z-scaled  : {scaled_count:,}   "
          f"[{'differs (good)' if ok_cf else 'SAME AS EMITTED — threshold still on scaled X!'}]")

    # SPOT: sample emitted edges, confirm each has log-norm > thr.
    ei = graph[REL_EXPR].edge_index
    n = min(2000, ei.shape[1])
    sel = torch.randperm(ei.shape[1])[:n]
    rows = ei[0, sel].numpy(); cols = ei[1, sel].numpy()
    vals = np.asarray(raw[rows, cols]).ravel()
    n_bad = int((vals <= threshold).sum())
    print(f"  spot-check {n} emitted edges: {n_bad} with log-norm ≤ {threshold} "
          f"[{'ok' if n_bad == 0 else 'FAIL'}]  (min sampled value {vals.min():.4f})")
    return ok_exact and ok_cf and n_bad == 0


def check_coexpression(graph, adata, threshold: float, top_n: int) -> bool:
    print("\n" + "=" * 76)
    print(f"gene–gene 'coexpressed_with' edges — |PCC| via numpy on top-{top_n} variable genes")
    print("=" * 76)
    emitted_ei = graph[REL_COEXPR].edge_index
    emitted = int(emitted_ei.shape[1])

    var_col = next((c for c in ("dispersions_norm", "dispersions", "variances_norm")
                    if c in adata.var.columns), None)
    if var_col is None:
        raise SystemExit(f"no variability column in adata.var; have {list(adata.var.columns)[:8]}")
    n = min(top_n, adata.n_vars)
    top_idx = np.argsort(adata.var[var_col].values)[::-1][:n]
    top_set = set(int(i) for i in top_idx)

    # MEMBERSHIP: every emitted endpoint must be one of the top-n most-variable genes.
    ep = np.unique(emitted_ei.numpy())
    outside = [int(g) for g in ep if int(g) not in top_set]
    ok_member = (len(outside) == 0)
    print(f"  emitted endpoints among top-{n} variable genes: "
          f"{'ALL' if ok_member else f'{len(outside)} OUTSIDE (remap bug?)'}")

    # EXACT: independent Pearson via numpy.corrcoef (not the build's manual unit-norming),
    # on the scaled X restricted to the top-n genes. Count directed off-diagonal > thr.
    X = adata.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    sub = X[:, top_idx].astype(np.float64)
    C = np.corrcoef(sub, rowvar=False)
    np.fill_diagonal(C, 0.0)
    indep_count = int((C > threshold).sum())
    ok_exact = (indep_count == emitted)
    print(f"  graph emitted              : {emitted:,}")
    print(f"  independent corrcoef >thr  : {indep_count:,}   [{'MATCH' if ok_exact else 'MISMATCH'}]")

    # COUNTERFACTUAL: the pre-fix bug used the FIRST top_n columns in var order.
    first = X[:, :n].astype(np.float64)
    Cf = np.corrcoef(first, rowvar=False)
    np.fill_diagonal(Cf, 0.0)
    cf_count = int((Cf > threshold).sum())
    ok_cf = (cf_count != emitted)
    print(f"  counterfactual first-{n}     : {cf_count:,}   "
          f"[{'differs (good)' if ok_cf else 'SAME AS EMITTED — selection unchanged!'}]")

    # SPOT: sample emitted edges, recompute their pairwise PCC independently, confirm > thr.
    n_spot = min(500, emitted)
    sel = torch.randperm(emitted)[:n_spot]
    g2col = {int(g): j for j, g in enumerate(top_idx)}
    n_bad = 0; worst = 1.0
    for k in sel.tolist():
        a = int(emitted_ei[0, k]); b = int(emitted_ei[1, k])
        if a not in g2col or b not in g2col:
            n_bad += 1; continue
        r = float(np.corrcoef(sub[:, g2col[a]], sub[:, g2col[b]])[0, 1])
        worst = min(worst, r)
        if r <= threshold:
            n_bad += 1
    print(f"  spot-check {n_spot} emitted edges: {n_bad} with PCC ≤ {threshold} "
          f"[{'ok' if n_bad == 0 else 'FAIL'}]  (weakest sampled corr {worst:.4f})")
    return ok_member and ok_exact and ok_cf and n_bad == 0


def check_mirna_gene(graph, index_maps, cfg) -> bool:
    print("\n" + "=" * 76)
    print("miRNA→gene 'regulates' edges — invariant to the two fixes (sanity)")
    print("=" * 76)
    emitted = int(graph[REL_FWD].edge_index.shape[1])
    interactions_file = cfg["data"].get("mirna", {}).get("interactions_file", "mirtarbase_hsa.tsv")
    path = os.path.join(cfg["data"]["raw_dir"], interactions_file)
    df = pd.read_csv(path, sep="\t")
    gene2idx = index_maps["gene2idx"]; mirna2idx = index_maps["mirna2idx"]
    df = df[df["target_gene"].isin(gene2idx) & df["mirna"].isin(mirna2idx)]
    recomputed = len(df)
    ok = (recomputed == emitted)
    print(f"  graph emitted        : {emitted:,}")
    print(f"  recomputed from {interactions_file}: {recomputed:,}   [{'MATCH' if ok else 'MISMATCH'}]")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/config_v3fixed_edgesplit.yaml")
    args = p.parse_args()

    root = str(Path(__file__).resolve().parents[1])
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    graphs_dir = cfg["data"]["graphs_dir"]
    proc_dir = cfg["data"]["processed_dir"]
    gcfg = cfg["data"]["graph"]

    print(f"config : {args.config}")
    print(f"graph  : {graphs_dir}")
    print(f"adata  : {proc_dir}/scrna_processed.h5ad")

    graph = torch.load(os.path.join(graphs_dir, "hetero_graph.pt"), weights_only=False)
    with open(os.path.join(graphs_dir, "index_maps.pkl"), "rb") as fh:
        index_maps = pickle.load(fh)
    adata = ad.read_h5ad(os.path.join(proc_dir, "scrna_processed.h5ad"))

    results = {
        "expresses": check_expresses(graph, adata, gcfg["cell_gene_expression_threshold"]),
        "coexpression": check_coexpression(graph, adata, gcfg["coexpression_threshold"],
                                           gcfg["coexpression_top_n_genes"]),
        "mirna_gene": check_mirna_gene(graph, index_maps, cfg),
    }

    print("\n" + "=" * 76)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 76)
    if all(results.values()):
        print("ALL INDEPENDENT CHECKS PASSED — the emitted edges match a from-scratch recompute.")
    else:
        print("VERIFICATION FAILED — see the MISMATCH/FAIL lines above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
