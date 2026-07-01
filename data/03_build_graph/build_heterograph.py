"""
build_heterograph.py — Construct the heterogeneous PyG graph for the miRNA-MS model.

Node types:
  miRNA  — one node per miRNA present in miRTarBase (and filtered GEO data)
  gene   — one node per highly-variable gene in the scRNA-seq dataset
  cell   — one node per cell in the processed scRNA-seq dataset

Edge types:
  (miRNA, regulates,      gene) — from miRTarBase validated interactions
  (gene,  regulated_by,  miRNA) — reverse of above
  (cell,  expresses,      gene) — log-normalized expression > threshold (sparse)
  (gene,  expressed_in,   cell) — reverse of above
  (gene,  coexpressed_with, gene) — Pearson correlation > threshold

Node features:
  miRNA  x: (n_mirna, mirna_init_dim) — random init, treated as learned embeddings
  gene   x: (n_gene, 1)              — mean log-normalized expression across cells
  cell   x: (n_cell, n_pcs)          — Scanpy PCA embeddings (50 dims)

Labels:
  cell   y: (n_cell,) int64          — cell type index (from annotation)

Output:
  data/graphs/hetero_graph.pt   — torch.save(HeteroData)
  data/graphs/index_maps.pkl    — gene2idx, mirna2idx, cell2idx, cell_type_labels
"""

import os
import pickle
import argparse
import yaml
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.data import HeteroData
import anndata as ad


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    return p.parse_args()


# ── Index builders ─────────────────────────────────────────────────────────────

def build_indices(
    adata: ad.AnnData, df_mirt: pd.DataFrame
) -> tuple[dict, dict, dict]:
    genes  = list(adata.var_names)
    gene2idx = {g: i for i, g in enumerate(genes)}

    # Keep only miRNAs that target at least one gene in our scRNA set
    df_mirt = df_mirt[df_mirt["target_gene"].isin(gene2idx)]
    mirnas = sorted(df_mirt["mirna"].unique())
    mirna2idx = {m: i for i, m in enumerate(mirnas)}

    cells = list(adata.obs_names)
    cell2idx = {c: i for i, c in enumerate(cells)}

    return gene2idx, mirna2idx, cell2idx, df_mirt


# ── Node features ──────────────────────────────────────────────────────────────

def build_gene_features(adata: ad.AnnData) -> torch.Tensor:
    """Mean log-normalized expression per gene → shape (n_genes, 1).

    Uses adata.raw (log-normalized, pre-scaling) when available so that
    features are not all near-zero after sc.pp.scale."""
    if adata.raw is not None and adata.raw.X is not None:
        # raw is stored with all genes; subset to the HVG var_names
        raw_var_names = list(adata.raw.var_names)
        hvg_names = list(adata.var_names)
        col_idx = [raw_var_names.index(g) for g in hvg_names if g in raw_var_names]
        X = adata.raw.X[:, col_idx]
    else:
        X = adata.X
    if sp.issparse(X):
        mean_expr = np.asarray(X.mean(axis=0)).ravel()
    else:
        mean_expr = X.mean(axis=0).ravel()
    return torch.tensor(mean_expr, dtype=torch.float32).unsqueeze(1)


def build_cell_features(adata: ad.AnnData) -> torch.Tensor:
    """PCA embeddings from Scanpy → shape (n_cells, n_pcs)."""
    if "X_pca" in adata.obsm:
        return torch.tensor(adata.obsm["X_pca"], dtype=torch.float32)
    # Fallback: scaled expression (heavy but functional)
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    return torch.tensor(X, dtype=torch.float32)


def build_cell_labels(adata: ad.AnnData) -> tuple[torch.Tensor, list[str]]:
    if "cell_type" not in adata.obs.columns:
        # Fallback to leiden cluster index
        cat = adata.obs["leiden"].astype("category")
    else:
        cat = adata.obs["cell_type"].astype("category")
    return (
        torch.tensor(cat.cat.codes.values, dtype=torch.long),
        list(cat.cat.categories),
    )


def build_mirna_features(n_mirna: int, mirna_init_dim: int) -> torch.Tensor:
    """Random normal init — treated as learned embeddings during training."""
    torch.manual_seed(42)
    return torch.randn(n_mirna, mirna_init_dim)


# ── Edge builders ──────────────────────────────────────────────────────────────

def build_mirtarbase_edges(
    df_mirt: pd.DataFrame, mirna2idx: dict, gene2idx: dict
) -> torch.Tensor:
    src = [mirna2idx[m] for m in df_mirt["mirna"] if m in mirna2idx]
    dst = [gene2idx[g] for g in df_mirt.loc[df_mirt["mirna"].isin(mirna2idx), "target_gene"]]
    return torch.tensor([src, dst], dtype=torch.long)


def build_expression_edges(
    adata: ad.AnnData, threshold: float
) -> torch.Tensor:
    """Sparse edges where log-normalized expression > threshold."""
    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X_coo = X.tocoo()
    mask = X_coo.data > threshold
    rows = X_coo.row[mask].astype(np.int64)
    cols = X_coo.col[mask].astype(np.int64)
    return torch.tensor(np.stack([rows, cols]), dtype=torch.long)


def build_coexpression_edges(
    adata: ad.AnnData, threshold: float, top_n: int
) -> torch.Tensor:
    """Gene-gene co-expression edges based on PCC > threshold."""
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    # Use top_n most variable genes to keep computation tractable
    n_genes = min(top_n, X.shape[1])
    X_sub = X[:, :n_genes].astype(np.float32)

    # Zero-center per gene before computing correlation
    X_sub -= X_sub.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(X_sub, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    X_sub /= norms

    # After unit-norming, dot product == Pearson correlation directly.
    # Do NOT divide by n_cells again (that was a bug causing all values → ~0).
    corr = X_sub.T @ X_sub  # (n_genes, n_genes), values in [-1, 1]

    src, dst = np.where((corr > threshold) & (np.eye(n_genes) == 0))
    src = src.astype(np.int64)
    dst = dst.astype(np.int64)
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    proc_dir   = cfg["data"]["processed_dir"]
    graphs_dir = cfg["data"]["graphs_dir"]
    os.makedirs(graphs_dir, exist_ok=True)

    graph_out = os.path.join(graphs_dir, "hetero_graph.pt")
    index_out = os.path.join(graphs_dir, "index_maps.pkl")

    if os.path.exists(graph_out) and os.path.exists(index_out):
        print(f"Graph already exists: {graph_out}")
        return

    gcfg = cfg["data"]["graph"]
    mirna_init_dim = cfg["model"]["mirna_init_dim"]

    # ── Load processed data ──────────────────────────────────────────────────
    scrna_path   = os.path.join(proc_dir, "scrna_processed.h5ad")
    mirtarbase_p = os.path.join(cfg["data"]["raw_dir"], "mirtarbase_hsa.tsv")

    print(f"Loading scRNA-seq: {scrna_path}")
    adata = ad.read_h5ad(scrna_path)
    print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    print(f"Loading miRTarBase: {mirtarbase_p}")
    df_mirt = pd.read_csv(mirtarbase_p, sep="\t")
    print(f"  {len(df_mirt):,} validated interactions ({df_mirt['mirna'].nunique():,} miRNAs)")

    # ── Build indices ────────────────────────────────────────────────────────
    gene2idx, mirna2idx, cell2idx, df_mirt_filtered = build_indices(adata, df_mirt)
    n_genes  = len(gene2idx)
    n_mirnas = len(mirna2idx)
    n_cells  = len(cell2idx)

    print(f"\nGraph nodes → genes: {n_genes:,} | miRNAs: {n_mirnas:,} | cells: {n_cells:,}")

    # ── Build HeteroData ──────────────────────────────────────────────────────
    data = HeteroData()

    # Node features
    print("\nBuilding node features...")
    data["gene"].x  = build_gene_features(adata)
    data["cell"].x  = build_cell_features(adata)
    data["miRNA"].x = build_mirna_features(n_mirnas, mirna_init_dim)

    cell_labels, cell_type_list = build_cell_labels(adata)
    data["cell"].y = cell_labels
    print(f"  Cell types: {len(cell_type_list)}")

    # miRNA→gene edges (miRTarBase)
    print("Building miRNA→gene edges (miRTarBase)...")
    mirt_edges = build_mirtarbase_edges(df_mirt_filtered, mirna2idx, gene2idx)
    data["miRNA", "regulates", "gene"].edge_index = mirt_edges
    data["gene", "regulated_by", "miRNA"].edge_index = mirt_edges.flip(0)
    print(f"  miRNA→gene edges: {mirt_edges.shape[1]:,}")

    # Cell→gene edges (expression)
    print(f"Building cell→gene edges (threshold={gcfg['cell_gene_expression_threshold']})...")
    expr_edges = build_expression_edges(adata, gcfg["cell_gene_expression_threshold"])
    data["cell", "expresses", "gene"].edge_index   = expr_edges
    data["gene", "expressed_in", "cell"].edge_index = expr_edges.flip(0)
    print(f"  cell→gene edges: {expr_edges.shape[1]:,}")

    # Gene-gene co-expression edges
    print(f"Building gene co-expression edges (PCC>{gcfg['coexpression_threshold']}, top {gcfg['coexpression_top_n_genes']} genes)...")
    coexpr_edges = build_coexpression_edges(
        adata,
        gcfg["coexpression_threshold"],
        gcfg["coexpression_top_n_genes"],
    )
    data["gene", "coexpressed_with", "gene"].edge_index = coexpr_edges
    print(f"  gene-gene edges: {coexpr_edges.shape[1]:,}")

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f"\nSaving graph: {graph_out}")
    torch.save(data, graph_out)

    index_maps = {
        "gene2idx":         gene2idx,
        "mirna2idx":        mirna2idx,
        "cell2idx":         cell2idx,
        "cell_type_labels": cell_type_list,
        "idx2gene":         {v: k for k, v in gene2idx.items()},
        "idx2mirna":        {v: k for k, v in mirna2idx.items()},
    }
    with open(index_out, "wb") as fh:
        pickle.dump(index_maps, fh)
    print(f"Saved index maps: {index_out}")

    print("\nGraph summary:")
    print(data)
    print("\nHeterogeneity graph construction complete.")


if __name__ == "__main__":
    main()
