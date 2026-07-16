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
import json
import pickle
import hashlib
import argparse
import subprocess
from datetime import datetime, timezone

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
    p.add_argument("--force", action="store_true",
                   help="Rebuild even if a graph exists at graphs_dir.")
    return p.parse_args()


# ── Build fingerprint ─────────────────────────────────────────────────────────

def config_fingerprint(cfg: dict) -> str:
    """Hash of the config subtrees that determine the graph's contents.

    Deliberately narrow: seeds, SLURM settings and training hyperparameters do not
    change the graph, so including them would make every unrelated edit look like a
    stale graph and train people to pass --force reflexively.
    """
    material = {
        "interactions_file": cfg["data"].get("mirna", {}).get("interactions_file",
                                                              "mirtarbase_hsa.tsv"),
        "graph":          cfg["data"]["graph"],
        "processed_dir":  cfg["data"]["processed_dir"],
        "raw_dir":        cfg["data"]["raw_dir"],
        "mirna_init_dim": cfg["model"]["mirna_init_dim"],
    }
    blob = json.dumps(material, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def check_existing_graph(graphs_dir: str, cfg: dict, force: bool) -> bool:
    """Return True if an existing graph may be reused; raise if it is stale.

    The old behaviour was an unconditional early return: change the config, re-run,
    and the previous graph came back while every log said success — then you train on
    it believing it matched the config you just edited. A skip is only safe when the
    graph on disk was built from a config that produces the same graph.
    """
    graph_out    = os.path.join(graphs_dir, "hetero_graph.pt")
    index_out    = os.path.join(graphs_dir, "index_maps.pkl")
    manifest_out = os.path.join(graphs_dir, "graph_manifest.json")

    if not (os.path.exists(graph_out) and os.path.exists(index_out)):
        return False
    if force:
        print(f"--force: rebuilding {graph_out}")
        return False

    want = config_fingerprint(cfg)
    if not os.path.exists(manifest_out):
        # Graphs built before manifests existed (including the miRDB graph every
        # published number comes from). Reusing them is what keeps Path A
        # reproducible, so do not refuse — but do not pretend it was verified.
        print(f"Graph already exists: {graph_out}")
        print(f"  WARNING: no {os.path.basename(manifest_out)} beside it, so it predates "
              f"build fingerprinting and CANNOT be verified against {want[:12]}.")
        print( "  If you changed anything under data.graph / data.mirna.interactions_file, "
               "this graph is stale — re-run with --force or point graphs_dir elsewhere.")
        return True

    with open(manifest_out) as fh:
        have = json.load(fh).get("config_fingerprint")
    if have == want:
        print(f"Graph already exists and matches this config: {graph_out}")
        return True

    raise SystemExit(
        f"\nRefusing to reuse a stale graph.\n"
        f"  graph:            {graph_out}\n"
        f"  built from config fingerprint: {have}\n"
        f"  this config's fingerprint:     {want}\n"
        f"They differ, so the graph on disk is NOT what this config describes. Either\n"
        f"re-run with --force to rebuild in place, or set data.graphs_dir to a new\n"
        f"directory so both graphs survive."
    )


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

    graph_out    = os.path.join(graphs_dir, "hetero_graph.pt")
    index_out    = os.path.join(graphs_dir, "index_maps.pkl")
    manifest_out = os.path.join(graphs_dir, "graph_manifest.json")

    if check_existing_graph(graphs_dir, cfg, args.force):
        return

    gcfg = cfg["data"]["graph"]
    mirna_init_dim = cfg["model"]["mirna_init_dim"]

    # ── Load processed data ──────────────────────────────────────────────────
    scrna_path   = os.path.join(proc_dir, "scrna_processed.h5ad")
    # Interaction-source filename is configurable so an alternate source (e.g. a
    # real miRTarBase pull) can be built into its own graph without overwriting
    # the miRDB one. Defaults to the legacy miRDB file for backward compatibility.
    interactions_file = cfg["data"].get("mirna", {}).get("interactions_file", "mirtarbase_hsa.tsv")
    mirtarbase_p = os.path.join(cfg["data"]["raw_dir"], interactions_file)

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

    # The manifest is what makes a future skip verifiable instead of hopeful.
    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, check=True).stdout.strip()
    except Exception:
        git_sha = "unknown"
    manifest = {
        "config_path":        os.path.abspath(args.config),
        "config_fingerprint": config_fingerprint(cfg),
        "interactions_file":  interactions_file,
        "git_sha":            git_sha,
        "slurm_job_id":       os.environ.get("SLURM_JOB_ID", "local"),
        "built":              datetime.now(timezone.utc).isoformat(),
        "n_nodes":            {nt: int(data[nt].num_nodes) for nt in data.node_types},
        "n_edges":            {str(et): int(data[et].edge_index.shape[1])
                               for et in data.edge_types},
    }
    with open(manifest_out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Saved manifest:   {manifest_out}  (fingerprint {manifest['config_fingerprint'][:12]})")

    print("\nGraph summary:")
    print(data)
    print("\nHeterogeneity graph construction complete.")


if __name__ == "__main__":
    main()
