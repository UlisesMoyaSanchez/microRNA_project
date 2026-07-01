"""
preprocess_scrna.py — QC, normalization, and cell-type annotation of scRNA-seq data.

Pipeline (Scanpy standard):
  1. Merge MS + healthy control AnnData objects (add 'condition' column)
  2. QC filtering (min_genes, min_cells, max_pct_mito)
  3. Normalization (log1p, CPM)
  4. Highly variable gene selection
  5. PCA → neighbors → Leiden → UMAP
  6. Rule-based cell type annotation using marker genes from config
  7. Save processed AnnData to data/processed/scrna_processed.h5ad

Usage: python data/02_preprocess/preprocess_scrna.py --config configs/config.yaml
"""

import os
import argparse
import yaml
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


# ── Cell-type annotation ──────────────────────────────────────────────────────

def annotate_cell_types(adata: ad.AnnData, markers: dict[str, list[str]]) -> ad.AnnData:
    """
    Score each Leiden cluster against marker gene sets (Scanpy score_genes).
    Assigns the highest-scoring label to each cluster.
    """
    sc.settings.verbosity = 0
    # Use raw var_names (all genes) so marker genes excluded from HVG selection are still found
    gene_universe = set(adata.raw.var_names) if adata.raw is not None else set(adata.var_names)
    for cell_type, gene_list in markers.items():
        present = [g for g in gene_list if g in gene_universe]
        if not present:
            continue
        sc.tl.score_genes(adata, gene_list=present, score_name=f"score_{cell_type}")

    score_cols = [c for c in adata.obs.columns if c.startswith("score_")]
    if not score_cols:
        adata.obs["cell_type"] = "Unknown"
        return adata

    score_df = adata.obs[score_cols].copy()
    score_df.columns = [c.replace("score_", "") for c in score_cols]

    # Per-cell: argmax of scores
    adata.obs["cell_type"] = score_df.idxmax(axis=1).astype("category")

    # Report distribution
    print("  Cell type distribution:")
    print(adata.obs["cell_type"].value_counts().to_string())
    return adata


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_qc(adata: ad.AnnData, cfg: dict) -> ad.AnnData:
    sc.pp.filter_cells(adata, min_genes=cfg["min_genes"])
    sc.pp.filter_genes(adata, min_cells=cfg["min_cells"])

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    before = adata.n_obs
    adata = adata[adata.obs["pct_counts_mt"] < cfg["max_pct_mito"]].copy()
    print(f"  After MT filter: {before:,} → {adata.n_obs:,} cells")
    return adata


def run_normalization(adata: ad.AnnData, n_top_genes: int) -> ad.AnnData:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata  # freeze normalized counts before HVG selection
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key="condition")
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    return adata


def run_dim_reduction(adata: ad.AnnData) -> ad.AnnData:
    sc.tl.pca(adata, svd_solver="arpack", n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
    sc.tl.leiden(adata, resolution=0.8, key_added="leiden")
    sc.tl.umap(adata)
    return adata


def main() -> None:
    args = parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    sc.settings.n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    sc.settings.verbosity = 1

    raw_dir       = cfg["data"]["raw_dir"]
    ccfg          = cfg["data"]["cellxgene"]
    out_dir       = args.out_dir or cfg["data"]["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "scrna_processed.h5ad")
    if os.path.exists(out_path):
        print(f"Already exists: {out_path}")
        return

    # ── Load ─────────────────────────────────────────────────────────────────
    ms_path   = os.path.join(raw_dir, "cellxgene_ms.h5ad")
    ctrl_path = os.path.join(raw_dir, "cellxgene_ctrl.h5ad")

    print("Loading AnnData files...")
    adata_ms   = sc.read_h5ad(ms_path)
    adata_ctrl = sc.read_h5ad(ctrl_path)

    # CellxGene h5ad files use integer _index as var_names;
    # gene symbols are stored in var['feature_name'].
    for adt, label in [(adata_ms, "MS"), (adata_ctrl, "ctrl")]:
        if "feature_name" in adt.var.columns:
            adt.var_names = adt.var["feature_name"].values.astype(str)
            adt.var_names_make_unique()

    adata_ms.obs["condition"]   = "MS"
    adata_ctrl.obs["condition"] = "Control"

    print(f"  MS cells:      {adata_ms.n_obs:,}")
    print(f"  Control cells: {adata_ctrl.n_obs:,}")

    # Concatenate (outer join on var; missing genes → 0)
    adata = ad.concat(
        [adata_ms, adata_ctrl],
        join="outer",
        label="source",
        keys=["ms", "ctrl"],
    )
    adata.obs_names_make_unique()
    print(f"  Merged: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # ── QC ───────────────────────────────────────────────────────────────────
    print("\nRunning QC...")
    adata = run_qc(adata, ccfg)

    # ── Normalization + HVG ──────────────────────────────────────────────────
    print("\nNormalizing and selecting HVG...")
    adata = run_normalization(adata, ccfg["n_top_genes"])
    print(f"  After HVG: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # ── Dimensionality reduction + clustering ────────────────────────────────
    print("\nPCA → neighbors → Leiden → UMAP...")
    adata = run_dim_reduction(adata)
    print(f"  Leiden clusters: {adata.obs['leiden'].nunique()}")

    # ── Cell-type annotation ─────────────────────────────────────────────────
    print("\nAnnotating cell types...")
    markers: dict = cfg["cell_type_markers"]
    adata = annotate_cell_types(adata, markers)

    # ── Save ────────────────────────────────────────────────────────────────
    print(f"\nSaving: {out_path}")
    adata.write_h5ad(out_path)
    print(f"  Final: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # Save cell-type label mapping for downstream use
    label_map = (
        adata.obs["cell_type"]
        .astype("category")
        .cat
        .categories
        .tolist()
    )
    pd.Series(label_map).to_csv(
        os.path.join(out_dir, "cell_type_labels.txt"), index=True, header=False
    )

    print("scRNA-seq preprocessing complete.")


if __name__ == "__main__":
    main()
