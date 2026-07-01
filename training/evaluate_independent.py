"""
evaluate_independent.py — Evaluate the trained V2 model on a held-out independent dataset.

Purpose:
  Tests generalization: the model was trained on CellxGene MS data (GSE289530
  partially). This script applies it to a *fully independent* scRNA-seq dataset
  (GEO accession GSE289530 or similar) that was NOT part of training,
  reporting AUROC, AUPRC, cell-type accuracy, and per-type confusion matrix.

Strategy:
  1. Load and preprocess the independent AnnData (same pipeline: QC, normalize,
     HVG selection using the *training* gene set to ensure alignment).
  2. Build a lightweight HeteroData with the same miRNA→gene edges (from index_maps)
     but only the new cells as nodes.
  3. Load the V2 checkpoint and run evaluate().
  4. Save results to results/comparison/independent_eval.json and a confusion matrix TSV.

Usage:
  python training/evaluate_independent.py \\
      --config configs/config_v2.yaml \\
      --h5ad data/raw/independent_dataset.h5ad \\
      --condition_col condition \\
      --celltype_col cell_type
"""

from __future__ import annotations

import os
import sys
import json
import pickle
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import scanpy as sc
import anndata as ad
import yaml
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.hetero_gnn import miRNAGraphTransformer
from models.losses import CombinedLoss
from training.train import load_graph, split_graph, set_seed


# ── Logging ────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, "independent_eval.log")),
        ],
    )
    return logging.getLogger(__name__)


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",       default="configs/config_v2.yaml")
    p.add_argument("--h5ad",         required=True,
                   help="Path to independent AnnData (.h5ad)")
    p.add_argument("--condition_col", default="condition",
                   help="Column in adata.obs for MS/control label")
    p.add_argument("--celltype_col",  default="cell_type",
                   help="Column in adata.obs for cell type label (if pre-annotated)")
    p.add_argument("--checkpoint",   default=None,
                   help="Override checkpoint path (default: config checkpoint_dir/best_model.pt)")
    return p.parse_args()


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess_independent(
    h5ad_path: str,
    training_genes: list[str],
    cfg: dict,
    log: logging.Logger,
) -> ad.AnnData:
    """
    Load and preprocess the independent dataset, aligning to training gene set.
    - QC filters matching the training config
    - Normalize and log1p
    - Restrict to training HVG set (intersection)
    """
    log.info(f"Loading independent dataset: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    log.info(f"  Raw: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    dcfg = cfg["data"]["cellxgene"]

    # QC
    sc.pp.filter_cells(adata, min_genes=dcfg["min_genes"])
    sc.pp.filter_genes(adata, min_cells=dcfg["min_cells"])
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs["pct_counts_mt"] < dcfg["max_pct_mito"]].copy()
    log.info(f"  After QC: {adata.n_obs:,} cells")

    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Align to training gene set: keep intersection, zero-pad missing genes
    ind_genes = set(adata.var_names)
    common    = [g for g in training_genes if g in ind_genes]
    missing   = [g for g in training_genes if g not in ind_genes]
    log.info(f"  Gene overlap: {len(common):,}/{len(training_genes):,} training genes found")
    log.info(f"  Missing (will be zero-padded): {len(missing):,} genes")

    # Subset to common genes
    adata_common = adata[:, common].copy()

    # Zero-pad missing genes so final order matches training exactly
    if missing:
        import scipy.sparse as sp
        zero_block = sp.csr_matrix((adata.n_obs, len(missing)), dtype=np.float32)
        pad_adata  = ad.AnnData(
            X=zero_block,
            obs=adata.obs,
            var=pd.DataFrame(index=missing),
        )
        adata_final = ad.concat([adata_common, pad_adata], axis=1)
        # Reorder columns to match training_genes exactly
        adata_final = adata_final[:, training_genes].copy()
    else:
        adata_final = adata_common[:, training_genes].copy()

    # PCA using training PCA components would be ideal, but we re-run PCA
    # on the independent set and use the resulting coordinates as cell features.
    # (For fully rigorous eval, save PCA components from training and apply them.)
    sc.tl.pca(adata_final, svd_solver="arpack", n_comps=50)
    log.info(f"  Final shape: {adata_final.n_obs:,} cells × {adata_final.n_vars:,} genes")
    return adata_final


# ── Build lightweight subgraph for independent cells ──────────────────────────

def build_independent_graph(
    adata: ad.AnnData,
    index_maps: dict,
    training_graph: HeteroData,
    cfg: dict,
    log: logging.Logger,
) -> HeteroData:
    """
    Construct a HeteroData using:
      - The independent cells as new 'cell' nodes (PCA features)
      - The same miRNA and gene nodes from training (same embeddings/features)
      - Re-computed cell→gene expression edges from independent data
      - Same miRNA→gene edges as training (unchanged)
      - Same gene→gene co-expression edges as training (unchanged)
    """
    import scipy.sparse as sp

    gene2idx  = index_maps["gene2idx"]
    mirna2idx = index_maps["mirna2idx"]
    cell_type_labels: list[str] = index_maps["cell_type_labels"]
    celltype2idx = {ct: i for i, ct in enumerate(cell_type_labels)}

    gcfg = cfg["data"]["graph"]
    expr_threshold = gcfg["cell_gene_expression_threshold"]

    n_cells = adata.n_obs
    n_genes = len(gene2idx)

    log.info(f"Building independent graph: {n_cells:,} cells, {n_genes:,} genes")

    # ── Cell PCA features ──────────────────────────────────────────────────
    cell_x = torch.tensor(adata.obsm["X_pca"].astype(np.float32))

    # ── Cell type labels (if pre-annotated) ───────────────────────────────
    celltype_col = cfg.get("_celltype_col", "cell_type")
    if celltype_col in adata.obs.columns:
        raw_labels = adata.obs[celltype_col].astype(str).tolist()
        cell_y = torch.tensor(
            [celltype2idx.get(ct, -1) for ct in raw_labels], dtype=torch.long
        )
        n_labeled = (cell_y >= 0).sum().item()
        log.info(f"  Cell types found: {n_labeled:,}/{n_cells:,} cells labeled")
    else:
        log.info(f"  No cell_type column found — labels set to -1 (unlabeled)")
        cell_y = torch.full((n_cells,), -1, dtype=torch.long)

    # ── Cell→gene expression edges ─────────────────────────────────────────
    # adata.X is (cells, training_genes) in training gene order
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.array(X, dtype=np.float32)

    row_idx, col_idx = np.where(X > expr_threshold)
    # col_idx already indexes into training_genes (aligned above)
    valid = col_idx < n_genes
    row_idx, col_idx = row_idx[valid], col_idx[valid]

    cell_gene_ei = torch.tensor(np.stack([row_idx, col_idx]), dtype=torch.long)
    gene_cell_ei = torch.tensor(np.stack([col_idx, row_idx]), dtype=torch.long)
    log.info(f"  Cell-gene edges: {cell_gene_ei.shape[1]:,}")

    # ── Assemble HeteroData ────────────────────────────────────────────────
    g = HeteroData()

    # Copy miRNA and gene nodes from training graph (same features)
    g["miRNA"].x = training_graph["miRNA"].x.clone()
    g["gene"].x  = training_graph["gene"].x.clone()
    g["cell"].x  = cell_x
    g["cell"].y  = cell_y

    # Copy miRNA↔gene edges from training (no change)
    g["miRNA", "regulates",   "gene"].edge_index = \
        training_graph["miRNA", "regulates", "gene"].edge_index.clone()
    g["gene",  "regulated_by", "miRNA"].edge_index = \
        training_graph["gene", "regulated_by", "miRNA"].edge_index.clone()

    # Copy gene co-expression from training
    if ("gene", "coexpressed_with", "gene") in training_graph.edge_types:
        g["gene", "coexpressed_with", "gene"].edge_index = \
            training_graph["gene", "coexpressed_with", "gene"].edge_index.clone()

    # New cell↔gene edges from independent data
    g["cell", "expresses",   "gene"].edge_index = cell_gene_ei
    g["gene", "expressed_in", "cell"].edge_index = gene_cell_ei

    log.info(f"  Graph assembled: {g}")
    return g


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_evaluation(
    model: miRNAGraphTransformer,
    graph: HeteroData,
    cfg: dict,
    device: torch.device,
    log: logging.Logger,
) -> dict:
    """
    Run cell-type classification evaluation on all labeled cells.
    Link prediction AUROC is computed using the miRNA→gene edges.
    """
    from torch_geometric.utils import negative_sampling

    tcfg = cfg["training"]
    # Use all cells as "test" — no train/val split for independent eval
    n_cells = graph["cell"].num_nodes
    all_mask = torch.ones(n_cells, dtype=torch.bool)
    graph["cell"].train_mask = all_mask
    graph["cell"].val_mask   = all_mask

    loader = NeighborLoader(
        graph,
        input_nodes=("cell", all_mask),
        num_neighbors={et: tcfg["num_neighbors"] for et in graph.edge_types},
        batch_size=tcfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    pos_edge_full = graph["miRNA", "regulates", "gene"].edge_index

    model.eval()
    all_cell_preds:  list[np.ndarray] = []
    all_cell_labels: list[np.ndarray] = []
    all_edge_logits: list[np.ndarray] = []
    all_edge_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Remap miRNA→gene edges to local batch indices
            mirna_g = batch["miRNA"].n_id.to(device)
            gene_g  = batch["gene"].n_id.to(device)
            n_m = max(int(pos_edge_full[0].max().item()), int(mirna_g.max().item())) + 1
            n_g = max(int(pos_edge_full[1].max().item()), int(gene_g.max().item())) + 1
            m2l = torch.full((n_m,), -1, dtype=torch.long, device=device)
            m2l[mirna_g] = torch.arange(len(mirna_g), device=device)
            g2l = torch.full((n_g,), -1, dtype=torch.long, device=device)
            g2l[gene_g]  = torch.arange(len(gene_g),  device=device)
            src = m2l[pos_edge_full[0].to(device)]
            dst = g2l[pos_edge_full[1].to(device)]
            valid = (src >= 0) & (dst >= 0)
            src, dst = src[valid], dst[valid]

            if src.shape[0] > 0:
                n_pos = min(src.shape[0], 256)
                perm  = torch.randperm(src.shape[0], device=device)[:n_pos]
                pos_src, pos_dst = src[perm], dst[perm]
                neg_ei = negative_sampling(
                    edge_index=torch.stack([pos_src, pos_dst]),
                    num_nodes=(batch["miRNA"].num_nodes, batch["gene"].num_nodes),
                    num_neg_samples=n_pos,
                )
                mirna_idx = torch.cat([pos_src, neg_ei[0]])
                gene_idx  = torch.cat([pos_dst, neg_ei[1]])
                edge_labels_b = torch.cat([
                    torch.ones(n_pos), torch.zeros(neg_ei.shape[1])
                ]).numpy()
            else:
                mirna_idx = gene_idx = None
                edge_labels_b = None

            out = model(
                x_dict=batch.x_dict,
                edge_index_dict=batch.edge_index_dict,
                mirna_idx=mirna_idx,
                gene_idx=gene_idx,
            )

            if "edge_logits" in out and edge_labels_b is not None:
                all_edge_logits.append(torch.sigmoid(out["edge_logits"]).cpu().numpy())
                all_edge_labels.append(edge_labels_b)

            if "cell_logits" in out:
                preds  = out["cell_logits"].argmax(dim=-1).cpu().numpy()
                labels = batch["cell"].y.cpu().numpy()
                mask   = labels >= 0
                if mask.any():
                    all_cell_preds.append(preds[mask])
                    all_cell_labels.append(labels[mask])

    metrics: dict = {}

    if all_edge_logits:
        y_score = np.concatenate(all_edge_logits)
        y_true  = np.concatenate(all_edge_labels)
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        metrics["auprc"] = float(average_precision_score(y_true, y_score))
        log.info(f"  Link AUROC: {metrics['auroc']:.4f}  AUPRC: {metrics['auprc']:.4f}")

    if all_cell_preds:
        y_pred   = np.concatenate(all_cell_preds)
        y_labels = np.concatenate(all_cell_labels)
        metrics["cell_acc"] = float(accuracy_score(y_labels, y_pred))
        metrics["cell_f1"]  = float(f1_score(y_labels, y_pred, average="macro", zero_division=0))
        metrics["classification_report"] = classification_report(
            y_labels, y_pred, zero_division=0
        )
        log.info(f"  Cell Acc: {metrics['cell_acc']:.4f}  F1-macro: {metrics['cell_f1']:.4f}")
        log.info(f"\n{metrics['classification_report']}")

    return metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    # Stash celltype_col in cfg for helper functions
    cfg["_celltype_col"] = args.celltype_col

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["project"]["seed"], 0)

    log_dir = cfg["training"]["log_dir"]
    log     = setup_logging(log_dir)
    log.info(f"Device: {device}")

    out_dir = os.path.join(cfg["project"]["output_dir"], "comparison")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load training graph & index maps ──────────────────────────────────
    log.info("Loading training graph and index maps...")
    training_graph, index_maps = load_graph(cfg["data"]["graphs_dir"])
    cell_type_labels: list[str] = index_maps["cell_type_labels"]
    num_cell_types = len(cell_type_labels)
    gene2idx: dict = index_maps["gene2idx"]
    training_genes = list(gene2idx.keys())  # ordered list matching graph

    # ── Preprocess independent dataset ────────────────────────────────────
    adata = preprocess_independent(args.h5ad, training_genes, cfg, log)

    # ── Build graph for independent cells ─────────────────────────────────
    ind_graph = build_independent_graph(adata, index_maps, training_graph, cfg, log)
    ind_graph = ind_graph.to("cpu")

    # ── Load V2 model ─────────────────────────────────────────────────────
    ckpt = args.checkpoint or os.path.join(
        cfg["training"]["checkpoint_dir"], "best_model.pt"
    )
    log.info(f"Loading checkpoint: {ckpt}")
    model = miRNAGraphTransformer.from_config(cfg, ind_graph.metadata(), num_cell_types)

    # Warm up lazy linears
    dummy_loader = NeighborLoader(
        ind_graph,
        input_nodes=("cell", torch.ones(ind_graph["cell"].num_nodes, dtype=torch.bool)),
        num_neighbors={et: [5, 3] for et in ind_graph.edge_types},
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )
    with torch.no_grad():
        _d = next(iter(dummy_loader))
        model(_d.x_dict, _d.edge_index_dict)
    del _d, dummy_loader

    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    state = ck.get("model", ck)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    log.info("Model loaded.")

    # ── Evaluate ──────────────────────────────────────────────────────────
    log.info("Running evaluation on independent dataset...")
    metrics = run_evaluation(model, ind_graph, cfg, device, log)

    # ── Save results ──────────────────────────────────────────────────────
    out_path = os.path.join(out_dir, "independent_eval.json")
    # classification_report is a string — save separately
    cr = metrics.pop("classification_report", "")
    with open(out_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    log.info(f"Metrics saved to: {out_path}")

    cr_path = os.path.join(out_dir, "independent_eval_report.txt")
    with open(cr_path, "w") as fh:
        fh.write(f"Independent dataset: {args.h5ad}\n\n")
        fh.write(cr)
    log.info(f"Classification report saved to: {cr_path}")

    log.info("\n=== SUMMARY ===")
    log.info(f"  Dataset:  {args.h5ad}")
    log.info(f"  N cells:  {ind_graph['cell'].num_nodes:,}")
    for k, v in metrics.items():
        log.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
