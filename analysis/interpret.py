"""
interpret.py — Model interpretation for the miRNA-MS paper results.

Two complementary methods:
  1. Gradient-based attribution (saliency):
     For each cell type, compute |∂cell_loss/∂miRNA_embedding|.
     High gradient norm → that miRNA strongly influences classification of that cell type.

  2. Edge attention ranking:
     Score each (miRNA, gene) pair using the trained TargetPredictor head.
     Reveals the top predicted regulatory circuits per cell type.

  3. GO/KEGG Pathway enrichment:
     For the top target genes per miRNA per cell type, run enrichment via gseapy.

Outputs (in results/interpretation/):
  mirna_saliency_by_celltype.tsv   — miRNA saliency score per cell type
  top_circuits_by_celltype.tsv     — top (miRNA, gene) pairs per cell type
  enrichment/<celltype>_GO.tsv     — GO Biological Process enrichment
  enrichment/<celltype>_KEGG.tsv   — KEGG pathway enrichment
"""

from __future__ import annotations

import os
import sys
import pickle
import argparse
import logging
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.hetero_gnn import miRNAGraphTransformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--top_k",      type=int, default=20, help="Top K miRNAs per cell type")
    p.add_argument("--top_genes",  type=int, default=50, help="Top target genes for enrichment")
    return p.parse_args()


# ── Gradient-based attribution ─────────────────────────────────────────────────

# Stratified cell sample per saliency worker.
# 2 K cells ≈ 288 K cell-gene edges → ~5 GB peak per GPU in backprop through 4 HGT layers.
# (15 K cells → 2.16 M edges → ~78 GB — too large for an A100-80 GB with gradient storage.)
SALIENCY_CELL_SAMPLE = 2_000


def _build_cell_subgraph(graph: HeteroData, cell_idx: torch.Tensor) -> HeteroData:
    """Return a HeteroData containing only the sampled cell nodes and their incident edges."""
    n_cells   = graph["cell"].num_nodes
    cell_mask = torch.zeros(n_cells, dtype=torch.bool)
    cell_mask[cell_idx] = True
    remap = torch.full((n_cells,), -1, dtype=torch.long)
    remap[cell_idx] = torch.arange(len(cell_idx))

    sub = HeteroData()
    sub["miRNA"].x = graph["miRNA"].x
    sub["gene"].x  = graph["gene"].x
    sub["cell"].x  = graph["cell"].x[cell_idx]
    if hasattr(graph["cell"], "y"):
        sub["cell"].y = graph["cell"].y[cell_idx]

    # miRNA-gene and gene-gene: pass through unchanged
    for et in [
        ("miRNA", "regulates",        "gene"),
        ("gene",  "regulated_by",     "miRNA"),
        ("gene",  "coexpressed_with", "gene"),
    ]:
        if et in graph.edge_types:
            sub[et].edge_index = graph[et].edge_index

    # cell-gene edges: filter rows and remap cell node indices
    for et, cell_side in [
        (("cell", "expresses",    "gene"), 0),
        (("gene", "expressed_in", "cell"), 1),
    ]:
        if et not in graph.edge_types:
            continue
        ei     = graph[et].edge_index
        mask   = cell_mask[ei[cell_side]]
        new_ei = ei[:, mask].clone()
        new_ei[cell_side] = remap[new_ei[cell_side]]
        sub[et].edge_index = new_ei

    return sub


def compute_mirna_saliency(
    model:          miRNAGraphTransformer,
    graph:          HeteroData,
    device:         torch.device,
    num_cell_types: int,
    sample_cells:   int = SALIENCY_CELL_SAMPLE,
) -> np.ndarray:
    """
    Returns saliency matrix of shape (n_mirna, num_cell_types).
    saliency[i, c] = mean |∂(mean class-c logit)/∂miRNA_i_embedding|

    Builds a stratified cell subsample (sample_cells cells) to keep the backward
    pass within GPU memory limits (2 K cells ≈ 288 K cell-gene edges ≈ 5 GB peak).
    Each class backward is run independently with empty_cache() between passes.
    Note: mp.spawn is intentionally avoided — CUDA is already initialised in the
    calling process and forking a second CUDA context causes illegal memory access.
    """
    labels     = graph["cell"].y.numpy()
    rng        = np.random.default_rng(seed=42)
    unique_cls = np.unique(labels[labels >= 0])
    per_cls    = max(1, sample_cells // len(unique_cls))
    selected   = []
    for c in unique_cls:
        idx = np.where(labels == c)[0]
        selected.append(rng.choice(idx, size=min(per_cls, len(idx)), replace=False))
    cell_idx = torch.from_numpy(np.sort(np.concatenate(selected)))

    subgraph = _build_cell_subgraph(graph, cell_idx).to(device)
    n_mirna  = graph["miRNA"].num_nodes
    saliency = np.zeros((n_mirna, num_cell_types))

    model.eval()
    for class_idx in range(num_cell_types):
        torch.cuda.empty_cache()
        mirna_x = subgraph["miRNA"].x.clone().detach().requires_grad_(True)
        x_dict  = {
            "miRNA": mirna_x,
            "gene":  subgraph["gene"].x.detach(),
            "cell":  subgraph["cell"].x.detach(),
        }
        out = model(x_dict=x_dict, edge_index_dict=subgraph.edge_index_dict)
        out["cell_logits"][:, class_idx].mean().backward()
        if mirna_x.grad is not None:
            saliency[:, class_idx] = mirna_x.grad.abs().mean(dim=-1).detach().cpu().numpy()
        del out, mirna_x

    del subgraph
    torch.cuda.empty_cache()
    return saliency


# ── Edge scoring ───────────────────────────────────────────────────────────────

def score_all_mirna_gene_pairs(
    model:  miRNAGraphTransformer,
    graph:  HeteroData,
    device: torch.device,
    batch_size: int = 4096,
    sample_cells: int = SALIENCY_CELL_SAMPLE,
) -> np.ndarray:
    """
    Score every validated (miRNA, gene) pair in miRTarBase using the trained predictor.
    Returns scores array of shape (n_edges,).

    Uses the same stratified cell subsample as saliency to encode the graph —
    the full 16 M cell-gene edges do not fit on a single A100 for encode().
    miRNA and gene embeddings are contextualised through the subgraph cell context.
    """
    model.eval()

    # Build stratified cell subgraph (same logic as compute_mirna_saliency)
    labels     = graph["cell"].y.numpy()
    rng        = np.random.default_rng(seed=42)
    unique_cls = np.unique(labels[labels >= 0])
    per_cls    = max(1, sample_cells // len(unique_cls))
    selected   = []
    for c in unique_cls:
        idx = np.where(labels == c)[0]
        selected.append(rng.choice(idx, size=min(per_cls, len(idx)), replace=False))
    cell_idx  = torch.from_numpy(np.sort(np.concatenate(selected)))
    subgraph  = _build_cell_subgraph(graph, cell_idx).to(device)

    with torch.no_grad():
        h = model.encode(subgraph.x_dict, subgraph.edge_index_dict)

    mirna_emb = h["miRNA"]  # (n_mirna, hidden)
    gene_emb  = h["gene"]   # (n_gene,  hidden)

    del subgraph
    torch.cuda.empty_cache()

    pos_edges = graph["miRNA", "regulates", "gene"].edge_index
    n_edges   = pos_edges.shape[1]
    scores    = np.empty(n_edges, dtype=np.float32)

    for start in range(0, n_edges, batch_size):
        end    = min(start + batch_size, n_edges)
        mi_idx = pos_edges[0, start:end].to(device)
        ge_idx = pos_edges[1, start:end].to(device)
        with torch.no_grad():
            logits = model.target_predictor(mirna_emb[mi_idx], gene_emb[ge_idx])
            scores[start:end] = torch.sigmoid(logits).cpu().numpy()

    return scores


# ── GO/KEGG enrichment ─────────────────────────────────────────────────────────

def run_enrichment(
    gene_list: list[str],
    cell_type: str,
    out_dir: str,
) -> None:
    try:
        import gseapy as gp
    except ImportError:
        logging.warning("gseapy not installed; skipping enrichment.")
        return

    os.makedirs(out_dir, exist_ok=True)
    gene_sets_to_run = {
        "GO_Biological_Process_2021": "GO",
        "KEGG_2021_Human":           "KEGG",
    }
    for gene_set, tag in gene_sets_to_run.items():
        try:
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=gene_set,
                organism="human",
                outdir=None,
                verbose=False,
            )
            df = enr.results.sort_values("Adjusted P-value")
            out_file = os.path.join(out_dir, f"{cell_type}_{tag}.tsv")
            df.to_csv(out_file, sep="\t", index=False)
            logging.info(f"  Saved enrichment: {out_file} ({len(df)} terms)")
        except Exception as exc:
            logging.warning(f"  Enrichment failed for {cell_type}/{gene_set}: {exc}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    out_dir  = os.path.join(cfg["project"]["output_dir"], "interpretation")
    enr_dir  = os.path.join(out_dir, "enrichment")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(enr_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load graph + index maps ──────────────────────────────────────────────
    graphs_dir = cfg["data"]["graphs_dir"]
    graph = torch.load(os.path.join(graphs_dir, "hetero_graph.pt"), weights_only=False)
    with open(os.path.join(graphs_dir, "index_maps.pkl"), "rb") as fh:
        index_maps = pickle.load(fh)

    cell_type_labels: list[str] = index_maps["cell_type_labels"]
    idx2mirna:  dict = index_maps["idx2mirna"]
    idx2gene:   dict = index_maps["idx2gene"]
    num_cell_types = len(cell_type_labels)

    # ── Load model ───────────────────────────────────────────────────────────
    metadata  = graph.metadata()
    model     = miRNAGraphTransformer.from_config(cfg, metadata, num_cell_types).to(device)
    ckpt_path = args.checkpoint or os.path.join(
        cfg["training"]["checkpoint_dir"], "best_model.pt"
    )
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    log.info(f"Model loaded from: {ckpt_path}")

    # ── 1. Gradient saliency ─────────────────────────────────────────────────
    log.info(
        f"Computing gradient saliency "
        f"(cell_sample={SALIENCY_CELL_SAMPLE} stratified, {num_cell_types} classes sequential)..."
    )
    saliency = compute_mirna_saliency(model, graph, device, num_cell_types)

    mirna_names = [idx2mirna[i] for i in range(saliency.shape[0])]
    df_saliency = pd.DataFrame(saliency, index=mirna_names, columns=cell_type_labels)
    df_saliency.index.name = "mirna"
    saliency_out = os.path.join(out_dir, "mirna_saliency_by_celltype.tsv")
    df_saliency.to_csv(saliency_out, sep="\t")
    log.info(f"Saved: {saliency_out}")

    # Top-K miRNAs per cell type
    for cell_type in cell_type_labels:
        top = df_saliency[cell_type].nlargest(args.top_k)
        log.info(f"\n  Top-{args.top_k} miRNAs for [{cell_type}]:")
        for mirna, score in top.items():
            log.info(f"    {mirna:<25} {score:.6f}")

    # ── 2. Edge scoring (top regulatory circuits) ────────────────────────────
    log.info("\nScoring miRNA→gene pairs...")
    edge_scores = score_all_mirna_gene_pairs(model, graph, device)

    pos_edges = graph["miRNA", "regulates", "gene"].edge_index.numpy()
    df_edges  = pd.DataFrame({
        "mirna":      [idx2mirna[i] for i in pos_edges[0]],
        "target_gene":[idx2gene[i]  for i in pos_edges[1]],
        "score":      edge_scores,
    }).sort_values("score", ascending=False)

    edges_out = os.path.join(out_dir, "all_edge_scores.tsv")
    df_edges.to_csv(edges_out, sep="\t", index=False)
    log.info(f"Saved: {edges_out}")

    # Per-cell-type: use top saliency miRNAs to filter circuits
    circuits_rows = []
    for cell_type in cell_type_labels:
        top_mirnas = set(df_saliency[cell_type].nlargest(args.top_k).index)
        ct_edges = df_edges[df_edges["mirna"].isin(top_mirnas)].head(200).copy()
        ct_edges["cell_type"] = cell_type
        circuits_rows.append(ct_edges)

    df_circuits = pd.concat(circuits_rows, ignore_index=True)
    circuits_out = os.path.join(out_dir, "top_circuits_by_celltype.tsv")
    df_circuits.to_csv(circuits_out, sep="\t", index=False)
    log.info(f"Saved: {circuits_out}")

    # ── 3. Enrichment analysis per cell type ─────────────────────────────────
    log.info("\nRunning GO/KEGG enrichment...")
    for cell_type in cell_type_labels:
        safe_name = cell_type.replace(" ", "_").replace("/", "_")
        top_genes = (
            df_circuits[df_circuits["cell_type"] == cell_type]["target_gene"]
            .head(args.top_genes)
            .tolist()
        )
        if len(top_genes) < 5:
            log.info(f"  Skipping {cell_type}: too few target genes ({len(top_genes)})")
            continue
        log.info(f"  Enrichment for {cell_type} ({len(top_genes)} genes)...")
        run_enrichment(top_genes, safe_name, enr_dir)

    log.info("\nInterpretation complete.")


if __name__ == "__main__":
    main()
