"""
evaluate.py — Evaluation metrics for miRNAGraphTransformer.

Metrics:
  Link prediction  → AUROC, AUPRC (for miRNA→gene target prediction)
  Classification   → Accuracy, macro-F1 (for cell type annotation)

The evaluate() function is called during training (on val set) and
standalone evaluation (on test set).
"""

from __future__ import annotations

import sys
import pickle
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    classification_report,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MIRNA_GENE_REL = ("miRNA", "regulates", "gene")


def get_mirna_gene_edges(graph: HeteroData) -> torch.Tensor | None:
    """
    Positive miRNA→gene edges, or None when the relation is absent (ablation_no_mirna).

    Must not use HeteroData.get(): for a tuple edge-type key it returns None even when
    the relation exists, which silently disables link prediction in both training and
    evaluation. Membership has to be tested against edge_types.
    """
    if MIRNA_GENE_REL not in graph.edge_types:
        return None
    return graph[MIRNA_GENE_REL].edge_index


def evaluate(
    model,
    loader: NeighborLoader,
    criterion,
    device: torch.device,
    full_graph: HeteroData,
    sampler=None,
    sup_edges: torch.Tensor | None = None,
) -> dict[str, float]:
    """
    Run one pass over the val/test loader and return metrics.

    sampler + sup_edges (a splits.LinkSampler and a global miRNA→gene supervision edge
    set) give the held-out-edge metric: only sup_edges are scored as positives, and
    negatives are degree-matched and excluded against every known edge.

    Without them the old transductive path runs — every miRNA→gene edge in the graph is
    fair game as a positive, negatives are uniform. That number is kept deliberately: it
    is the original protocol, so the comparison table can show both side by side.
    """
    from torch_geometric.utils import negative_sampling

    model.eval()
    all_edge_logits: list[np.ndarray] = []
    all_edge_labels: list[np.ndarray] = []
    all_cell_preds:  list[np.ndarray] = []
    all_cell_labels: list[np.ndarray] = []
    totals = {"loss": 0.0, "link_loss": 0.0, "clf_loss": 0.0}
    n_batches  = 0

    # Positive edges for link prediction (absent in ablation_no_mirna)
    pos_edge_full = get_mirna_gene_edges(full_graph)

    def _map_global_to_local(batch, max_pairs: int = 256):
        """Remap global miRNA→gene edges to local batch indices via n_id."""
        if pos_edge_full is None:
            return None
        mirna_g = batch["miRNA"].n_id.to(device)
        gene_g  = batch["gene"].n_id.to(device)
        n_mirna_g = max(int(pos_edge_full[0].max().item()), int(mirna_g.max().item())) + 1
        n_gene_g  = max(int(pos_edge_full[1].max().item()), int(gene_g.max().item())) + 1
        mirna_g2l = torch.full((n_mirna_g,), -1, dtype=torch.long, device=device)
        mirna_g2l[mirna_g] = torch.arange(len(mirna_g), device=device)
        gene_g2l = torch.full((n_gene_g,), -1, dtype=torch.long, device=device)
        gene_g2l[gene_g] = torch.arange(len(gene_g), device=device)
        src_l = mirna_g2l[pos_edge_full[0].to(device)]
        dst_l = gene_g2l[pos_edge_full[1].to(device)]
        valid = (src_l >= 0) & (dst_l >= 0)
        src_l, dst_l = src_l[valid], dst_l[valid]
        if src_l.shape[0] == 0:
            return None
        bp = torch.stack([src_l, dst_l])
        if bp.shape[1] > max_pairs:
            perm = torch.randperm(bp.shape[1], device=device)[:max_pairs]
            bp = bp[:, perm]
        return bp

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            if sampler is not None and sup_edges is not None:
                # Held-out-edge path: score only the supervision edges of this split.
                mirna_idx, gene_idx, labels = sampler.sample(
                    batch, sup_edges, device, max_pairs=256
                )
                if mirna_idx is None:
                    continue  # no supervision edge landed in this batch
            elif pos_edge_full is not None:
                # Transductive path (the original number).
                batch_pos = _map_global_to_local(batch, max_pairs=256)
                if batch_pos is None:
                    continue  # batch has no miRNA/gene overlap — skip
                n_mirna = batch["miRNA"].num_nodes
                n_gene  = batch["gene"].num_nodes
                n_pos = batch_pos.shape[1]
                neg_edge = negative_sampling(
                    edge_index=batch_pos,
                    num_nodes=(n_mirna, n_gene),
                    num_neg_samples=n_pos,
                    method="sparse",
                )
                mirna_idx = torch.cat([batch_pos[0], neg_edge[0]]).to(device)
                gene_idx  = torch.cat([batch_pos[1], neg_edge[1]]).to(device)
                labels    = torch.cat([
                    torch.ones(n_pos),
                    torch.zeros(neg_edge.shape[1]),
                ]).to(device)
            else:
                # Ablation: no miRNA→gene edges — classification only
                mirna_idx, gene_idx, labels = None, None, None

            out = model(
                x_dict=batch.x_dict,
                edge_index_dict=batch.edge_index_dict,
                mirna_idx=mirna_idx,
                gene_idx=gene_idx,
            )

            mirna_emb = out["embeddings"].get("miRNA")
            loss_dict = criterion(
                edge_logits=out.get("edge_logits"),
                edge_labels=labels,
                cell_logits=out["cell_logits"],
                cell_labels=batch["cell"].y,
                mirna_emb=mirna_emb,
            )
            for k in totals:
                totals[k] += loss_dict[k].item()

            if labels is not None and "edge_logits" in out:
                all_edge_logits.append(torch.sigmoid(out["edge_logits"]).cpu().numpy())
                all_edge_labels.append(labels.cpu().numpy())

            cell_preds = out["cell_logits"].argmax(dim=-1).cpu().numpy()
            all_cell_preds.append(cell_preds)
            all_cell_labels.append(batch["cell"].y.cpu().numpy())

            n_batches += 1

    # ── Metrics ───────────────────────────────────────────────────────────────
    # link_loss and clf_loss are reported separately: a model without a link head
    # optimizes a strictly smaller objective, so its total `loss` is not comparable
    # across rows of the comparison table. The per-task terms are.
    metrics: dict[str, float] = {k: v / max(n_batches, 1) for k, v in totals.items()}

    if all_edge_logits:
        edge_scores = np.concatenate(all_edge_logits)
        edge_labels = np.concatenate(all_edge_labels)
        if edge_labels.sum() > 0 and edge_labels.sum() < len(edge_labels):
            metrics["auroc"] = float(roc_auc_score(edge_labels, edge_scores))
            metrics["auprc"] = float(average_precision_score(edge_labels, edge_scores))

    cell_preds  = np.concatenate(all_cell_preds)
    cell_labels = np.concatenate(all_cell_labels)
    valid       = cell_labels >= 0
    if valid.sum() > 0:
        metrics["cell_acc"] = float(accuracy_score(cell_labels[valid], cell_preds[valid]))
        metrics["cell_f1"]  = float(
            f1_score(cell_labels[valid], cell_preds[valid], average="macro", zero_division=0)
        )

    return metrics


# ── Standalone evaluation on test set ─────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate trained model on test set")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()

    import yaml
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    import os
    from models.hetero_gnn import miRNAGraphTransformer
    from models.losses import CombinedLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    graphs_dir = cfg["data"]["graphs_dir"]
    graph = torch.load(os.path.join(graphs_dir, "hetero_graph.pt"))
    with open(os.path.join(graphs_dir, "index_maps.pkl"), "rb") as fh:
        index_maps = pickle.load(fh)

    num_cell_types = len(index_maps["cell_type_labels"])
    metadata = graph.metadata()
    model = miRNAGraphTransformer.from_config(cfg, metadata, num_cell_types).to(device)

    ckpt_path = args.checkpoint or os.path.join(
        cfg["training"]["checkpoint_dir"], "best_model.pt"
    )
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    log.info(f"Loaded checkpoint: {ckpt_path} (epoch {ck['epoch']})")

    tcfg = cfg["training"]
    n_cells = graph["cell"].num_nodes
    idx     = torch.randperm(n_cells, generator=torch.Generator().manual_seed(cfg["project"]["seed"]))
    n_test  = int(n_cells * tcfg["test_ratio"])
    n_val   = int(n_cells * tcfg["val_ratio"])
    test_idx = idx[:n_test]
    test_mask = torch.zeros(n_cells, dtype=torch.bool)
    test_mask[test_idx] = True
    graph["cell"].test_mask = test_mask

    test_loader = NeighborLoader(
        graph,
        num_neighbors={et: tcfg["num_neighbors"] for et in graph.edge_types},
        batch_size=tcfg["batch_size"],
        input_nodes=("cell", test_mask),
        shuffle=False,
    )
    criterion = CombinedLoss(
        reconstruction_weight=tcfg["loss_reconstruction_weight"],
        classification_weight=tcfg["loss_classification_weight"],
        sparsity_weight=tcfg["loss_sparsity_weight"],
    ).to(device)

    metrics = evaluate(model, test_loader, criterion, device, graph)

    log.info("─" * 50)
    log.info("TEST SET RESULTS")
    for k, v in metrics.items():
        log.info(f"  {k:<15}: {v:.4f}")
    log.info("─" * 50)


if __name__ == "__main__":
    main()
