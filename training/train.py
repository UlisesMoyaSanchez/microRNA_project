"""
train.py — Multi-GPU distributed training for miRNAGraphTransformer.

Execution:
  Single GPU:    python training/train.py --config configs/config.yaml
  Multi-GPU DDP: torchrun --nproc_per_node=4 training/train.py --config configs/config.yaml

Training strategy:
  - Full graph lives on CPU; NeighborLoader samples mini-batches per GPU process.
  - Model wrapped in DistributedDataParallel (NCCL backend) for multi-GPU.
  - Negative sampling for link prediction: for each positive (miRNA, gene) edge,
    sample an equal number of random (miRNA, gene') negatives.
  - Early stopping on validation loss with configurable patience.
  - Checkpointing: saves best model by val loss; resumes from checkpoint if found.
"""

from __future__ import annotations

import os
import sys
import pickle
import argparse
import random
import logging
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import negative_sampling

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.hetero_gnn import miRNAGraphTransformer
from models.losses import CombinedLoss
from training.evaluate import evaluate


# ── Setup ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    return p.parse_args()


def setup_logging(rank: int, log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format=f"[rank{rank}] %(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, f"train_rank{rank}.log")),
        ],
    )


def set_seed(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_graph(graphs_dir: str) -> tuple[HeteroData, dict]:
    graph = torch.load(os.path.join(graphs_dir, "hetero_graph.pt"))
    with open(os.path.join(graphs_dir, "index_maps.pkl"), "rb") as fh:
        index_maps = pickle.load(fh)
    return graph, index_maps


def split_graph(
    graph: HeteroData,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split cell nodes into train/val/test masks."""
    n_cells = graph["cell"].num_nodes
    idx     = torch.randperm(n_cells, generator=torch.Generator().manual_seed(seed))
    n_test  = int(n_cells * test_ratio)
    n_val   = int(n_cells * val_ratio)

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test: n_test + n_val]
    train_idx = idx[n_test + n_val:]

    train_mask = torch.zeros(n_cells, dtype=torch.bool)
    val_mask   = torch.zeros(n_cells, dtype=torch.bool)
    test_mask  = torch.zeros(n_cells, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    return train_mask, val_mask, test_mask


def sample_link_prediction_pairs(
    pos_edge_index: torch.Tensor,
    n_mirna: int,
    n_gene: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (mirna_idx, gene_idx, labels) with balanced pos/neg pairs.
    """
    n_pos = pos_edge_index.shape[1]
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=(n_mirna, n_gene),
        num_neg_samples=n_pos,
        method="sparse",
    )
    mirna_idx = torch.cat([pos_edge_index[0], neg_edge_index[0]], dim=0).to(device)
    gene_idx  = torch.cat([pos_edge_index[1], neg_edge_index[1]], dim=0).to(device)
    labels    = torch.cat([
        torch.ones(n_pos, device=device),
        torch.zeros(neg_edge_index.shape[1], device=device),
    ])
    return mirna_idx, gene_idx, labels


# ── Index remapping helper ────────────────────────────────────────────────────

def map_global_edges_to_local(
    pos_edge_global: torch.Tensor,
    batch,
    device: torch.device,
    max_pairs: int = 512,
) -> torch.Tensor | None:
    """
    NeighborLoader mini-batches use *local* node indices (0..n_local-1).
    pos_edge_global uses *global* indices (0..n_total_nodes-1).
    This remaps global miRNA→gene edges to local batch indices using
    batch["miRNA"].n_id and batch["gene"].n_id, then subsamples to max_pairs.
    Returns (2, k) local edge index, or None if no valid edges in this batch.
    """
    mirna_g = batch["miRNA"].n_id.to(device)  # (n_local_mirna,) global IDs
    gene_g  = batch["gene"].n_id.to(device)   # (n_local_gene,)  global IDs

    # Size lookup tables to cover ALL global IDs seen in both the edge list
    # and the current batch — not every miRNA/gene appears in pos_edge_full.
    n_mirna_g = max(int(pos_edge_global[0].max().item()), int(mirna_g.max().item())) + 1
    n_gene_g  = max(int(pos_edge_global[1].max().item()), int(gene_g.max().item())) + 1

    mirna_g2l = torch.full((n_mirna_g,), -1, dtype=torch.long, device=device)
    mirna_g2l[mirna_g] = torch.arange(len(mirna_g), device=device)

    gene_g2l = torch.full((n_gene_g,), -1, dtype=torch.long, device=device)
    gene_g2l[gene_g] = torch.arange(len(gene_g), device=device)

    src_l = mirna_g2l[pos_edge_global[0].to(device)]
    dst_l = gene_g2l[pos_edge_global[1].to(device)]

    valid = (src_l >= 0) & (dst_l >= 0)
    src_l, dst_l = src_l[valid], dst_l[valid]

    if src_l.shape[0] == 0:
        return None

    batch_pos = torch.stack([src_l, dst_l])
    if batch_pos.shape[1] > max_pairs:
        perm = torch.randperm(batch_pos.shape[1], device=device)[:max_pairs]
        batch_pos = batch_pos[:, perm]

    return batch_pos


# ── Training loop ───────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     DDP | miRNAGraphTransformer,
    loader:    NeighborLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CombinedLoss,
    device:    torch.device,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {"loss": 0.0, "link_loss": 0.0, "clf_loss": 0.0}
    n_batches = 0

    # Safely get positive edges for link prediction (absent in ablation_no_mirna)
    _edge_store   = loader.data.get(("miRNA", "regulates", "gene"))
    pos_edge_full = _edge_store.edge_index if (_edge_store is not None and hasattr(_edge_store, "edge_index")) else None

    for batch in loader:
        batch = batch.to(device)

        if pos_edge_full is not None:
            # Remap global pos edges to local batch indices (NeighborLoader uses local IDs)
            batch_pos = map_global_edges_to_local(pos_edge_full, batch, device, max_pairs=512)
            if batch_pos is None:
                continue  # batch has no miRNA/gene overlap — skip
            n_mirna = batch["miRNA"].num_nodes
            n_gene  = batch["gene"].num_nodes
            mirna_idx, gene_idx, labels = sample_link_prediction_pairs(
                batch_pos, n_mirna, n_gene, device
            )
        else:
            # Ablation: no miRNA→gene edges — classification only
            mirna_idx, gene_idx, labels = None, None, None

        optimizer.zero_grad()
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
        loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in loss_dict.items():
            if k in totals:
                totals[k] += v.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ── Checkpointing ──────────────────────────────────────────────────────────────

def save_checkpoint(
    model: miRNAGraphTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    path: str,
) -> None:
    raw = model.module if hasattr(model, "module") else model
    torch.save({
        "epoch":      epoch,
        "val_loss":   val_loss,
        "model":      raw.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }, path)


def load_checkpoint(
    model: miRNAGraphTransformer,
    optimizer: torch.optim.Optimizer,
    path: str,
) -> tuple[int, float]:
    ck = torch.load(path, weights_only=False)
    raw = model.module if hasattr(model, "module") else model
    raw.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    return ck["epoch"], ck["val_loss"]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    # ── Distributed init ─────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    set_seed(cfg["project"]["seed"], local_rank)
    setup_logging(local_rank, cfg["training"]["log_dir"])
    log = logging.getLogger(__name__)

    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["training"]["log_dir"], exist_ok=True)

    # ── Load graph ────────────────────────────────────────────────────────────
    log.info("Loading graph...")
    graph, index_maps = load_graph(cfg["data"]["graphs_dir"])
    cell_type_labels: list[str] = index_maps["cell_type_labels"]
    num_cell_types = len(cell_type_labels)
    log.info(f"Graph: {graph}")
    log.info(f"Cell types: {num_cell_types}")

    # ── Data splits ───────────────────────────────────────────────────────────
    tcfg = cfg["training"]
    train_mask, val_mask, test_mask = split_graph(
        graph, tcfg["val_ratio"], tcfg["test_ratio"], cfg["project"]["seed"]
    )
    graph["cell"].train_mask = train_mask
    graph["cell"].val_mask   = val_mask
    graph["cell"].test_mask  = test_mask

    # ── DataLoaders ───────────────────────────────────────────────────────────
    num_workers = max(0, int(os.environ.get("SLURM_CPUS_PER_TASK", 4)) // world_size() - 1)
    loader_kwargs = dict(
        num_neighbors={et: tcfg["num_neighbors"] for et in graph.edge_types},
        batch_size=tcfg["batch_size"],
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    train_loader = NeighborLoader(
        graph, input_nodes=("cell", train_mask), shuffle=True, **loader_kwargs
    )
    val_loader = NeighborLoader(
        graph, input_nodes=("cell", val_mask), shuffle=False, **loader_kwargs
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    metadata = graph.metadata()
    model = miRNAGraphTransformer.from_config(cfg, metadata, num_cell_types).to(device)

    # PyG lazy Linear(-1, ...) parameters are uninitialized until the first forward
    # pass. DDP refuses to wrap such a model, so we run a dummy forward here to
    # materialize all parameter shapes before wrapping.
    with torch.no_grad():
        _dummy = next(iter(train_loader))
        _dummy = _dummy.to(device)
        model(
            x_dict=_dummy.x_dict,
            edge_index_dict=_dummy.edge_index_dict,
        )
    del _dummy

    if is_dist():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tcfg["num_epochs"]
    )
    criterion = CombinedLoss(
        reconstruction_weight=tcfg["loss_reconstruction_weight"],
        classification_weight=tcfg["loss_classification_weight"],
        sparsity_weight=tcfg["loss_sparsity_weight"],
    ).to(device)

    # ── Resume ───────────────────────────────────────────────────────────────
    best_ck  = os.path.join(cfg["training"]["checkpoint_dir"], "best_model.pt")
    start_epoch = 0
    best_val = float("inf")
    patience_counter = 0

    if args.resume and os.path.exists(best_ck):
        start_epoch, best_val = load_checkpoint(model, optimizer, best_ck)
        log.info(f"Resumed from epoch {start_epoch}, val_loss={best_val:.4f}")

    # ── Training ──────────────────────────────────────────────────────────────
    log.info(f"Starting training (epochs={tcfg['num_epochs']}, patience={tcfg['patience']})")
    for epoch in range(start_epoch, tcfg["num_epochs"]):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        if rank() == 0:
            val_metrics = evaluate(model, val_loader, criterion, device, graph)
            log.info(
                f"Epoch {epoch + 1:03d} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_auroc={val_metrics.get('auroc', 0):.4f} | "
                f"val_acc={val_metrics.get('cell_acc', 0):.4f}"
            )

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                patience_counter = 0
                raw = model.module if hasattr(model, "module") else model
                save_checkpoint(raw, optimizer, epoch + 1, best_val, best_ck)
                log.info(f"  → New best model saved (val_loss={best_val:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= tcfg["patience"]:
                    log.info(f"Early stopping at epoch {epoch + 1}")
                    break

        if is_dist():
            dist.barrier()

    if rank() == 0:
        log.info(f"Training complete. Best val_loss={best_val:.4f}")
        log.info(f"Best checkpoint: {best_ck}")

    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
