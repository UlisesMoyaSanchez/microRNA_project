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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.hetero_gnn import miRNAGraphTransformer
from models.losses import CombinedLoss
from training.evaluate import evaluate, get_mirna_gene_edges
from training.splits import (
    LinkSampler,
    REL_FWD,
    assert_no_edge_leakage,
    build_edge_split,
    gene_in_degree,
)


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


# ── Training loop ───────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     DDP | miRNAGraphTransformer,
    loader:    NeighborLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CombinedLoss,
    device:    torch.device,
    sampler:   LinkSampler | None = None,
    sup_edges: torch.Tensor | None = None,
) -> dict[str, float]:
    """
    sampler + sup_edges supervise the link head on the training edge split only, with
    degree-matched negatives. Without them the link head is not trained at all — which is
    correct for the no_mirna ablation and a bug anywhere else, hence the check in main().
    """
    model.train()
    totals: dict[str, float] = {"loss": 0.0, "link_loss": 0.0, "clf_loss": 0.0}
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)

        if sampler is not None and sup_edges is not None:
            mirna_idx, gene_idx, labels = sampler.sample(
                batch, sup_edges, device, max_pairs=512
            )
            if mirna_idx is None:
                continue  # no training supervision edge landed in this batch
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

    # State the link-prediction status out loud. A silent "None" here previously
    # disabled the link head for every model without any visible signal.
    _pos = get_mirna_gene_edges(graph)
    if _pos is None:
        log.warning(
            "No (miRNA, regulates, gene) edges — LINK PREDICTION DISABLED "
            "(classification only). Expected only for the no_mirna ablation."
        )
    else:
        log.info(f"Link prediction ACTIVE — {_pos.shape[1]:,} positive miRNA→gene edges")

    # ── Cell-node split (unchanged: the classification task is unaffected) ────
    tcfg = cfg["training"]
    seed = cfg["project"]["seed"]
    train_mask, val_mask, test_mask = split_graph(
        graph, tcfg["val_ratio"], tcfg["test_ratio"], seed
    )
    graph["cell"].train_mask = train_mask
    graph["cell"].val_mask   = val_mask
    graph["cell"].test_mask  = test_mask

    # ── Edge-level split ──────────────────────────────────────────────────────
    # Without this the encoder is handed the very edges the link head is asked to
    # predict, and the AUROC is a reconstruction score rather than a prediction.
    sampler:   LinkSampler | None   = None
    train_sup: torch.Tensor | None  = None
    val_sup:   torch.Tensor | None  = None

    if _pos is not None:
        edge_split = build_edge_split(
            graph,
            val_ratio=tcfg["val_ratio"],
            test_ratio=tcfg["test_ratio"],
            seed=seed,
            disjoint_train_ratio=tcfg.get("disjoint_train_ratio", 0.3),
        )
        assert_no_edge_leakage(edge_split)

        # From here on the encoder only ever sees training message-passing edges.
        graph     = edge_split.mp_graph
        train_sup = edge_split.train_sup
        val_sup   = edge_split.val_sup

        # Bin genes by in-degree over TRAINING edges only — binning on the full edge
        # set would leak held-out structure into the choice of negatives.
        train_edges = torch.cat([graph[REL_FWD].edge_index, train_sup], dim=1)
        deg = gene_in_degree(train_edges, graph["gene"].num_nodes)
        sampler = LinkSampler(
            all_pos_global=edge_split.all_pos,
            deg=deg,
            seed=seed,
            hard=tcfg.get("hard_negatives", True),
        )
        log.info(
            f"Negatives: {'degree-matched (hard)' if sampler.hard else 'uniform'}"
        )

        # build_edge_split reseeds the global RNG so every DDP rank derives the same
        # split; restore this rank's own seed so the ranks do not train in lockstep.
        set_seed(seed, local_rank)

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
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            sampler=sampler, sup_edges=train_sup,
        )
        scheduler.step()

        if rank() == 0:
            # val_auroc is now a held-out-edge number: val_sup edges were never message-
            # passed and never supervised, and the negatives are degree-matched.
            val_metrics = evaluate(
                model, val_loader, criterion, device, graph,
                sampler=sampler, sup_edges=val_sup,
            )
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
        if sampler is not None and sampler.hard:
            # A high rate means many negatives could not be degree-matched and fell back
            # to uniform, which drags the metric back toward the inflated one.
            log.info(f"Degree-matched negative fallback rate: {sampler.fallback_pct:.1f}%")

    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
