"""
run_baselines.py — Train and evaluate all baseline & ablation models.

Runs the following experiments sequentially on a single GPU, saving
results to results/comparison/comparison_table.tsv:

  random     — RandomBaseline (no training, floor reference)
  mlp        — MLPBaseline    (no graph structure)
  homo_gcn   — HomoGCNBaseline (homogeneous GCN, no type semantics)
  no_mirna   — miRNAGraphTransformer (V2) without miRNA→gene edges (ablation A)
  no_coexpr  — miRNAGraphTransformer (V2) without gene co-expression edges (ablation B)

Usage:
  python training/run_baselines.py --config configs/config_v2.yaml

The best V2 metrics are read from logs/best_v2_metrics.json if present,
otherwise re-evaluated from the checkpoint.
"""

from __future__ import annotations

import os
import sys
import json
import pickle
import argparse
import logging
from pathlib import Path

import yaml
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.hetero_gnn import miRNAGraphTransformer
from models.baselines import MLPBaseline, HomoGCNBaseline, RandomBaseline
from models.losses import CombinedLoss
from training.evaluate import evaluate, get_mirna_gene_edges
from training.train import (
    load_graph,
    split_graph,
    train_one_epoch,
    save_checkpoint,
    set_seed,
)
from training.splits import (
    LinkSampler,
    REL_FWD,
    assert_no_edge_leakage,
    build_edge_split,
    gene_in_degree,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config_v2.yaml",
                   help="Base V2 config (used for all experiments)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override num_epochs for baselines (default: use config value)")
    p.add_argument("--skip_training", action="store_true",
                   help="Skip training — only evaluate existing checkpoints")
    return p.parse_args()


def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, "baselines.log")),
        ],
    )


def drop_edge_types(graph: HeteroData, drop_keys: list[str]) -> HeteroData:
    """
    Return a copy of graph with specified edge types removed.
    drop_keys: list of "src,rel,dst" strings matching graph edge_types.
    Expected format from ablation_drop_edge_types in config.
    """
    if not drop_keys:
        return graph

    drop_set = set()
    for k in drop_keys:
        parts = k.split(",")
        if len(parts) == 3:
            drop_set.add(tuple(parts))

    import copy
    g = copy.copy(graph)
    for et in list(g.edge_types):
        if et in drop_set:
            # Remove both edge_index and any edge attributes
            del g[et]
    return g


def train_model(
    model,
    train_loader: NeighborLoader,
    val_loader: NeighborLoader,
    graph: HeteroData,
    cfg: dict,
    device: torch.device,
    checkpoint_path: str,
    num_epochs: int | None,
    log: logging.Logger,
    sampler: LinkSampler | None = None,
    train_sup: torch.Tensor | None = None,
    val_sup: torch.Tensor | None = None,
) -> dict[str, float]:
    """Full train + early stopping loop. Returns best val metrics."""
    tcfg = cfg["training"]
    epochs = num_epochs if num_epochs is not None else tcfg["num_epochs"]
    patience = tcfg["patience"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
    )
    criterion = CombinedLoss(
        reconstruction_weight=tcfg["loss_reconstruction_weight"],
        classification_weight=tcfg["loss_classification_weight"],
        sparsity_weight=tcfg["loss_sparsity_weight"],
    ).to(device)

    best_val   = float("inf")
    pat_count  = 0
    best_metrics: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            sampler=sampler, sup_edges=train_sup,
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device, graph,
            sampler=sampler, sup_edges=val_sup,
        )

        log.info(
            f"  epoch {epoch:03d} | train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_auroc={val_metrics.get('auroc', 0):.4f} | "
            f"val_acc={val_metrics.get('cell_acc', 0):.4f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            pat_count = 0
            best_metrics = val_metrics
            torch.save(model.state_dict(), checkpoint_path)
        else:
            pat_count += 1
            if pat_count >= patience:
                log.info(f"  Early stopping at epoch {epoch}")
                break

    return best_metrics


def evaluate_both(
    model,
    val_loader: NeighborLoader,
    graph: HeteroData,
    cfg: dict,
    device: torch.device,
    checkpoint_path: str,
    log: logging.Logger,
    sampler: LinkSampler | None = None,
    val_sup: torch.Tensor | None = None,
) -> dict[str, float]:
    """
    Warm up lazy linears, load the checkpoint, and score it two ways on the same cells:

      auroc            — held-out edges, degree-matched negatives. The honest number.
      auroc_transd     — every miRNA→gene edge is fair game, uniform negatives. This is
                         the protocol that produced the published 0.9836, kept so the
                         table shows the drop rather than quietly replacing the number.

    Both come from the *best* checkpoint, not whatever was last in memory after training.
    """
    # PyG uses Linear(-1, ...) (lazy) — must run one forward pass to
    # materialize parameter shapes before load_state_dict can work.
    with torch.no_grad():
        try:
            _dummy = next(iter(val_loader)).to(device)
            model(_dummy.x_dict, _dummy.edge_index_dict)
            del _dummy
            torch.cuda.empty_cache()
        except Exception as e:
            log.warning(f"  Warm-up pass failed (will attempt load anyway): {e}")

    if checkpoint_path and os.path.exists(checkpoint_path):
        # Checkpoint is saved as {"epoch":..., "val_loss":..., "model":..., "optimizer":...}
        ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ck.get("model", ck)  # handle both raw and nested checkpoint formats
        model.load_state_dict(state_dict)
        epoch = ck.get("epoch", "?")
        val_loss = ck.get("val_loss", "?")
        log.info(f"  Loaded checkpoint: {checkpoint_path}  (epoch={epoch}, val_loss={val_loss})")
    else:
        log.warning(f"  No checkpoint at '{checkpoint_path}'; evaluating untrained model")

    tcfg = cfg["training"]
    criterion = CombinedLoss(
        reconstruction_weight=tcfg["loss_reconstruction_weight"],
        classification_weight=tcfg["loss_classification_weight"],
        sparsity_weight=tcfg["loss_sparsity_weight"],
    ).to(device)

    metrics = evaluate(
        model, val_loader, criterion, device, graph,
        sampler=sampler, sup_edges=val_sup,
    )
    transd = evaluate(model, val_loader, criterion, device, graph)
    metrics["auroc_transd"] = transd.get("auroc", float("nan"))
    metrics["auprc_transd"] = transd.get("auprc", float("nan"))
    return metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["project"]["seed"], 0)
    log_dir = cfg["training"]["log_dir"]
    setup_logging(log_dir)
    log = logging.getLogger(__name__)
    log.info(f"Device: {device}")

    out_dir = os.path.join(cfg["project"]["output_dir"], "comparison")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load graph ─────────────────────────────────────────────────────────
    log.info("Loading graph...")
    graph, index_maps = load_graph(cfg["data"]["graphs_dir"])
    cell_type_labels: list[str] = index_maps["cell_type_labels"]
    num_cell_types = len(cell_type_labels)
    metadata = graph.metadata()

    tcfg = cfg["training"]
    seed = cfg["project"]["seed"]
    train_mask, val_mask, _ = split_graph(
        graph, tcfg["val_ratio"], tcfg["test_ratio"], seed
    )
    graph["cell"].train_mask = train_mask
    graph["cell"].val_mask   = val_mask

    # ── Edge-level split ───────────────────────────────────────────────────
    # Same split for every row, so the models are compared on identical held-out
    # edges. graph is replaced by the message-passing graph: val/test edges are
    # absent from it in both directions.
    edge_split = build_edge_split(
        graph,
        val_ratio=tcfg["val_ratio"],
        test_ratio=tcfg["test_ratio"],
        seed=seed,
        disjoint_train_ratio=tcfg.get("disjoint_train_ratio", 0.3),
    )
    assert_no_edge_leakage(edge_split)

    # The pre-split graph, kept only to reproduce the *published* V2 number under the
    # protocol that produced it. Nothing else may be evaluated on it.
    intact_graph = graph

    graph     = edge_split.mp_graph
    train_sup = edge_split.train_sup
    val_sup   = edge_split.val_sup

    train_edges = torch.cat([graph[REL_FWD].edge_index, train_sup], dim=1)
    deg = gene_in_degree(train_edges, graph["gene"].num_nodes)
    sampler = LinkSampler(
        all_pos_global=edge_split.all_pos,
        deg=deg,
        seed=seed,
        hard=tcfg.get("hard_negatives", True),
    )
    metadata = graph.metadata()
    set_seed(seed, 0)  # build_edge_split reseeds the global RNG

    num_workers = min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", 4)) - 1)
    loader_kwargs = dict(
        num_neighbors={et: tcfg["num_neighbors"] for et in graph.edge_types},
        batch_size=tcfg["batch_size"],
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    train_loader = NeighborLoader(graph, input_nodes=("cell", train_mask), shuffle=True,  **loader_kwargs)
    val_loader   = NeighborLoader(graph, input_nodes=("cell", val_mask),   shuffle=False, **loader_kwargs)

    # ── Experiment registry ────────────────────────────────────────────────
    # Each entry: (name, model_factory, graph_override, checkpoint_path)
    project_dir = Path(args.config).parent.parent

    experiments = [
        # (label, model_class, graph_to_use, ckpt_path)
        (
            # The headline row: the same V2 architecture retrained under the held-out-edge
            # split with degree-matched negatives. Loads the checkpoint produced by
            #   sbatch --export=ALL,CONFIG=configs/config_v2_edgesplit.yaml training/slurm_train.sh
            # when it exists, so this does not silently duplicate that run.
            "hgt_v2_edgesplit",
            lambda: miRNAGraphTransformer.from_config(cfg, metadata, num_cell_types),
            graph,
            os.path.join(tcfg["checkpoint_dir"], "best_model.pt"),
        ),
        (
            "random",
            lambda: RandomBaseline.from_config(cfg, metadata, num_cell_types),
            graph,
            None,  # no checkpoint needed
        ),
        (
            "mlp",
            lambda: MLPBaseline.from_config(cfg, metadata, num_cell_types),
            graph,
            str(project_dir / "checkpoints_baseline_mlp" / "best_model.pt"),
        ),
        (
            "homo_gcn",
            lambda: HomoGCNBaseline.from_config(cfg, metadata, num_cell_types),
            graph,
            str(project_dir / "checkpoints_baseline_gcn" / "best_model.pt"),
        ),
        (
            "ablation_no_mirna",
            lambda: miRNAGraphTransformer.from_config(
                cfg,
                drop_edge_types(graph, ["miRNA,regulates,gene", "gene,regulated_by,miRNA"]).metadata(),
                num_cell_types,
            ),
            drop_edge_types(graph, ["miRNA,regulates,gene", "gene,regulated_by,miRNA"]),
            str(project_dir / "checkpoints_ablation_no_mirna" / "best_model.pt"),
        ),
        (
            "ablation_no_coexpr",
            lambda: miRNAGraphTransformer.from_config(
                cfg,
                drop_edge_types(graph, ["gene,coexpressed_with,gene"]).metadata(),
                num_cell_types,
            ),
            drop_edge_types(graph, ["gene,coexpressed_with,gene"]),
            str(project_dir / "checkpoints_ablation_no_coexpr" / "best_model.pt"),
        ),
    ]

    # ── V2 (the published model) — transductive reference row only ─────────
    # This checkpoint was trained with every miRNA→gene edge as a supervision target,
    # so the "held-out" edges of the split above are not held out *for it*. Scoring it
    # on them would report a memorized number in the honest column. Its held-out cells
    # are left nan on purpose, and it is evaluated on the intact graph — the protocol
    # that actually produced the published 0.9836.
    v2_metrics_path = os.path.join(out_dir, "v2_metrics.json")
    if os.path.exists(v2_metrics_path):
        with open(v2_metrics_path) as fh:
            v2_metrics = json.load(fh)
        log.info(f"Loaded V2 metrics from {v2_metrics_path}")
    else:
        log.info("Evaluating V2 (published) from checkpoint on the INTACT graph...")
        v2_ckpt = str(project_dir / "checkpoints_v2" / "best_model.pt")
        v2_model = miRNAGraphTransformer.from_config(
            cfg, intact_graph.metadata(), num_cell_types
        ).to(device)
        intact_val_loader = NeighborLoader(
            intact_graph,
            num_neighbors={et: tcfg["num_neighbors"] for et in intact_graph.edge_types},
            batch_size=tcfg["batch_size"],
            input_nodes=("cell", val_mask),
            shuffle=False,
        )
        v2_metrics = evaluate_both(
            v2_model, intact_val_loader, intact_graph, cfg, device, v2_ckpt, log,
            sampler=None, val_sup=None,   # transductive only — see comment above
        )
        with open(v2_metrics_path, "w") as fh:
            json.dump(v2_metrics, fh, indent=2)
        del v2_model
        torch.cuda.empty_cache()

    all_results: list[dict] = [{
        "model":        "hgt_v2_published",
        "link_loss":    v2_metrics.get("link_loss", float("nan")),
        "clf_loss":     v2_metrics.get("clf_loss", float("nan")),
        "auroc":        float("nan"),   # no honest held-out number exists for this ckpt
        "auprc":        float("nan"),
        "auroc_transd": v2_metrics.get("auroc_transd", v2_metrics.get("auroc", float("nan"))),
        "cell_acc":     v2_metrics.get("cell_acc", float("nan")),
        "cell_f1":      v2_metrics.get("cell_f1", float("nan")),
    }]

    # ── Run each experiment ────────────────────────────────────────────────
    for name, model_fn, exp_graph, ckpt_path in experiments:
        log.info(f"\n{'='*60}")
        log.info(f"Experiment: {name}")
        log.info(f"{'='*60}")

        # Only no_mirna may legitimately lack link prediction. Anything else reporting
        # "disabled" here means AUROC will come back nan — which is a bug, not a result.
        _pos = get_mirna_gene_edges(exp_graph)
        if _pos is None:
            level = log.info if name == "ablation_no_mirna" else log.warning
            level(f"  Link prediction DISABLED for '{name}' (no miRNA→gene edges) — AUROC will be nan")
            exp_sampler, exp_train_sup, exp_val_sup = None, None, None
        else:
            log.info(f"  Link prediction active — {_pos.shape[1]:,} positive miRNA→gene edges")
            exp_sampler, exp_train_sup, exp_val_sup = sampler, train_sup, val_sup

        # Build loaders for this graph (may differ for ablations)
        if exp_graph is not graph:
            exp_loader_kwargs = dict(
                num_neighbors={et: tcfg["num_neighbors"] for et in exp_graph.edge_types},
                batch_size=tcfg["batch_size"],
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
            )
            # reuse same masks (cell nodes unchanged)
            exp_graph["cell"].train_mask = train_mask
            exp_graph["cell"].val_mask   = val_mask
            exp_train_loader = NeighborLoader(exp_graph, input_nodes=("cell", train_mask), shuffle=True,  **exp_loader_kwargs)
            exp_val_loader   = NeighborLoader(exp_graph, input_nodes=("cell", val_mask),   shuffle=False, **exp_loader_kwargs)
        else:
            exp_train_loader = train_loader
            exp_val_loader   = val_loader

        model = model_fn().to(device)

        # Warm up lazy linears BEFORE counting parameters or loading checkpoints.
        # LazyLinear(-1, ...) parameters are UninitializedParameter until the
        # first forward pass — calling numel() on them raises ValueError.
        with torch.no_grad():
            try:
                _dummy = next(iter(exp_train_loader)).to(device)
                model(x_dict=_dummy.x_dict, edge_index_dict=_dummy.edge_index_dict)
                del _dummy
                torch.cuda.empty_cache()
            except Exception as e:
                log.warning(f"  Warm-up skipped: {e}")

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"  Parameters: {n_params:,}")

        if name == "random":
            # Random baseline: no training, just evaluate
            eval_ckpt = ""
        elif args.skip_training and ckpt_path and os.path.exists(ckpt_path):
            log.info("  Skip-training mode: loading existing checkpoint")
            eval_ckpt = ckpt_path
        else:
            if ckpt_path:
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            eval_ckpt = ckpt_path or "/tmp/baseline_tmp.pt"
            train_model(
                model, exp_train_loader, exp_val_loader, exp_graph,
                cfg, device, eval_ckpt, args.epochs, log,
                sampler=exp_sampler, train_sup=exp_train_sup, val_sup=exp_val_sup,
            )

        # Always score the best checkpoint, both ways — not whatever weights training
        # happened to end on.
        metrics = evaluate_both(
            model, exp_val_loader, exp_graph, cfg, device, eval_ckpt, log,
            sampler=exp_sampler, val_sup=exp_val_sup,
        )

        all_results.append({
            "model":        name,
            "link_loss":    metrics.get("link_loss",    float("nan")),
            "clf_loss":     metrics.get("clf_loss",     float("nan")),
            "auroc":        metrics.get("auroc",        float("nan")),
            "auprc":        metrics.get("auprc",        float("nan")),
            "auroc_transd": metrics.get("auroc_transd", float("nan")),
            "cell_acc":     metrics.get("cell_acc",     float("nan")),
            "cell_f1":      metrics.get("cell_f1",      float("nan")),
        })

        torch.cuda.empty_cache()

    # ── Save comparison table ──────────────────────────────────────────────
    # auroc       = held-out edges, degree-matched negatives  ← the number to quote
    # auroc_transd= all edges scorable, uniform negatives     ← the published protocol
    # The two are not comparable and deliberately never share a cell. `val_loss` is gone
    # as a cross-model column: a model with no link head optimizes a strictly smaller
    # objective, so its total loss looked "best" while being the worst model. link_loss
    # and clf_loss are per-task and can be compared.
    cols = ["model", "link_loss", "clf_loss", "auroc", "auprc", "auroc_transd",
            "cell_acc", "cell_f1"]
    tsv_path = os.path.join(out_dir, "comparison_table.tsv")
    with open(tsv_path, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in all_results:
            fh.write(r["model"] + "\t" + "\t".join(f"{r[c]:.4f}" for c in cols[1:]) + "\n")

    log.info(f"\nComparison table saved to: {tsv_path}")
    log.info("  auroc = held-out edges + degree-matched negatives (honest)")
    log.info("  auroc_transd = all edges + uniform negatives (published protocol)")
    log.info("\n" + "\t".join(cols))
    for r in all_results:
        log.info(
            f"{r['model']:<20} link={r['link_loss']:.4f}  clf={r['clf_loss']:.4f}  "
            f"auroc={r['auroc']:.4f}  auprc={r['auprc']:.4f}  "
            f"auroc_transd={r['auroc_transd']:.4f}  "
            f"acc={r['cell_acc']:.4f}  f1={r['cell_f1']:.4f}"
        )
    if sampler.hard:
        log.info(f"\nDegree-matched negative fallback rate: {sampler.fallback_pct:.1f}%")

    # ── Also save as JSON for downstream plotting ──────────────────────────
    json_path = os.path.join(out_dir, "comparison_table.json")
    with open(json_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    log.info(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
