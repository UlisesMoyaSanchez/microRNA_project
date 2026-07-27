"""
run_baselines.py — Train and evaluate all baseline & ablation models.

Runs the following experiments sequentially on a single GPU, saving results to
results/comparison/comparison_table_<checkpoint-dir-stem>.tsv:

  hgt_v2     — miRNAGraphTransformer, this config's protocol and negative sampler
  random     — RandomBaseline (no training, floor reference)
  mlp        — MLPBaseline    (no graph structure)
  homo_gcn   — HomoGCNBaseline (homogeneous GCN, no type semantics)
  no_mirna   — miRNAGraphTransformer (V2) without miRNA→gene edges (ablation A)
  no_coexpr  — miRNAGraphTransformer (V2) without gene co-expression edges (ablation B)

Each run covers ONE cell of the protocol × negative-sampler grid, selected by
`training.edge_split` and `training.hard_negatives` in the config, and every model is
scored under BOTH samplers on that cell's supervision edges. Run it once per cell and
compare across the resulting tables: if the inflation shows up for every architecture,
the finding is about the evaluation protocol rather than about any one model.

Usage:
  python training/run_baselines.py --config configs/config_v3fixed_baselines_edgesplit.yaml

Optionally, `evaluation.reference_checkpoint` names an already-trained transductive
checkpoint to include as a reference row; without it that row is simply absent.
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

    # Select the checkpoint on the metric the model is *for* — identical to train.py:389-394,
    # and for the same reason. Selecting on val_loss picked epoch 1 in job 5603: the link head
    # overfits from the very first epoch, so val_loss rises monotonically while val_auroc is
    # still climbing. That saved a "best model" scoring at chance (0.5324) for a model that
    # reaches 0.6268. train.py was fixed; this path kept the broken criterion, so every
    # baseline row was being selected by the one rule known to produce a chance-level
    # checkpoint. A model with no link head (sampler is None) has no auroc to monitor and
    # correctly falls back to loss.
    monitor  = "auroc" if sampler is not None else "loss"
    maximize = monitor == "auroc"
    best_val = -float("inf") if maximize else float("inf")
    log.info(f"  Model selection on val_{monitor} ({'max' if maximize else 'min'})")

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

        current = val_metrics.get(monitor, float("nan"))
        improved = current > best_val if maximize else current < best_val
        if improved:
            best_val = current
            pat_count = 0
            best_metrics = val_metrics
            torch.save(model.state_dict(), checkpoint_path)
        else:
            pat_count += 1
            if pat_count >= patience:
                log.info(f"  Early stopping at epoch {epoch}")
                break

    log.info(f"  Best val_{monitor}={best_val:.4f}")
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
    deg: torch.Tensor | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    """
    Warm up lazy linears, load the checkpoint, and score it on the same cells under BOTH
    negative samplers, plus the no-holdout reference:

      auroc / auprc                — the configured supervision edges, degree-matched negatives
      auroc_uniform / auprc_uniform — the SAME edges, uniform negatives
      auroc_transd / auprc_transd  — every miRNA→gene edge is fair game, default negatives

    The first two differ only in the negative sampler: identical positives, identical encoder
    view, fresh sampler per condition at the same seed. That is what makes the pair an
    attribution rather than two unrelated numbers — the same requirement eval_heldout_grid.py
    enforces, and the "mismatch trap" the audit documents comes from violating it.

    All come from the *best* checkpoint, not whatever was last in memory after training.
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

    # Same positives, same encoder view, only the negative sampler differs. A fresh
    # LinkSampler at the same seed rather than mutating `sampler` in place — the caller
    # reuses it across experiments, and flipping .hard under it would silently change
    # every later row. deg/seed come from the caller because LinkSampler keeps neither
    # (it stores the derived bins and generator, not the inputs).
    if sampler is not None and deg is not None and seed is not None:
        alt = LinkSampler(
            all_pos_global=sampler.all_pos,
            deg=deg,
            seed=seed,
            hard=not sampler.hard,
        )
        alt_metrics = evaluate(
            model, val_loader, criterion, device, graph,
            sampler=alt, sup_edges=val_sup,
        )
        # Name the columns by what the negatives ARE, not by which one happened to be
        # configured — a column called "uniform" must mean uniform in every row of the table.
        hard_m, unif_m = (metrics, alt_metrics) if sampler.hard else (alt_metrics, metrics)
        metrics = dict(hard_m)
        metrics["auroc_matched"] = hard_m.get("auroc", float("nan"))
        metrics["auprc_matched"] = hard_m.get("auprc", float("nan"))
        metrics["auroc_uniform"] = unif_m.get("auroc", float("nan"))
        metrics["auprc_uniform"] = unif_m.get("auprc", float("nan"))

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
    #
    # edge_split=false selects the transductive protocol instead — the same flag, with the
    # same meaning and the same deliberate leak, as train.py:277-336. It exists here so
    # every architecture can be run through BOTH protocols: showing the inflation only for
    # the HGT shows that *our* model was evaluated badly, which is not the claim. The
    # branches below mirror train.py's; keep them in step if either changes.
    do_edge_split = tcfg.get("edge_split", True)

    # The pre-split graph, kept only to reproduce a reference number under the protocol
    # that produced it. Nothing else may be evaluated on it.
    intact_graph = graph

    if do_edge_split:
        edge_split = build_edge_split(
            graph,
            val_ratio=tcfg["val_ratio"],
            test_ratio=tcfg["test_ratio"],
            seed=seed,
            disjoint_train_ratio=tcfg.get("disjoint_train_ratio", 0.3),
        )
        assert_no_edge_leakage(edge_split)

        graph     = edge_split.mp_graph
        train_sup = edge_split.train_sup
        val_sup   = edge_split.val_sup

        # Bin genes by in-degree over TRAINING edges only — binning on the full edge
        # set would leak held-out structure into the choice of negatives.
        train_edges = torch.cat([graph[REL_FWD].edge_index, train_sup], dim=1)
        deg = gene_in_degree(train_edges, graph["gene"].num_nodes)
        all_pos_global = edge_split.all_pos
    else:
        # TRANSDUCTIVE (leak on purpose): no holdout. Every positive edge is both a
        # message-passing edge and a supervision target, degrees are binned on all of
        # them, and only the split differs from the branch above.
        _all_pos = get_mirna_gene_edges(graph)
        if _all_pos is None:
            raise SystemExit(
                "edge_split=false, but this graph has no miRNA→gene edges to supervise on. "
                "The transductive protocol has nothing to measure here — check "
                f"data.graphs_dir ({cfg['data']['graphs_dir']})."
            )
        all_pos_global = _all_pos.clone()
        train_sup = all_pos_global
        val_sup   = all_pos_global
        deg = gene_in_degree(all_pos_global, graph["gene"].num_nodes)
        log.warning(
            "TRANSDUCTIVE protocol (edge_split=false): miRNA→gene edges are NOT held out. "
            "Every row in this table is a reconstruction score on memorized edges — the "
            "leak the paper is about, reintroduced deliberately behind this flag."
        )

    sampler = LinkSampler(
        all_pos_global=all_pos_global,
        deg=deg,
        seed=seed,
        hard=tcfg.get("hard_negatives", True),
    )
    log.info(
        f"Protocol: {'held-out edges' if do_edge_split else 'transductive (seen edges)'} | "
        f"negatives: {'degree-matched (hard)' if sampler.hard else 'uniform'}"
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
    #
    # Per-architecture checkpoints live UNDER this config's checkpoint_dir rather than in
    # fixed `checkpoints_baseline_*` / `checkpoints_ablation_*` trees at the project root.
    # This script is run once per protocol × sampler cell, and shared paths meant all four
    # cells wrote the same files — four different models behind one filename, with whichever
    # ran last winning any subsequent `--skip-training`. The metrics were never wrong (each
    # cell retrains before scoring), but the artifacts could not be told apart, which is the
    # same "you cannot verify which computation produced this" failure the audit is about.
    ckpt_root = Path(tcfg["checkpoint_dir"])

    experiments = [
        # (label, model_class, graph_to_use, ckpt_path)
        (
            # The headline row: the same V2 architecture under this config's protocol and
            # negative sampler. Writes to the config's own checkpoint_dir, so it is the
            # direct counterpart of the corresponding train.py run.
            "hgt_v2",
            lambda: miRNAGraphTransformer.from_config(cfg, metadata, num_cell_types),
            graph,
            str(ckpt_root / "best_model.pt"),
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
            str(ckpt_root / "baseline_mlp" / "best_model.pt"),
        ),
        (
            "homo_gcn",
            lambda: HomoGCNBaseline.from_config(cfg, metadata, num_cell_types),
            graph,
            str(ckpt_root / "baseline_gcn" / "best_model.pt"),
        ),
        (
            "ablation_no_mirna",
            lambda: miRNAGraphTransformer.from_config(
                cfg,
                drop_edge_types(graph, ["miRNA,regulates,gene", "gene,regulated_by,miRNA"]).metadata(),
                num_cell_types,
            ),
            drop_edge_types(graph, ["miRNA,regulates,gene", "gene,regulated_by,miRNA"]),
            str(ckpt_root / "ablation_no_mirna" / "best_model.pt"),
        ),
        (
            "ablation_no_coexpr",
            lambda: miRNAGraphTransformer.from_config(
                cfg,
                drop_edge_types(graph, ["gene,coexpressed_with,gene"]).metadata(),
                num_cell_types,
            ),
            drop_edge_types(graph, ["gene,coexpressed_with,gene"]),
            str(ckpt_root / "ablation_no_coexpr" / "best_model.pt"),
        ),
    ]

    # ── Pre-trained transductive reference row (optional) ──────────────────
    # A checkpoint trained with every miRNA→gene edge as a supervision target, so the
    # "held-out" edges of the split above are not held out *for it*. Scoring it on them
    # would report a memorized number in the honest column; its held-out cells are left
    # nan on purpose and it is evaluated on the intact graph.
    #
    # The checkpoint is declared in the config rather than hardcoded. It used to be
    # `checkpoints_v2/best_model.pt` — the PRE-FIX graph's — which any run on another graph
    # would have silently pulled into its table, and the cache filename carried no graph
    # identity either, so a v3fixed run would have *loaded* the pre-fix numbers rather than
    # recomputing them. Same bug class as the reference_seen_edges constants
    # (eval_heldout_grid.py) and the checkpoint stems (aggregate_seeds.py): a default
    # standing in for a computation that never ran. A config that declares no reference
    # checkpoint gets no row, never a borrowed one.
    #
    # Note this row is largely superseded by `training.edge_split: false`, which trains the
    # whole grid transductively on THIS graph. It is kept for configs that want to cite an
    # existing checkpoint without retraining.
    ref_ckpt = (cfg.get("evaluation") or {}).get("reference_checkpoint")
    all_results: list[dict] = []

    if not ref_ckpt:
        log.info(
            "No evaluation.reference_checkpoint declared for this config — the pre-trained "
            "transductive reference row is omitted rather than borrowed from another graph."
        )
    else:
        ref_ckpt = str(ref_ckpt)
        # Cache keyed on the checkpoint's own directory, so two graphs cannot share a file.
        ref_tag = Path(ref_ckpt).parent.name
        ref_metrics_path = os.path.join(out_dir, f"reference_metrics_{ref_tag}.json")
        if os.path.exists(ref_metrics_path):
            with open(ref_metrics_path) as fh:
                ref_metrics = json.load(fh)
            log.info(f"Loaded reference metrics from {ref_metrics_path}")
        elif not os.path.exists(ref_ckpt):
            log.warning(
                f"evaluation.reference_checkpoint '{ref_ckpt}' does not exist — omitting the "
                "reference row rather than reporting an untrained model."
            )
            ref_metrics = None
        else:
            log.info(f"Evaluating reference checkpoint on the INTACT graph: {ref_ckpt}")
            ref_model = miRNAGraphTransformer.from_config(
                cfg, intact_graph.metadata(), num_cell_types
            ).to(device)
            intact_val_loader = NeighborLoader(
                intact_graph,
                num_neighbors={et: tcfg["num_neighbors"] for et in intact_graph.edge_types},
                batch_size=tcfg["batch_size"],
                input_nodes=("cell", val_mask),
                shuffle=False,
            )
            ref_metrics = evaluate_both(
                ref_model, intact_val_loader, intact_graph, cfg, device, ref_ckpt, log,
                sampler=None, val_sup=None,   # transductive only — see comment above
            )
            with open(ref_metrics_path, "w") as fh:
                json.dump(ref_metrics, fh, indent=2)
            del ref_model
            torch.cuda.empty_cache()

        if ref_metrics:
            all_results.append({
                "model":         f"reference_transductive ({ref_tag})",
                "protocol":      "transductive_seen",
                "trained_with":  "uniform",
                "link_loss":     ref_metrics.get("link_loss", float("nan")),
                "clf_loss":      ref_metrics.get("clf_loss", float("nan")),
                "auroc":         float("nan"),   # no honest held-out number exists for this ckpt
                "auprc":         float("nan"),
                "auroc_matched": float("nan"),
                "auprc_matched": float("nan"),
                "auroc_uniform": float("nan"),
                "auprc_uniform": float("nan"),
                "auroc_transd":  ref_metrics.get("auroc_transd",
                                                 ref_metrics.get("auroc", float("nan"))),
                "cell_acc":      ref_metrics.get("cell_acc", float("nan")),
                "cell_f1":       ref_metrics.get("cell_f1", float("nan")),
            })

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
            sampler=exp_sampler, val_sup=exp_val_sup, deg=deg, seed=seed,
        )

        all_results.append({
            "model":         name,
            "protocol":      "held_out" if do_edge_split else "transductive_seen",
            "trained_with":  "degree_matched" if tcfg.get("hard_negatives", True) else "uniform",
            "link_loss":     metrics.get("link_loss",     float("nan")),
            "clf_loss":      metrics.get("clf_loss",      float("nan")),
            "auroc":         metrics.get("auroc",         float("nan")),
            "auprc":         metrics.get("auprc",         float("nan")),
            "auroc_matched": metrics.get("auroc_matched", float("nan")),
            "auprc_matched": metrics.get("auprc_matched", float("nan")),
            "auroc_uniform": metrics.get("auroc_uniform", float("nan")),
            "auprc_uniform": metrics.get("auprc_uniform", float("nan")),
            "auroc_transd":  metrics.get("auroc_transd",  float("nan")),
            "cell_acc":      metrics.get("cell_acc",      float("nan")),
            "cell_f1":       metrics.get("cell_f1",       float("nan")),
        })

        torch.cuda.empty_cache()

    # ── Save comparison table ──────────────────────────────────────────────
    # auroc_matched / auroc_uniform = the SAME supervision edges under the two negative
    #   samplers. Under `edge_split: true` those edges are held out; under `false` they are
    #   the seen edges. `protocol` and `trained_with` say which, per row, so a cell can
    #   never be read as the wrong one.
    # auroc_transd = all edges scorable, default negatives — the original protocol.
    # These are not interchangeable and deliberately never share a cell. `val_loss` is gone
    # as a cross-model column: a model with no link head optimizes a strictly smaller
    # objective, so its total loss looked "best" while being the worst model. link_loss
    # and clf_loss are per-task and can be compared.
    #
    # The filename carries the checkpoint-dir stem: this script is run once per protocol ×
    # sampler cell, and a fixed `comparison_table.tsv` meant four runs silently overwriting
    # each other until only the last survived.
    run_tag = Path(tcfg["checkpoint_dir"]).name
    cols = ["model", "protocol", "trained_with", "link_loss", "clf_loss",
            "auroc_matched", "auprc_matched", "auroc_uniform", "auprc_uniform",
            "auroc_transd", "cell_acc", "cell_f1"]
    numeric = set(cols) - {"model", "protocol", "trained_with"}
    tsv_path = os.path.join(out_dir, f"comparison_table_{run_tag}.tsv")
    with open(tsv_path, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in all_results:
            fh.write("\t".join(
                f"{r[c]:.4f}" if c in numeric else str(r[c]) for c in cols
            ) + "\n")

    log.info(f"\nComparison table saved to: {tsv_path}")
    log.info(f"  protocol = {'held-out edges' if do_edge_split else 'transductive (seen edges)'}, "
             f"trained with {'degree-matched' if tcfg.get('hard_negatives', True) else 'uniform'} negatives")
    log.info("  auroc_matched / auroc_uniform = same edges, the two negative samplers")
    log.info("  auroc_transd = all edges + default negatives (original protocol)")
    for r in all_results:
        log.info(
            f"{r['model']:<34} link={r['link_loss']:.4f}  clf={r['clf_loss']:.4f}  "
            f"auroc_matched={r['auroc_matched']:.4f}  auroc_uniform={r['auroc_uniform']:.4f}  "
            f"auroc_transd={r['auroc_transd']:.4f}  "
            f"acc={r['cell_acc']:.4f}  f1={r['cell_f1']:.4f}"
        )
    if sampler.hard:
        log.info(f"\nDegree-matched negative fallback rate: {sampler.fallback_pct:.1f}%")

    # ── Also save as JSON for downstream plotting ──────────────────────────
    json_path = os.path.join(out_dir, f"comparison_table_{run_tag}.json")
    with open(json_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    log.info(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
