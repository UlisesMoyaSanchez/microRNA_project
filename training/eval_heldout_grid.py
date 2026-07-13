"""
eval_heldout_grid.py — Attribute the 0.9836 → 0.6268 collapse to its two causes.

The retrain changed two things at once: edges became genuinely held out, AND negatives
became degree-matched. So the drop is currently not attributable, and that is the first
thing a reviewer will ask. This fills in the 2x2:

                        | uniform negatives | degree-matched negatives
  ----------------------+-------------------+-------------------------
  edges seen in training|   0.9836 (pub.)   |   0.8828  (jobs 5594/5596)
  edges held out        |        ???        |   0.6268  (job 5603)

The bottom-left cell is the missing one. With it:
  - (top-left → top-right)    = cost of honest negatives, holding the split fixed
  - (top-left → bottom-left)  = cost of an honest split, holding negatives fixed
  - (top-left → bottom-right) = the total, which is what the paper reports

Both rows of the bottom half are scored here on the same checkpoint, the same held-out
edges, and the same encoder view (the message-passing graph, which never contained the
scored edges in either direction). Only the negative sampler differs.

Usage:
  python training/eval_heldout_grid.py --config configs/config_v2_edgesplit.yaml \
      --checkpoint checkpoints_v2_edgesplit/best_model.pt
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import yaml
import torch
from torch_geometric.loader import NeighborLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.hetero_gnn import miRNAGraphTransformer
from models.losses import CombinedLoss
from training.train import load_graph, split_graph
from training.evaluate import evaluate
from training.splits import (
    REL_FWD,
    LinkSampler,
    assert_no_edge_leakage,
    build_edge_split,
    gene_in_degree,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/config_v2_edgesplit.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", default="val", choices=["val", "test"])
    # Suffixed by split AND checkpoint: this script is run against two different models
    # (matched-negative and uniform-negative) on two splits, and a fixed path would have
    # each run silently clobber the last.
    p.add_argument("--out", default=None,
                   help="Default: results/comparison/heldout_grid_<ckptdir>_<split>.json")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    tcfg = cfg["training"]
    seed = cfg["project"]["seed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph, index_maps = load_graph(cfg["data"]["graphs_dir"])
    num_cell_types = len(index_maps["cell_type_labels"])
    train_mask, val_mask, test_mask = split_graph(
        graph, tcfg["val_ratio"], tcfg["test_ratio"], seed
    )
    graph["cell"].train_mask = train_mask
    graph["cell"].val_mask = val_mask
    graph["cell"].test_mask = test_mask

    split = build_edge_split(
        graph,
        val_ratio=tcfg["val_ratio"],
        test_ratio=tcfg["test_ratio"],
        seed=seed,
        disjoint_train_ratio=tcfg.get("disjoint_train_ratio", 0.3),
    )
    assert_no_edge_leakage(split)

    mp_graph = split.mp_graph
    sup = split.val_sup if args.split == "val" else split.test_sup
    eval_mask = val_mask if args.split == "val" else test_mask
    log.info(f"Scoring {sup.shape[1]:,} held-out {args.split} edges")

    train_edges = torch.cat([mp_graph[REL_FWD].edge_index, split.train_sup], dim=1)
    deg = gene_in_degree(train_edges, mp_graph["gene"].num_nodes)

    loader = NeighborLoader(
        mp_graph,
        num_neighbors={et: tcfg["num_neighbors"] for et in mp_graph.edge_types},
        batch_size=tcfg["batch_size"],
        input_nodes=("cell", eval_mask),
        shuffle=False,
    )

    model = miRNAGraphTransformer.from_config(
        cfg, mp_graph.metadata(), num_cell_types
    ).to(device)
    ckpt = args.checkpoint or os.path.join(tcfg["checkpoint_dir"], "best_model.pt")
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    log.info(f"Loaded {ckpt} (epoch {ck.get('epoch', '?')})")

    if args.out is None:
        tag = Path(ckpt).parent.name
        args.out = f"results/comparison/heldout_grid_{tag}_{args.split}.json"

    criterion = CombinedLoss(
        reconstruction_weight=tcfg["loss_reconstruction_weight"],
        classification_weight=tcfg["loss_classification_weight"],
        sparsity_weight=tcfg["loss_sparsity_weight"],
    ).to(device)

    results: dict[str, dict[str, float]] = {}
    for name, hard in (("uniform", False), ("degree_matched", True)):
        # Fresh sampler per condition, same seed: identical positives, identical encoder
        # view, only the negatives differ.
        sampler = LinkSampler(split.all_pos, deg, seed, hard=hard)
        m = evaluate(model, loader, criterion, device, mp_graph,
                     sampler=sampler, sup_edges=sup)
        results[name] = {
            "auroc": m.get("auroc", float("nan")),
            "auprc": m.get("auprc", float("nan")),
            "cell_acc": m.get("cell_acc", float("nan")),
        }
        if hard:
            log.info(f"  degree-matched fallback rate: {sampler.fallback_pct:.1f}%")

    u = results["uniform"]["auroc"]
    d = results["degree_matched"]["auroc"]

    log.info("=" * 76)
    log.info("HELD-OUT EDGES — attributing the collapse")
    log.info("=" * 76)
    log.info(f"{'':<26}{'uniform neg':>16}{'degree-matched neg':>22}")
    log.info("-" * 76)
    log.info(f"{'edges seen (published)':<26}{0.9836:>16.4f}{0.8828:>22.4f}")
    log.info(f"{'edges held out (honest)':<26}{u:>16.4f}{d:>22.4f}")
    log.info("-" * 76)
    log.info(f"Cost of honest negatives alone : {0.9836 - 0.8828:+.4f}")
    log.info(f"Cost of an honest split alone  : {0.9836 - u:+.4f}")
    log.info(f"Total, published -> honest     : {0.9836 - d:+.4f}")
    log.info("=" * 76)

    summary = {
        "checkpoint": str(ckpt),
        "epoch": ck.get("epoch"),
        "split": args.split,
        "n_held_out_edges": int(sup.shape[1]),
        "held_out": results,
        "reference_seen_edges": {"uniform": 0.9836, "degree_matched": 0.8828},
        "attribution": {
            "negatives_only": 0.9836 - 0.8828,
            "split_only": 0.9836 - u,
            "total": 0.9836 - d,
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
