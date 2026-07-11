"""
diagnose_leakage.py — Quantify how much of the reported link-prediction AUROC comes
from the model reading the answer off the input graph.

The (miRNA, regulates, gene) edges are both the *supervision targets* of the link
head and *inputs to the encoder*: build_heterograph.py adds the relation and its
reverse (gene, regulated_by, miRNA), and miRNAGraphTransformer.encode() runs HGT
message passing over the whole edge_index_dict. So when TargetPredictor scores the
pair (m, g), gene g's embedding has already aggregated miRNA m's embedding along the
very edge being predicted.

This script scores the SAME pairs with the SAME checkpoint under three encoder
conditions, and reports the metrics side by side:

  (a) intact      — edge_index_dict unchanged. Should reproduce the published number.
  (c) self-masked — only the scored positive pairs (and their reverses) are removed
                    from message passing. Other miRNA->gene edges stay. This isolates
                    the self-leak, and mirrors what a correct disjoint-supervision
                    split would give the encoder.
  (b) relation-off— the whole (miRNA, regulates, gene) relation and its reverse are
                    removed from message passing. Upper bound on the relation's total
                    contribution: this removes the shortcut AND all legitimate
                    miRNA-gene structural context, so a drop here is not attributable
                    to the shortcut alone. (c) is the number to reason from.

Sampling is done on the intact graph in all cases, so all three conditions see the
same batches, the same nodes and the same scored pairs — only the encoder's view of
the edges differs.

Caveat: (c) is still not a clean generalization estimate. This checkpoint was trained
with these pairs as supervision targets, so it may have memorized them in the weights.
(c) is an optimistic ceiling; a true number needs retraining under an edge-level split.

Usage:
  python training/diagnose_leakage.py --config configs/config_v2.yaml \
      --checkpoint checkpoints_v2/best_model.pt
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
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.hetero_gnn import miRNAGraphTransformer
from training.train import load_graph, split_graph

REL_FWD = ("miRNA", "regulates", "gene")
REL_REV = ("gene", "regulated_by", "miRNA")


# ── Edge bookkeeping ───────────────────────────────────────────────────────────

def all_local_pos_edges(
    pos_edge_global: torch.Tensor,
    batch,
    device: torch.device,
) -> torch.Tensor | None:
    """
    Every true (miRNA, gene) edge whose BOTH endpoints are in this batch, in local
    indices. Unlike train.map_global_edges_to_local this does not subsample — the
    full set is needed as the exclusion set for negative sampling, otherwise
    "negatives" can be genuine positives.
    """
    mirna_g = batch["miRNA"].n_id.to(device)
    gene_g = batch["gene"].n_id.to(device)

    n_mirna_g = max(int(pos_edge_global[0].max()), int(mirna_g.max())) + 1
    n_gene_g = max(int(pos_edge_global[1].max()), int(gene_g.max())) + 1

    mirna_g2l = torch.full((n_mirna_g,), -1, dtype=torch.long, device=device)
    mirna_g2l[mirna_g] = torch.arange(len(mirna_g), device=device)
    gene_g2l = torch.full((n_gene_g,), -1, dtype=torch.long, device=device)
    gene_g2l[gene_g] = torch.arange(len(gene_g), device=device)

    src = mirna_g2l[pos_edge_global[0].to(device)]
    dst = gene_g2l[pos_edge_global[1].to(device)]
    keep = (src >= 0) & (dst >= 0)
    if int(keep.sum()) == 0:
        return None
    return torch.stack([src[keep], dst[keep]])


def _pair_keys(edge_index: torch.Tensor, n_gene: int) -> torch.Tensor:
    """Encode (src, dst) pairs as single ints so they can be set-compared."""
    return edge_index[0].long() * n_gene + edge_index[1].long()


def mask_pairs_from_edge_index(
    edge_index: torch.Tensor,
    pairs_to_drop: torch.Tensor,
    n_gene: int,
    flip: bool = False,
) -> torch.Tensor:
    """
    Drop every column of edge_index whose (src, dst) pair appears in pairs_to_drop.
    If flip, edge_index is the reverse relation (gene -> miRNA) and pairs_to_drop is
    still given as (miRNA, gene), so compare against the flipped edge_index.
    """
    if edge_index.numel() == 0 or pairs_to_drop.numel() == 0:
        return edge_index
    probe = edge_index.flip(0) if flip else edge_index
    drop = _pair_keys(pairs_to_drop, n_gene)
    keys = _pair_keys(probe, n_gene)
    keep = ~torch.isin(keys, drop)
    return edge_index[:, keep]


def build_edge_index_dicts(
    batch,
    scored_pos: torch.Tensor,
    n_gene: int,
) -> dict[str, dict]:
    """The three encoder views of the same batch."""
    intact = dict(batch.edge_index_dict)

    # (c) remove only the pairs being scored, in both directions
    self_masked = dict(intact)
    if REL_FWD in self_masked:
        self_masked[REL_FWD] = mask_pairs_from_edge_index(
            self_masked[REL_FWD], scored_pos, n_gene, flip=False
        )
    if REL_REV in self_masked:
        self_masked[REL_REV] = mask_pairs_from_edge_index(
            self_masked[REL_REV], scored_pos, n_gene, flip=True
        )

    # (b) remove the relation entirely
    relation_off = {k: v for k, v in intact.items() if k not in (REL_FWD, REL_REV)}

    return {"a_intact": intact, "c_self_masked": self_masked, "b_relation_off": relation_off}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/config_v2.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--max_pairs", type=int, default=256,
                   help="Positive pairs scored per batch (matches evaluate.py)")
    p.add_argument("--out", default="results/comparison/leakage_diagnostic.json")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Graph + test split (identical to evaluate.py) ──────────────────────────
    graph, index_maps = load_graph(cfg["data"]["graphs_dir"])
    num_cell_types = len(index_maps["cell_type_labels"])
    tcfg = cfg["training"]

    _, _, test_mask = split_graph(graph, tcfg["val_ratio"], tcfg["test_ratio"], seed)
    graph["cell"].test_mask = test_mask
    log.info(f"Test cells: {int(test_mask.sum())} / {graph['cell'].num_nodes}")

    pos_edge_full = graph[REL_FWD].edge_index
    log.info(f"miRNA->gene edges in graph: {pos_edge_full.shape[1]:,}")

    test_loader = NeighborLoader(
        graph,
        num_neighbors={et: tcfg["num_neighbors"] for et in graph.edge_types},
        batch_size=tcfg["batch_size"],
        input_nodes=("cell", test_mask),
        shuffle=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = miRNAGraphTransformer.from_config(cfg, graph.metadata(), num_cell_types).to(device)
    ckpt = args.checkpoint or os.path.join(tcfg["checkpoint_dir"], "best_model.pt")
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    log.info(f"Loaded checkpoint {ckpt} (epoch {ck.get('epoch', '?')})")

    conditions = ["a_intact", "c_self_masked", "b_relation_off"]
    scores: dict[str, list[np.ndarray]] = {c: [] for c in conditions}
    cell_preds: dict[str, list[np.ndarray]] = {c: [] for c in conditions}
    labels_all: list[np.ndarray] = []
    cell_labels_all: list[np.ndarray] = []

    n_batches = 0
    n_skipped = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            all_pos = all_local_pos_edges(pos_edge_full, batch, device)
            if all_pos is None:
                n_skipped += 1
                continue

            n_mirna = batch["miRNA"].num_nodes
            n_gene = batch["gene"].num_nodes

            # Scored positives: subsample of the true edges in this batch.
            k = min(args.max_pairs, all_pos.shape[1])
            perm = torch.randperm(all_pos.shape[1], device=device)[:k]
            scored_pos = all_pos[:, perm]

            # Negatives excluded against ALL true edges in the batch, not just the
            # scored subsample — otherwise sampled negatives can be real positives.
            neg = negative_sampling(
                edge_index=all_pos,
                num_nodes=(n_mirna, n_gene),
                num_neg_samples=k,
                method="sparse",
            ).to(device)

            mirna_idx = torch.cat([scored_pos[0], neg[0]])
            gene_idx = torch.cat([scored_pos[1], neg[1]])
            y = torch.cat([torch.ones(k), torch.zeros(neg.shape[1])]).numpy()
            labels_all.append(y)
            cell_labels_all.append(batch["cell"].y.cpu().numpy())

            eid_dicts = build_edge_index_dicts(batch, scored_pos, n_gene)

            for cond in conditions:
                out = model(
                    x_dict=batch.x_dict,
                    edge_index_dict=eid_dicts[cond],
                    mirna_idx=mirna_idx,
                    gene_idx=gene_idx,
                )
                scores[cond].append(torch.sigmoid(out["edge_logits"]).cpu().numpy())
                cell_preds[cond].append(out["cell_logits"].argmax(dim=-1).cpu().numpy())

            n_batches += 1

    if n_batches == 0:
        log.error("No usable batches — every batch lacked miRNA/gene overlap.")
        sys.exit(1)

    log.info(f"Scored {n_batches} batches ({n_skipped} skipped, no miRNA/gene overlap)")

    # ── Metrics ───────────────────────────────────────────────────────────────
    y = np.concatenate(labels_all)
    cy = np.concatenate(cell_labels_all)
    cvalid = cy >= 0

    results: dict[str, dict[str, float]] = {}
    for cond in conditions:
        s = np.concatenate(scores[cond])
        cp = np.concatenate(cell_preds[cond])
        results[cond] = {
            "auroc": float(roc_auc_score(y, s)),
            "auprc": float(average_precision_score(y, s)),
            "cell_acc": float(accuracy_score(cy[cvalid], cp[cvalid])),
            "cell_f1": float(f1_score(cy[cvalid], cp[cvalid], average="macro", zero_division=0)),
        }

    a, c, b = results["a_intact"], results["c_self_masked"], results["b_relation_off"]
    summary = {
        "n_pairs_scored": int(len(y)),
        "n_positives": int(y.sum()),
        "checkpoint": str(ckpt),
        "conditions": results,
        "self_leak_auroc_drop": a["auroc"] - c["auroc"],
        "relation_off_auroc_drop": a["auroc"] - b["auroc"],
    }

    label = {
        "a_intact": "(a) intact          ",
        "c_self_masked": "(c) self-masked     ",
        "b_relation_off": "(b) relation-off    ",
    }
    log.info("=" * 72)
    log.info("LEAKAGE DIAGNOSTIC — same checkpoint, same pairs, different encoder view")
    log.info("=" * 72)
    log.info(f"{'condition':<22}{'AUROC':>9}{'AUPRC':>9}{'cell_acc':>10}{'cell_f1':>9}")
    for cond in conditions:
        r = results[cond]
        log.info(
            f"{label[cond]:<22}{r['auroc']:>9.4f}{r['auprc']:>9.4f}"
            f"{r['cell_acc']:>10.4f}{r['cell_f1']:>9.4f}"
        )
    log.info("-" * 72)
    log.info(f"Self-leak AUROC drop      (a) - (c) : {summary['self_leak_auroc_drop']:+.4f}")
    log.info(f"Relation-off AUROC drop   (a) - (b) : {summary['relation_off_auroc_drop']:+.4f}")
    log.info("=" * 72)
    log.info("(c) is the number to reason from. It still assumes no weight-level")
    log.info("memorization of these pairs — a clean estimate needs retraining under")
    log.info("an edge-level split.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
