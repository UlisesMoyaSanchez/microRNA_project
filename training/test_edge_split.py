"""
test_edge_split.py — Gating check for the edge-level split.

Everything downstream of the split (the retrained checkpoint, the comparison table, the
circuit rankings) is only worth as much as this test. If a held-out edge is still visible
to the encoder — in either direction — the "held-out" AUROC is just the old leaked number
wearing a different name, and it will look fine while being wrong. So this runs first, and
the retrain does not get submitted until it passes.

CPU only, no GPU needed:
  python training/test_edge_split.py --config configs/config_v2_edgesplit.yaml
"""

from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path

import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.train import load_graph, split_graph
from training.splits import (
    LinkSampler,
    REL_FWD,
    REL_REV,
    assert_no_edge_leakage,
    build_edge_split,
    gene_in_degree,
    pair_keys,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/config_v2_edgesplit.yaml")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    tcfg = cfg["training"]
    seed = cfg["project"]["seed"]

    graph, _ = load_graph(cfg["data"]["graphs_dir"])
    train_mask, val_mask, test_mask = split_graph(
        graph, tcfg["val_ratio"], tcfg["test_ratio"], seed
    )
    graph["cell"].train_mask = train_mask
    graph["cell"].val_mask = val_mask
    graph["cell"].test_mask = test_mask

    n_pos_before = graph[REL_FWD].edge_index.shape[1]

    log.info("=" * 72)
    log.info("EDGE SPLIT INTEGRITY")
    log.info("=" * 72)

    split = build_edge_split(
        graph,
        val_ratio=tcfg["val_ratio"],
        test_ratio=tcfg["test_ratio"],
        seed=seed,
        disjoint_train_ratio=tcfg.get("disjoint_train_ratio", 0.3),
    )
    assert_no_edge_leakage(split)

    # ── Conservation: every positive lands in exactly one place ────────────────
    n_mp    = split.mp_graph[REL_FWD].edge_index.shape[1]
    n_train = split.train_sup.shape[1]
    n_val   = split.val_sup.shape[1]
    n_test  = split.test_sup.shape[1]
    total   = n_mp + n_train + n_val + n_test
    assert split.all_pos.shape[1] == n_pos_before, "all_pos was not the original edge set"
    assert total == n_pos_before, (
        f"edges lost or duplicated: mp {n_mp} + train {n_train} + val {n_val} + "
        f"test {n_test} = {total}, expected {n_pos_before}"
    )
    log.info(f"ok    conservation: {n_mp} + {n_train} + {n_val} + {n_test} = {n_pos_before}")

    # The point of the exercise: the encoder must have lost the val/test edges.
    assert n_mp < n_pos_before, "message-passing graph still holds every edge"

    # ── Determinism: same seed, same split ────────────────────────────────────
    graph2, _ = load_graph(cfg["data"]["graphs_dir"])
    split2 = build_edge_split(
        graph2,
        val_ratio=tcfg["val_ratio"],
        test_ratio=tcfg["test_ratio"],
        seed=seed,
        disjoint_train_ratio=tcfg.get("disjoint_train_ratio", 0.3),
    )
    assert torch.equal(split.test_sup, split2.test_sup), (
        "split is not reproducible from the seed — every DDP rank would hold out a "
        "different edge set, and evaluation could not be reproduced"
    )
    log.info("ok    determinism: same seed reproduces the same held-out edges")

    # ── Negative sampler: negatives must never be real edges ──────────────────
    n_gene = graph["gene"].num_nodes
    train_edges = torch.cat([split.mp_graph[REL_FWD].edge_index, split.train_sup], dim=1)
    deg = gene_in_degree(train_edges, n_gene)
    sampler = LinkSampler(split.all_pos, deg, seed, hard=tcfg.get("hard_negatives", True))

    from training.splits import sample_degree_matched_negatives

    # Score the full global edge set as if it were one batch (local == global here).
    scored = split.test_sup[:, :2000]
    neg, n_fb = sample_degree_matched_negatives(
        scored, split.all_pos, sampler.gene_bins_global, n_gene,
        sampler.generator, torch.device("cpu"),
    )
    assert neg.shape[1] == scored.shape[1], "sampler dropped pairs"

    true_keys = pair_keys(split.all_pos, n_gene)
    neg_keys  = pair_keys(neg, n_gene)
    n_false_neg = int(torch.isin(neg_keys, true_keys).sum())
    assert n_false_neg == 0, (
        f"{n_false_neg} sampled 'negatives' are real miRNA→gene edges — the exclusion "
        "set is wrong, and the model is being trained to call true edges false"
    )
    log.info(f"ok    negatives: 0/{neg.shape[1]} sampled negatives are real edges")

    assert torch.equal(neg[0], scored[0]), "degree-matched negatives changed the miRNA"

    fb_pct = 100.0 * n_fb / neg.shape[1]
    log.info(f"ok    degree match: fallback rate {fb_pct:.1f}% (job 5595 saw 0.0%)")
    if fb_pct > 5.0:
        log.warning(
            f"fallback rate {fb_pct:.1f}% is high — that many negatives are NOT "
            "degree-matched, so the metric drifts back toward the uniform-negative one."
        )

    log.info("=" * 72)
    log.info("ALL CHECKS PASSED — the split is safe to train on.")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
