"""
eval_topology_baseline.py — Is 0.62 the model's achievement, or the task's ceiling?

The retrain under a real held-out edge split (job 5603) scored AUROC 0.6268 on unseen
miRNA→gene pairs, down from a published 0.9836. Before concluding anything about the
model, we need to know what a *model-free* scorer gets on the identical pairs.

The stakes: the graph carries no sequence information, and miRDB edges are derived from
seed-sequence complementarity. So for a pair the model has never seen, the only signal
available is topology — who else targets this gene, and what else does this miRNA target.
If a no-learning heuristic already reaches ~0.62, then the HGT has learned nothing beyond
trivial graph structure and 0.62 is the information ceiling of the task as posed. If the
HGT clearly beats the heuristics, the architecture is doing real work and it is worth
trying to lift it.

Heuristics (all computed on TRAINING edges only — message-passing + train-supervision;
val/test edges are never visible, exactly as for the model):

  gene_degree     score(m,g) = deg(g). The popularity prior. Should be ~0.5 against
                  degree-matched negatives by construction — it is the sanity check.
  pref_attach     score(m,g) = deg(m) * deg(g).
  common_neigh    Σ_{m'≠m} |N(m) ∩ N(m')| * A[m',g]. "How many miRNAs that target g
                  share targets with m, weighted by how much they overlap."
  adamic_adar     Same, but shared target genes are down-weighted by 1/log(deg(g')):
                  co-targeting a promiscuously-regulated gene is weak evidence,
                  co-targeting a selectively-regulated one is strong evidence. This is
                  the strongest of the four and the one to compare against.

Each is scored against both negative samplers, giving the same 2×2 the model gets.

Usage (DGX, CPU is enough but the GPU makes the matmuls instant):
  python training/eval_topology_baseline.py --config configs/config_v2_edgesplit.yaml
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import yaml
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.train import load_graph, split_graph
from training.splits import (
    REL_FWD,
    build_edge_split,
    assert_no_edge_leakage,
    degree_bins,
    gene_in_degree,
    pair_keys,
    sample_degree_matched_negatives,
)


def uniform_negatives(
    scored_pos: torch.Tensor,
    all_pos: torch.Tensor,
    n_mirna: int,
    n_gene: int,
    generator: torch.Generator,
    n_tries: int = 32,
) -> torch.Tensor:
    """One uniformly-random non-edge per positive, keeping the miRNA fixed so the only
    thing that differs from the degree-matched sampler is *which* gene is drawn."""
    k = scored_pos.shape[1]
    m = scored_pos[0].long()
    true_keys = pair_keys(all_pos, n_gene)

    cand = torch.randint(0, n_gene, (k, n_tries), generator=generator)
    valid = ~torch.isin(
        pair_keys(torch.stack([
            m.unsqueeze(1).expand_as(cand).reshape(-1), cand.reshape(-1)
        ]), n_gene),
        true_keys,
    ).view(k, n_tries)
    first = valid.float().argmax(dim=1)
    return torch.stack([m, cand.gather(1, first.unsqueeze(1)).squeeze(1)])


def build_scorers(A: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    A: (n_mirna, n_gene) binary adjacency over TRAINING edges only.
    Returns a dense score matrix per heuristic, so any (m, g) pair can be looked up.
    """
    A = A.float()
    deg_m = A.sum(dim=1)                       # (n_mirna,) targets per miRNA
    deg_g = A.sum(dim=0)                       # (n_gene,)  regulators per gene

    scores: dict[str, torch.Tensor] = {}

    # Popularity prior: identical for every miRNA.
    scores["gene_degree"] = deg_g.unsqueeze(0).expand(A.shape[0], -1).contiguous()

    scores["pref_attach"] = torch.outer(deg_m, deg_g)

    # miRNA-miRNA co-targeting overlap, then propagate to genes those miRNAs regulate.
    # Zero the diagonal: a miRNA must not vouch for itself, or the score degenerates
    # into deg(m) * A[m, g] and leaks the edge for any pair still present in A.
    S = A @ A.T                                # (n_mirna, n_mirna) shared-target counts
    S.fill_diagonal_(0)
    scores["common_neigh"] = S @ A

    # Adamic-Adar: discount shared targets that everything regulates.
    inv_log = 1.0 / torch.log(deg_g.clamp(min=2.0))
    A_w = A * inv_log.unsqueeze(0)
    S_aa = A @ A_w.T
    S_aa.fill_diagonal_(0)
    scores["adamic_adar"] = S_aa @ A

    return scores


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/config_v2_edgesplit.yaml")
    p.add_argument("--split", default="val", choices=["val", "test"],
                   help="Which held-out edges to score (model selection used val).")
    p.add_argument("--out", default="results/comparison/topology_baseline.json")
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

    # The identical split the model trained under — same seed, same config.
    split = build_edge_split(
        graph,
        val_ratio=tcfg["val_ratio"],
        test_ratio=tcfg["test_ratio"],
        seed=seed,
        disjoint_train_ratio=tcfg.get("disjoint_train_ratio", 0.3),
    )
    assert_no_edge_leakage(split)

    n_mirna = graph["miRNA"].num_nodes
    n_gene = graph["gene"].num_nodes

    # Everything the model was allowed to learn from: message-passing + supervision.
    train_edges = torch.cat(
        [split.mp_graph[REL_FWD].edge_index, split.train_sup], dim=1
    )
    log.info(
        f"Training edges visible to the heuristics: {train_edges.shape[1]:,} "
        f"(mp {split.mp_graph[REL_FWD].edge_index.shape[1]:,} + "
        f"train_sup {split.train_sup.shape[1]:,})"
    )

    A = torch.zeros((n_mirna, n_gene))
    A[train_edges[0].long(), train_edges[1].long()] = 1.0

    scored_pos = split.val_sup if args.split == "val" else split.test_sup
    log.info(f"Scoring {scored_pos.shape[1]:,} held-out {args.split} edges")

    # Same negatives as the model saw: degree bins from TRAINING edges only.
    deg = gene_in_degree(train_edges, n_gene)
    bins = degree_bins(deg)
    gen = torch.Generator().manual_seed(seed)

    hard_neg, n_fb = sample_degree_matched_negatives(
        scored_pos, split.all_pos, bins, n_gene, gen, torch.device("cpu")
    )
    unif_neg = uniform_negatives(scored_pos, split.all_pos, n_mirna, n_gene, gen)
    log.info(
        f"Negatives — degree-matched: {hard_neg.shape[1]:,} "
        f"(fallback {100.0 * n_fb / max(hard_neg.shape[1], 1):.1f}%), "
        f"uniform: {unif_neg.shape[1]:,}"
    )

    log.info("Building topology score matrices...")
    scorers = build_scorers(A)

    k = scored_pos.shape[1]
    results: dict[str, dict[str, dict[str, float]]] = {}

    for sampler_name, neg in (("uniform", unif_neg), ("degree_matched", hard_neg)):
        y = np.concatenate([np.ones(k), np.zeros(neg.shape[1])])
        m_idx = torch.cat([scored_pos[0], neg[0]]).long()
        g_idx = torch.cat([scored_pos[1], neg[1]]).long()

        results[sampler_name] = {}
        for name, M in scorers.items():
            s = M[m_idx, g_idx].numpy().astype(np.float64)
            results[sampler_name][name] = {
                "auroc": float(roc_auc_score(y, s)),
                "auprc": float(average_precision_score(y, s)),
                "n_pairs": int(len(y)),
            }

    # ── Report ────────────────────────────────────────────────────────────────
    order = ["gene_degree", "pref_attach", "common_neigh", "adamic_adar"]
    log.info("=" * 78)
    log.info("TOPOLOGY-ONLY BASELINES on held-out edges — no learning, no model")
    log.info("=" * 78)
    log.info(f"{'heuristic':<24}{'uniform neg':>16}{'degree-matched neg':>22}")
    log.info("-" * 78)
    for name in order:
        u = results["uniform"][name]["auroc"]
        d = results["degree_matched"][name]["auroc"]
        log.info(f"{name:<24}{u:>16.4f}{d:>22.4f}")
    log.info("-" * 78)
    best = max(results["degree_matched"][n]["auroc"] for n in order)
    log.info(f"Best model-free heuristic (degree-matched): {best:.4f}")
    log.info("HGT V2 retrained on the same split      :  0.6268  (job 5603, peak)")
    log.info("")
    log.info("Read: if the best heuristic is close to 0.6268, the HGT has learned")
    log.info("nothing beyond trivial graph structure, and ~0.62 is the ceiling of the")
    log.info("task as posed — the graph carries no sequence signal to generalize from.")
    log.info("=" * 78)

    summary = {
        "split": args.split,
        "n_held_out_edges": int(k),
        "train_edges_visible": int(train_edges.shape[1]),
        "hgt_v2_edgesplit_auroc_reference": 0.6268,
        "best_model_free_degree_matched": float(best),
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
