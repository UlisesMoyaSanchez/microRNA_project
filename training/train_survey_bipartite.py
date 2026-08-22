"""
train_survey_bipartite.py — Train this project's own model on a surveyed paper's graph,
across the same 2x2 protocol grid the model-free scorers were measured on.

The model-free grid (Table 6) shows the *floor* moves by +0.32 to +0.43 between the
conventional and corrected protocols on other papers' graphs. It cannot show whether a
*trained* model moves the same way, because we do not have the surveyed papers' models.
This trains ours on their graph instead. It is explicitly not a reproduction of anyone's
architecture -- the claim it supports is about the protocol, which is architecture-
independent, and MEAHNE and DiGAMN are chosen as the two ends of the observed gap range
(+0.033 and -0.092) so the answer cannot be an artifact of one end.

Why a separate trainer instead of training/train.py: train.py seeds its mini-batches from
cell nodes (`NeighborLoader(graph, input_nodes=("cell", train_mask))`), and these graphs
have no cells. Rather than fork that path, this script trains **full-batch** -- entirely
practical at 5k-18k edges -- and reuses everything that carries methodological weight
unchanged: the model class (`models.hetero_gnn.miRNAGraphTransformer`), the edge split
and its leakage assertion (`training.splits.build_edge_split` / `assert_no_edge_leakage`),
and both negative samplers. Only the batching differs, which is a scale concession, not a
methodological one.

The edge regime controls **only what the encoder may see**, exactly as it does for the
model-free scorers in eval_hmdd_survey_topology_baseline.py:
  held_out  the encoder message-passes over training edges only.
  seen      the encoder message-passes over every edge, including the scored ones.
Supervision, the split, and the scored test pairs are identical either way, so the
difference between the two rows is attributable to the protocol alone. This is a tighter
control than train.py's `edge_split: false` branch, which also moves supervision onto the
full edge set; here supervision is held fixed on purpose.

Usage:
  python training/train_survey_bipartite.py --config configs/config_survey_meahne_heldout.yaml
  python training/train_survey_bipartite.py --config configs/config_survey_digamn_seen.yaml
"""

from __future__ import annotations

import os
import sys
import json
import copy
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

import yaml
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.hetero_gnn import miRNAGraphTransformer
from training.eval_topology_baseline import uniform_negatives
from training.splits import (
    REL_FWD, REL_REV, build_edge_split, assert_no_edge_leakage,
    degree_bins, gene_in_degree, sample_degree_matched_negatives,
)


def load_graph(graphs_dir: str) -> "HeteroData":
    """Loaded here rather than via training.train.load_graph, which omits
    weights_only=False and so fails on torch >= 2.6, and which also expects an
    index_maps.pkl this trainer has no use for."""
    return torch.load(os.path.join(graphs_dir, "hetero_graph.pt"), weights_only=False)


def edge_dict(edge_index: torch.Tensor) -> dict:
    """Encoder view: the forward and its reverse relation, both from the same edges."""
    return {REL_FWD: edge_index, REL_REV: edge_index.flip(0)}


def draw_negatives(
    pos: torch.Tensor, all_pos: torch.Tensor, bins: torch.Tensor,
    n_mirna: int, n_gene: int, hard: bool, gen: torch.Generator,
) -> torch.Tensor:
    if hard:
        neg, _ = sample_degree_matched_negatives(
            pos, all_pos, bins, n_gene, gen, torch.device("cpu")
        )
        return neg
    return uniform_negatives(pos, all_pos, n_mirna, n_gene, gen)


@torch.no_grad()
def score(model, x_dict, ei_dict, pairs: torch.Tensor, device) -> np.ndarray:
    model.eval()
    out = model(x_dict, ei_dict,
                mirna_idx=pairs[0].to(device), gene_idx=pairs[1].to(device))
    return out["edge_logits"].float().cpu().numpy()


def evaluate(model, x_dict, ei_dict, pos, neg, device) -> dict:
    s = np.concatenate([score(model, x_dict, ei_dict, pos, device),
                        score(model, x_dict, ei_dict, neg, device)])
    y = np.concatenate([np.ones(pos.shape[1]), np.zeros(neg.shape[1])])
    return {"auroc": float(roc_auc_score(y, s)),
            "auprc": float(average_precision_score(y, s)),
            "n_pairs": int(len(y))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--negatives", default=None, choices=["uniform", "degree_matched"],
                   help="Override the config's training negatives. Needed because the "
                        "protocol grid follows Figure 4's train/eval-MATCHED convention: "
                        "a model trained on degree-matched negatives and scored against "
                        "uniform ones measures a train/eval distribution mismatch, not "
                        "protocol difficulty (the same caveat Table 2 flags with a "
                        "dagger). Each grid cell must be filled from the arm trained on "
                        "the negatives it is scored against.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override the config's seed, so one config covers all seeds. "
                        "The split is seeded from this too, so each seed is a different "
                        "split AND a different init -- the same convention as the "
                        "multi-seed runs on our own graph.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    tcfg, mcfg = cfg["training"], cfg["model"]
    paper = cfg["project"]["name"]
    seed = args.seed if args.seed is not None else cfg["project"]["seed"]
    regime = tcfg["edge_regime"]
    assert regime in ("held_out", "seen"), regime

    if args.out is None:
        neg_tag = ("" if args.negatives is None
                   else ("_trainuniform" if args.negatives == "uniform"
                         else "_traindm"))
        args.out = (f"results/comparison/survey_trained_grid_{paper}_{regime}"
                    f"{neg_tag}_s{seed}.json")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph = load_graph(cfg["data"]["graphs_dir"])
    n_mirna, n_gene = graph["miRNA"].num_nodes, graph["gene"].num_nodes
    log.info("=" * 78)
    log.info(f"{paper} — trained bipartite model, edge regime {regime}, seed {seed}")
    log.info(f"{n_mirna} miRNA x {n_gene} diseases (stored as 'gene'), "
             f"{graph[REL_FWD].edge_index.shape[1]:,} associations, device {device}")
    log.info("=" * 78)

    # One split, built identically in both regimes so the scored pairs match exactly.
    split = build_edge_split(
        graph, tcfg["val_ratio"], tcfg["test_ratio"], seed,
        disjoint_train_ratio=tcfg.get("disjoint_train_ratio", 0.3),
    )
    assert_no_edge_leakage(split)
    train_sup, val_sup, test_sup = split.train_sup, split.val_sup, split.test_sup
    mp_edges = split.mp_graph[REL_FWD].edge_index
    train_edges = torch.cat([mp_edges, train_sup], dim=1)

    # The ONLY thing the regime changes.
    encoder_edges = train_edges if regime == "held_out" else split.all_pos
    log.info(f"Encoder sees {encoder_edges.shape[1]:,} edges "
             f"({'training only' if regime == 'held_out' else 'ALL, incl. the scored ones'}); "
             f"supervision {train_sup.shape[1]:,}, val {val_sup.shape[1]:,}, "
             f"test {test_sup.shape[1]:,}")

    # Degree bins always from training edges: binning on the full set would leak
    # held-out structure into the choice of negatives, in either regime.
    bins = degree_bins(gene_in_degree(train_edges, n_gene))
    hard = (tcfg.get("hard_negatives", True) if args.negatives is None
            else args.negatives == "degree_matched")
    log.info(f"Training negatives: {'degree-matched' if hard else 'uniform'}")

    x_dict = {k: graph[k].x.to(device) for k in ("miRNA", "gene")}
    ei_dict = {k: v.to(device) for k, v in edge_dict(encoder_edges).items()}

    model = miRNAGraphTransformer.from_config(cfg, graph.metadata(), num_cell_types=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"])
    gen = torch.Generator().manual_seed(seed)

    best_auroc, best_state, patience = -1.0, None, 0
    val_neg = draw_negatives(val_sup, split.all_pos, bins, n_mirna, n_gene, hard, gen)

    for epoch in range(1, tcfg["num_epochs"] + 1):
        model.train()
        neg = draw_negatives(train_sup, split.all_pos, bins, n_mirna, n_gene, hard, gen)
        pairs = torch.cat([train_sup, neg], dim=1)
        labels = torch.cat([torch.ones(train_sup.shape[1]),
                            torch.zeros(neg.shape[1])]).to(device)

        opt.zero_grad()
        out = model(x_dict, ei_dict,
                    mirna_idx=pairs[0].to(device), gene_idx=pairs[1].to(device))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out["edge_logits"], labels)
        loss.backward()
        opt.step()

        if epoch % tcfg.get("eval_every", 5) == 0 or epoch == tcfg["num_epochs"]:
            v = evaluate(model, x_dict, ei_dict, val_sup, val_neg, device)["auroc"]
            if v > best_auroc:
                best_auroc, patience = v, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience += 1
            log.info(f"  epoch {epoch:>4}  loss {loss.item():.4f}  val AUROC {v:.4f}"
                     f"{'  *' if patience == 0 else ''}")
            if patience >= tcfg["patience"]:
                log.info(f"  early stop at epoch {epoch} (best val AUROC {best_auroc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Both test cells from the one trained model: same pairs, same encoder view, only
    # the negative sampler differs -- mirroring eval_heldout_grid.py on our own graph.
    results = {}
    for name, is_hard in (("uniform", False), ("degree_matched", True)):
        g_eval = torch.Generator().manual_seed(seed)
        neg = draw_negatives(test_sup, split.all_pos, bins, n_mirna, n_gene, is_hard, g_eval)
        results[name] = evaluate(model, x_dict, ei_dict, test_sup, neg, device)
        log.info(f"  TEST [{name:<15}] AUROC={results[name]['auroc']:.4f}  "
                 f"AUPRC={results[name]['auprc']:.4f}")

    summary = {
        "paper": paper,
        "edge_regime": regime,
        "seed": seed,
        "n_mirna": int(n_mirna),
        "n_disease": int(n_gene),
        "n_associations": int(graph[REL_FWD].edge_index.shape[1]),
        "encoder_edges_visible": int(encoder_edges.shape[1]),
        "n_train_sup": int(train_sup.shape[1]),
        "n_test": int(test_sup.shape[1]),
        "training_negatives": "degree_matched" if hard else "uniform",
        "best_val_auroc": float(best_auroc),
        "results": results,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
