"""
eval_hard_negatives.py — Is the link-prediction AUROC regulatory specificity, or
popularity bias?

diagnose_leakage.py showed the model is not reading the scored edge off the graph
(masking it costs only ~0.009 AUROC), but that its ability collapses to chance when
the miRNA<->gene relation is removed entirely. So the signal lives in the topology of
the interaction graph. Two very different mechanisms produce that:

  (1) real structure — genes co-targeted by similar miRNAs, 2-hop generalization;
  (2) popularity bias — a gene targeted by many miRNAs is a good guess for ANY miRNA.

Against uniformly-random negatives these are indistinguishable, because a random
(miRNA, gene) pair usually lands on a low-degree gene. This script separates them by
crossing two negative samplers with two scorers:

                    | uniform negatives | degree-matched negatives
  ------------------+-------------------+-------------------------
  degree heuristic  |        A          |        B (~0.5 by design)
  HGT V2            |        C          |        D   <-- the number that matters

  A = how far a model-free "score the gene by how many miRNAs target it" heuristic
      gets. If A is already high, uniform negatives are trivially separable and the
      published metric is weak evidence no matter what the model does.
  D = the model against negatives that are popularity-matched to the positives. If D
      holds up, the model has specificity beyond degree. If D falls toward A's floor,
      the headline number was mostly popularity.

Degree-matched negatives: for a positive (m, g), sample a negative (m, g') where g'
is NOT a target of m but has an in-degree in the same log-spaced bin as g. Same miRNA,
equally-popular gene, so neither miRNA promiscuity nor gene popularity can separate
them.

Scored pairs are masked from message passing throughout (condition (c) of the leakage
diagnostic), so this is the honest encoder view.

Usage:
  python training/eval_hard_negatives.py --config configs/config_v2.yaml \
      --checkpoint checkpoints_v2/best_model.pt
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
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.hetero_gnn import miRNAGraphTransformer
from training.train import load_graph, split_graph
from training.evaluate import get_mirna_gene_edges
from training.diagnose_leakage import (
    all_local_pos_edges,
    mask_pairs_from_edge_index,
    REL_FWD,
    REL_REV,
)

N_DEGREE_BINS = 8


def degree_bins(gene_in_degree: torch.Tensor, n_bins: int = N_DEGREE_BINS) -> torch.Tensor:
    """Assign each gene to a log-spaced in-degree bin. Bin 0 = never targeted."""
    deg = gene_in_degree.float()
    logd = torch.log1p(deg)
    hi = float(logd.max()) if float(logd.max()) > 0 else 1.0
    edges = torch.linspace(0, hi, n_bins + 1)[1:-1]
    return torch.bucketize(logd, edges)


def sample_degree_matched_negatives(
    scored_pos: torch.Tensor,      # (2, k) local (miRNA, gene)
    all_pos_local: torch.Tensor,   # (2, P) every true edge among batch nodes, local
    gene_bins_local: torch.Tensor, # (n_gene_local,) degree bin per local gene
    n_gene_local: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """
    For each positive (m, g), a negative (m, g') with g' in the same degree bin as g
    and (m, g') not a true edge. Falls back to any non-target of m when the bin has no
    valid candidate, so the returned tensor always has k columns.
    """
    k = scored_pos.shape[1]
    # true-target lookup: key = m * n_gene_local + g
    true_keys = (all_pos_local[0].long() * n_gene_local + all_pos_local[1].long())
    true_set = set(true_keys.tolist())

    bins_by_gene = gene_bins_local.tolist()
    genes_in_bin: dict[int, list[int]] = {}
    for g_idx, b in enumerate(bins_by_gene):
        genes_in_bin.setdefault(int(b), []).append(g_idx)

    rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,), generator=generator)))

    neg_m: list[int] = []
    neg_g: list[int] = []
    n_fallback = 0

    m_list = scored_pos[0].tolist()
    g_list = scored_pos[1].tolist()

    for m, g in zip(m_list, g_list):
        b = int(bins_by_gene[g])
        candidates = genes_in_bin.get(b, [])
        picked = -1
        # try within-bin candidates first
        for _ in range(20):
            if not candidates:
                break
            gp = int(candidates[rng.integers(len(candidates))])
            if (m * n_gene_local + gp) not in true_set:
                picked = gp
                break
        if picked < 0:
            # fall back: any gene that m does not target
            for _ in range(50):
                gp = int(rng.integers(n_gene_local))
                if (m * n_gene_local + gp) not in true_set:
                    picked = gp
                    n_fallback += 1
                    break
        if picked < 0:
            continue
        neg_m.append(m)
        neg_g.append(picked)

    if not neg_m:
        return torch.empty((2, 0), dtype=torch.long, device=device), 0
    return torch.tensor([neg_m, neg_g], dtype=torch.long, device=device), n_fallback


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/config_v2.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--max_pairs", type=int, default=256)
    p.add_argument("--out", default="results/comparison/hard_negatives.json")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    graph, index_maps = load_graph(cfg["data"]["graphs_dir"])
    num_cell_types = len(index_maps["cell_type_labels"])
    tcfg = cfg["training"]

    _, _, test_mask = split_graph(graph, tcfg["val_ratio"], tcfg["test_ratio"], seed)
    graph["cell"].test_mask = test_mask

    pos_edge_full = get_mirna_gene_edges(graph)
    if pos_edge_full is None:
        log.error("Graph has no miRNA→gene edges.")
        sys.exit(1)

    n_gene_global = graph["gene"].num_nodes
    gene_in_degree = torch.zeros(n_gene_global, dtype=torch.long)
    gene_in_degree.scatter_add_(
        0, pos_edge_full[1].long(), torch.ones(pos_edge_full.shape[1], dtype=torch.long)
    )
    gene_bins_global = degree_bins(gene_in_degree)
    log.info(
        f"Gene in-degree: mean={gene_in_degree.float().mean():.1f} "
        f"max={int(gene_in_degree.max())} "
        f"targeted={int((gene_in_degree > 0).sum()):,}/{n_gene_global:,}"
    )

    test_loader = NeighborLoader(
        graph,
        num_neighbors={et: tcfg["num_neighbors"] for et in graph.edge_types},
        batch_size=tcfg["batch_size"],
        input_nodes=("cell", test_mask),
        shuffle=False,
    )

    model = miRNAGraphTransformer.from_config(cfg, graph.metadata(), num_cell_types).to(device)
    ckpt = args.checkpoint or os.path.join(tcfg["checkpoint_dir"], "best_model.pt")
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    log.info(f"Loaded {ckpt} (epoch {ck.get('epoch', '?')})")

    # scores[sampler] = {"model": [...], "degree": [...], "y": [...]}
    acc: dict[str, dict[str, list[np.ndarray]]] = {
        s: {"model": [], "degree": [], "y": []} for s in ("uniform", "degree_matched")
    }
    total_fallback = 0
    total_hard = 0
    n_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            all_pos = all_local_pos_edges(pos_edge_full, batch, device)
            if all_pos is None:
                continue

            n_mirna = batch["miRNA"].num_nodes
            n_gene = batch["gene"].num_nodes
            gene_nid = batch["gene"].n_id.cpu()
            gene_bins_local = gene_bins_global[gene_nid]
            gene_deg_local = gene_in_degree[gene_nid]

            k = min(args.max_pairs, all_pos.shape[1])
            perm = torch.randperm(all_pos.shape[1], generator=gen)[:k].to(device)
            scored_pos = all_pos[:, perm]

            negs: dict[str, torch.Tensor] = {}
            negs["uniform"] = negative_sampling(
                edge_index=all_pos,
                num_nodes=(n_mirna, n_gene),
                num_neg_samples=k,
                method="sparse",
            ).to(device)

            hard, n_fb = sample_degree_matched_negatives(
                scored_pos, all_pos, gene_bins_local, n_gene, gen, device
            )
            if hard.shape[1] == 0:
                continue
            negs["degree_matched"] = hard
            total_fallback += n_fb
            total_hard += hard.shape[1]

            # Honest encoder view: scored pairs masked from message passing.
            eid = dict(batch.edge_index_dict)
            if REL_FWD in eid:
                eid[REL_FWD] = mask_pairs_from_edge_index(eid[REL_FWD], scored_pos, n_gene, flip=False)
            if REL_REV in eid:
                eid[REL_REV] = mask_pairs_from_edge_index(eid[REL_REV], scored_pos, n_gene, flip=True)

            for sampler, neg in negs.items():
                n_neg = neg.shape[1]
                mirna_idx = torch.cat([scored_pos[0], neg[0]])
                gene_idx = torch.cat([scored_pos[1], neg[1]])
                y = np.concatenate([np.ones(k), np.zeros(n_neg)])

                out = model(
                    x_dict=batch.x_dict,
                    edge_index_dict=eid,
                    mirna_idx=mirna_idx,
                    gene_idx=gene_idx,
                )
                s_model = torch.sigmoid(out["edge_logits"]).cpu().numpy()

                # Model-free baseline: score the pair purely by how many miRNAs
                # target that gene. Knows nothing about the miRNA.
                s_degree = gene_deg_local[gene_idx.cpu()].float().numpy()

                acc[sampler]["model"].append(s_model)
                acc[sampler]["degree"].append(s_degree)
                acc[sampler]["y"].append(y)

            n_batches += 1

    if n_batches == 0:
        log.error("No usable batches.")
        sys.exit(1)

    results: dict[str, dict[str, dict[str, float]]] = {}
    for sampler in ("uniform", "degree_matched"):
        y = np.concatenate(acc[sampler]["y"])
        results[sampler] = {}
        for scorer in ("model", "degree"):
            s = np.concatenate(acc[sampler][scorer])
            results[sampler][scorer] = {
                "auroc": float(roc_auc_score(y, s)),
                "auprc": float(average_precision_score(y, s)),
                "n_pairs": int(len(y)),
            }

    fb_pct = 100.0 * total_fallback / max(total_hard, 1)

    log.info("=" * 74)
    log.info("HARD NEGATIVES — is the AUROC specificity, or popularity bias?")
    log.info("=" * 74)
    log.info(f"{'scorer':<26}{'uniform neg':>16}{'degree-matched neg':>22}")
    log.info("-" * 74)
    for scorer, name in (("degree", "gene-degree heuristic"), ("model", "HGT V2")):
        u = results["uniform"][scorer]["auroc"]
        d = results["degree_matched"][scorer]["auroc"]
        log.info(f"{name:<26}{u:>16.4f}{d:>22.4f}")
    log.info("-" * 74)
    log.info(f"degree-matched fallback rate: {fb_pct:.1f}% (lower is a cleaner match)")
    log.info("")
    log.info("Read: if the degree heuristic already scores high under uniform negatives,")
    log.info("the published metric is weak evidence. If HGT falls to that same floor")
    log.info("under degree-matched negatives, the result was popularity, not regulation.")
    log.info("=" * 74)

    summary = {
        "checkpoint": str(ckpt),
        "n_batches": n_batches,
        "degree_matched_fallback_pct": fb_pct,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
