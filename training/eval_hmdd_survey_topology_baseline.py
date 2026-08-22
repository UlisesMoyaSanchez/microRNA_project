"""
eval_hmdd_survey_topology_baseline.py — Does a no-learning topology heuristic close
the gap to a surveyed paper's own reported AUROC, on that paper's own data?

Part of the HMDD topology-baseline audit (see data/01_download/
download_hmdd_survey_canonical5430.py and results/HMDD_TOPOLOGY_AUDIT.md). Extends
this project's own model-free-baseline finding (training/eval_topology_baseline.py:
0.8712 on our miRNA-gene graph, beating a trained HGT's 0.8056) to the miRNA-disease
/ HMDD cluster of the 22-paper literature survey (results/literature_survey.tsv).

Reuses training.eval_topology_baseline.build_scorers() UNCHANGED -- it already
operates on a plain (rows, cols) binary adjacency tensor, so no bipartite-vs-
homogeneous reimplementation is needed the way the OGB extension required (OGB's
graphs are homogeneous; miRNA-disease is bipartite just like this project's own
miRNA-gene graph).

Two negative-sampling regimes per paper, exactly mirroring
eval_topology_baseline.py's own uniform vs. degree-matched comparison:
  uniform          training.eval_topology_baseline.uniform_negatives
  degree_matched   training.splits.sample_degree_matched_negatives
Ratio r (1:1, 1:5, 1:10, 50:1 -- see literature_survey.tsv) is handled by tiling the
positive-pairs tensor r times before calling either sampler; both samplers derive
their negative count from scored_pos.shape[1], so this produces exactly r negatives
per positive with zero changes to training/splits.py (verified locally this session
across r in {1, 5, 10, 50} with a leak-free check against the true-positive set).

Two data regimes per paper (see each paper's own docstring note in
download_hmdd_survey_canonical5430.py for why it's tiered the way it is):
  Tier 1 (exact split)   a paper's own bundled, labeled train/test file is used
                          directly -- no split is generated here.
  Tier 2 (ratio-matched) only the exact positive-association matrix is confirmed;
                          the split/negatives are generated here at the paper's
                          stated ratio and are an honest approximation, not a
                          reproduction. Every output JSON records which regime
                          applies.

Two edge regimes per paper (--edge-regime, see build_scoring_matrix), crossing the
two negative regimes above into the same 2x2 protocol grid this project measures on
its own graph (manuscript Figure 4):
  held_out  heuristics see training positives only -- the corrected protocol, and
            the default, so an un-flagged run reproduces the published numbers.
  seen      heuristics see every positive including the scored ones -- what a paper
            does when it never masks held-out edges from its message-passing graph.
Only the graph view differs; the split and the negatives are identical across the
two, so the difference between rows is attributable to the protocol alone.

MGCNSS is the exception: its Mode-A split carries the paper's own negatives, so the
negative axis does not exist there and it contributes a 1x2 row, not a 2x2 grid.

Usage:
  python training/eval_hmdd_survey_topology_baseline.py --config configs/config_hmdd_survey_mgcnss.yaml
  python training/eval_hmdd_survey_topology_baseline.py --config configs/config_hmdd_survey_meahne.yaml --edge-regime seen
  python training/eval_hmdd_survey_topology_baseline.py --config configs/config_hmdd_survey_nimgsa.yaml
  python training/eval_hmdd_survey_topology_baseline.py --config configs/config_hmdd_survey_hlgnn_mda.yaml
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

import yaml
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.eval_topology_baseline import build_scorers, uniform_negatives
from training.splits import degree_bins, gene_in_degree, pair_keys, sample_degree_matched_negatives

ORDER = ["gene_degree", "pref_attach", "common_neigh", "adamic_adar"]


def load_matrix(path: str) -> torch.Tensor:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append([float(v) for v in line.split(",")])
    return torch.tensor(rows, dtype=torch.float32)


def load_labeled_split(train_path: str, test_path: str, index_base: int = 0) -> dict[str, torch.Tensor]:
    """Tier-1 path: paper's own (node1, node2, label) triples, whitespace-separated.

    index_base: 0 if the file already uses 0-based (mirna, disease) local indices,
    1 if 1-based (subtracted here). MGCNSS's train7.txt/test7_1.txt are 1-based --
    confirmed from src/link_prediction_evaluate.py's own indexing
    (`edge[0]-1` for miRNA, `edge[1]+494` i.e. `(edge[1]-1)+495` for disease)."""

    def read(p: str) -> tuple[torch.Tensor, torch.Tensor]:
        pos, neg = [], []
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                a, b, y = line.split()
                a, b, y = int(float(a)), int(float(b)), int(float(y))
                a -= index_base
                b -= index_base
                (pos if y == 1 else neg).append((a, b))
        pos_t = torch.tensor(pos, dtype=torch.long).T if pos else torch.empty((2, 0), dtype=torch.long)
        neg_t = torch.tensor(neg, dtype=torch.long).T if neg else torch.empty((2, 0), dtype=torch.long)
        return pos_t, neg_t

    train_pos, train_neg = read(train_path)
    test_pos, test_neg = read(test_path)
    return {
        "train_pos": train_pos, "train_neg": train_neg,
        "test_pos": test_pos, "test_neg": test_neg,
    }


def load_pair_list(path: str) -> torch.Tensor:
    """Plain (row_idx, col_idx) CSV, no header, no label column -- positives only."""
    pairs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            a, b = line.split(",")
            pairs.append((int(a), int(b)))
    return torch.tensor(pairs, dtype=torch.long).T if pairs else torch.empty((2, 0), dtype=torch.long)


def build_scoring_matrix(
    train_pairs: torch.Tensor,
    all_pairs: torch.Tensor,
    n_mirna: int,
    n_disease: int,
    edge_regime: str,
) -> torch.Tensor:
    """The adjacency the heuristics are allowed to see. Only this differs between
    the two edge regimes -- the split and the negatives are identical either way,
    so the same pairs are scored and any AUROC difference is the protocol alone.

    held_out (corrected)     training positives only. The scored edge was never in
                             the graph, so nothing about it can leak into its score.
    seen (conventional)      every positive, including the ones about to be scored.
                             build_scorers() zeroes its miRNA-miRNA diagonal, so a
                             test edge still cannot vouch for itself directly -- but
                             it does inflate the co-targeting counts that feed
                             common_neigh/adamic_adar, and it raises the disease's
                             own degree. That indirect inflation IS the conventional
                             protocol's leak, not a bug in this function.
    """
    pairs = train_pairs if edge_regime == "held_out" else all_pairs
    A = torch.zeros((n_mirna, n_disease))
    A[pairs[0], pairs[1]] = 1.0
    return A


def score_pairs(scorers: dict[str, torch.Tensor], pos: torch.Tensor, neg: torch.Tensor) -> dict:
    k = pos.shape[1]
    y = np.concatenate([np.ones(k), np.zeros(neg.shape[1])])
    m_idx = torch.cat([pos[0], neg[0]]).long()
    d_idx = torch.cat([pos[1], neg[1]]).long()
    out = {}
    for name, M in scorers.items():
        s = M[m_idx, d_idx].numpy().astype(np.float64)
        out[name] = {
            "auroc": float(roc_auc_score(y, s)),
            "auprc": float(average_precision_score(y, s)),
            "n_pairs": int(len(y)),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--edge-regime", default="held_out", choices=["held_out", "seen"],
                   help="Which edges the heuristics may see (see build_scoring_matrix). "
                        "held_out is the corrected protocol and the default, so an "
                        "un-flagged run reproduces the published numbers exactly.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    paper = cfg["project"]["name"]
    seed = cfg["project"]["seed"]
    dcfg = cfg["data"]
    ref = cfg["evaluation"]["paper_reference"]
    tier = ref["tier"]

    # Unsuffixed for held_out: the already-published artifacts keep their paths.
    if args.out is None:
        suffix = "" if args.edge_regime == "held_out" else f"_{args.edge_regime}"
        args.out = f"results/comparison/hmdd_survey_topology_baseline_{paper}{suffix}.json"

    log.info("=" * 78)
    log.info(f"HMDD survey topology baseline — {paper}  (tier {tier}, "
             f"edges {args.edge_regime.replace('_', ' ')})")
    log.info(f"Paper's own reported AUROC: {ref['headline_auroc']}  ({ref['url']})")
    log.info("=" * 78)

    gen = torch.Generator().manual_seed(seed)

    if "train_path" in dcfg and "test_path" in dcfg:
        # Mode A: exact labeled (m, d, label) triples for both train and test
        # (MGCNSS) -- fully exact, no negatives generated.
        A_full = load_matrix(dcfg["matrix_path"])
        n_mirna, n_disease = A_full.shape
        log.info(f"Full association matrix: {n_mirna} x {n_disease}, "
                 f"{int(A_full.sum()):,} positives")
        index_base = int(dcfg.get("index_base", 0))
        split = load_labeled_split(dcfg["train_path"], dcfg["test_path"], index_base=index_base)
        train_edges = split["train_pos"]  # message-passing graph: TRAIN positives only
        test_pos, test_neg = split["test_pos"], split["test_neg"]
        log.info(f"Exact labeled split: train_pos={train_edges.shape[1]:,}  "
                 f"test_pos={test_pos.shape[1]:,}  test_neg={test_neg.shape[1]:,}")
        # Mode A carries the paper's OWN negatives, so the negative-sampling axis
        # does not exist here -- this paper contributes a 1x2 row (held_out vs
        # seen), not a full 2x2 grid.
        scorers = build_scorers(build_scoring_matrix(
            train_edges, A_full.nonzero().T, n_mirna, n_disease, args.edge_regime
        ))
        results = {"paper_split": score_pairs(scorers, test_pos, test_neg)}
        n_positives_total = int(A_full.sum())

    elif "train_pos_path" in dcfg and "test_pos_path" in dcfg:
        # Mode B: exact positive train/test split (CoupleMDA), no bundled
        # negatives -- generated here at the configured ratio, exactly as the
        # Tier-2 path does, but skipping the random positive split since we
        # already have the paper's own exact one.
        n_mirna, n_disease = int(dcfg["n_mirna"]), int(dcfg["n_disease"])
        ratio = int(dcfg.get("negative_ratio", 1))
        train_pos = load_pair_list(dcfg["train_pos_path"])
        test_pos = load_pair_list(dcfg["test_pos_path"])
        all_pos = torch.cat([train_pos, test_pos], dim=1)
        log.info(f"Exact positive split: {n_mirna} x {n_disease}  "
                 f"train_pos={train_pos.shape[1]:,}  test_pos={test_pos.shape[1]:,}  "
                 f"negative_ratio=1:{ratio}")

        scorers = build_scorers(build_scoring_matrix(
            train_pos, all_pos, n_mirna, n_disease, args.edge_regime
        ))

        tiled = test_pos.repeat(1, ratio)
        unif_neg = uniform_negatives(tiled, all_pos, n_mirna, n_disease, gen)
        # Degree bins stay on train_pos in BOTH regimes: the negatives must be the
        # same pairs across the two rows, or the row difference would confound the
        # edge regime with a different negative set.
        deg = gene_in_degree(train_pos, n_disease)
        bins = degree_bins(deg)
        dm_neg, n_fb = sample_degree_matched_negatives(
            tiled, all_pos, bins, n_disease, gen, torch.device("cpu")
        )
        log.info(f"  degree-matched fallback rate: "
                 f"{100.0 * n_fb / max(dm_neg.shape[1], 1):.1f}%")
        results = {
            "uniform": score_pairs(scorers, tiled, unif_neg),
            "degree_matched": score_pairs(scorers, tiled, dm_neg),
        }
        n_positives_total = int(all_pos.shape[1])

    else:
        # Mode C: only the full association matrix is exact (Tier 2 default) --
        # both the split and the negatives are generated here.
        A_full = load_matrix(dcfg["matrix_path"])
        n_mirna, n_disease = A_full.shape
        log.info(f"Full association matrix: {n_mirna} x {n_disease}, "
                 f"{int(A_full.sum()):,} positives")
        ratio = int(dcfg.get("negative_ratio", 1))
        test_frac = float(dcfg.get("test_fraction", 0.1))
        all_pos = A_full.nonzero().T  # (2, n_pos), (mirna, disease)
        n_pos = all_pos.shape[1]
        perm = torch.randperm(n_pos, generator=gen)
        n_test = max(1, int(round(n_pos * test_frac)))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        test_pos = all_pos[:, test_idx]
        train_pos = all_pos[:, train_idx]
        log.info(f"Tier-2 generated split ({int((1 - test_frac) * 100)}/{int(test_frac * 100)}, "
                 f"seed={seed}): train_pos={train_pos.shape[1]:,}  test_pos={test_pos.shape[1]:,}  "
                 f"negative_ratio=1:{ratio}")

        scorers = build_scorers(build_scoring_matrix(
            train_pos, all_pos, n_mirna, n_disease, args.edge_regime
        ))

        tiled = test_pos.repeat(1, ratio)
        unif_neg = uniform_negatives(tiled, all_pos, n_mirna, n_disease, gen)

        # Degree bins stay on train_pos in BOTH regimes -- see the Mode B note.
        deg = gene_in_degree(train_pos, n_disease)
        bins = degree_bins(deg)
        dm_neg, n_fb = sample_degree_matched_negatives(
            tiled, all_pos, bins, n_disease, gen, torch.device("cpu")
        )
        log.info(f"  degree-matched fallback rate: "
                 f"{100.0 * n_fb / max(dm_neg.shape[1], 1):.1f}%")

        results = {
            "uniform": score_pairs(scorers, tiled, unif_neg),
            "degree_matched": score_pairs(scorers, tiled, dm_neg),
        }
        n_positives_total = int(A_full.sum())

    log.info("-" * 78)
    for sampler_name, per_heur in results.items():
        for name in ORDER:
            log.info(f"  [{sampler_name:<12}] {name:<16} "
                     f"AUROC={per_heur[name]['auroc']:.4f}  AUPRC={per_heur[name]['auprc']:.4f}")
    best_auroc = max(
        per_heur[name]["auroc"] for per_heur in results.values() for name in ORDER
    )
    log.info("-" * 78)
    log.info(f"Best model-free heuristic AUROC : {best_auroc:.4f}")
    log.info(f"Paper's own reported AUROC      : {ref['headline_auroc']}")
    log.info("=" * 78)

    summary = {
        "paper": paper,
        "tier": tier,
        "edge_regime": args.edge_regime,
        "n_mirna": n_mirna,
        "n_disease": n_disease,
        "n_positives_total": n_positives_total,
        "seed": seed,
        "best_model_free_auroc": float(best_auroc),
        "results": results,
        "paper_reference": ref,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
