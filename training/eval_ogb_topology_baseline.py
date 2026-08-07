"""
eval_ogb_topology_baseline.py — Does the audit's story replicate on someone else's
biological graph?

Part of the "light" extension of this project's evaluation-methodology audit
(training/eval_topology_baseline.py) beyond its own miRNA graph. Runs the same kind
of no-learning, topology-only heuristic scoring on two public, standardized OGB
link-prediction benchmarks — ogbl-ddi (drug-drug interaction) and ogbl-ppa
(protein-protein association) — and compares against OGB's own published
leaderboard numbers for the same heuristics. No GNN is trained here; the trained-
model comparison point is the leaderboard, not a model this script fits.

Unlike this project's own miRNA graph (bipartite: miRNA rows, gene columns),
ogbl-ddi and ogbl-ppa are HOMOGENEOUS, undirected graphs — a single node type,
square adjacency. The four heuristics from eval_topology_baseline.py are
reimplemented with the textbook 1-hop formulas (not that file's bipartite 2-hop
propagation, which does not apply here), plus a fifth heuristic, Resource
Allocation, added because it is the strongest entry on OGB's own ogbl-ppa
leaderboard (Hits@100 = 0.4933, ahead of several trained GNNs) and gives the most
direct cross-check available.

  node_degree(u,v)     = deg(v)                                   [popularity prior]
  pref_attach(u,v)      = deg(u) * deg(v)
  common_neigh(u,v)     = |N(u) ∩ N(v)|
  adamic_adar(u,v)      = sum_{w in N(u)∩N(v)} 1 / log(deg(w))
  resource_alloc(u,v)   = sum_{w in N(u)∩N(v)} 1 / deg(w)

ogbl-ddi (4,267 nodes) is scored with a dense adjacency matrix — trivial at this
scale. ogbl-ppa (576,289 nodes, ~30.3M training edges) CANNOT use a dense matrix
(576,289^2 entries, ~1.3 TB) and cannot even use a full sparse A@A (output nnz
would still be intractable given ~105 average degree). Instead, only the specific
(u,v) pairs that need a score are ever scored, via chunked scipy.sparse row
intersections — no N x N matrix, dense or sparse, is ever materialized for ppa.

Negative sampling — two regimes scored independently, mirroring
eval_topology_baseline.py's uniform vs. degree_matched comparison:
  ogb_official    split_edge[split]['edge_neg'] used as-is. Reports both OGB's own
                  official Hits@K (via ogb.linkproppred.Evaluator) and AUROC/AUPRC.
  degree_matched  this project's own training.splits.sample_degree_matched_negatives,
                  reused UNMODIFIED — adapted only at the call site (edges
                  symmetrized, a synthetic self-loop added to the exclusion set so
                  no negative ever has u==v).

Every trustworthiness check before citing a number from this script anywhere:
results/comparison/ogb_{ddi,ppa}_topology_baseline_{valid,test}.json embeds the
OGB leaderboard reference alongside the results specifically so that check travels
with the artifact. See data/01_download/inspect_ogb_split.py (run first) for the
runtime-verified leak-free assertion this script's own gate re-checks.

Usage (DGX):
  python training/eval_ogb_topology_baseline.py --config configs/config_ogb_ddi.yaml --split test
  python training/eval_ogb_topology_baseline.py --config configs/config_ogb_ppa.yaml --split test
"""

from __future__ import annotations

import os
import sys
import json
import time
import resource
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import yaml
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.splits import degree_bins, gene_in_degree, pair_keys, sample_degree_matched_negatives

HEURISTIC_NAMES = ["node_degree", "pref_attach", "common_neigh", "adamic_adar", "resource_alloc"]


def git(*args: str, cwd: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                               text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def provenance(root: str) -> dict:
    return {
        "git_sha": git("rev-parse", "HEAD", cwd=root),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no", cwd=root)),
        "git_untracked_files": len(
            [ln for ln in git("status", "--porcelain", cwd=root).splitlines()
             if ln.startswith("??")]),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def to_2xN(t: torch.Tensor) -> torch.Tensor:
    """OGB ships edges as (N,2); this project's own code uses (2,N). Normalize once."""
    if t.dim() == 2 and t.shape[1] == 2 and t.shape[0] != 2:
        return t.T.contiguous()
    return t


def drop_self_loops(edge_index: torch.Tensor, log: logging.Logger, tag: str) -> torch.Tensor:
    mask = edge_index[0] != edge_index[1]
    n_dropped = int((~mask).sum())
    if n_dropped:
        log.warning(f"  {tag}: dropped {n_dropped} self-loop pair(s) (unexpected — flag if large)")
    return edge_index[:, mask]


def symmetrize(edge_index: torch.Tensor) -> torch.Tensor:
    """Add the reverse of every edge, then dedup. Cheap defensive step regardless
    of what inspect_ogb_split.py found about dataset[0]'s own symmetry."""
    both = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return torch.unique(both, dim=1)


# ── Dense path (ogbl-ddi) ────────────────────────────────────────────────────────

def build_dense_scorers(A: torch.Tensor) -> dict[str, torch.Tensor]:
    """A: (n,n) dense symmetric 0/1 adjacency, diagonal already zero."""
    A = A.float()
    deg = A.sum(0)
    scores: dict[str, torch.Tensor] = {}

    scores["node_degree"] = deg.unsqueeze(0).expand(A.shape[0], -1).contiguous()
    scores["pref_attach"] = torch.outer(deg, deg)

    scores["common_neigh"] = A @ A
    scores["common_neigh"].fill_diagonal_(0)

    inv_log = 1.0 / torch.log(deg.clamp(min=2.0))
    A_aa = A * inv_log.unsqueeze(1)          # row w scaled by 1/log(deg(w))
    scores["adamic_adar"] = A @ A_aa
    scores["adamic_adar"].fill_diagonal_(0)

    inv_deg = 1.0 / deg.clamp(min=1.0)
    A_ra = A * inv_deg.unsqueeze(1)          # row w scaled by 1/deg(w)
    scores["resource_alloc"] = A @ A_ra
    scores["resource_alloc"].fill_diagonal_(0)

    return scores


def score_dense(scorers: dict[str, torch.Tensor], idx0: torch.Tensor, idx1: torch.Tensor) -> dict[str, np.ndarray]:
    return {k: M[idx0, idx1].numpy().astype(np.float64) for k, M in scorers.items()}


# ── Sparse / pairwise path (ogbl-ppa) ────────────────────────────────────────────

def build_sparse_adjacency(edge_index: torch.Tensor, n: int) -> sp.csr_matrix:
    rows = edge_index[0].numpy()
    cols = edge_index[1].numpy()
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A.sum_duplicates()
    A.data[:] = 1.0
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def score_pairs_sparse(
    A: sp.csr_matrix,
    deg: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    chunk_size: int,
    log: logging.Logger,
) -> dict[str, np.ndarray]:
    """Never materializes an N x N matrix. For each chunk of (u,v) pairs, gathers
    just those rows of A (and of the column-reweighted variants for AA/RA) and
    scores via elementwise sparse row intersection — O(chunk * avg_degree), not
    O(n^2)."""
    inv_log = 1.0 / np.log(np.clip(deg, 2.0, None))
    inv_deg = 1.0 / np.clip(deg, 1.0, None)
    # Column w scaled by weight(w); A is symmetric so this equals row-scaling too,
    # but we index rows of these by v below, so column-scaling is what we need.
    A_aa_cols = A.multiply(inv_log[None, :]).tocsr()
    A_ra_cols = A.multiply(inv_deg[None, :]).tocsr()

    n_pairs = len(U)
    out = {k: np.zeros(n_pairs, dtype=np.float64) for k in HEURISTIC_NAMES}

    t0 = time.time()
    for start in range(0, n_pairs, chunk_size):
        end = min(start + chunk_size, n_pairs)
        u_chunk = U[start:end]
        v_chunk = V[start:end]

        out["node_degree"][start:end] = deg[v_chunk]
        out["pref_attach"][start:end] = deg[u_chunk] * deg[v_chunk]

        A_u = A[u_chunk, :]
        A_v = A[v_chunk, :]
        out["common_neigh"][start:end] = np.asarray(A_u.multiply(A_v).sum(axis=1)).ravel()

        A_aa_v = A_aa_cols[v_chunk, :]
        out["adamic_adar"][start:end] = np.asarray(A_u.multiply(A_aa_v).sum(axis=1)).ravel()

        A_ra_v = A_ra_cols[v_chunk, :]
        out["resource_alloc"][start:end] = np.asarray(A_u.multiply(A_ra_v).sum(axis=1)).ravel()

        if (start // chunk_size) % 10 == 0:
            log.info(f"    scored {end:,}/{n_pairs:,} pairs ({time.time() - t0:.1f}s elapsed)")
    return out


# ── Metrics ──────────────────────────────────────────────────────────────────────

def auroc_auprc(scores_pos: dict[str, np.ndarray], scores_neg: dict[str, np.ndarray]) -> dict[str, dict]:
    out = {}
    for k in scores_pos:
        y = np.concatenate([np.ones(len(scores_pos[k])), np.zeros(len(scores_neg[k]))])
        s = np.concatenate([scores_pos[k], scores_neg[k]])
        out[k] = {
            "auroc": float(roc_auc_score(y, s)),
            "auprc": float(average_precision_score(y, s)),
            "n_pairs": int(len(y)),
        }
    return out


def official_hits(evaluator, scores_pos: dict[str, np.ndarray], scores_neg: dict[str, np.ndarray]) -> dict[str, dict]:
    out = {}
    for k in scores_pos:
        result = evaluator.eval({
            "y_pred_pos": torch.tensor(scores_pos[k]),
            "y_pred_neg": torch.tensor(scores_neg[k]),
        })
        out[k] = {kk: float(vv) for kk, vv in result.items()}
    return out


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True,
                   help="configs/config_ogb_ddi.yaml or configs/config_ogb_ppa.yaml")
    # OGB's own split naming is "valid", not this project's usual "val" — kept as
    # OGB spells it to avoid a silent KeyError against split_edge.
    p.add_argument("--split", default="test", choices=["valid", "test"])
    p.add_argument("--out", default=None,
                   help="Default: results/comparison/ogb_<dataset>_topology_baseline_<split>.json")
    args = p.parse_args()

    from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)
    root = str(Path(__file__).resolve().parents[1])

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    ogb_name = cfg["data"]["ogb_name"]           # "ogbl-ddi" / "ogbl-ppa"
    dataset_short = ogb_name.split("-")[1]        # "ddi" / "ppa"
    seed = cfg["project"]["seed"]
    chunk_size = cfg.get("scoring", {}).get("pair_chunk_size", 100_000)

    if args.out is None:
        args.out = f"results/comparison/ogb_{dataset_short}_topology_baseline_{args.split}.json"

    log.info("=" * 78)
    log.info(f"{ogb_name} — model-free topology baseline, split={args.split}")
    log.info("=" * 78)

    dataset = PygLinkPropPredDataset(name=ogb_name, root=cfg["data"]["ogb_root"])
    data = dataset[0]
    n = int(data.num_nodes)
    split_edge = dataset.get_edge_split()

    mp_edges = drop_self_loops(data.edge_index, log, "dataset[0].edge_index")
    mp_edges = symmetrize(mp_edges)
    log.info(f"  nodes: {n:,}   mp (training) edges, symmetrized: {mp_edges.shape[1]:,}")

    val_pos = drop_self_loops(to_2xN(split_edge["valid"]["edge"]), log, "valid positives")
    test_pos = drop_self_loops(to_2xN(split_edge["test"]["edge"]), log, "test positives")

    # Gate: re-verify leak-free before trusting anything downstream, mirroring
    # training/splits.py::assert_no_edge_leakage's philosophy (never trust a doc).
    mp_keys = pair_keys(mp_edges, n)
    for tag, pos_edges in (("valid", val_pos), ("test", test_pos)):
        overlap = int(torch.isin(pair_keys(pos_edges, n), mp_keys).sum())
        overlap_rev = int(torch.isin(pair_keys(pos_edges.flip(0), n), mp_keys).sum())
        if overlap or overlap_rev:
            raise AssertionError(
                f"LEAK: {tag} positives overlap the training message-passing graph "
                f"({overlap} fwd + {overlap_rev} rev) — do not trust any number from "
                f"this run. Re-check data/01_download/inspect_ogb_split.py output."
            )
        log.info(f"  [ok] {tag} positives: 0 overlap with mp graph — leak-free confirmed")

    # Exclusion set for the degree-matched negative sampler: every known positive
    # (train + val + test), symmetrized, plus a synthetic self-loop on every node
    # so the sampler never proposes u==v as a negative. NOT the same as mp_edges
    # (which is train-only and is what the heuristics are computed on).
    self_loops = torch.arange(n).unsqueeze(0).repeat(2, 1)
    neg_exclusion_edges = torch.unique(
        torch.cat([mp_edges, symmetrize(val_pos), symmetrize(test_pos), self_loops], dim=1),
        dim=1,
    )

    # Degree/bins computed on TRAINING edges only (mp_edges) — binning on the full
    # edge set would leak held-out structure into the negative sampler's choices,
    # same rule training/splits.py::gene_in_degree's docstring states.
    deg_t = gene_in_degree(mp_edges, n)
    bins = degree_bins(deg_t)

    pos = val_pos if args.split == "valid" else test_pos
    neg_official = drop_self_loops(to_2xN(split_edge[args.split]["edge_neg"]), log, f"{args.split} edge_neg")

    log.info(f"  {args.split}: {pos.shape[1]:,} positives, {neg_official.shape[1]:,} official negatives")

    generator = torch.Generator().manual_seed(seed)
    neg_dm, n_fallback = sample_degree_matched_negatives(
        pos, neg_exclusion_edges, bins, n, generator, torch.device("cpu"),
    )
    fallback_pct = 100.0 * n_fallback / max(neg_dm.shape[1], 1)
    log.info(f"  degree-matched negatives: {neg_dm.shape[1]:,} (fallback rate {fallback_pct:.1f}%)")

    dense = dataset_short == "ddi"
    if dense:
        log.info("  Dense path (ddi-scale) — building full adjacency matrix...")
        A = torch.zeros((n, n))
        A[mp_edges[0], mp_edges[1]] = 1.0
        A.fill_diagonal_(0)
        scorers = build_dense_scorers(A)
        scores_pos = score_dense(scorers, pos[0], pos[1])
        scores_neg_official = score_dense(scorers, neg_official[0], neg_official[1])
        scores_neg_dm = score_dense(scorers, neg_dm[0], neg_dm[1])
    else:
        log.info("  Sparse/pairwise path (ppa-scale) — no N x N matrix will be built...")
        A_sp = build_sparse_adjacency(mp_edges, n)
        deg_np = np.asarray(A_sp.sum(axis=1)).ravel()
        log.info(f"    adjacency nnz: {A_sp.nnz:,}   avg degree: {deg_np.mean():.1f}")
        log.info("    scoring test/val positives...")
        scores_pos = score_pairs_sparse(A_sp, deg_np, pos[0].numpy(), pos[1].numpy(), chunk_size, log)
        log.info("    scoring official negatives...")
        scores_neg_official = score_pairs_sparse(
            A_sp, deg_np, neg_official[0].numpy(), neg_official[1].numpy(), chunk_size, log
        )
        log.info("    scoring degree-matched negatives...")
        scores_neg_dm = score_pairs_sparse(
            A_sp, deg_np, neg_dm[0].numpy(), neg_dm[1].numpy(), chunk_size, log
        )

    results = {
        "ogb_official": auroc_auprc(scores_pos, scores_neg_official),
        "degree_matched": auroc_auprc(scores_pos, scores_neg_dm),
    }

    evaluator = Evaluator(name=ogb_name)
    official_metric_name = str(getattr(evaluator, "eval_metric", "hits@k"))
    hits = official_hits(evaluator, scores_pos, scores_neg_official)

    leaderboard_reference = cfg.get("evaluation", {}).get("leaderboard_reference", {})

    log.info("=" * 78)
    log.info(f"{ogb_name} TOPOLOGY-ONLY BASELINES — no learning, no model, split={args.split}")
    log.info("=" * 78)
    log.info(f"{'heuristic':<18}{'ogb_official AUROC':>20}{'degree_matched AUROC':>22}{official_metric_name:>16}")
    log.info("-" * 78)
    for name in HEURISTIC_NAMES:
        log.info(
            f"{name:<18}{results['ogb_official'][name]['auroc']:>20.4f}"
            f"{results['degree_matched'][name]['auroc']:>22.4f}"
            f"{list(hits[name].values())[0] if hits[name] else float('nan'):>16.4f}"
        )
    log.info("-" * 78)
    if leaderboard_reference:
        log.info(f"OGB leaderboard reference: {leaderboard_reference}")
    log.info(f"node_degree AUROC under degree_matched negatives should be ~0.5 by "
              f"construction — sanity check, not a finding: "
              f"{results['degree_matched']['node_degree']['auroc']:.4f}")
    log.info("=" * 78)

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    log.info(f"Peak RSS this run: {peak_rss_mb:,.0f} MB "
             f"(log this for future --mem sizing of the sparse/ppa path)")

    summary = {
        "dataset": ogb_name,
        "split": args.split,
        "n_nodes": n,
        "n_mp_edges": int(mp_edges.shape[1]),
        "n_positives": int(pos.shape[1]),
        "n_negatives_official": int(neg_official.shape[1]),
        "n_negatives_degree_matched": int(neg_dm.shape[1]),
        "degree_matched_fallback_pct": fallback_pct,
        "official_metric": official_metric_name,
        "leaderboard_reference": leaderboard_reference,
        "results": results,
        "official_hits": hits,
        "peak_rss_mb": peak_rss_mb,
        "provenance": provenance(root),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
