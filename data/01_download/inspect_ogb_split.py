"""
inspect_ogb_split.py — What does OGB's official split actually look like?

Part of the "light" extension of the evaluation-methodology audit to two public,
standardized link-prediction benchmarks: ogbl-ddi (drug-drug interaction) and
ogbl-ppa (protein-protein association). Before writing any heuristic-scoring code
against these graphs (eval_ogb_topology_baseline.py), this script answers questions
that must be confirmed against the actual downloaded bytes, not assumed from OGB's
documentation or from memory:

  1. Is `dataset[0].edge_index` already leak-free — i.e. do the val/test positive
     edges actually appear ABSENT from it? OGB's docs say yes; this re-derives it
     the same way training/splits.py::assert_no_edge_leakage() never trusts a doc,
     it asserts against the tensors.
  2. Is `dataset[0].edge_index` already symmetrized (both (u,v) and (v,u) present
     for every training edge), or does it need symmetrizing before use as an
     undirected adjacency?
  3. Are there self-loops in the graph at all (train or held-out)?
  4. Is `edge_neg` (the official negative pool) a set of free (u,v) pairs, or does
     it fix one endpoint per row the way this project's own uniform_negatives()
     (training/eval_topology_baseline.py) does? This determines whether the
     homogeneous heuristics need a source-aware scoring convention.
  5. What does Evaluator(name=...) actually expect/support (K values, input
     format) — printed directly from the object, not assumed from the leaderboard
     page.

Writes every finding to results/comparison/ogb_protocol_verification.json. This
artifact must be reviewed by hand before eval_ogb_topology_baseline.py is written,
per the plan this script is step 1 of.

Usage (DGX, needs outbound internet on first run to trigger OGB's download):
  python data/01_download/inspect_ogb_split.py --dataset both
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


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


def pair_keys(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """Encode (u, v) pairs as single ints for set-comparison. n must exceed every
    node index that appears, so the encoding is injective."""
    return edge_index[0].long() * n + edge_index[1].long()


def inspect_one(name: str, out_dir: str, log: logging.Logger) -> dict:
    from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

    ogb_name = f"ogbl-{name}"
    log.info("=" * 78)
    log.info(f"Inspecting {ogb_name}")
    log.info("=" * 78)

    dataset = PygLinkPropPredDataset(name=ogb_name, root=out_dir)
    data = dataset[0]
    n = int(data.num_nodes)
    ei = data.edge_index

    log.info(f"  nodes: {n:,}   edge_index shape: {tuple(ei.shape)}")

    # ── Self-loops ────────────────────────────────────────────────────────────
    n_self_loops = int((ei[0] == ei[1]).sum())
    log.info(f"  self-loops in dataset[0].edge_index: {n_self_loops:,}")

    # ── Symmetrization check ─────────────────────────────────────────────────
    # Sample up to 20,000 directed edges and confirm the reverse also appears.
    n_sample = min(20_000, ei.shape[1])
    perm = torch.randperm(ei.shape[1])[:n_sample]
    fwd_keys = pair_keys(ei[:, perm], n)
    rev_keys = pair_keys(ei[:, perm].flip(0), n)
    all_keys = pair_keys(ei, n)
    rev_present = torch.isin(rev_keys, all_keys)
    frac_symmetrized = float(rev_present.float().mean())
    log.info(f"  fraction of sampled edges whose reverse is also present: "
              f"{frac_symmetrized:.4f} (1.0 = already symmetric)")

    # ── Official split ────────────────────────────────────────────────────────
    split_edge = dataset.get_edge_split()
    split_shapes = {}
    for split_name, split_dict in split_edge.items():
        split_shapes[split_name] = {
            k: list(v.shape) if torch.is_tensor(v) else str(type(v))
            for k, v in split_dict.items()
        }
        log.info(f"  split[{split_name}] keys/shapes: {split_shapes[split_name]}")

    # ── Leak-free assertion: val/test positives must be ABSENT from dataset[0] ──
    mp_keys = pair_keys(ei, n)
    leak_report = {}
    for split_name in ("valid", "test"):
        if split_name not in split_edge or "edge" not in split_edge[split_name]:
            continue
        pos = split_edge[split_name]["edge"]
        # split_edge stores edges as (n_edges, 2), NOT (2, n_edges) — confirm shape
        # convention directly rather than assuming.
        if pos.shape[1] == 2 and pos.shape[0] != 2:
            pos_ei = pos.T
        else:
            pos_ei = pos
        pos_keys = pair_keys(pos_ei, n)
        pos_keys_rev = pair_keys(pos_ei.flip(0), n)
        overlap_fwd = int(torch.isin(pos_keys, mp_keys).sum())
        overlap_rev = int(torch.isin(pos_keys_rev, mp_keys).sum())
        leak_report[split_name] = {
            "n_positive_edges": int(pos_ei.shape[1]),
            "overlap_with_mp_graph_forward": overlap_fwd,
            "overlap_with_mp_graph_reverse": overlap_rev,
            "leak_free": overlap_fwd == 0 and overlap_rev == 0,
        }
        status = "OK leak-free" if leak_report[split_name]["leak_free"] else "FAIL — LEAK DETECTED"
        log.info(f"  [{status}] {split_name}: {overlap_fwd} fwd + {overlap_rev} rev "
                  f"overlaps out of {pos_ei.shape[1]:,} positives")

    # ── train-edge-count vs dataset[0] edge-count (symmetrization arithmetic) ───
    n_train_edges = None
    if "train" in split_edge and "edge" in split_edge["train"]:
        train_pos = split_edge["train"]["edge"]
        n_train_edges = int(train_pos.shape[0] if train_pos.shape[1] == 2 else train_pos.shape[1])
        log.info(f"  train['edge'] count: {n_train_edges:,}   "
                 f"dataset[0].edge_index count: {ei.shape[1]:,}   "
                 f"ratio: {ei.shape[1] / max(n_train_edges, 1):.3f} "
                 f"(~2.0 implies dataset[0] is train['edge'] symmetrized)")

    # ── Negative pool structure ──────────────────────────────────────────────
    neg_report = {}
    for split_name in ("valid", "test"):
        if split_name not in split_edge or "edge_neg" not in split_edge[split_name]:
            continue
        neg = split_edge[split_name]["edge_neg"]
        neg_ei = neg.T if (neg.dim() == 2 and neg.shape[1] == 2 and neg.shape[0] != 2) else neg
        if neg_ei.dim() == 2:
            n_unique_src = int(torch.unique(neg_ei[0]).numel())
            n_rows = int(neg_ei.shape[1])
            neg_report[split_name] = {
                "shape": list(neg.shape),
                "n_unique_source_nodes": n_unique_src,
                "n_rows": n_rows,
                "looks_fixed_source_scheme": n_unique_src < n_rows * 0.5,
            }
        else:
            # Some OGB datasets ship edge_neg as (n_pos, n_neg_per_pos, 2) — a
            # per-positive candidate set rather than a flat shared pool. Record
            # the raw shape either way rather than guessing.
            neg_report[split_name] = {"shape": list(neg.shape), "note": "non-2D — inspect shape manually"}
        log.info(f"  {split_name} edge_neg: {neg_report[split_name]}")

    # ── Evaluator introspection ──────────────────────────────────────────────
    evaluator = Evaluator(name=ogb_name)
    evaluator_report = {
        "expected_input_format": str(getattr(evaluator, "expected_input_format", "N/A")),
        "eval_metric": str(getattr(evaluator, "eval_metric", "N/A")),
    }
    log.info(f"  Evaluator.eval_metric: {evaluator_report['eval_metric']}")
    log.info(f"  Evaluator.expected_input_format:\n{evaluator_report['expected_input_format']}")

    return {
        "dataset": ogb_name,
        "n_nodes": n,
        "n_edges_dataset0": int(ei.shape[1]),
        "n_self_loops": n_self_loops,
        "fraction_sampled_edges_symmetric": frac_symmetrized,
        "split_shapes": split_shapes,
        "leak_report": leak_report,
        "n_train_edges_official": n_train_edges,
        "negative_pool_report": neg_report,
        "evaluator": evaluator_report,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["ddi", "ppa", "both"], default="both")
    p.add_argument("--out_dir", default="data/raw/ogb")
    p.add_argument("--out", default="results/comparison/ogb_protocol_verification.json")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)
    root = str(Path(__file__).resolve().parents[2])

    names = ["ddi", "ppa"] if args.dataset == "both" else [args.dataset]
    findings = {name: inspect_one(name, args.out_dir, log) for name in names}

    summary = {
        "findings": findings,
        "provenance": provenance(root),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("=" * 78)
    log.info(f"Wrote {args.out}")
    log.info("Read this file by hand before writing any scoring code — it determines")
    log.info("whether the homogeneous heuristics need a source-aware negative-scoring")
    log.info("convention (negative_pool_report.looks_fixed_source_scheme) and confirms")
    log.info("the leak-free property empirically rather than by trusting OGB's docs.")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
