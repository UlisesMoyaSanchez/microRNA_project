"""
aggregate_seeds.py — collapse the per-seed held-out grids into the 2x2 with variance.

Every number in the audit so far has been a single seed. This reads the per-seed
outputs of eval_heldout_grid.py and reports mean +/- std for each cell of

                            | evaluated with     | evaluated with
                            | uniform negatives  | degree-matched negatives
    ------------------------+--------------------+-------------------------
    trained w/ degree-matched (hard) negatives
    trained w/ uniform negatives

The model-free gene-degree heuristic is printed alongside, because the cell that
matters is not "is the model good" but "does the model beat counting edges".

CAVEAT, stated in the output: the "edges seen in training" reference row
(0.9836 / 0.8828) is a hardcoded constant in eval_heldout_grid.py from the
original single-seed transductive run. It is NOT recomputed per seed, so every
attribution that subtracts from it inherits n=1 and carries no variance.

Usage:
    python training/aggregate_seeds.py --split test
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

COMPARISON_DIR = Path("results/comparison")

# checkpoint-dir stem -> how that run was TRAINED
TRAIN_CONDITIONS = {
    "checkpoints_v2_edgesplit": "degree_matched",
    "checkpoints_v2_edgesplit_uniform": "uniform",
}
EVAL_CONDITIONS = ("uniform", "degree_matched")

PRETTY = {
    "degree_matched": "degree-matched (hard)",
    "uniform": "uniform",
}


def mean_std(xs: list[float]) -> tuple[float, float]:
    """Sample mean and std (ddof=1). std is 0.0 for a single observation."""
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def collect(split: str, seeds: list[int]) -> dict:
    """Load every per-seed grid JSON, keyed by (train condition, seed)."""
    runs: dict[str, dict[int, dict]] = {c: {} for c in TRAIN_CONDITIONS.values()}
    missing: list[str] = []

    for stem, train_cond in TRAIN_CONDITIONS.items():
        for seed in seeds:
            path = COMPARISON_DIR / f"heldout_grid_{stem}_s{seed}_{split}.json"
            if not path.exists():
                missing.append(str(path))
                continue
            with open(path) as fh:
                runs[train_cond][seed] = json.load(fh)

    if missing:
        raise SystemExit(
            "Missing per-seed grid outputs:\n  "
            + "\n  ".join(missing)
            + "\n\nRun training/slurm_heldout_grid.sh for each seed first."
        )
    return runs


def topology_reference(split: str, metric: str) -> dict | None:
    """The model-free heuristic this whole paper is about beating."""
    path = COMPARISON_DIR / f"topology_baseline_{split}.json"
    if not path.exists():
        return None
    with open(path) as fh:
        topo = json.load(fh)
    return {
        cond: topo["results"][cond]["gene_degree"][metric] for cond in EVAL_CONDITIONS
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--seeds", type=int, nargs="+", default=[123, 777, 2024, 7])
    p.add_argument("--metric", default="auroc", choices=["auroc", "auprc"])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    runs = collect(args.split, args.seeds)
    topo = topology_reference(args.split, args.metric)

    # cells[train_cond][eval_cond] = (mean, std, [per-seed values])
    cells: dict[str, dict[str, tuple[float, float, list[float]]]] = {}
    for train_cond, by_seed in runs.items():
        cells[train_cond] = {}
        for eval_cond in EVAL_CONDITIONS:
            vals = [
                by_seed[s]["held_out"][eval_cond][args.metric] for s in args.seeds
            ]
            m, sd = mean_std(vals)
            cells[train_cond][eval_cond] = (m, sd, vals)

    # The seen-edges row is a hardcoded single-seed constant; carry it through as such.
    any_run = next(iter(runs["degree_matched"].values()))
    seen = any_run["reference_seen_edges"]

    n = len(args.seeds)
    width = 78
    lines: list[str] = []
    lines.append("=" * width)
    lines.append(
        f"HELD-OUT {args.split.upper()} EDGES — {args.metric.upper()}, "
        f"mean +/- std over n={n} seeds {args.seeds}"
    )
    lines.append("=" * width)
    lines.append(f"{'trained with':<28}{'eval: uniform neg':>24}{'eval: degree-matched':>26}")
    lines.append("-" * width)
    for train_cond in ("degree_matched", "uniform"):
        row = f"{PRETTY[train_cond] + ' neg':<28}"
        for eval_cond in EVAL_CONDITIONS:
            m, sd, _ = cells[train_cond][eval_cond]
            row += f"{m:>17.4f} +/- {sd:.4f}" if eval_cond == "uniform" else f"{m:>19.4f} +/- {sd:.4f}"
        lines.append(row)
    lines.append("-" * width)

    if topo:
        row = f"{'gene-degree heuristic (n=1)':<28}"
        row += f"{topo['uniform']:>17.4f}          "
        row += f"{topo['degree_matched']:>9.4f}"
        lines.append(row)
    if args.metric == "auroc":
        # The hardcoded reference constants are AUROC; there is no AUPRC equivalent.
        lines.append(
            f"{'edges seen in training':<28}{seen['uniform']:>17.4f}          "
            f"{seen['degree_matched']:>9.4f}   <- n=1, hardcoded"
        )
    lines.append("=" * width)

    # The claim the paper actually rests on.
    if topo:
        for eval_cond in EVAL_CONDITIONS:
            best_model = max(
                cells[tc][eval_cond][0] for tc in cells
            )
            margin = best_model - topo[eval_cond]
            verdict = "BEATS" if margin > 0 else "LOSES TO"
            lines.append(
                f"eval={eval_cond:<15} best model {best_model:.4f}  "
                f"{verdict} heuristic {topo[eval_cond]:.4f}  ({margin:+.4f})"
            )
        lines.append("=" * width)
    lines.append(
        "NOTE: 'edges seen in training' is a single-seed constant hardcoded in\n"
        "eval_heldout_grid.py. Every attribution derived from it (cost of an honest\n"
        "split, cost of honest negatives) therefore has NO variance and must not be\n"
        "reported as mean +/- std."
    )
    lines.append("=" * width)

    report = "\n".join(lines)
    print(report)

    summary = {
        "split": args.split,
        "metric": args.metric,
        "seeds": args.seeds,
        "n_seeds": n,
        "cells": {
            tc: {
                ec: {
                    "mean": cells[tc][ec][0],
                    "std": cells[tc][ec][1],
                    "per_seed": dict(zip(args.seeds, cells[tc][ec][2])),
                }
                for ec in EVAL_CONDITIONS
            }
            for tc in cells
        },
        "gene_degree_heuristic": topo,
        "reference_seen_edges": seen if args.metric == "auroc" else None,
        "reference_seen_edges_caveat": (
            "hardcoded single-seed constant from the original transductive run; "
            "not recomputed per seed, carries no variance"
        ),
    }
    out = args.out or str(
        COMPARISON_DIR / f"multiseed_{args.metric}_{args.split}.json"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
