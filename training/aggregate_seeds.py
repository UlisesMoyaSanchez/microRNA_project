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

# checkpoint-dir stem -> how that run was TRAINED. Derived from a prefix rather than
# hardcoded to checkpoints_v2_*: with the stems fixed, this script silently aggregated
# the miRDB runs no matter which graph the caller meant, so a miRTarBase sweep would
# either report miRDB's numbers or claim its own outputs were missing.
def train_conditions(prefix: str = "checkpoints_v2") -> dict:
    return {
        f"{prefix}_edgesplit": "degree_matched",
        f"{prefix}_edgesplit_uniform": "uniform",
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


def collect(split: str, seeds: list[int], prefix: str = "checkpoints_v2") -> dict:
    """Load every per-seed grid JSON, keyed by (train condition, seed)."""
    conds = train_conditions(prefix)
    runs: dict[str, dict[int, dict]] = {c: {} for c in conds.values()}
    missing: list[str] = []

    for stem, train_cond in conds.items():
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


def topology_reference(split: str, metric: str, path: Path | None = None) -> dict | None:
    """The model-free heuristic this whole paper is about beating.

    `path` is explicit because the default filename carries no graph identity: an
    aggregate over a different interaction source would otherwise compare its models
    against miRDB's gene_degree and call it a margin.
    """
    path = path or COMPARISON_DIR / f"topology_baseline_{split}.json"
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
    p.add_argument("--checkpoint-prefix", default="checkpoints_v2",
                   help="Checkpoint-dir stem to aggregate, e.g. checkpoints_mirtarbase. "
                        "Default reproduces the miRDB sweep.")
    p.add_argument("--topology-baseline", default=None,
                   help="Path to the topology_baseline JSON for THIS graph. Default: "
                        "results/comparison/topology_baseline_<split>.json (the miRDB one) "
                        "— pass explicitly for any other interaction source.")
    args = p.parse_args()

    if len(args.seeds) < 2:
        # mean_std reports std 0.0 for a single observation, which renders as a
        # confident '+/- 0.0000' — indistinguishable from a real zero-variance result.
        raise SystemExit(
            f"--seeds got {args.seeds}: n={len(args.seeds)} cannot carry an error bar, "
            f"and the report would print '+/- 0.0000' as if it could. Pass >=2 seeds."
        )

    runs = collect(args.split, args.seeds, args.checkpoint_prefix)
    topo = topology_reference(
        args.split, args.metric,
        Path(args.topology_baseline) if args.topology_baseline else None)

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

    # The seen-edges row is a single-seed constant declared by the graph's own config;
    # carry it through as such. None for any graph on which the transductive protocol
    # was never measured (e.g. miRTarBase) — that row is then simply absent, rather
    # than borrowed from miRDB.
    any_run = next(iter(runs["degree_matched"].values()))
    seen = any_run.get("reference_seen_edges")

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
    if args.metric == "auroc" and seen:
        # The reference constants are AUROC; there is no AUPRC equivalent.
        lines.append(
            f"{'edges seen in training':<28}{seen['uniform']:>17.4f}          "
            f"{seen['degree_matched']:>9.4f}   <- n=1, from config"
        )
    elif args.metric == "auroc":
        lines.append(
            f"{'edges seen in training':<28}{'n/a':>17}          {'n/a':>9}"
            f"   <- not measured on this graph"
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
    if seen:
        lines.append(
            "NOTE: 'edges seen in training' is a single-seed constant, declared by this\n"
            "graph's config under evaluation.reference_seen_edges. Every attribution\n"
            "derived from it (cost of an honest split, cost of honest negatives) therefore\n"
            "has NO variance and must not be reported as mean +/- std."
        )
    else:
        lines.append(
            "NOTE: this graph declares no evaluation.reference_seen_edges, so the\n"
            "transductive row and every attribution derived from it are absent. Do not\n"
            "fill them in from another graph's constants — the held-out numbers above\n"
            "stand on their own."
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
