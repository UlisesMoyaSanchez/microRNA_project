"""
density_sweep.py — Does graph sparsity itself raise the model-free floor?

results/HMDD_TOPOLOGY_AUDIT.md left a hypothesis open: *sparser graphs make the
popularity ceiling higher relative to what a trained model can add*. The obvious test
-- correlate the gap against density across the surveyed graphs -- does not work: the
seven papers span only five distinct graphs (three share the canonical HMDD matrix),
and Spearman on n=5 gives rho=+0.70, p=0.19. Direction as predicted, nowhere near
significance, and DiGAMN breaks the ordering. Five points that also differ in source
database, version, size and domain cannot carry a mechanism claim.

So test it WITHIN a graph instead. Subsample positives from one association matrix at
a decreasing retention rate, and watch the model-free floor move. This turns five
confounded between-graph points into a dose-response curve per graph, with the graph
itself held fixed -- the only thing that changes is how many edges it has.

The falsifiable prediction, and its control:
  uniform negatives        the floor RISES as the graph gets sparser. In a sparse
                           graph a random unlabeled pair is almost surely a true
                           negative and almost surely touches a low-degree disease,
                           so degree alone separates the classes well.
  degree-matched negatives the floor stays FLAT near 0.55 at every density, because
                           matching the candidate's popularity bin removes exactly the
                           signal the uniform arm is exploiting.
If the uniform arm rises while the degree-matched arm does not, the effect is
popularity, not some general "sparse graphs are easier" artifact. If both rise, the
mechanism is not what we claimed and the hypothesis is wrong.

Runs per distinct GRAPH, not per paper -- MGCNSS, NIMGSA and HLGNN-MDA share one
matrix, so the canonical-5430 graph is swept once (see HMDD_TOPOLOGY_AUDIT.md).

Usage:
  python analysis/density_sweep.py --graph canonical5430
  python analysis/density_sweep.py --graph all --seeds 5
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.eval_topology_baseline import build_scorers, uniform_negatives
from training.eval_hmdd_survey_topology_baseline import (
    ORDER, load_matrix, load_pair_list, score_pairs,
)
from training.splits import degree_bins, gene_in_degree, sample_degree_matched_negatives

# One entry per DISTINCT graph. `papers` is recorded in the output so a reader can see
# which surveyed papers each curve stands for without re-deriving the sharing.
GRAPHS = {
    "canonical5430": {
        "matrix": "data/raw/hmdd_survey/nimgsa/matrix.csv",
        "papers": ["MGCNSS", "NIMGSA", "HLGNN-MDA"],
        "test_fraction": 0.1,
        "negative_ratio": 1,
    },
    "cksnp_gnn": {
        "matrix": "data/raw/hmdd_survey/cksnp_gnn/matrix.csv",
        "papers": ["CKSNP-GNN"],
        "test_fraction": 0.1,
        "negative_ratio": 1,
    },
    "digamn": {
        "matrix": "data/raw/hmdd_survey/digamn/matrix.csv",
        "papers": ["DiGAMN"],
        "test_fraction": 0.1,
        "negative_ratio": 1,
    },
    "meahne": {
        "matrix": "data/raw/hmdd_survey/meahne/matrix.csv",
        "papers": ["MEAHNE"],
        "test_fraction": 0.2,
        "negative_ratio": 1,
    },
    "couplemda": {
        # Pair lists, not a matrix: CoupleMDA publishes its positive split directly.
        # The sweep pools them and re-splits, since it needs to control the edge count.
        "pair_lists": [
            "data/raw/hmdd_survey/couplemda/train_pos.csv",
            "data/raw/hmdd_survey/couplemda/test_pos.csv",
        ],
        "n_mirna": 2090,
        "n_disease": 1754,
        "papers": ["CoupleMDA"],
        "test_fraction": 0.1,
        "negative_ratio": 1,
    },
}

RETENTIONS = [1.0, 0.75, 0.5, 0.25, 0.10]


def load_positives(spec: dict) -> tuple[torch.Tensor, int, int]:
    """Every graph reduced to the same (2, n_pos) form, whatever it ships as."""
    if "matrix" in spec:
        A = load_matrix(spec["matrix"])
        return A.nonzero().T, A.shape[0], A.shape[1]
    pairs = torch.cat([load_pair_list(p) for p in spec["pair_lists"]], dim=1)
    return pairs, int(spec["n_mirna"]), int(spec["n_disease"])


def one_run(
    all_pos: torch.Tensor,
    n_mirna: int,
    n_disease: int,
    retention: float,
    test_fraction: float,
    ratio: int,
    seed: int,
) -> dict:
    """Subsample to `retention` of the positives, then run the standard protocol.

    Held-out edges only: the scorers never see the pairs they score, in every cell.
    The edge-seen regime is a separate axis (eval_hmdd_survey_topology_baseline.py
    --edge-regime) and is deliberately not crossed in here -- this sweep isolates
    density, and crossing both at once would make the curve unattributable.
    """
    gen = torch.Generator().manual_seed(seed)

    n_all = all_pos.shape[1]
    n_keep = max(2, int(round(n_all * retention)))
    kept = all_pos[:, torch.randperm(n_all, generator=gen)[:n_keep]]

    perm = torch.randperm(n_keep, generator=gen)
    n_test = max(1, int(round(n_keep * test_fraction)))
    test_pos = kept[:, perm[:n_test]]
    train_pos = kept[:, perm[n_test:]]

    A_train = torch.zeros((n_mirna, n_disease))
    A_train[train_pos[0], train_pos[1]] = 1.0
    scorers = build_scorers(A_train)

    tiled = test_pos.repeat(1, ratio)
    # Negatives are drawn against the SUBSAMPLED positive set: a pair dropped by the
    # subsample is genuinely absent from this graph, so treating it as a candidate
    # negative is correct here -- that is what the sparser graph actually looks like.
    unif_neg = uniform_negatives(tiled, kept, n_mirna, n_disease, gen)
    deg = gene_in_degree(train_pos, n_disease)
    dm_neg, n_fb = sample_degree_matched_negatives(
        tiled, kept, degree_bins(deg), n_disease, gen, torch.device("cpu")
    )

    out = {
        "retention": retention,
        "seed": seed,
        "n_positives": n_keep,
        "density": n_keep / (n_mirna * n_disease),
        "n_test": int(test_pos.shape[1]),
        "degree_matched_fallback_rate": n_fb / max(dm_neg.shape[1], 1),
        "uniform": score_pairs(scorers, tiled, unif_neg),
        "degree_matched": score_pairs(scorers, tiled, dm_neg),
    }
    for reg in ("uniform", "degree_matched"):
        out[f"best_{reg}"] = max(out[reg][h]["auroc"] for h in ORDER)
    return out


def sweep(name: str, spec: dict, seeds: list[int], log: logging.Logger) -> dict:
    all_pos, n_mirna, n_disease = load_positives(spec)
    log.info(f"{name}: {n_mirna} x {n_disease}, {all_pos.shape[1]:,} positives, "
             f"density {all_pos.shape[1] / (n_mirna * n_disease):.5f}  "
             f"[{', '.join(spec['papers'])}]")

    runs = [
        one_run(all_pos, n_mirna, n_disease, r, spec["test_fraction"],
                spec["negative_ratio"], s)
        for r in RETENTIONS
        for s in seeds
    ]

    points = []
    for r in RETENTIONS:
        sel = [x for x in runs if x["retention"] == r]
        pt = {
            "retention": r,
            "n_positives": sel[0]["n_positives"],
            "density": sel[0]["density"],
        }
        for reg in ("uniform", "degree_matched"):
            v = np.array([x[f"best_{reg}"] for x in sel], dtype=np.float64)
            pt[reg] = {"mean": float(v.mean()), "std": float(v.std(ddof=0))}
        points.append(pt)
        log.info(f"  retention {r:>5.0%}  n={pt['n_positives']:>7,}  "
                 f"density={pt['density']:.5f}  "
                 f"uniform={pt['uniform']['mean']:.4f}±{pt['uniform']['std']:.4f}  "
                 f"deg-matched={pt['degree_matched']['mean']:.4f}"
                 f"±{pt['degree_matched']['std']:.4f}")

    # Spearman over the sweep points. Negative rho on the uniform arm = the floor rises
    # as density falls, which is the prediction.
    from scipy.stats import spearmanr
    dens = [p["density"] for p in points]
    trend = {}
    for reg in ("uniform", "degree_matched"):
        rho, pv = spearmanr(dens, [p[reg]["mean"] for p in points])
        trend[reg] = {"spearman_rho": float(rho), "p_value": float(pv)}
        log.info(f"  trend [{reg:<14}] rho={rho:+.3f} vs density  p={pv:.4f}")

    return {
        "graph": name,
        "papers": spec["papers"],
        "n_mirna": n_mirna,
        "n_disease": n_disease,
        "n_positives_full": int(all_pos.shape[1]),
        "seeds": seeds,
        "retentions": RETENTIONS,
        "points": points,
        "trend_vs_density": trend,
        "runs": runs,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }


PADS = [0, 500, 2000, 5000, 11400]


def pad_sweep(seeds: list[int], log: logging.Logger) -> dict:
    """Second experiment: the density sweep REFUTES the sparsity hypothesis, so what
    does explain MEAHNE's outlying 0.9848 floor?

    Across the five graphs the floor tracks the fraction of *dead candidate columns* --
    diseases with degree zero -- far better than it tracks density (MEAHNE: 92.4% dead
    of 11,783 columns; CoupleMDA 55.7%; the other three 0%). With only five graphs that
    is still a correlation, so manipulate it directly: take canonical-5430, which has no
    dead columns at all, and pad it with empty ones. Padding adds no information of any
    kind -- no edges, no features, just candidate slots that can never be positive.

    If the floor rises anyway, an AUROC reported under uniform negatives is partly a
    measure of how a dataset padded its candidate space, not of what a model learned.
    """
    spec = GRAPHS["canonical5430"]
    pos, n_mirna, n_disease = load_positives(spec)
    log.info(f"padding canonical5430 ({n_mirna} x {n_disease}, "
             f"{pos.shape[1]:,} positives, 0% dead columns)")

    points = []
    for pad in PADS:
        vals = [one_run(pos, n_mirna, n_disease + pad, 1.0, spec["test_fraction"],
                        spec["negative_ratio"], s)["best_uniform"] for s in seeds]
        v = np.array(vals, dtype=np.float64)
        points.append({
            "pad_columns": pad,
            "total_columns": n_disease + pad,
            "dead_column_fraction": pad / (n_disease + pad),
            "density": pos.shape[1] / (n_mirna * (n_disease + pad)),
            "uniform": {"mean": float(v.mean()), "std": float(v.std(ddof=0))},
        })
        log.info(f"  pad {pad:>6,}  cols={n_disease + pad:>7,}  "
                 f"dead={pad / (n_disease + pad):>6.1%}  "
                 f"uniform={v.mean():.4f}±{v.std(ddof=0):.4f}")

    return {
        "experiment": "empty_column_padding",
        "base_graph": "canonical5430",
        "n_mirna": n_mirna,
        "n_disease_base": n_disease,
        "n_positives": int(pos.shape[1]),
        "seeds": seeds,
        "points": points,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", default="all", choices=["all", *GRAPHS])
    p.add_argument("--seeds", type=int, default=5, help="Number of seeds per point.")
    p.add_argument("--skip-padding", action="store_true",
                   help="Run only the density sweep, not the follow-up padding test.")
    p.add_argument("--out-dir", default="results/comparison")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    seeds = [42 + i for i in range(args.seeds)]
    names = list(GRAPHS) if args.graph == "all" else [args.graph]
    os.makedirs(args.out_dir, exist_ok=True)

    summaries = []
    for name in names:
        log.info("=" * 78)
        s = sweep(name, GRAPHS[name], seeds, log)
        out = os.path.join(args.out_dir, f"density_sweep_{name}.json")
        with open(out, "w") as fh:
            json.dump(s, fh, indent=2)
        log.info(f"Wrote {out}")
        summaries.append(s)

    if len(summaries) > 1:
        log.info("=" * 78)
        log.info("Within-graph trend of the model-free floor vs. density:")
        for s in summaries:
            u = s["trend_vs_density"]["uniform"]
            d = s["trend_vs_density"]["degree_matched"]
            log.info(f"  {s['graph']:<15} uniform rho={u['spearman_rho']:+.3f} "
                     f"(p={u['p_value']:.3f})   deg-matched rho={d['spearman_rho']:+.3f} "
                     f"(p={d['p_value']:.3f})")
        log.info("A POSITIVE rho means the floor FALLS as the graph gets sparser -- the "
                 "opposite of the hypothesis this sweep was built to test.")

    if not args.skip_padding:
        log.info("=" * 78)
        pad = pad_sweep(seeds, log)
        out = os.path.join(args.out_dir, "density_sweep_padding.json")
        with open(out, "w") as fh:
            json.dump(pad, fh, indent=2)
        log.info(f"Wrote {out}")


if __name__ == "__main__":
    main()
