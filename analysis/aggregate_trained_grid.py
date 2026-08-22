"""
aggregate_trained_grid.py — Assemble the trained protocol grid, and the discrimination
margin against the model-free floor, across every graph with a trained model.

Two things come out of this, matching the paper's two theses:

  the grid          each graph's trained AUROC across edge regime x negative sampler,
                    mean +- std over seeds. Thesis 1's evidence on other people's graphs.
  the margin        trained minus model-free floor, under each protocol. Thesis 2's
                    evidence: under the conventional protocol that margin is small
                    everywhere, so the protocol cannot support the trained-vs-baseline
                    comparison the subfield uses it for.

**The matched convention.** A cell must be filled from the arm trained on the negatives it
is scored against. A degree-matched-trained model scored against uniform negatives measures
a train/eval distribution mismatch, not protocol difficulty -- the caveat Table 2 already
flags with a dagger. Ignoring this produced a visibly wrong grid on the first pass:
MEAHNE's held_out/uniform cell landed ABOVE its seen/uniform cell, with nine times the
variance of every other cell. This module encodes the convention so it cannot be forgotten:

  seen  / uniform          uniform-trained,        uniform eval
  seen  / degree_matched   uniform-trained,        degree-matched eval
  held  / uniform          uniform-trained,        uniform eval
  held  / degree_matched   degree-matched-trained, degree-matched eval

which is exactly what analysis/make_manuscript_figures.py::load_headline_grid does for our
own graph (seen row from the uniform-trained model, held row on the matched diagonal).

Usage:
  python analysis/aggregate_trained_grid.py
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
import statistics as st
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COMP = Path("results/comparison")

# Graph -> the paper artifact carrying that graph's model-free floor. canonical5430's floor
# is NIMGSA's: same matrix, same generated split, same seed (HLGNN-MDA is bit-identical to
# it; MGCNSS differs only by using its own bundled split).
FLOOR_ARTIFACT = {
    "canonical5430": "nimgsa",
    "cksnp_gnn": "cksnp_gnn",
    "digamn": "digamn",
    "meahne": "meahne",
    "couplemda": "couplemda",
}

SEEDS = [42, 7, 123, 2024]

# (regime, eval sampler) -> which training arm fills it. See the docstring.
CELL_ARM = {
    ("seen", "uniform"): "uniform",
    ("seen", "degree_matched"): "uniform",
    ("held_out", "uniform"): "uniform",
    ("held_out", "degree_matched"): "degree_matched",
}


def trained_cell(graph: str, regime: str, ev: str) -> tuple[float, float, int]:
    arm = CELL_ARM[(regime, ev)]
    tag = "_trainuniform" if arm == "uniform" else ""
    vals = []
    for s in SEEDS:
        f = COMP / f"survey_trained_grid_{graph}_{regime}{tag}_s{s}.json"
        if not f.exists():
            raise FileNotFoundError(f"missing {f} — run training/slurm_survey_bipartite.sh "
                                    f"for graph={graph} regime={regime} "
                                    f"negatives={arm} seed={s}")
        vals.append(json.load(open(f))["results"][ev]["auroc"])
    return st.mean(vals), st.pstdev(vals), len(vals)


def floor_cell(graph: str, regime: str, ev: str) -> float:
    """Best of the four model-free heuristics, same graph, same protocol cell."""
    suffix = "_seen" if regime == "seen" else ""
    f = COMP / f"hmdd_survey_topology_baseline_{FLOOR_ARTIFACT[graph]}{suffix}.json"
    res = json.load(open(f))["results"]
    return max(v["auroc"] for v in res[ev].values())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(COMP / "trained_grid_summary.json"))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    graphs = []
    for g in FLOOR_ARTIFACT:
        cells, floors = {}, {}
        for regime in ("seen", "held_out"):
            for ev in ("uniform", "degree_matched"):
                m, sd, n = trained_cell(g, regime, ev)
                cells[f"{regime}__{ev}"] = {"mean": m, "std": sd, "n_seeds": n,
                                            "trained_on": CELL_ARM[(regime, ev)]}
                floors[f"{regime}__{ev}"] = floor_cell(g, regime, ev)

        # The conventional corner is (seen, uniform); the corrected one is
        # (held_out, degree_matched) -- the same two corners Figure 4 connects.
        conv_t, conv_f = cells["seen__uniform"]["mean"], floors["seen__uniform"]
        corr_t, corr_f = cells["held_out__degree_matched"]["mean"], floors["held_out__degree_matched"]

        graphs.append({
            "graph": g,
            "trained_cells": cells,
            "model_free_cells": floors,
            "protocol_cost_trained": conv_t - corr_t,
            "discrimination": {
                "conventional": {"trained": conv_t, "model_free": conv_f,
                                 "margin": conv_t - conv_f},
                "corrected": {"trained": corr_t, "model_free": corr_f,
                              "margin": corr_t - corr_f},
            },
        })
        log.info(f"{g:<15} cost {conv_t - corr_t:+.3f}   "
                 f"margin conventional {conv_t - conv_f:+.4f}   "
                 f"corrected {corr_t - corr_f:+.4f}")

    margins = [g["discrimination"]["conventional"]["margin"] for g in graphs]
    log.info("-" * 78)
    log.info(f"Conventional-protocol margin over {len(graphs)} graphs: "
             f"max {max(margins):+.4f}, min {min(margins):+.4f}")

    with open(args.out, "w") as fh:
        json.dump({"graphs": graphs, "seeds": SEEDS,
                   "cell_arm_convention": {f"{k[0]}__{k[1]}": v for k, v in CELL_ARM.items()},
                   "generated_utc": datetime.now(timezone.utc).isoformat()}, fh, indent=2)
    log.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
