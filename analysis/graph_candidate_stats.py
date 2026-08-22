"""
graph_candidate_stats.py — Shape, density and dead-candidate-column stats per graph.

`analysis/density_sweep.py` showed that what moves the model-free floor is not a graph's
density but the fraction of its candidate columns that can never be positive -- columns
with degree zero. This script emits those descriptive stats as an artifact so the
manuscript's protocol-grid table can cite them instead of carrying hand-typed numbers.

Covers the five distinct miRNA-disease graphs behind the seven surveyed papers, plus
this project's own miRNA-gene graph when its .pt is present (DGX only -- the graph is
untracked and ~550MB, so on a machine without it that row is simply omitted and the
rest still writes).

Usage:
  python analysis/graph_candidate_stats.py
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

from analysis.density_sweep import GRAPHS, load_positives

OWN_GRAPH = "data/graphs_v3fixed/hetero_graph.pt"


def gini(x: np.ndarray) -> float:
    """Inequality of the candidate-column degree distribution. 0 = every column equally
    connected, 1 = all edges on one column."""
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def stats(name: str, pos: torch.Tensor, n_rows: int, n_cols: int, papers: list[str]) -> dict:
    deg = torch.zeros(n_cols)
    deg.index_add_(0, pos[1].long(), torch.ones(pos.shape[1]))
    d = deg.numpy()
    return {
        "graph": name,
        "papers": papers,
        "n_rows": int(n_rows),
        "n_columns": int(n_cols),
        "n_positives": int(pos.shape[1]),
        "density": float(pos.shape[1] / (n_rows * n_cols)),
        "dead_column_fraction": float((d == 0).mean()),
        "n_dead_columns": int((d == 0).sum()),
        "column_degree_gini": gini(d),
        "mean_column_degree": float(d.mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/comparison/graph_candidate_stats.json")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    rows = []
    for name, spec in GRAPHS.items():
        pos, n_m, n_d = load_positives(spec)
        rows.append(stats(name, pos, n_m, n_d, spec["papers"]))

    if os.path.exists(OWN_GRAPH):
        g = torch.load(OWN_GRAPH, weights_only=False)
        ei = g["miRNA", "regulates", "gene"].edge_index
        rows.append(stats("own_graph", ei, g["miRNA"].num_nodes, g["gene"].num_nodes,
                          ["this paper (primary case)"]))
    else:
        log.warning(f"{OWN_GRAPH} not found -- own-graph row omitted (expected off the DGX)")

    rows.sort(key=lambda r: r["dead_column_fraction"])
    log.info(f"{'graph':<16}{'rows':>7}{'cols':>9}{'positives':>11}"
             f"{'density':>10}{'dead cols':>11}{'gini':>7}")
    for r in rows:
        log.info(f"{r['graph']:<16}{r['n_rows']:>7,}{r['n_columns']:>9,}"
                 f"{r['n_positives']:>11,}{r['density']:>10.5f}"
                 f"{r['dead_column_fraction']:>11.1%}{r['column_degree_gini']:>7.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"graphs": rows,
                   "generated_utc": datetime.now(timezone.utc).isoformat()}, fh, indent=2)
    log.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
