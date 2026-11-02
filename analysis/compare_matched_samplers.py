"""
compare_matched_samplers.py -- why is our corrected floor ~0.01 above the independent audit's?

Averaging our floor over four splits (job 566) closed the gap on canonical5430 but left the
other four survey graphs 0.007-0.017 above the audit's in the corrected cell, while the
conventional cell agreed. Data and split fractions are identical, so the difference is in the
degree-matched sampler. Two candidate causes, varied one at a time on the SAME splits:

  n_tries  how many draws from the positive's degree bin before falling back to a uniform
           non-edge (ours 16, the audit's 50). A uniform fallback negative is easy to separate,
           so a higher fallback rate raises the floor.
  bins     our log-linear bins from 0 to max (training/splits.py::degree_bins) or the audit's
           from min to max (its trackA.py::make_bins, np.digitize).

Floor = best of the four bipartite heuristics (protocheck.TopologyScorer, which reproduces
training/eval_topology_baseline.py::build_scorers), heuristics seeing training edges only.

Usage: python analysis/compare_matched_samplers.py   (writes results/comparison/matched_sampler_comparison.json)
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocheck.checks import (  # noqa: E402
    column_degree, degree_bins, degree_matched_negatives, model_free_floor,
)

RAW = Path("data/raw/hmdd_survey")
GRAPHS = {  # graph -> (loader kind, test fraction)
    "canonical5430": ("nimgsa", 0.1),
    "cksnp_gnn": ("cksnp_gnn", 0.1),
    "digamn": ("digamn", 0.1),
    "meahne": ("meahne", 0.2),
    "couplemda": ("couplemda", None),   # the paper's own positive split
}
SEEDS = [0, 1, 2, 3, 4]
N_TRIES = [16, 50, 1000]


def audit_bins(deg: np.ndarray, nbins: int = 8) -> np.ndarray:
    """The independent audit's bins: linear on log1p(degree) from min to max, np.digitize."""
    x = np.log1p(deg)
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    return np.clip(np.digitize(x, edges[1:-1]), 0, nbins - 1)


def load(graph: str):
    src, frac = GRAPHS[graph]
    if frac is None:
        tr = np.loadtxt(RAW / src / "train_pos.csv", delimiter=",", dtype=np.int64).T
        te = np.loadtxt(RAW / src / "test_pos.csv", delimiter=",", dtype=np.int64).T
        allp = np.concatenate([tr, te], axis=1)
        return allp, int(allp[0].max()) + 1, int(allp[1].max()) + 1, (tr, te)
    m = np.loadtxt(RAW / src / "matrix.csv", delimiter=",")
    return np.argwhere(m > 0).T.astype(np.int64), m.shape[0], m.shape[1], frac


def main() -> None:
    out: dict = {}
    for graph in GRAPHS:
        allp, n_rows, n_cols, how = load(graph)
        rows = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            if isinstance(how, tuple):
                train, test = how
            else:
                perm = rng.permutation(allp.shape[1])
                n_test = int(round(allp.shape[1] * how))
                test, train = allp[:, perm[:n_test]], allp[:, perm[n_test:]]
            deg = column_degree(train, n_cols)
            for bins_name, bins in (("ours", degree_bins(deg)), ("audit", audit_bins(deg))):
                for n_tries in N_TRIES:
                    neg, n_fb = degree_matched_negatives(
                        test, allp, n_rows, n_cols, bins, np.random.default_rng(1000 + seed),
                        n_tries=n_tries)
                    f = model_free_floor(train, test, neg, n_rows, n_cols)
                    rows.append({"seed": seed, "bins": bins_name, "n_tries": n_tries,
                                 "fallback_rate": n_fb / test.shape[1], "floor": f["floor"],
                                 "best": f["best"]})
        summ = {}
        for b in ("ours", "audit"):
            for t in N_TRIES:
                sel = [r for r in rows if r["bins"] == b and r["n_tries"] == t]
                summ[f"{b}_tries{t}"] = {
                    "floor_mean": st.mean(r["floor"] for r in sel),
                    "floor_sd": st.stdev(r["floor"] for r in sel),
                    "fallback_mean": st.mean(r["fallback_rate"] for r in sel),
                }
        out[graph] = {"summary": summ, "rows": rows}
        print(graph)
        for k, v in summ.items():
            print(f"  {k:16} floor {v['floor_mean']:.4f} ± {v['floor_sd']:.4f}   "
                  f"fallback {100 * v['fallback_mean']:.1f}%")
    Path("results/comparison").mkdir(parents=True, exist_ok=True)
    with open("results/comparison/matched_sampler_comparison.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
