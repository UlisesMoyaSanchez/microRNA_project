"""
test_fallback_policy.py -- how much of the degree-matched floor is still uniform fallback, and
does a better fallback change it?

compare_matched_samplers.py (job 568) showed that raising the retry budget from 16 to 50 lowers
the survey graphs' corrected floor by 0.008-0.011, and that 1000 retries change nothing more:
a residual 2-5% of positives sit in a degree bin where their miRNA already links to every
column, and those still get a UNIFORM negative. Three policies, on the same splits:

  published   16 random draws from the positive's bin, then a uniform non-edge
              (training/splits.py::sample_degree_matched_negatives, numpy port in protocheck)
  exact       enumerate the bin's valid columns instead of drawing (= infinite retries);
              still uniform when the bin is exhausted
  neighbour   exact, but an exhausted bin falls back to the nearest bin with a valid column
              (b-1/b+1, then b-2/b+2, ...), never to a uniform non-edge

Reported per policy: best model-free floor, the gene-degree heuristic alone (0.5 = perfectly
degree-matched), share of negatives that are uniform fallbacks / neighbour-bin fallbacks, and
the TV distance between the negatives' and the positives' degree-bin histograms.

CoupleMDA is run twice: on the paper's own positive split (what our tables use) and on a random
10% split of the same matrix (what the independent audit did), to test whether that difference
explains the audit's 0.010-lower floor on that graph.

Usage: python analysis/test_fallback_policy.py   (writes results/comparison/fallback_policy_test.json)
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocheck.checks import (  # noqa: E402
    TopologyScorer, column_degree, degree_bins, degree_matched_negatives,
)

RAW = Path("data/raw/hmdd_survey")
SEEDS = [0, 1, 2, 3, 4]
CASES = {  # case -> (source, test fraction or None for the paper's own split)
    "canonical5430": ("nimgsa", 0.1),
    "cksnp_gnn": ("cksnp_gnn", 0.1),
    "digamn": ("digamn", 0.1),
    "meahne": ("meahne", 0.2),
    "couplemda": ("couplemda", None),
    "couplemda_randomsplit": ("couplemda", 0.1),
}


def load(src: str):
    if src == "couplemda":
        tr = np.loadtxt(RAW / src / "train_pos.csv", delimiter=",", dtype=np.int64).T
        te = np.loadtxt(RAW / src / "test_pos.csv", delimiter=",", dtype=np.int64).T
        return np.concatenate([tr, te], axis=1), 2090, 1754, (tr, te)
    m = np.loadtxt(RAW / src / "matrix.csv", delimiter=",")
    return np.argwhere(m > 0).T.astype(np.int64), m.shape[0], m.shape[1], None


def exact_negatives(test, allp, n_rows, n_cols, bins, rng, neighbour: bool):
    """One negative per positive, drawn uniformly from ALL valid columns of its bin."""
    known = [set() for _ in range(n_rows)]
    for r, c in allp.T:
        known[r].add(int(c))
    n_bins = int(bins.max()) + 1
    by_bin = [np.flatnonzero(bins == b) for b in range(n_bins)]
    out, n_unif, n_nb = [], 0, 0
    for m, g in test.T:
        b = int(bins[g])
        valid = [c for c in by_bin[b] if c not in known[m]]
        if not valid and neighbour:
            for d in range(1, n_bins):
                cand = [c for bb in (b - d, b + d) if 0 <= bb < n_bins
                        for c in by_bin[bb] if c not in known[m]]
                if cand:
                    valid, n_nb = cand, n_nb + 1
                    break
        if not valid:  # uniform non-edge
            n_unif += 1
            valid = [c for c in range(n_cols) if c not in known[m]]
        out.append((m, valid[rng.integers(len(valid))]))
    return np.array(out).T, n_unif, n_nb


def metrics(train, test, neg, n_rows, n_cols, bins):
    scorer = TopologyScorer(train, n_rows, n_cols)
    y = np.r_[np.ones(test.shape[1]), np.zeros(neg.shape[1])]
    sc = scorer.score(np.concatenate([test, neg], axis=1))
    au = {k: float(roc_auc_score(y, v)) for k, v in sc.items()}
    nb = int(bins.max()) + 1
    hp = np.bincount(bins[test[1]], minlength=nb) / test.shape[1]
    hn = np.bincount(bins[neg[1]], minlength=nb) / neg.shape[1]
    return {"floor": max(au.values()), "gene_degree": au["gene_degree"],
            "tv_neg_vs_pos": float(0.5 * np.abs(hp - hn).sum())}


def main() -> None:
    out = {}
    for case, (src, frac) in CASES.items():
        allp, n_rows, n_cols, own = load(src)
        rows = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            if frac is None:
                train, test = own
            else:
                perm = rng.permutation(allp.shape[1])
                nt = int(round(allp.shape[1] * frac))
                test, train = allp[:, perm[:nt]], allp[:, perm[nt:]]
            bins = degree_bins(column_degree(train, n_cols))
            k = test.shape[1]
            neg, fb = degree_matched_negatives(test, allp, n_rows, n_cols, bins,
                                               np.random.default_rng(1000 + seed), n_tries=16)
            rows.append({"seed": seed, "policy": "published", "uniform_pct": 100 * fb / k,
                         "neighbour_pct": 0.0, **metrics(train, test, neg, n_rows, n_cols, bins)})
            for pol, nbr in (("exact", False), ("neighbour", True)):
                neg, nu, nn = exact_negatives(test, allp, n_rows, n_cols, bins,
                                              np.random.default_rng(1000 + seed), nbr)
                rows.append({"seed": seed, "policy": pol, "uniform_pct": 100 * nu / k,
                             "neighbour_pct": 100 * nn / k,
                             **metrics(train, test, neg, n_rows, n_cols, bins)})
        summ = {}
        print(case)
        for pol in ("published", "exact", "neighbour"):
            sel = [r for r in rows if r["policy"] == pol]
            summ[pol] = {k: (st.mean(r[k] for r in sel), st.stdev(r[k] for r in sel))
                         for k in ("floor", "gene_degree", "uniform_pct", "neighbour_pct",
                                   "tv_neg_vs_pos")}
            s = summ[pol]
            print(f"  {pol:10} floor {s['floor'][0]:.4f}±{s['floor'][1]:.4f}  "
                  f"gene_degree {s['gene_degree'][0]:.4f}  uniform {s['uniform_pct'][0]:.1f}%  "
                  f"neighbour {s['neighbour_pct'][0]:.1f}%  TV {s['tv_neg_vs_pos'][0]:.3f}")
        out[case] = {"summary": summ, "rows": rows}
    with open("results/comparison/fallback_policy_test.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
