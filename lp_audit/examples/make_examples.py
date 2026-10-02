"""
Synthetic datasets that each trigger one failure mode from the paper -- and one that
triggers none. Fictional data: the graph is random with a skewed column-degree profile,
and `model_auroc` is a made-up number standing in for "the AUROC you would report".

    python -m lp_audit.examples.make_examples            # writes lp_audit/examples/data/
    python -m lp_audit --train-pos lp_audit/examples/data/leak/train_pos.npy \
        --heldout-pos lp_audit/examples/data/leak/heldout_pos.npy \
        --encoder-edges lp_audit/examples/data/leak/encoder_edges.npy \
        --eval-neg lp_audit/examples/data/leak/eval_neg.npy \
        --n-rows 300 --n-cols 250

Scenario -> what it should flag
  clean                 nothing (leak-free split, degree-matched negatives on both sides)
  leak                  edge_leakage        (held-out edges left in the encoder's graph)
  mismatched_negatives  negative_matching   (degree-matched train negatives, uniform eval negatives)
  dead_columns          dead_candidates     (candidate space padded with empty columns)
  weak_margin           model_free_baseline (reported AUROC barely above the heuristic floor)
"""

from __future__ import annotations

import json
import os

import numpy as np

from lp_audit.checks import (
    column_degree, degree_bins, degree_matched_negatives, model_free_floor, uniform_negatives,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
N_ROWS, N_COLS, N_EDGES = 300, 250, 6000


def make_graph(rng, n_rows=N_ROWS, n_cols=N_COLS, n_edges=N_EDGES) -> np.ndarray:
    """Random bipartite graph, skewed on both sides (a few promiscuous rows, popular columns)."""
    pc = np.arange(1, n_cols + 1, dtype=float) ** -0.8
    pr = rng.lognormal(0.0, 0.8, n_rows)
    pc, pr = pc / pc.sum(), pr / pr.sum()
    pairs = set()
    while len(pairs) < n_edges:
        k = n_edges - len(pairs)
        pairs |= set(zip(rng.choice(n_rows, k, p=pr).tolist(), rng.choice(n_cols, k, p=pc).tolist()))
    return np.array(sorted(pairs), dtype=np.int64).T


def split(pos, rng, heldout_frac=0.2):
    perm = rng.permutation(pos.shape[1])
    n_h = int(heldout_frac * pos.shape[1])
    return pos[:, perm[n_h:]], pos[:, perm[:n_h]]


def _save(name, meta, **arrays):
    d = os.path.join(DATA, name)
    os.makedirs(d, exist_ok=True)
    for k, v in arrays.items():
        np.save(os.path.join(d, f"{k}.npy"), v.astype(np.int32))
    with open(os.path.join(d, "scenario.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


def build(seed: int = 0) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    pos = make_graph(rng)
    train, held = split(pos, rng)
    bins = degree_bins(column_degree(train, N_COLS))     # TRAINING edges only
    matched = lambda p: degree_matched_negatives(p, pos, N_ROWS, N_COLS, bins, rng)[0]  # noqa: E731
    uniform = lambda p: uniform_negatives(p, pos, N_ROWS, N_COLS, rng)  # noqa: E731

    base = dict(n_rows=N_ROWS, n_cols=N_COLS, train_pos=train, heldout_pos=held,
                encoder_edges=train, train_neg=matched(train), eval_neg=matched(held))
    out = {"clean": dict(base, model_auroc=0.90,
                         declared_samplers={"train": "degree_matched", "eval": "degree_matched"})}
    out["leak"] = dict(out["clean"], encoder_edges=np.concatenate([train, held], axis=1))
    out["mismatched_negatives"] = dict(out["clean"], eval_neg=uniform(held),
                                       declared_samplers=None)

    # Padding: same positives, 3x the candidate columns, none of the new ones ever positive.
    pad_cols = 4 * N_COLS
    padded = dict(base, n_cols=pad_cols, model_auroc=0.99, declared_samplers=None,
                  train_neg=uniform_negatives(train, pos, N_ROWS, pad_cols, rng),
                  eval_neg=uniform_negatives(held, pos, N_ROWS, pad_cols, rng))
    out["dead_columns"] = padded

    floor = model_free_floor(train, held, base["eval_neg"], N_ROWS, N_COLS)["floor"]
    out["weak_margin"] = dict(out["clean"], model_auroc=round(floor + 0.01, 4))
    return out


def write(seed: int = 0) -> None:
    for name, kw in build(seed).items():
        meta = {k: v for k, v in kw.items() if not isinstance(v, np.ndarray)}
        _save(name, meta, **{k: v for k, v in kw.items() if isinstance(v, np.ndarray)})
        print(f"wrote {os.path.join(DATA, name)}")


def load_scenario(name: str) -> dict:
    """Read a written scenario back as kwargs for lp_audit.audit."""
    d = os.path.join(DATA, name)
    with open(os.path.join(d, "scenario.json")) as fh:
        kw = json.load(fh)
    for f in os.listdir(d):
        if f.endswith(".npy"):
            kw[f[:-4]] = np.load(os.path.join(d, f)).astype(np.int64)
    return kw


if __name__ == "__main__":
    write()
