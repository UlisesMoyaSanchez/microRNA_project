"""
lp_audit.checks -- the four evaluation checks, as pure numpy/scipy functions.

Every function takes (2, N) integer arrays of (row, col) pairs and returns a Finding.
Nothing here imports torch or PyG: the logic was extracted from

  training/splits.py                  pair_keys, assert_no_edge_leakage, degree_bins,
                                      gene_in_degree, sample_degree_matched_negatives
  training/eval_topology_baseline.py  build_scorers (bipartite heuristics)
  training/eval_ogb_topology_baseline.py  build_dense_scorers (homogeneous heuristics)
  analysis/graph_candidate_stats.py   gini, stats (dead candidate columns)

and tests/test_lp_audit.py checks that the scorers here reproduce the torch originals.

Status levels: ok / info / warn / error / unchecked. "unchecked" means the input needed
to run the check was not supplied -- it is never reported as "ok".

Thresholds are heuristics, NOT validated cut-offs. Each finding carries the measured
number in `evidence`; read that, not the status word, before drawing a conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

N_DEGREE_BINS = 8  # same as training/splits.py

DEFAULT_THRESHOLDS = {
    # Dead candidate columns: the graphs in the paper sit at 0% (three) or 47-92% (three);
    # 0.25 falls in the empty gap between them. A heuristic split, not a measured knee.
    "dead_column_fraction_warn": 0.25,
    "negatives_on_dead_columns_warn": 0.25,
    # Total-variation distance between two degree-bin histograms (0 = identical, 1 = disjoint).
    "neg_mismatch_tv_warn": 0.10,   # train negatives vs eval negatives
    "neg_vs_pos_tv_warn": 0.20,     # eval negatives vs the held-out positives
    # AUROC points a trained model must clear the best no-learning heuristic by.
    "margin_warn": 0.05,
}

SCOPE_NOTE = (
    "Scope: these checks inspect the pairs you supplied. They cannot see leakage that does "
    "not pass through exact edges (e.g. node features derived from held-out labels), cannot "
    "tell a correct sampler with a different seed/parameter from a matched one, and do not "
    "test whether any published number is inflated -- only whether the evaluation set-up "
    "has the failure modes described in the paper."
)


@dataclass
class Finding:
    check: str
    status: str
    message: str
    evidence: dict = field(default_factory=dict)


# ── Pair utilities ────────────────────────────────────────────────────────────

def as_pairs(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.int64)
    if a.ndim != 2 or a.shape[0] != 2:
        raise ValueError(f"expected a (2, N) array of (row, col) pairs, got shape {a.shape}")
    return a


def pair_keys(pairs: np.ndarray, n_cols: int, undirected: bool = False) -> np.ndarray:
    """Encode (row, col) pairs as single ints so they can be set-compared. For an
    undirected graph (u, v) and (v, u) get the same key."""
    r, c = pairs[0], pairs[1]
    if undirected:
        r, c = np.minimum(r, c), np.maximum(r, c)
    return r * n_cols + c


def _overlap(a_keys: np.ndarray, b_keys: np.ndarray) -> int:
    return int(np.isin(a_keys, b_keys).sum())


def degree_bins(deg: np.ndarray, n_bins: int = N_DEGREE_BINS) -> np.ndarray:
    """Log-spaced degree bin per column. Bin 0 = never targeted. Same edges as
    training/splits.py::degree_bins (torch.bucketize(right=False) == searchsorted 'left')."""
    logd = np.log1p(deg.astype(np.float64))
    hi = float(logd.max()) if logd.max() > 0 else 1.0
    edges = np.linspace(0, hi, n_bins + 1)[1:-1]
    return np.searchsorted(edges, logd, side="left")


def column_degree(pairs: np.ndarray, n_cols: int, undirected: bool = False) -> np.ndarray:
    """Degree per candidate column. For an undirected graph both endpoints count."""
    cols = np.concatenate([pairs[1], pairs[0]]) if undirected else pairs[1]
    return np.bincount(cols, minlength=n_cols)


def _cols(pairs: np.ndarray, undirected: bool) -> np.ndarray:
    return np.concatenate([pairs[1], pairs[0]]) if undirected else pairs[1]


def gini(x: np.ndarray) -> float:
    """Inequality of a degree distribution (analysis/graph_candidate_stats.py::gini)."""
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def _bin_hist(cols: np.ndarray, col_bins: np.ndarray, n_bins: int) -> np.ndarray:
    h = np.bincount(col_bins[cols], minlength=n_bins).astype(float)
    return h / max(h.sum(), 1.0)


# ── Negative samplers (used by examples, tests and the baseline when none is supplied) ──

def uniform_negatives(pos, exclude, n_rows, n_cols, rng, undirected=False, n_tries=32):
    """One random non-edge per positive, row kept fixed. Pairs still invalid after
    n_tries redraws are dropped, so the result can have fewer than pos.shape[1] columns."""
    pos = as_pairs(pos)
    excl = pair_keys(as_pairs(exclude), n_cols, undirected)
    rows = pos[0].copy()
    cols = rng.integers(0, n_cols, size=rows.shape[0])
    for _ in range(n_tries):
        bad = np.isin(pair_keys(np.stack([rows, cols]), n_cols, undirected), excl)
        if undirected:
            bad |= rows == cols
        if not bad.any():
            break
        cols[bad] = rng.integers(0, n_cols, size=int(bad.sum()))
    bad = np.isin(pair_keys(np.stack([rows, cols]), n_cols, undirected), excl)
    if undirected:
        bad |= rows == cols
    return np.stack([rows[~bad], cols[~bad]])


def degree_matched_negatives(pos, exclude, n_rows, n_cols, col_bins, rng,
                             undirected=False, n_tries=16):
    """For each positive (m, g): a negative (m, g') with g' in g's degree bin. Falls back
    to a uniform non-edge when the bin has no valid candidate. Returns (negatives, n_fallback)."""
    pos = as_pairs(pos)
    excl = pair_keys(as_pairs(exclude), n_cols, undirected)
    order = np.argsort(col_bins, kind="stable")
    n_bins = int(col_bins.max()) + 1
    counts = np.bincount(col_bins, minlength=n_bins)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rows = pos[0]
    b = col_bins[pos[1]]
    cols = np.full(rows.shape[0], -1, dtype=np.int64)
    todo = np.arange(rows.shape[0])
    for _ in range(n_tries):
        if todo.size == 0:
            break
        bt = b[todo]
        pick = starts[bt] + np.floor(rng.random(todo.size) * counts[bt]).astype(np.int64)
        cand = order[pick]
        bad = np.isin(pair_keys(np.stack([rows[todo], cand]), n_cols, undirected), excl)
        if undirected:
            bad |= rows[todo] == cand
        cols[todo[~bad]] = cand[~bad]
        todo = todo[bad]
    n_fb = int(todo.size)
    if n_fb:
        c = rng.integers(0, n_cols, size=n_fb)
        for _ in range(2 * n_tries):
            bad = np.isin(pair_keys(np.stack([rows[todo], c]), n_cols, undirected), excl)
            if undirected:
                bad |= rows[todo] == c
            if not bad.any():
                break
            c[bad] = rng.integers(0, n_cols, size=int(bad.sum()))
        cols[todo] = c
    # Anything still invalid after the fallback is dropped, as in uniform_negatives.
    bad = np.isin(pair_keys(np.stack([rows, cols]), n_cols, undirected), excl) | (cols < 0)
    if undirected:
        bad |= rows == cols
    return np.stack([rows[~bad], cols[~bad]]), n_fb


# ── Model-free scorers ────────────────────────────────────────────────────────

def _csr(pairs: np.ndarray, shape, undirected: bool) -> sp.csr_matrix:
    A = sp.coo_matrix((np.ones(pairs.shape[1], dtype=np.float32), (pairs[0], pairs[1])),
                      shape=shape).tocsr()
    if undirected:
        A = A + A.T
    A.sum_duplicates()
    A.data[:] = 1.0
    if undirected:
        A.setdiag(0)
        A.eliminate_zeros()
    return A


def _row_dot(X: sp.csr_matrix, Y: sp.csr_matrix, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    return np.asarray(X[i].multiply(Y[j]).sum(axis=1)).ravel()


class TopologyScorer:
    """No-learning link scores computed from the edges a scorer is allowed to see.

    mode='bipartite' reproduces training/eval_topology_baseline.py::build_scorers
    (gene_degree, pref_attach, common_neigh, adamic_adar); mode='homogeneous' reproduces
    training/eval_ogb_topology_baseline.py::build_dense_scorers on a symmetrised graph
    (node_degree, pref_attach, common_neigh, adamic_adar, resource_alloc).

    Scores are computed per queried pair, never as a dense matrix, so memory is bounded
    by the (n_rows x n_rows) co-neighbour matrix in bipartite mode.
    """

    def __init__(self, visible_edges, n_rows: int, n_cols: int, mode: str = "bipartite"):
        if mode not in ("bipartite", "homogeneous"):
            raise ValueError("mode must be 'bipartite' or 'homogeneous'")
        self.mode = mode
        visible_edges = as_pairs(visible_edges)
        if mode == "homogeneous":
            if n_rows != n_cols:
                raise ValueError("homogeneous mode needs n_rows == n_cols")
            A = _csr(visible_edges, (n_rows, n_cols), True)
            deg = np.asarray(A.sum(axis=0)).ravel()
            self._deg_m = self._deg_g = deg
            inv_log = 1.0 / np.log(np.clip(deg, 2.0, None))
            inv_deg = 1.0 / np.clip(deg, 1.0, None)
            self._A, self._Aaa = A, A.multiply(inv_log[None, :]).tocsr()
            self._Ara = A.multiply(inv_deg[None, :]).tocsr()
        else:
            A = _csr(visible_edges, (n_rows, n_cols), False)
            self._deg_m = np.asarray(A.sum(axis=1)).ravel()
            self._deg_g = np.asarray(A.sum(axis=0)).ravel()
            inv_log = 1.0 / np.log(np.clip(self._deg_g, 2.0, None))
            Aw = A.multiply(inv_log[None, :]).tocsr()
            # The diagonal is zeroed: a miRNA must not vouch for itself (see
            # eval_topology_baseline.build_scorers), or the score degenerates into
            # deg(m) * A[m, g] and leaks the edge for any pair still present in A.
            T = (A @ A.T).tocsr()
            T.setdiag(0)
            T.eliminate_zeros()
            Taa = (A @ Aw.T).tocsr()
            Taa.setdiag(0)
            Taa.eliminate_zeros()
            self._A, self._AT, self._T, self._Taa = A, A.T.tocsr(), T, Taa

    @property
    def names(self) -> list[str]:
        if self.mode == "homogeneous":
            return ["node_degree", "pref_attach", "common_neigh", "adamic_adar", "resource_alloc"]
        return ["gene_degree", "pref_attach", "common_neigh", "adamic_adar"]

    def score(self, pairs, chunk: int = 100_000) -> dict[str, np.ndarray]:
        pairs = as_pairs(pairs)
        n = pairs.shape[1]
        out = {k: np.empty(n, dtype=np.float64) for k in self.names}
        for s in range(0, n, chunk):
            m, g = pairs[0, s:s + chunk], pairs[1, s:s + chunk]
            sl = slice(s, s + m.shape[0])
            if self.mode == "homogeneous":
                out["node_degree"][sl] = self._deg_g[g]
                out["pref_attach"][sl] = self._deg_m[m] * self._deg_g[g]
                same = m == g
                for name, Y in (("common_neigh", self._A), ("adamic_adar", self._Aaa),
                                ("resource_alloc", self._Ara)):
                    v = _row_dot(self._A, Y, m, g)
                    v[same] = 0.0
                    out[name][sl] = v
            else:
                out["gene_degree"][sl] = self._deg_g[g]
                out["pref_attach"][sl] = self._deg_m[m] * self._deg_g[g]
                out["common_neigh"][sl] = _row_dot(self._T, self._AT, m, g)
                out["adamic_adar"][sl] = _row_dot(self._Taa, self._AT, m, g)
        return out


def model_free_floor(visible_edges, pos, neg, n_rows, n_cols, mode="bipartite") -> dict:
    """AUROC of each no-learning heuristic on pos-vs-neg, scoring from `visible_edges`."""
    pos, neg = as_pairs(pos), as_pairs(neg)
    scorer = TopologyScorer(visible_edges, n_rows, n_cols, mode)
    pairs = np.concatenate([pos, neg], axis=1)
    y = np.concatenate([np.ones(pos.shape[1]), np.zeros(neg.shape[1])])
    scores = scorer.score(pairs)
    aurocs = {k: float(roc_auc_score(y, v)) for k, v in scores.items()}
    best = max(aurocs, key=aurocs.get)
    return {"heuristics": aurocs, "best": best, "floor": aurocs[best],
            "n_pos": int(pos.shape[1]), "n_neg": int(neg.shape[1])}


# ── The four checks ───────────────────────────────────────────────────────────

def check_leakage(n_cols, heldout_pos, *, encoder_edges=None, encoder_edges_rev=None,
                  train_pos=None, eval_neg=None, undirected=False) -> Finding:
    """Held-out edges reachable from the encoder's input, in either direction; held-out
    edges that are also training edges; evaluation negatives that are real edges."""
    name = "edge_leakage"
    heldout = as_pairs(heldout_pos)
    hk = pair_keys(heldout, n_cols, undirected)
    if encoder_edges is None:
        return Finding(name, "unchecked",
                       "No encoder_edges supplied, so leakage into the message-passing "
                       "graph was not checked. Pass the edges your model's encoder sees.")
    ev: dict = {"n_heldout": int(heldout.shape[1])}
    ev["heldout_in_encoder_fwd"] = _overlap(hk, pair_keys(as_pairs(encoder_edges), n_cols, undirected))
    if encoder_edges_rev is not None:
        rev = as_pairs(encoder_edges_rev).copy()[::-1]  # (col,row) -> (row,col)
        ev["heldout_in_encoder_rev"] = _overlap(hk, pair_keys(rev, n_cols, undirected))
    if train_pos is not None:
        tr = as_pairs(train_pos)
        ev["heldout_in_train_pos"] = _overlap(hk, pair_keys(tr, n_cols, undirected))
    if eval_neg is not None:
        known = [heldout] + ([as_pairs(train_pos)] if train_pos is not None else [])
        kk = pair_keys(np.concatenate(known, axis=1), n_cols, undirected)
        ev["eval_negatives_that_are_real_edges"] = _overlap(
            pair_keys(as_pairs(eval_neg), n_cols, undirected), kk)
    total = sum(v for k, v in ev.items() if k != "n_heldout")
    ev["fraction_heldout_in_encoder"] = ev["heldout_in_encoder_fwd"] / max(ev["n_heldout"], 1)
    if total == 0:
        return Finding(name, "ok", "No held-out edge is visible to the encoder, shared with "
                       "training, or used as an evaluation negative.", ev)
    parts = [f"{k}={v:,}" for k, v in ev.items()
             if k not in ("n_heldout", "fraction_heldout_in_encoder") and v]
    return Finding(name, "error",
                   f"Evaluation is not leak-free: {', '.join(parts)} "
                   f"({ev['fraction_heldout_in_encoder']:.1%} of {ev['n_heldout']:,} held-out "
                   "edges are in the encoder's input). A model can recover these edges from "
                   "its own neighbourhood, so a metric computed here measures memorisation.", ev)


def check_negative_matching(n_rows, n_cols, train_pos, heldout_pos, *, train_neg=None,
                            eval_neg=None, declared_samplers=None, thresholds=None,
                            n_bins=N_DEGREE_BINS, undirected=False) -> Finding:
    """Empirical train-vs-eval negative distribution, in column-degree bins computed from
    TRAINING edges only (training/splits.py: binning on the full edge set leaks structure)."""
    name = "negative_matching"
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    ev: dict = {}
    if declared_samplers:
        ev["declared_samplers"] = dict(declared_samplers)
        d = {k: str(v).strip().lower() for k, v in declared_samplers.items()}
        if d.get("train") and d.get("eval") and d["train"] != d["eval"]:
            return Finding(name, "error",
                           f"Declared samplers differ: train='{d['train']}', eval='{d['eval']}'. "
                           "Evaluating against a different negative distribution than the one "
                           "trained on yields an artefact, not a difficulty measurement.", ev)
    if eval_neg is None:
        return Finding(name, "unchecked", "No eval_neg supplied, so the evaluation negatives "
                       "were not inspected. Pass the negatives your reported metric used.", ev)
    tr = as_pairs(train_pos)
    col_bins = degree_bins(column_degree(tr, n_cols, undirected), n_bins)
    hist = lambda p: _bin_hist(_cols(as_pairs(p), undirected), col_bins, n_bins)  # noqa: E731
    eval_h, pos_h = hist(eval_neg), hist(heldout_pos)
    ev["tv_eval_neg_vs_heldout_pos"] = _tv(eval_h, pos_h)
    ev["n_eval_neg"] = int(as_pairs(eval_neg).shape[1])
    problems, notes = [], []
    if train_neg is not None:
        train_h = hist(train_neg)
        ev["tv_train_neg_vs_eval_neg"] = _tv(train_h, eval_h)
        ev["tv_train_neg_vs_train_pos"] = _tv(train_h, hist(tr))
        if ev["tv_train_neg_vs_eval_neg"] > thr["neg_mismatch_tv_warn"]:
            problems.append(
                f"train and eval negatives differ (TV={ev['tv_train_neg_vs_eval_neg']:.3f} > "
                f"{thr['neg_mismatch_tv_warn']}): the model is scored on a different negative "
                "distribution than it trained on")
    else:
        notes.append("train_neg not supplied: train-vs-eval matching was not checked")
    if ev["tv_eval_neg_vs_heldout_pos"] > thr["neg_vs_pos_tv_warn"]:
        problems.append(
            f"eval negatives do not mirror the held-out positives' column-degree profile "
            f"(TV={ev['tv_eval_neg_vs_heldout_pos']:.3f} > {thr['neg_vs_pos_tv_warn']}): a "
            "popularity-only scorer can separate them, so part of the metric is column degree")
    if problems:
        return Finding(name, "warn", "; ".join(problems) + ".", ev)
    msg = "Negative degree profiles are consistent" + (" (" + "; ".join(notes) + ")" if notes else "") + "."
    return Finding(name, "info" if notes else "ok", msg, ev)


def check_baseline_margin(n_rows, n_cols, train_pos, heldout_pos, eval_neg=None, *,
                          encoder_edges=None, model_auroc=None, mode="bipartite",
                          thresholds=None, seed=0) -> Finding:
    """The model-free floor on the supplied held-out pairs, and the margin over it."""
    name = "model_free_baseline"
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    train_pos, heldout = as_pairs(train_pos), as_pairs(heldout_pos)
    undirected = mode == "homogeneous"
    ev: dict = {}
    if eval_neg is None:
        rng = np.random.default_rng(seed)
        all_pos = np.concatenate([train_pos, heldout], axis=1)
        eval_neg = uniform_negatives(heldout, all_pos, n_rows, n_cols, rng, undirected)
        ev["negatives"] = "generated by lp_audit: uniform (no eval_neg supplied)"
    else:
        ev["negatives"] = "supplied by the user"
    corrected = model_free_floor(train_pos, heldout, eval_neg, n_rows, n_cols, mode)
    ev["floor_corrected"] = corrected
    reference, ref_label = corrected["floor"], "corrected (heuristics see training edges only)"
    if encoder_edges is not None:
        enc = as_pairs(encoder_edges)
        if _overlap(pair_keys(heldout, n_cols, undirected), pair_keys(enc, n_cols, undirected)):
            seen = model_free_floor(np.concatenate([train_pos, enc], axis=1), heldout,
                                    eval_neg, n_rows, n_cols, mode)
            ev["floor_as_evaluated"] = seen
            reference = seen["floor"]
            ref_label = "as evaluated (heuristics see the same leaked edges the encoder does)"
    ev["reference_floor"] = reference
    ev["reference"] = ref_label
    best = corrected["best"]
    if model_auroc is None:
        return Finding(name, "info",
                       f"Best model-free heuristic reaches AUROC {reference:.4f} ({ref_label}). "
                       "No model_auroc supplied, so no margin was computed: report your model's "
                       "margin over this floor alongside its AUROC.", ev)
    margin = float(model_auroc) - reference
    ev["model_auroc"], ev["margin"] = float(model_auroc), margin
    if margin < 0:
        return Finding(name, "error",
                       f"Your model (AUROC {model_auroc:.4f}) does not beat a no-learning "
                       f"heuristic ({best}, {reference:.4f}); margin {margin:+.4f}.", ev)
    if margin < thr["margin_warn"]:
        return Finding(name, "warn",
                       f"Margin over the model-free floor is only {margin:+.4f} "
                       f"(model {model_auroc:.4f} vs {best} {reference:.4f}); "
                       f"below {thr['margin_warn']}, so most of the number may be graph structure.", ev)
    return Finding(name, "ok", f"Model clears the model-free floor by {margin:+.4f} "
                   f"({model_auroc:.4f} vs {best} {reference:.4f}).", ev)


def check_dead_columns(n_rows, n_cols, train_pos, heldout_pos, *, eval_neg=None,
                       thresholds=None, undirected=False) -> Finding:
    """Candidate columns (and rows) with no positive anywhere. Negatives drawn there are
    separable by degree alone, which inflates the model-free floor (paper's Table 6 /
    HMDD topology audit)."""
    name = "dead_candidates"
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    allp = np.concatenate([as_pairs(train_pos), as_pairs(heldout_pos)], axis=1)
    cdeg = column_degree(allp, n_cols, undirected)
    rdeg = np.bincount(_cols(allp[::-1], undirected), minlength=n_rows)
    ev = {
        "n_rows": int(n_rows), "n_columns": int(n_cols), "n_positives": int(allp.shape[1]),
        "density": float(allp.shape[1] / (n_rows * n_cols)),
        "dead_column_fraction": float((cdeg == 0).mean()), "n_dead_columns": int((cdeg == 0).sum()),
        "dead_row_fraction": float((rdeg == 0).mean()), "column_degree_gini": gini(cdeg),
    }
    over = []
    if ev["dead_column_fraction"] > thr["dead_column_fraction_warn"]:
        over.append(f"{ev['dead_column_fraction']:.1%} of candidate columns have no positive "
                    f"({ev['n_dead_columns']:,} of {n_cols:,})")
    if eval_neg is not None:
        share = float((cdeg[_cols(as_pairs(eval_neg), undirected)] == 0).mean())
        ev["eval_negatives_on_dead_columns"] = share
        if share > thr["negatives_on_dead_columns_warn"]:
            over.append(f"{share:.1%} of evaluation negatives sit on such columns")
    if over:
        return Finding(name, "warn",
                       "; ".join(over) + ". A scorer that just ranks by column degree "
                       "separates these negatives for free, inflating the model-free floor; "
                       "restrict the candidate space to columns that can be positive, or report "
                       "the floor next to the model.", ev)
    return Finding(name, "ok", f"{ev['dead_column_fraction']:.1%} dead candidate columns.", ev)
