"""protocheck.io -- loaders that turn common dataset layouts into (2, N) (row, col) arrays.

File convention for pair files: one pair per line / row, columns = (row_index, col_index),
.npy or delimited text. A (2, N) .npy is also accepted (unambiguous unless N == 2).
"""

from __future__ import annotations

import numpy as np


def _from_array(a: np.ndarray, index_base: int = 0) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"pair file must be 2-D, got shape {a.shape}")
    if a.shape[1] == 2:
        a = a.T
    elif a.shape[0] != 2:
        raise ValueError(f"expected N x 2 or 2 x N pairs, got {a.shape}")
    return a.astype(np.int64) - index_base


def load_pairs(path: str, index_base: int = 0) -> np.ndarray:
    """Pairs file -> (2, N). `index_base=1` for 1-based indices."""
    if path.endswith(".npy"):
        return _from_array(np.load(path), index_base)
    a = np.loadtxt(path, delimiter="," if _sniff_comma(path) else None, dtype=float, ndmin=2)
    return _from_array(a, index_base)


def _sniff_comma(path: str) -> bool:
    with open(path) as fh:
        return "," in fh.readline()


def load_binary_matrix(path: str) -> tuple[np.ndarray, int, int]:
    """HMDD-style 0/1 matrix CSV (no header) -> (positives (2, N), n_rows, n_cols)."""
    m = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2)
    return np.argwhere(m > 0).T.astype(np.int64), m.shape[0], m.shape[1]


def load_labeled_pairs(path: str, index_base: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Whitespace/comma-separated (row, col, label) triples -> (positives, negatives)."""
    a = np.loadtxt(path, dtype=float, ndmin=2, delimiter="," if _sniff_comma(path) else None)
    if a.shape[1] != 3:
        raise ValueError(f"expected (row, col, label) triples, got {a.shape[1]} columns")
    pairs, y = a[:, :2].astype(np.int64) - index_base, a[:, 2]
    return pairs[y == 1].T, pairs[y != 1].T


def subtract_pairs(a: np.ndarray, b: np.ndarray, n_cols: int) -> np.ndarray:
    """Pairs in `a` that are not in `b` (e.g. matrix positives minus the held-out ones)."""
    keep = ~np.isin(a[0] * n_cols + a[1], b[0] * n_cols + b[1])
    return a[:, keep]


def from_ogb_split_edge(split_edge: dict, num_nodes: int) -> dict:
    """OGB link-prediction `split_edge` (ogbl-ddi style) -> kwargs for protocheck.audit.

    Uses the validation split as the held-out set. ogbl-ddi's encoder sees only the
    training edges, which is what encoder_edges is set to here; override if yours differs.
    Convert torch tensors with .numpy() first, or pass them as-is (np.asarray handles CPU
    tensors).
    """
    tr = np.asarray(split_edge["train"]["edge"]).T.astype(np.int64)
    va = np.asarray(split_edge["valid"]["edge"]).T.astype(np.int64)
    out = dict(n_rows=num_nodes, n_cols=num_nodes, train_pos=tr, heldout_pos=va,
               encoder_edges=tr, mode="homogeneous")
    if "edge_neg" in split_edge["valid"]:
        out["eval_neg"] = np.asarray(split_edge["valid"]["edge_neg"]).T.astype(np.int64)
    return out


def from_pyg_edge_split(split, fwd=("miRNA", "regulates", "gene"), which: str = "test") -> dict:
    """training.splits.EdgeSplit -> kwargs for protocheck.audit (lazy: needs torch only here).

    Lets this repo's own pipeline be audited: encoder_edges are the message-passing graph's
    forward edges, so a regression in the leak-free split shows up as a failing check.
    """
    mp = split.mp_graph[fwd]
    gene_type, mirna_type = fwd[2], fwd[0]
    held = split.test_sup if which == "test" else split.val_sup
    return dict(
        n_rows=split.mp_graph[mirna_type].num_nodes,
        n_cols=split.mp_graph[gene_type].num_nodes,
        train_pos=np.concatenate([mp.edge_index.cpu().numpy(),
                                  split.train_sup.cpu().numpy()], axis=1),
        heldout_pos=held.cpu().numpy(),
        encoder_edges=mp.edge_index.cpu().numpy(),
        mode="bipartite",
    )
