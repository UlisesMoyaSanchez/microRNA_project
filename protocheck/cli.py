"""
Command line: python -m protocheck ...

Three ways to describe a dataset (pair files are N x 2 (row, col), .npy or CSV):

  1. Edge files
       --train-pos F --heldout-pos F --n-rows N --n-cols M
       [--encoder-edges F] [--encoder-edges-rev F] [--train-neg F] [--eval-neg F]
  2. HMDD-style matrix: the full positive matrix plus the held-out pairs
       --matrix M.csv --heldout-pos F [--encoder-edges matrix]
     (train positives = matrix minus held-out; `--encoder-edges matrix` states that your
      model's graph input was the whole matrix, the conventional protocol)
  3. (row, col, label) triples
       --labeled-train F --labeled-test F --n-rows N --n-cols M [--index-base 1]

Exit status: 0 clean, 1 if any check fails, 2 on a usage error. --fail-on-warn also
fails on warnings.
"""

from __future__ import annotations

import argparse
import sys

from . import io as lio
from .audit import audit


def _build(a: argparse.Namespace) -> dict:
    ib = a.index_base
    kw: dict = dict(mode=a.mode, model_auroc=a.model_auroc, seed=a.seed)
    if a.matrix:
        pos, n_rows, n_cols = lio.load_binary_matrix(a.matrix)
        if not a.heldout_pos:
            raise ValueError("--matrix needs --heldout-pos")
        held = lio.load_pairs(a.heldout_pos, ib)
        kw.update(n_rows=n_rows, n_cols=n_cols, heldout_pos=held,
                  train_pos=lio.subtract_pairs(pos, held, n_cols))
        if a.encoder_edges == "matrix":
            kw["encoder_edges"] = pos
        elif a.encoder_edges:
            kw["encoder_edges"] = lio.load_pairs(a.encoder_edges, ib)
    elif a.labeled_train:
        if not (a.labeled_test and a.n_rows and a.n_cols):
            raise ValueError("--labeled-train needs --labeled-test, --n-rows, --n-cols")
        tp, tn = lio.load_labeled_pairs(a.labeled_train, ib)
        hp, hn = lio.load_labeled_pairs(a.labeled_test, ib)
        kw.update(n_rows=a.n_rows, n_cols=a.n_cols, train_pos=tp, heldout_pos=hp,
                  train_neg=tn, eval_neg=hn)
    else:
        if not (a.train_pos and a.heldout_pos and a.n_rows and a.n_cols):
            raise ValueError("give --train-pos, --heldout-pos, --n-rows, --n-cols "
                             "(or --matrix, or --labeled-train)")
        kw.update(n_rows=a.n_rows, n_cols=a.n_cols, train_pos=lio.load_pairs(a.train_pos, ib),
                  heldout_pos=lio.load_pairs(a.heldout_pos, ib))
    # Optional files override whatever the layout above supplied.
    for flag, key in (("encoder_edges_rev", "encoder_edges_rev"), ("train_neg", "train_neg"),
                      ("eval_neg", "eval_neg")):
        if getattr(a, flag):
            kw[key] = lio.load_pairs(getattr(a, flag), ib)
    if a.encoder_edges and a.encoder_edges != "matrix":
        kw["encoder_edges"] = lio.load_pairs(a.encoder_edges, ib)
    if a.train_sampler or a.eval_sampler:
        kw["declared_samplers"] = {"train": a.train_sampler, "eval": a.eval_sampler}
    return kw


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m protocheck", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("train-pos", "heldout-pos", "encoder-edges", "encoder-edges-rev", "train-neg",
              "eval-neg", "matrix", "labeled-train", "labeled-test"):
        p.add_argument(f"--{f}")
    p.add_argument("--n-rows", type=int)
    p.add_argument("--n-cols", type=int)
    p.add_argument("--index-base", type=int, default=0, choices=[0, 1])
    p.add_argument("--mode", default="bipartite", choices=["bipartite", "homogeneous"])
    p.add_argument("--model-auroc", type=float, help="The AUROC you report, to compute the margin.")
    p.add_argument("--train-sampler", help="Declared train negative sampler (string).")
    p.add_argument("--eval-sampler", help="Declared eval negative sampler (string).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", help="Also write the full report here.")
    p.add_argument("--fail-on-warn", action="store_true")
    a = p.parse_args(argv)
    try:
        report = audit(**_build(a))
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(report.render())
    if a.json:
        report.save(a.json)
    return 1 if report.has_errors or (a.fail_on_warn and report.has_warnings) else 0
