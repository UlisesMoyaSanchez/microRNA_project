"""
eval_celltype_baseline.py — Is 0.9916 a real result, or a reconstruction of its own label?

The headline cell-type number (test accuracy 0.9916,
results/comparison/heldout_grid_checkpoints_v2_edgesplit_test.json) has been cited
throughout this project as the one result "unaffected" by the link-prediction evaluation
problems. It has never had a no-learning control next to it, unlike every link-prediction
number (gene_degree, adamic_adar, pref_attach, common_neigh, mlp, random).

There is a specific reason to check: cell_type labels are the per-cell argmax of
sc.tl.score_genes marker scores (data/02_preprocess/preprocess_scrna.py:34-58), computed
from expression. The cell node feature the model sees is X_pca — a PCA of that same
expression matrix (data/03_build_graph/build_heterograph.py:184-191). If a classifier with
no graph and no gradient-based learning at all reaches anywhere near 0.9916 on raw X_pca,
the graph and the transformer are buying nothing on this task, and 0.9916 mostly measures
how separable the marker-gene-defined classes already are in PCA space.

Two controls, both on graph["cell"].x alone (no message passing):

  nearest_centroid     Zero-hyperparameter, purely geometric — the closest analog in
                        spirit to gene_degree/adamic_adar (a formula, not a fit).
  logistic_regression   A fitted-but-simple, convex, graph-free classifier (with feature
                        scaling for numerical conditioning only — this does not add
                        representational capacity beyond a linear decision boundary).

Plus one diagnostic that is not a baseline: how much of the stored cell_type label is
literally reproducible as argmax(marker score) read back from the processed h5ad. Expected
near 1.0 by construction; it documents the label-generation mechanism in the artifact
itself rather than requiring a reader to go find preprocess_scrna.py.

Usage (DGX, CPU-only):
  python training/eval_celltype_baseline.py --config configs/config_v2_edgesplit.yaml --split test
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import yaml
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.train import load_graph, split_graph

# config_v2_edgesplit.yaml, test, degree_matched row — the headline number this script
# checks. cell_acc does not depend on the link negative sampler (see the "uniform" row in
# the same file, 0.9915879343356665): both are reported here so a reader can see the
# spread is noise-level, not a second finding.
HGT_REFERENCE = {
    "hgt_cell_acc_degree_matched": 0.9916477077944339,
    "hgt_cell_acc_uniform": 0.9915879343356665,
    "hgt_cell_f1": None,  # not recorded for this checkpoint/split in the source artifact
    "source_artifact": "results/comparison/heldout_grid_checkpoints_v2_edgesplit_test.json",
    "source_checkpoint": "checkpoints_v2_edgesplit/best_model.pt",
}


def git(*args: str, cwd: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                               text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def provenance(root: str) -> dict:
    return {
        "git_sha": git("rev-parse", "HEAD", cwd=root),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no", cwd=root)),
        "git_untracked_files": len(
            [ln for ln in git("status", "--porcelain", cwd=root).splitlines()
             if ln.startswith("??")]),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def label_construction_check(processed_dir: str, y: np.ndarray, eval_mask: np.ndarray,
                              cell_type_labels: list[str], log: logging.Logger) -> dict:
    """How much of the stored cell_type label is literally argmax(marker score)?

    Reads the score_<celltype> columns straight from the processed h5ad — these are
    never dropped between preprocess_scrna.py and build_heterograph.py (confirmed by
    reading both files). Row order must match graph["cell"] node order: both are built
    from a single, un-reordered read of the same AnnData object, so indexing by the same
    eval_mask used for X/y is valid.
    """
    h5ad_path = os.path.join(processed_dir, "scrna_processed.h5ad")
    if not os.path.exists(h5ad_path):
        log.warning(f"{h5ad_path} not found — skipping label-construction check")
        return {"skipped": True, "reason": f"{h5ad_path} not found"}

    import scanpy as sc
    adata = sc.read_h5ad(h5ad_path)

    if adata.n_obs != len(y):
        log.warning(
            f"h5ad has {adata.n_obs:,} cells, graph has {len(y):,} — row order cannot be "
            "trusted, skipping label-construction check"
        )
        return {"skipped": True, "reason": "cell count mismatch between h5ad and graph"}

    score_cols = [c for c in adata.obs.columns if c.startswith("score_")]
    if not score_cols:
        log.warning("No score_<celltype> columns in the h5ad — skipping")
        return {"skipped": True, "reason": "no score_ columns found"}

    score_df = adata.obs[score_cols].copy()
    score_df.columns = [c.replace("score_", "") for c in score_cols]
    recomputed = score_df.idxmax(axis=1).to_numpy()

    stored_names = np.array([cell_type_labels[i] for i in y])
    match = recomputed == stored_names
    match_eval = match[eval_mask]

    return {
        "skipped": False,
        "note": "cell_type = argmax(sc.tl.score_genes marker scores); "
                "see data/02_preprocess/preprocess_scrna.py:34-58",
        "score_columns_found": len(score_cols),
        "argmax_matches_stored_label_fraction_all_cells": float(match.mean()),
        "argmax_matches_stored_label_fraction_eval_split": float(match_eval.mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/config_v2_edgesplit.yaml")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--out", default=None,
                   help="Default: results/comparison/celltype_baseline_<config-stem>_<split>.json")
    args = p.parse_args()

    config_stem = Path(args.config).stem
    if args.out is None:
        args.out = f"results/comparison/celltype_baseline_{config_stem}_{args.split}.json"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    tcfg = cfg["training"]
    seed = cfg["project"]["seed"]
    root = str(Path(__file__).resolve().parents[1])

    graph, index_maps = load_graph(cfg["data"]["graphs_dir"])
    train_mask, val_mask, test_mask = split_graph(
        graph, tcfg["val_ratio"], tcfg["test_ratio"], seed
    )
    eval_mask = test_mask if args.split == "test" else val_mask

    X = graph["cell"].x.numpy()
    y = graph["cell"].y.numpy()
    train_np = train_mask.numpy()
    eval_np = eval_mask.numpy()

    log.info(f"Cells — train: {train_np.sum():,}, {args.split}: {eval_np.sum():,}, "
             f"features: X_pca dim {X.shape[1]}")

    X_train, y_train = X[train_np], y[train_np]
    X_eval, y_eval = X[eval_np], y[eval_np]

    results: dict[str, dict[str, float]] = {}

    log.info("Fitting nearest_centroid (no gradient learning, purely geometric)...")
    nc = NearestCentroid()
    nc.fit(X_train, y_train)
    pred_nc = nc.predict(X_eval)
    results["nearest_centroid"] = {
        "acc": float(accuracy_score(y_eval, pred_nc)),
        "f1_macro": float(f1_score(y_eval, pred_nc, average="macro", zero_division=0)),
    }

    log.info("Fitting logistic_regression (scaled, convex, graph-free)...")
    lr = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_eval)
    results["logistic_regression"] = {
        "acc": float(accuracy_score(y_eval, pred_lr)),
        "f1_macro": float(f1_score(y_eval, pred_lr, average="macro", zero_division=0)),
    }

    log.info("=" * 78)
    log.info("CELL-TYPE, NO-GRAPH BASELINES — no message passing, X_pca only")
    log.info("=" * 78)
    log.info(f"{'baseline':<24}{'acc':>12}{'f1_macro':>14}")
    log.info("-" * 78)
    for name, r in results.items():
        log.info(f"{name:<24}{r['acc']:>12.4f}{r['f1_macro']:>14.4f}")
    log.info("-" * 78)
    log.info(f"HGT reference (degree_matched, {config_stem}, {args.split}): "
              f"{HGT_REFERENCE['hgt_cell_acc_degree_matched']:.4f}")
    log.info("")
    log.info("Read: if either baseline lands near the HGT reference, the graph/transformer")
    log.info("is not earning its complexity on this task, and 0.9916 mostly measures how")
    log.info("separable the marker-gene-defined classes already are in PCA space.")
    log.info("=" * 78)

    ct_check = label_construction_check(
        cfg["data"]["processed_dir"], y, eval_np, index_maps["cell_type_labels"], log
    )

    summary = {
        "split": args.split,
        "config": args.config,
        "n_train": int(train_np.sum()),
        "n_eval": int(eval_np.sum()),
        "reference": HGT_REFERENCE,
        "results": results,
        "label_construction_check": ct_check,
        "provenance": provenance(root),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
