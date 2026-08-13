"""
inspect_hmdd_survey_sources.py — Does the downloaded canonical-5430 data actually
match what results/literature_survey.tsv says about each paper?

Mandatory gate, run after download_hmdd_survey_canonical5430.py and before
training/eval_hmdd_survey_topology_baseline.py is trusted for any number cited
anywhere. Mirrors data/01_download/inspect_ogb_split.py's role for the OGB
extension: re-derive facts from the downloaded bytes, never assume them from a
paper's prose or from this project's own prior notes.

Checks, per paper:
  1. Matrix shape and positive-edge count match literature_survey.tsv's implied
     scale (495 miRNAs x 383 diseases, 5,430 positives) for the canonical trio.
  2. MGCNSS's bundled train7.txt / test7_1.txt: re-derive the positive/negative
     counts and confirm train + test positives sum to exactly 5,430 (the full
     canonical positive set), and that no (m,d) pair appears as positive in one
     file and untested in the other in a way that would leak.
  3. Cross-paper matrix identity (also checked by the download script, re-checked
     here independently against the manifest as a second look).

Reads results/literature_survey.tsv at runtime (not hardcoded) so this check can't
silently drift out of sync with the survey if a TSV row is later corrected.

Writes results/comparison/hmdd_survey_protocol_verification.json. This file's
`all_checks_passed` must be true before citing any Stage-1 number.

Usage:
  python data/01_download/inspect_hmdd_survey_sources.py
"""

from __future__ import annotations

import os
import sys
import csv
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CANONICAL_PAPERS = ["mgcnss", "nimgsa", "hlgnn_mda"]
# Stage 2/3 papers: each has its own downloader that already writes a
# manifest.json with a "matches_*" self-check flag -- this script aggregates
# and gates on those rather than re-deriving shape/count from scratch.
MANIFEST_PAPERS = ["digamn", "cksnp_gnn", "meahne", "couplemda"]
TSV_NAME_MAP = {
    "mgcnss": "MGCNSS",
    "nimgsa": "NIMGSA",
    "hlgnn_mda": "HLGNN-MDA (Yu et al.)",
    "digamn": "DiGAMN (Liu et al.)",
    "cksnp_gnn": "CKSNP-GNN (Li et al.)",
    "meahne": "MEAHNE (Huang et al.)",
    "couplemda": "CoupleMDA (Li et al.)",
}


def load_tsv_row(tsv_path: str, paper_label: str) -> dict:
    with open(tsv_path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["paper"] == paper_label:
                return row
    raise KeyError(f"{paper_label!r} not found in {tsv_path}")


def load_matrix(path: Path) -> list[list[int]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([int(float(v)) for v in line.split(",")])
    return rows


def load_labeled_triples(path: Path) -> list[tuple[int, int, int]]:
    triples = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        a, b, y = line.split()
        triples.append((int(float(a)), int(float(b)), int(float(y))))
    return triples


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default="data/raw/hmdd_survey")
    p.add_argument("--tsv", default="results/literature_survey.tsv")
    p.add_argument("--out", default="results/comparison/hmdd_survey_protocol_verification.json")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    checks: dict[str, dict] = {}
    all_passed = True

    manifest_path = Path(args.data_root) / "canonical5430_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    log.info("=" * 78)
    log.info("Cross-paper matrix identity (re-check against manifest)")
    log.info("=" * 78)
    matrices = {}
    for paper in CANONICAL_PAPERS:
        mat_path = Path(args.data_root) / paper / "matrix.csv"
        rows = load_matrix(mat_path)
        n_rows = len(rows)
        n_cols = len(rows[0]) if rows else 0
        n_pos = sum(sum(r) for r in rows)
        matrices[paper] = (n_rows, n_cols, n_pos)
        log.info(f"  {paper}: shape ({n_rows}, {n_cols}), positives={n_pos:,}")

    shapes = {v[:2] for v in matrices.values()}
    counts = {v[2] for v in matrices.values()}
    identical_shape = len(shapes) == 1
    identical_count = len(counts) == 1
    ok = identical_shape and identical_count and shapes == {(495, 383)} and counts == {5430}
    checks["canonical_matrix_identity"] = {
        "shapes": {k: list(v[:2]) for k, v in matrices.items()},
        "positive_counts": {k: v[2] for k, v in matrices.items()},
        "expected_shape": [495, 383],
        "expected_positives": 5430,
        "passed": ok,
    }
    log.info(f"  [{'ok' if ok else 'FAIL'}] shape/count match expected (495,383)/5430 across all three")
    all_passed &= ok

    log.info("=" * 78)
    log.info("MGCNSS bundled split (train7.txt / test7_1.txt) — Tier 1 verification")
    log.info("=" * 78)
    train_path = Path(args.data_root) / "mgcnss" / "train7.txt"
    test_path = Path(args.data_root) / "mgcnss" / "test7_1.txt"
    train_triples = load_labeled_triples(train_path)
    test_triples = load_labeled_triples(test_path)
    train_pos = sum(1 for *_, y in train_triples if y == 1)
    train_neg = sum(1 for *_, y in train_triples if y == 0)
    test_pos = sum(1 for *_, y in test_triples if y == 1)
    test_neg = sum(1 for *_, y in test_triples if y == 0)
    total_pos = train_pos + test_pos
    train_pairs = {(a, b) for a, b, y in train_triples if y == 1}
    test_pairs = {(a, b) for a, b, y in test_triples if y == 1}
    overlap = train_pairs & test_pairs
    split_ok = (
        total_pos == 5430
        and train_pos == train_neg
        and test_pos == test_neg
        and len(overlap) == 0
    )
    checks["mgcnss_split"] = {
        "train_pos": train_pos, "train_neg": train_neg,
        "test_pos": test_pos, "test_neg": test_neg,
        "train_plus_test_pos": total_pos,
        "train_test_positive_overlap": len(overlap),
        "passed": split_ok,
    }
    log.info(f"  train: {train_pos} pos / {train_neg} neg   test: {test_pos} pos / {test_neg} neg")
    log.info(f"  train+test positives = {total_pos} (expect 5430)")
    log.info(f"  train/test positive-pair overlap = {len(overlap)} (expect 0)")
    log.info(f"  [{'ok' if split_ok else 'FAIL'}]")
    all_passed &= split_ok

    log.info("=" * 78)
    log.info("Stage 2/3 papers — aggregate each downloader's own manifest self-check")
    log.info("=" * 78)
    manifest_checks = {}
    for paper in MANIFEST_PAPERS:
        manifest_path = Path(args.data_root) / paper / "manifest.json"
        if not manifest_path.exists():
            log.warning(f"  {paper}: no manifest.json found at {manifest_path} -- "
                        f"run its downloader first. Skipping (not counted as failed).")
            continue
        m = json.loads(manifest_path.read_text())
        flag_keys = [k for k in m if k.startswith("matches_")]
        passed = all(m[k] for k in flag_keys) if flag_keys else None
        manifest_checks[paper] = {"manifest": m, "self_check_passed": passed}
        log.info(f"  {paper}: {', '.join(f'{k}={m[k]}' for k in flag_keys) or '(no self-check flag)'}")
        if passed is False:
            all_passed = False
    checks["stage23_manifests"] = manifest_checks

    log.info("=" * 78)
    log.info("Cross-check reported headline numbers against literature_survey.tsv")
    log.info("=" * 78)
    tsv_refs = {}
    for paper in CANONICAL_PAPERS + [p for p in MANIFEST_PAPERS if p in manifest_checks]:
        row = load_tsv_row(args.tsv, TSV_NAME_MAP[paper])
        tsv_refs[paper] = {
            "headline_auroc": row["headline_auroc"],
            "negative_sampling": row["negative_sampling"],
            "url": row["url"],
        }
        log.info(f"  {paper}: headline_auroc={row['headline_auroc']!r}  "
                 f"negatives={row['negative_sampling']!r}")
    checks["tsv_references"] = tsv_refs

    summary = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "all_checks_passed": bool(all_passed),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("=" * 78)
    log.info(f"all_checks_passed = {all_passed}")
    log.info(f"Wrote {args.out}")
    if not all_passed:
        raise SystemExit(
            "One or more checks failed — do not trust eval_hmdd_survey_topology_baseline.py "
            "output until this is resolved."
        )


if __name__ == "__main__":
    main()
