"""
download_hmdd_survey_canonical5430.py — Fetch the shared HMDD-derived dataset behind
three of this project's literature-survey papers (MGCNSS, NIMGSA, HLGNN-MDA).

Part of the HMDD topology-baseline audit: quantifying, on the surveyed papers' own
data, how much a no-learning topology heuristic (training/eval_topology_baseline.py's
build_scorers) closes the gap to their reported AUROCs. See
results/literature_survey.tsv for each paper's row and results/HMDD_TOPOLOGY_AUDIT.md
(written after training/eval_hmdd_survey_topology_baseline.py runs) for the results.

Why these three together: MGCNSS, NIMGSA, and HLGNN-MDA all build on the identical
495 miRNA x 383 disease, 5,430-positive HMDD v2.0-derived association matrix -- a
benchmark recycled across this subfield. Verified this session by byte-diffing
MGCNSS's data/miRNA_disease_matrix.csv against NIMGSA's m-d.txt (identical) and by
summing HLGNN-MDA's association.txt (also 5,430 positives over the same 495x383
shape). One fetch of the canonical matrix serves all three; each paper's own
evaluation protocol (see per-paper notes below) determines how it's split/negatively
sampled downstream in eval_hmdd_survey_topology_baseline.py.

Per-paper protocol notes (each verified against that paper's own code or PMC text,
not assumed from a filename):

  MGCNSS   Tier 1 (exact split). Link_Prediction.py's default call
           (predict_model(..., dataset=1)) reads data/train/train7.txt as the
           validation set and data/train/test7_1.txt as the test set --
           src/link_prediction_evaluate.py::load_testing_data() confirms both files
           carry an explicit (node1, node2, label) triple. Verified: train7.txt is
           8,688 rows (4,344 pos / 4,344 neg), test7_1.txt is 2,172 rows (1,086 pos /
           1,086 neg); 4,344 + 1,086 = 5,430, the full positive set, split 80/20 --
           i.e. this is edge-level CV over the SAME 5,430 positives (no held-out
           node holdout), 1:1 uniform negatives, fixed pos/neg pairing already
           chosen by the paper. Their headline AUROC: 0.9874 (literature_survey.tsv).

  NIMGSA   Tier 2 (exact matrix, no bundled split). fivefoldcv.py builds folds at
           runtime from m-d.txt; no fixed split file is published. Reconstructed
           downstream as 5-fold CV, uniform 1:1 negatives (matches "not described"
           in literature_survey.tsv -- NIMGSA's own paper doesn't state its negative
           protocol explicitly; 1:1 is the field default we use as the honest,
           clearly-labeled approximation). Their headline AUROC: 0.9354.

  HLGNN-MDA  Tier 2 (exact matrix, no bundled split usable for the headline number).
           The repo DOES bundle train5430_idx.txt / test2792_idx.txt, but tracing
           Python/Main.py shows those are positive-only edge lists for an
           alternate, fixed-split SEAL invocation (--train-name/--test-name), which
           is NOT the invocation that produces the paper's headline number. The
           paper's own text (PMC9657597) states the headline result (0.93086 AUROC,
           hop4) comes from "randomly divided ... into ten parts" -- ordinary
           10-fold CV over the full 5,430 pos + 5,430 neg (1:1 uniform) set, the
           same protocol as NIMGSA. Reconstructed downstream as such; the bundled
           split files are fetched for provenance but not used for scoring.

Usage:
  python data/01_download/download_hmdd_survey_canonical5430.py
"""

from __future__ import annotations

import os
import sys
import json
import hashlib
import logging
import argparse
import subprocess
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUT_ROOT = "data/raw/hmdd_survey"

# Commit-pinned, not HEAD/main/master -- an upstream push must not silently change
# what this script calls "exact". Re-pin deliberately (bump the SHA + re-verify
# checksums) if a paper's repo is ever intentionally re-synced.
SOURCES = {
    "mgcnss": {
        "repo": "15136943622/MGCNSS",
        "commit": "f8a87e698b78696fb8cfe930a2b86ae53e61c81a",
        "files": {
            "data/miRNA_disease_matrix.csv": "matrix.csv",
            "data/train/train7.txt": "train7.txt",
            "data/train/test7_1.txt": "test7_1.txt",
        },
    },
    "nimgsa": {
        "repo": "zhanglabNKU/NIMGSA",
        "commit": "5bf10a93d32286a84bed641d0f85be19f2ba011f",
        "files": {
            "m-d.txt": "matrix.csv",
        },
    },
    "hlgnn_mda": {
        "repo": "LiangYu-Xidian/HLGNN-MDA",
        "commit": "e3ac824017b8d1bf246e7f695551a8874d231724",
        "files": {
            "Python/data/5430dataset/1.miRNA-disease associations/association.txt": "matrix.csv",
            "Python/data/train5430_idx.txt": "train5430_idx.txt",
            "Python/data/test2792_idx.txt": "test2792_idx.txt",
        },
    },
}


def fetch(repo: str, commit: str, path: str) -> bytes:
    # urllib.request hangs for minutes per file in this environment (DNS/IPv6
    # fallback stall observed directly -- curl on the identical URL returns in
    # well under a second), so shell out to curl instead. --fail so a 404 raises
    # rather than silently writing an HTML error page as if it were data.
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{urllib.parse.quote(path)}"
    result = subprocess.run(
        ["curl", "-sS", "--fail", "--max-time", "30", url],
        capture_output=True, check=True,
    )
    return result.stdout


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_matrix(raw: bytes, sep_hint: str) -> bytes:
    """MGCNSS/NIMGSA ship comma-separated, HLGNN-MDA ships whitespace-separated --
    normalize all three to comma-separated so downstream code has one format."""
    text = raw.decode("utf-8")
    if sep_hint == "whitespace":
        lines = [ln for ln in text.splitlines() if ln.strip()]
        text = "\n".join(",".join(ln.split()) for ln in lines)
    return text.encode("utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_root", default=OUT_ROOT)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    matrix_bytes: dict[str, bytes] = {}

    for paper, spec in SOURCES.items():
        paper_dir = Path(args.out_root) / paper
        paper_dir.mkdir(parents=True, exist_ok=True)
        log.info("=" * 78)
        log.info(f"{paper}  ({spec['repo']} @ {spec['commit'][:12]})")
        for src_path, dst_name in spec["files"].items():
            log.info(f"  fetching {src_path} -> {dst_name}")
            raw = fetch(spec["repo"], spec["commit"], src_path)
            if dst_name == "matrix.csv" and paper == "hlgnn_mda":
                raw = normalize_matrix(raw, sep_hint="whitespace")
            dst = paper_dir / dst_name
            dst.write_bytes(raw)
            log.info(f"    wrote {dst} ({len(raw):,} bytes, sha256={sha256(raw)[:16]}...)")
            if dst_name == "matrix.csv":
                matrix_bytes[paper] = raw

    # Cross-check: the three matrices should carry identical VALUES (mod harmless
    # formatting differences -- HLGNN-MDA ships "1.000...e+00" scientific-notation
    # floats where MGCNSS/NIMGSA ship plain "1"/"0", and one file lacks a trailing
    # newline). Raw byte hashes will therefore legitimately differ; compare parsed
    # int values instead, which is what actually matters for treating these as one
    # shared dataset. A parsed-value mismatch here would mean one paper's HMDD
    # snapshot has actually drifted -- that must not pass silently.
    log.info("=" * 78)
    log.info("Cross-checking canonical-5430 matrix identity across the three papers...")
    hashes = {p: sha256(b) for p, b in matrix_bytes.items()}
    for p, h in hashes.items():
        log.info(f"  {p}: sha256={h}  (raw-byte hash -- formatting differs by paper, see note above)")

    def parsed_rows(raw: bytes) -> list[list[int]]:
        return [
            [int(float(v)) for v in ln.split(",")]
            for ln in raw.decode("utf-8").splitlines() if ln.strip()
        ]

    parsed = {p: parsed_rows(b) for p, b in matrix_bytes.items()}
    value_hashes = {p: sha256(json.dumps(rows).encode()) for p, rows in parsed.items()}
    unique = set(value_hashes.values())
    if len(unique) == 1:
        log.info("  [ok] all three matrices are value-identical once parsed "
                 "(formatting differences are cosmetic only).")
    else:
        log.warning(
            "  [WARNING] parsed VALUES differ across papers despite matching "
            "provenance expectations -- do NOT treat this as one shared dataset "
            "until the difference is understood. Re-run data/01_download/"
            "inspect_hmdd_survey_sources.py and inspect by hand before trusting "
            "any downstream comparison."
        )

    manifest = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "sources": SOURCES,
        "matrix_sha256_raw_bytes": hashes,
        "matrix_sha256_parsed_values": value_hashes,
        "matrices_identical": len(unique) == 1,
    }
    manifest_path = Path(args.out_root) / "canonical5430_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
