"""
download_hmdd_survey_cksnp.py — Fetch CKSNP-GNN's data for the HMDD topology-
baseline audit, from the shared institutional mirror the paper cites.

CKSNP-GNN (Li et al., Genes 2022) links its data+code as a single zip at
http://public.aibiochem.net/DNA_RNA/Genes_Human-miRNA-disease-Associations/code.zip.
That zip's own bundled README.md/main.py describe a DIFFERENT model ("GAEMDA", a
graph auto-encoder) -- the code in this mirror is almost certainly not CKSNP-GNN's
own code. We do NOT use or claim that code. What we DO trust is the DATA: the zip's
data/all_mirna_disease_pairs.csv contains exactly 16,427 positive-labeled rows,
matching CKSNP-GNN's own reported count verbatim ("16,427 negative samples were
randomly selected from the unknown associations" -- literature_survey.tsv), on this
institutional mirror that multiple miRNA-disease papers in this literature reuse.
Byte-level count match is the confidence signal here, not code provenance.

The pairs file is the FULL labeled cartesian product: 790,177 rows = 901 miRNAs x
877 diseases exactly (verified: 901*877 = 790,177), with 16,427 positives (label=1)
and 773,750 negatives (label=0, i.e. every non-positive pair, not a sampled subset).
This is converted here to the same dense-matrix CSV format the other Tier-2 papers
use, so no eval-script changes were needed for this paper.

Only the two files actually needed are extracted from the 112MB zip (~290MB of
unrelated similarity-feature CSVs inside it are discarded).

Tier 2 (exact positives, high confidence via count match; no bundled split -- CKSNP-
GNN's own train/test split is generated downstream at their stated ratio).

Usage:
  python data/01_download/download_hmdd_survey_cksnp.py
"""

from __future__ import annotations

import os
import sys
import json
import zipfile
import hashlib
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ZIP_URL = "http://public.aibiochem.net/DNA_RNA/Genes_Human-miRNA-disease-Associations/code.zip"
OUT_DIR = "data/raw/hmdd_survey/cksnp_gnn"
EXPECTED_SHAPE = (901, 877)
EXPECTED_POSITIVES = 16427


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--zip_cache", default=None,
                   help="Reuse an already-downloaded copy of the zip instead of "
                        "re-fetching (it's 112MB, slow on a metered link).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.zip_cache and Path(args.zip_cache).exists():
        zip_path = Path(args.zip_cache)
        log.info(f"Reusing cached zip at {zip_path}")
    else:
        zip_path = out_dir / "_code.zip.tmp"
        log.info(f"Downloading {ZIP_URL} (112MB, may take a while)...")
        subprocess.run(
            ["curl", "-sS", "--fail", "--max-time", "300", "-o", str(zip_path), ZIP_URL],
            check=True,
        )
        log.info(f"  downloaded {zip_path.stat().st_size:,} bytes")

    with zipfile.ZipFile(zip_path) as zf:
        pairs_raw = zf.read("data/all_mirna_disease_pairs.csv")
        readme_raw = zf.read("README.md")
        main_py_raw = zf.read("main.py")

    if not args.zip_cache:
        zip_path.unlink()  # discard the 112MB zip, we only wanted 2 files from it

    (out_dir / "README.md").write_bytes(readme_raw)
    (out_dir / "main.py").write_bytes(main_py_raw)
    log.info(f"  bundled README/main.py model name: "
             f"{readme_raw.decode('utf-8', errors='replace').splitlines()[0]!r} "
             f"(recorded for the code/README-mismatch caveat, not used for scoring)")

    log.info("Converting all_mirna_disease_pairs.csv (edge-list + label) to a dense "
             "matrix CSV...")
    rows = pairs_raw.decode("utf-8").splitlines()
    max_m, max_d = 0, 0
    triples = []
    for ln in rows:
        if not ln.strip():
            continue
        m, d, y = ln.split(",")
        m, d, y = int(m), int(d), int(y)
        max_m, max_d = max(max_m, m), max(max_d, d)
        if y == 1:
            triples.append((m, d))
    n_rows, n_cols = max_m, max_d  # 1-indexed in the source; matrix uses 1..max as size
    n_pos = len(triples)
    log.info(f"  parsed shape ({n_rows}, {n_cols}), positives={n_pos:,} "
             f"(expect {EXPECTED_SHAPE}, {EXPECTED_POSITIVES:,})")

    ok = (n_rows, n_cols) == EXPECTED_SHAPE and n_pos == EXPECTED_POSITIVES
    if not ok:
        log.warning("  [WARNING] shape/count do NOT match CKSNP-GNN's own reported "
                     "16,427 -- do not trust this data until resolved.")

    matrix = [[0] * n_cols for _ in range(n_rows)]
    for m, d in triples:
        matrix[m - 1][d - 1] = 1  # source is 1-indexed
    matrix_csv = "\n".join(",".join(str(v) for v in row) for row in matrix)
    (out_dir / "matrix.csv").write_text(matrix_csv)
    log.info(f"  wrote {out_dir / 'matrix.csv'}")

    manifest = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source_zip_url": ZIP_URL,
        "source_zip_sha256": hashlib.sha256(Path(args.zip_cache).read_bytes()).hexdigest()
            if args.zip_cache else None,
        "shape": [n_rows, n_cols],
        "n_positives": n_pos,
        "matches_cksnp_gnn_reported_count": ok,
        "caveat": (
            "This zip's bundled README/main.py describe a different model "
            "('GAEMDA'), not CKSNP-GNN -- the code is not used or trusted, only "
            "the data, whose positive count (16,427) matches CKSNP-GNN's own "
            "reported figure exactly."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(f"Wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
