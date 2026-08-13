"""
download_hmdd_survey_meahne.py — Fetch MEAHNE's (Huang et al., Life 2022)
miRNA-disease association data for the HMDD topology-baseline audit.

MEAHNE's repo (yyx-hc/MEAHNE) ships a raw edge list (DATA/association/
"mir_disease .csv" -- note the literal space in the upstream filename) over a much
larger, non-canonical vocabulary than the other papers in this cluster: 1,296
miRNAs x 11,783 diseases (per DATA/node_list/{mir,disease}.csv), 17,972 positive
edges, 0-indexed already. Converted here to the same dense-matrix CSV format the
other Tier-2 papers use (~15.3M cells, ~30MB as text) so no eval-script changes
were needed for this paper.

Tier 2 (exact positive edge list; no bundled split -- MEAHNE's own 70/10/20 split
is described but not published as a file, so one is generated downstream at their
stated 1:1 ratio).

Usage:
  python data/01_download/download_hmdd_survey_meahne.py
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

REPO = "yyx-hc/MEAHNE"
COMMIT = "96f3faa6e143b6e70ef29c9692ffa399c2b5780a"
OUT_DIR = "data/raw/hmdd_survey/meahne"


def fetch(repo: str, commit: str, path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{urllib.parse.quote(path)}"
    result = subprocess.run(
        ["curl", "-sS", "--fail", "--max-time", "60", url],
        capture_output=True, check=True,
    )
    return result.stdout


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default=OUT_DIR)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Fetching mir.csv / disease.csv (node counts) and mir_disease.csv "
             f"(positive edges) from {REPO} @ {COMMIT[:12]}...")
    mir_raw = fetch(REPO, COMMIT, "DATA/node_list/mir.csv")
    disease_raw = fetch(REPO, COMMIT, "DATA/node_list/disease.csv")
    edges_raw = fetch(REPO, COMMIT, "DATA/association/mir_disease .csv")

    n_mirna = len(mir_raw.decode("utf-8").splitlines()) - 1  # minus header
    n_disease = len(disease_raw.decode("utf-8").splitlines()) - 1
    log.info(f"  {n_mirna:,} miRNAs, {n_disease:,} diseases")

    edges = []
    for i, ln in enumerate(edges_raw.decode("utf-8").splitlines()):
        if i == 0 or not ln.strip():
            continue  # header: ",mir,disease"
        _, m, d = ln.split(",")
        edges.append((int(m), int(d)))
    log.info(f"  {len(edges):,} positive edges (expect 17,972)")

    matrix = [[0] * n_disease for _ in range(n_mirna)]
    for m, d in edges:
        matrix[m][d] = 1
    matrix_csv = "\n".join(",".join(str(v) for v in row) for row in matrix)
    dst = out_dir / "matrix.csv"
    dst.write_text(matrix_csv)
    log.info(f"  wrote {dst} ({dst.stat().st_size:,} bytes)")

    ok = (n_mirna, n_disease, len(edges)) == (1296, 11783, 17972)
    if not ok:
        log.warning("  [WARNING] shape/count do not match the expected "
                     "(1296, 11783)/17,972 -- verify before trusting downstream results.")

    manifest = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "repo": REPO, "commit": COMMIT,
        "shape": [n_mirna, n_disease],
        "n_positives": len(edges),
        "matches_expected": ok,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(f"Wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
