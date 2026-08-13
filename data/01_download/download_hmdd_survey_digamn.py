"""
download_hmdd_survey_digamn.py — Fetch DiGAMN's (Liu et al., BMC Genomics 2024) own
"Data1" miRNA-disease matrix for the HMDD topology-baseline audit.

DiGAMN's repo is `yinboliu-git/GAMN` (confirmed match: authors' names in the repo's
commit history and `setproctitle.setproctitle("DiGAMN_Bio_gsearch")` in main_bio.py
both match; the paper's title differs slightly in wording from the README's but the
author list in manuscript/jbi/references.bib's liu2024digamn entry matches exactly).

Important correction found while tracing this paper: main_bio.py's own default run
(data_id=0) uses the repo's "AMHMDA" folder (853x591/12,446 positives) -- but that is
NOT what produces the paper's headline number. The paper's own text (PMC11610307)
states: "DiGAMN excelled, achieving AUC scores of 96.35%, 96.10%, 96.01%, and 95.89%
on the Data1 to Data4 datasets, respectively" with Data1 = 917 miRNAs, 792 diseases,
14,550 associations (Table 1) -- confirmed by fetching data/MDA-CF/m_d.csv from the
repo, which has exactly that shape and positive count. Data1 = MDA-CF, not AMHMDA.
Do not trust the code's own default run target over the paper's reported table.

Tier 2 (exact matrix, no bundled split): 1:1 balanced negatives were used for the
96.35% headline number ("we randomly selected an equal number of miRNA-disease pairs
with no recorded associations as negative samples"), but no split file is published,
so a split is generated downstream in eval_hmdd_survey_topology_baseline.py.

Usage:
  python data/01_download/download_hmdd_survey_digamn.py
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

REPO = "yinboliu-git/GAMN"
COMMIT = "6695f4e4b4f3ff7060d934a43807ae25925c574e"
SRC_PATH = "data/MDA-CF/m_d.csv"  # = paper's "Data1" (917x792/14,550) -- see docstring
OUT_DIR = "data/raw/hmdd_survey/digamn"


def fetch(repo: str, commit: str, path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{urllib.parse.quote(path)}"
    result = subprocess.run(
        ["curl", "-sS", "--fail", "--max-time", "30", url],
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

    log.info(f"Fetching {SRC_PATH} from {REPO} @ {COMMIT[:12]} (paper's Data1)...")
    raw = fetch(REPO, COMMIT, SRC_PATH)
    dst = out_dir / "matrix.csv"
    dst.write_bytes(raw)

    rows = [ln for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    n_rows = len(rows)
    n_cols = len(rows[0].split(","))
    n_pos = sum(int(float(v)) for ln in rows for v in ln.split(","))
    log.info(f"  wrote {dst} ({len(raw):,} bytes)")
    log.info(f"  shape ({n_rows}, {n_cols}), positives={n_pos:,}  "
             f"(expect (917, 792), 14,550 per the paper's Table 1)")

    ok = (n_rows, n_cols, n_pos) == (917, 792, 14550)
    if not ok:
        log.warning("  [WARNING] shape/count do NOT match the paper's reported Data1 "
                     "stats -- do not trust this as Data1 until resolved.")

    manifest = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "repo": REPO, "commit": COMMIT, "src_path": SRC_PATH,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": [n_rows, n_cols],
        "n_positives": n_pos,
        "matches_paper_table1": ok,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(f"Wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
