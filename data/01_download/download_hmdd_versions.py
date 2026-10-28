"""
download_hmdd_versions.py — Fetch the HMDD releases and node names needed to ask how many
of the canonical 5,430 matrix's "negatives" were later recorded as associations.

The canonical 495 x 383 matrix behind MGCNSS, NIMGSA and HLGNN-MDA is HMDD v2.0-derived
(download_hmdd_survey_canonical5430.py) but ships as bare indices. Its names come from
HLGNN-MDA's index files (commit-pinned; NIMGSA's name CSVs agree up to case and one comma
in three disease names). Read by analysis/hmdd_hidden_positives.py.

  hmdd2.zip         HMDD v2.0 (2014), to validate the index -> name mapping: every matrix
                    positive must reappear here.
  v3_alldata.txt    HMDD v3.2 (2019), the later release the negatives are checked against.

HMDD v4.0 (alldata_v4.txt / .xlsx, same site) is deliberately NOT used: its miRNA column is
damaged in both published formats -- no hsa-mir-30a/30b/30d/30e, 320a/320b or 302a-f at
all, an "hsa-mir-30" with 173 rows, and a first row labelled hsa-mir-40 whose description
is about miR-340 -- so a v2 association can go missing from it for naming reasons alone.

cuilab.cn serves files without versioned URLs, so they cannot be commit-pinned; the sha256
seen when this was written is checked instead and a mismatch is a hard error.

Usage:
  python data/01_download/download_hmdd_versions.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("data/raw/hmdd_versions")

CUILAB = "http://www.cuilab.cn/static/hmdd3/data/"
HLGNN = ("https://raw.githubusercontent.com/LiangYu-Xidian/HLGNN-MDA/"
         "e3ac824017b8d1bf246e7f695551a8874d231724/"
         "Python/data/5430dataset/1.miRNA-disease associations/")

# local name -> (url, sha256 observed 2026-10-27)
FILES = {
    "hmdd2.zip": (CUILAB + "hmdd2.zip",
                  "4b167c107cffbb44cd944989619c3f112accc4e726b45fe1821b60c5e1f02203"),
    "v3_alldata.txt": (CUILAB + "v3_alldata.txt",
                       "dcc871c880a2351428d994752cd57caa5438b951cd42a37e58e63fda420c033e"),
    "canonical5430_mirna_index.xlsx": (HLGNN + "miRNA_index.xlsx",
                                       "5f4b792fc64391d87a78b4b48f7a115403c9395ab2039f638c3dc8669d48269c"),
    "canonical5430_disease_index.xlsx": (HLGNN + "disease_index.xlsx",
                                         "3ea6c655a4acc76da7c702fbe7a78e84c030a70eaad21f846a60c86032818d8b"),
}


def fetch(url: str) -> bytes:
    # curl, not urllib: see download_hmdd_survey_canonical5430.py::fetch.
    quoted = urllib.parse.quote(url, safe=":/")
    return subprocess.run(["curl", "-sS", "--fail", "--max-time", "120", quoted],
                          capture_output=True, check=True).stdout


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"fetched_at": datetime.now(timezone.utc).isoformat(), "files": {}}
    for name, (url, expected) in FILES.items():
        raw = fetch(url)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected:
            raise SystemExit(f"{name}: sha256 {digest} != expected {expected} -- the source "
                             "changed; re-verify it before updating the hash")
        (OUT / name).write_bytes(raw)
        manifest["files"][name] = {"url": url, "sha256": digest, "bytes": len(raw)}
        print(f"wrote {OUT / name} ({len(raw):,} bytes)")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
