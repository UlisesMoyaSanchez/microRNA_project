"""
download_hmdd_survey_couplemda.py — Fetch CoupleMDA's (Li et al., IJMS 2025)
exact train/test miRNA-disease edges for the HMDD topology-baseline audit.

CoupleMDA's repo (lizhj39/CoupleMDA) ships a heterogeneous knowledge graph in the
HGTMDA data format (node.dat/link.dat/link.dat.test/info.dat, borrowed from
zht-code/HGTMDA) with 8 node types and 17 link types. info.dat confirms link type
12 = miRNA-disease (start type 0, end type 1). node.dat confirms node types are
laid out as contiguous global-id blocks in type order: miRNA = global ids 0-2089
(2,090 nodes), disease = global ids 2090-3843 (1,754 nodes) -- so local disease id
= global id - 2090, local miRNA id = global id directly.

link.dat.test's type-12 rows are ALL weight=1.0 (positives only, no negative-
labeled rows) -- 1,529 rows (1,523 unique pairs). link.dat's type-12 rows are the
training positives -- 13,744 rows (13,552 unique pairs). Checked here: 43 pairs
appear in BOTH files. Per this project's own leak-free-training-graph standard
(training/splits.py::assert_no_edge_leakage), those 43 pairs are excluded from the
training positives written here, so the message-passing graph this audit builds
never contains a held-out test edge.

CoupleMDA's own text states negatives are "sampled from node pairs that did not
have any existing edges in the graph at a 1:1 ratio" -- but the actual negative
PAIRS used are generated at runtime by the paper's own data loader
(scripts/data_loader.py), not saved to a file, so they cannot be recovered exactly.
This makes CoupleMDA a hybrid case: Tier 1 for the POSITIVE split (exact, from
link.dat/link.dat.test), Tier 2 for negatives (ratio-matched, self-generated).
Recorded downstream as tier=1 with an explicit `negatives_exact: false` flag.

Usage:
  python data/01_download/download_hmdd_survey_couplemda.py
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO = "lizhj39/CoupleMDA"
COMMIT = "c9128cb789cf32c13ad5e381d2055a03f91dcbe8"
OUT_DIR = "data/raw/hmdd_survey/couplemda"
MIRNA_DISEASE_LINK_TYPE = "12"
N_MIRNA = 2090
N_DISEASE_GLOBAL_OFFSET = 2090


def fetch(path: str) -> str:
    url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}"
    result = subprocess.run(
        ["curl", "-sS", "--fail", "--max-time", "60", url],
        capture_output=True, check=True,
    )
    # Source files use CRLF line endings; normalize.
    return result.stdout.decode("utf-8").replace("\r\n", "\n")


def parse_type12_pairs(link_text: str) -> set[tuple[int, int]]:
    pairs = set()
    for ln in link_text.splitlines():
        if not ln.strip():
            continue
        parts = ln.split("\t")
        start, end, ltype = parts[0], parts[1], parts[2]
        if ltype != MIRNA_DISEASE_LINK_TYPE:
            continue
        m_local = int(start)
        d_local = int(end) - N_DISEASE_GLOBAL_OFFSET
        pairs.add((m_local, d_local))
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default=OUT_DIR)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Fetching node.dat/link.dat/link.dat.test/info.dat from "
              f"{REPO} @ {COMMIT[:12]}...")
    node_text = fetch("data/Zou/node.dat")
    link_text = fetch("data/Zou/link.dat")
    linktest_text = fetch("data/Zou/link.dat.test")
    info_text = fetch("data/Zou/info.dat")
    (out_dir / "info.dat").write_text(info_text)

    n_mirna_nodes = sum(1 for ln in node_text.splitlines() if ln.split("\t")[2:3] == ["0"])
    n_disease_nodes = sum(1 for ln in node_text.splitlines() if ln.split("\t")[2:3] == ["1"])
    log.info(f"  node.dat: {n_mirna_nodes:,} miRNA nodes, {n_disease_nodes:,} disease nodes")

    train_pairs_raw = parse_type12_pairs(link_text)
    test_pairs = parse_type12_pairs(linktest_text)
    overlap = train_pairs_raw & test_pairs
    train_pairs = train_pairs_raw - overlap
    log.info(f"  link.dat (type 12): {len(train_pairs_raw):,} unique pairs")
    log.info(f"  link.dat.test (type 12): {len(test_pairs):,} unique pairs")
    log.info(f"  train/test overlap: {len(overlap)} pairs -- removed from train "
             f"to guarantee a leak-free message-passing graph")
    log.info(f"  final: train_pos={len(train_pairs):,}  test_pos={len(test_pairs):,}")

    def write_pairs(pairs: set[tuple[int, int]], path: Path) -> None:
        with open(path, "w") as fh:
            for m, d in sorted(pairs):
                fh.write(f"{m},{d}\n")

    write_pairs(train_pairs, out_dir / "train_pos.csv")
    write_pairs(test_pairs, out_dir / "test_pos.csv")
    log.info(f"  wrote {out_dir / 'train_pos.csv'} and {out_dir / 'test_pos.csv'}")

    manifest = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "repo": REPO, "commit": COMMIT,
        "n_mirna": n_mirna_nodes,
        "n_disease": n_disease_nodes,
        "n_train_pos_raw": len(train_pairs_raw),
        "n_test_pos": len(test_pairs),
        "n_train_test_overlap_removed": len(overlap),
        "n_train_pos_final": len(train_pairs),
        "tier": 1,
        "negatives_exact": False,
        "caveat": (
            "Positive train/test split is exact (from CoupleMDA's own bundled "
            "link.dat/link.dat.test). Negatives are NOT exact -- CoupleMDA's own "
            "1:1 negatives are generated at runtime by its data loader and not "
            "published as a file, so this audit generates its own 1:1 uniform "
            "negatives on this exact positive split."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(f"Wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
