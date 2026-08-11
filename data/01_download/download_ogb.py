"""
download_ogb.py — Download ogbl-ddi and/or ogbl-ppa via the official `ogb` package.

Part of the "light" extension of the evaluation-methodology audit to two public,
standardized OGB link-prediction benchmarks (no new GNN training — see
training/eval_ogb_topology_baseline.py for the actual scoring). This script's only
job is to make sure the data is on disk, logged, and its basic stats confirmed —
`inspect_ogb_split.py` (run first, see slurm_inspect_ogb.sh) already triggers the
same download as a side effect of its verification pass, so re-running this script
after that is normally a no-op (`PygLinkPropPredDataset` is idempotent: it skips
the download/processing step if the target files already exist under `root`).

Datasets:
  ogbl-ddi   4,267 nodes / 1,334,889 training edges — drug-drug interaction.
  ogbl-ppa   576,289 nodes / 30,326,273 training edges — protein-protein association.
             This is OGB's largest link-prediction dataset; check disk space first.

Usage:
  python data/01_download/download_ogb.py --dataset both --out_dir data/raw/ogb
"""

from __future__ import annotations

import os
import shutil
import argparse
import logging


def check_disk_space(path: str, min_free_gb: float, log: logging.Logger) -> None:
    os.makedirs(path, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / (1024 ** 3)
    log.info(f"  Free disk space at {path}: {free_gb:.1f} GB")
    if free_gb < min_free_gb:
        raise SystemExit(
            f"Only {free_gb:.1f} GB free at {path}, want at least {min_free_gb:.0f} GB "
            f"headroom before downloading ogbl-ppa (OGB's largest link-prediction "
            f"dataset). Free up space or point --out_dir elsewhere."
        )


def download_one(name: str, out_dir: str, log: logging.Logger) -> None:
    from ogb.linkproppred import PygLinkPropPredDataset

    ogb_name = f"ogbl-{name}"
    log.info("=" * 78)
    log.info(f"Downloading {ogb_name} into {out_dir} (skips if already present)...")
    dataset = PygLinkPropPredDataset(name=ogb_name, root=out_dir)
    data = dataset[0]
    split_edge = dataset.get_edge_split()

    log.info(f"  {ogb_name}: {data.num_nodes:,} nodes, "
              f"{data.edge_index.shape[1]:,} edges in dataset[0]")
    for split_name, split_dict in split_edge.items():
        n = None
        if "edge" in split_dict:
            e = split_dict["edge"]
            n = e.shape[0] if e.shape[-1] == 2 else e.shape[1]
        log.info(f"  split['{split_name}']: {n:,} positive edges" if n is not None
                  else f"  split['{split_name}']: (see keys {list(split_dict.keys())})")
    log.info(f"{ogb_name} download complete.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["ddi", "ppa", "both"], default="both")
    p.add_argument("--out_dir", default="data/raw/ogb")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    names = ["ddi", "ppa"] if args.dataset == "both" else [args.dataset]
    if "ppa" in names:
        check_disk_space(args.out_dir, min_free_gb=20.0, log=log)

    for name in names:
        download_one(name, args.out_dir, log)


if __name__ == "__main__":
    main()
