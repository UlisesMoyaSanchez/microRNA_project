"""
kmerpmtf_their_step3.py -- run kmerPMTF's own step3_train.py, unmodified, on the inputs kmerpmtf_rerun.py
prepares, to check that the rerun's reimplemented fold logic gives the same numbers as the released script.

Their step1 (k-mer counting with one pandas .loc per k-mer) is too slow to run on a genome, so the
process_data/ files step3 reads are written from kmerpmtf_rerun.py's cached preprocessing:
  <prefix>_G_G_cosine_graph.csv, <prefix>_M_M_cosine_graph.csv   dense cosine matrices; entries below 0.4
      are written as 0, which step3 does anyway (G_G[G_G < simthreshold] = 0) before using them
  <prefix>_G_M_cosine_graph.csv                                   gene x miRNA 0/1 matrix
  <prefix>_D256_node_attr_encode.csv                              the step-2 embeddings
Then step3_train.py runs in a scratch copy of the repository (its own file, byte-identical; model/ linked)
with the given arguments, and its per-fold AUROCs are parsed from its output.

Usage: python analysis/kmerpmtf_their_step3.py --repo PATH --data PATH --species Prunus_persica \
           --chebconvk 2 [--lr 0.0001 --epoch 3000 --kfold 5]
Writes results/comparison/kmerpmtf_their_step3_<species>_K<k>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from kmerpmtf_rerun import FILES  # noqa: E402


def write_dense(path, names, rows, cols, vals, n, offset=0):
    """Dense CSV (pandas layout: empty corner cell, header of names, index column) from sparse entries."""
    by_row = {}
    for r, c, v in zip(rows - offset, cols - offset, vals):
        by_row.setdefault(int(r), []).append((int(c), float(v)))
    with open(path, "w") as fh:
        fh.write("," + ",".join(names) + "\n")
        for i in range(n):
            line = ["0"] * n
            for c, v in by_row.get(i, ()):
                line[c] = repr(v)
            fh.write(names[i] + "," + ",".join(line) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--species", required=True, choices=list(FILES))
    ap.add_argument("--chebconvk", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.0001)
    ap.add_argument("--epoch", type=int, default=3000)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--work", type=Path, default=ROOT / "results/kmerpmtf_work")
    a = ap.parse_args()

    sys.path.insert(0, str(a.repo))
    import utils as U
    z = np.load(a.work / f"{a.species}_prep.npz")
    d = a.data / a.species
    genes = list(U.read_fasta_to_dict(str(d / "Genome" / FILES[a.species])))
    mirnas = list(U.read_fasta_to_dict(str(d / "PmiREN" / f"{a.species}_mature.fa")))
    n_g, n_m = len(genes), len(mirnas)
    assert n_g == int(z["n_g"]) and n_m == int(z["n_m"])

    run = a.work / f"their_step3_{a.species}"
    pdir = run / "process_data"
    pdir.mkdir(parents=True, exist_ok=True)
    (run / "train_result").mkdir(exist_ok=True)
    shutil.copy2(a.repo / "step3_train.py", run / "step3_train.py")
    if not (run / "model").exists():
        (run / "model").symlink_to(a.repo / "model")
    sha = hashlib.sha256((run / "step3_train.py").read_bytes()).hexdigest()
    assert sha == hashlib.sha256((a.repo / "step3_train.py").read_bytes()).hexdigest()

    pre = a.species
    if not (pdir / f"{pre}_G_G_cosine_graph.csv").exists():
        write_dense(pdir / f"{pre}_G_G_cosine_graph.csv", genes, z["gg"][0], z["gg"][1], z["gg_w"], n_g)
        write_dense(pdir / f"{pre}_M_M_cosine_graph.csv", mirnas, z["mm"][0], z["mm"][1], z["mm_w"], n_m, offset=n_g)
        GM = np.zeros((n_g, n_m), np.int64)
        GM[z["pos"][:, 0], z["pos"][:, 1]] = 1
        pd.DataFrame(GM, index=genes, columns=mirnas).to_csv(pdir / f"{pre}_G_M_cosine_graph.csv")
        pd.DataFrame(z["enc"], index=genes + mirnas).to_csv(pdir / f"{pre}_D256_node_attr_encode.csv",
                                                            float_format="%.10f")
    cmd = [sys.executable, "step3_train.py", "--method", "ChebConv", "--prefix", pre, "--device", "cuda:0",
           "--lr", str(a.lr), "--epoch", str(a.epoch), "--kfold", str(a.kfold), "--chebconvk", str(a.chebconvk),
           "--D", "256", "--sim", "cosine"]
    print("Running their script:", " ".join(cmd), "(sha256", sha[:16] + ")", flush=True)
    # their script imports torch before numpy; under this conda env MKL's Intel threading layer then clashes
    # with libgomp, so the layer is set from outside (an environment variable, the script is untouched)
    env = dict(os.environ, MKL_THREADING_LAYER="GNU")
    p = subprocess.run(cmd, cwd=run, capture_output=True, text=True, env=env)
    log = re.sub(r"\x1b\[[0-9;]*m", "", p.stdout + p.stderr)
    (run / f"step3_K{a.chebconvk}.log").write_text(log)
    folds = [float(m.group(1)) for m in re.finditer(r"on test dataset =+\s*\n\s*METHOD:\S+ LOSS:\S+ AUC:([\d.]+)", log)]
    print(log[-3000:])
    out = {"species": a.species, "args": vars(a) | {"repo": str(a.repo), "data": str(a.data), "work": str(a.work)},
           "step3_sha256": sha, "returncode": p.returncode, "fold_auroc": folds,
           "mean_auroc": float(np.mean(folds)) if folds else None}
    path = ROOT / f"results/comparison/kmerpmtf_their_step3_{a.species}_K{a.chebconvk}.json"
    path.write_text(json.dumps(out, indent=2))
    print("fold AUROCs:", folds, "\nWrote", path)
    if p.returncode != 0 or not folds:
        sys.exit(f"their step3_train.py failed (return code {p.returncode})")


if __name__ == "__main__":
    main()
