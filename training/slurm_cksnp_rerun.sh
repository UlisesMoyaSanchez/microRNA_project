#!/bin/bash
# =============================================================================
# slurm_cksnp_rerun.sh -- CKSNP-GNN's own released code, as released and corrected step by step
# (analysis/cksnp_rerun.py + analysis/cksnp_rerun_mx.py; BITACORA 2026-11-16).
#
# CKSNP-GNN's code.zip (public.aibiochem.net, linked from the article's supplement; sha256 3277b666...2952,
# no licence) is not vendored: it is unzipped into /raid/home/$USER/scripts/external_code/CKSNP-GNN.
# Their code needs mxnet 1.6.0 + DGL 0.4.3post1: conda env cksnp_py37 (Python 3.7). protocheck and the
# matched sampler run in mirna_ms.
#
# Usage: sbatch training/slurm_cksnp_rerun.sh        (EPOCHS=2 SEEDS=2008 for a smoke test)
# =============================================================================
#SBATCH --job-name=cksnp
#SBATCH --cpus-per-task=16
#SBATCH --mem=48gb
#SBATCH --output=logs/cksnp_rerun_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large

set -euo pipefail
P="/raid/home/${USER}/scripts/microRNA_project"
cd "$P"
mkdir -p logs results/comparison
R="/raid/home/${USER}/scripts/external_code/CKSNP-GNN"
W="$P/results/cksnp_work"
mkdir -p "$W"
[ -f "$R/train.py" ] && [ -f "$R/miRNA_MSSM/miRNA_CKSNAP5_pearson.csv" ] || { echo "CKSNP-GNN not unzipped"; exit 1; }
SEEDS="${SEEDS:-2008 1 2}"
EPOCHS="${EPOCHS:-100}"
SMOKE=""; [ "$EPOCHS" -lt 100 ] && SMOKE="--smoke"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 PYTHONNOUSERSITE=1

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
set -u
F='DeprecationWarning|LooseVersion|UserWarning|warnings.warn|^\s*$'
for s in $SEEDS; do
  set +u; conda activate cksnp_py37; set -u
  DGLBACKEND=mxnet python -W ignore analysis/cksnp_rerun_mx.py --repo "$R" --work "$W" --seed "$s" --dump 2>&1 | grep -v -E "$F" | grep -v -E "^(Building|Adding|Successfully)" || true
  set +u; conda activate mirna_ms; set -u
  python analysis/cksnp_rerun.py --prep --seed "$s" --work "$W"
  set +u; conda activate cksnp_py37; set -u
  DGLBACKEND=mxnet python -W ignore analysis/cksnp_rerun_mx.py --repo "$R" --work "$W" --seed "$s" --epochs "$EPOCHS" 2>&1 | grep --line-buffered -v -E "$F" | grep --line-buffered -v -E "^(Building|Adding|Successfully)"
done
set +u; conda activate mirna_ms; set -u
python analysis/cksnp_rerun.py --report --seeds $SEEDS --work "$W" $SMOKE 2>&1 | grep -v -E "UserWarning|warnings.warn"
echo "Done: $(date)"
