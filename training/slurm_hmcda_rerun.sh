#!/bin/bash
# =============================================================================
# slurm_hmcda_rerun.sh -- HMCDA's own released code, as released and corrected step by step
# (analysis/hmcda_rerun.py + analysis/hmcda_rerun_mg.py; BITACORA 2026-11-16).
#
# HMCDA (github.com/shiyangl/HMCDA @ 4d9e8bb, no licence) is not vendored: it is cloned into
# /raid/home/$USER/scripts/external_code/HMCDA. Its code needs Python 3.6, PyTorch 1.2.0 and DGL 0.3.1:
# conda env hmcda_py36 (CPU). protocheck and the matched sampler run in mirna_ms.
#
# Usage: sbatch training/slurm_hmcda_rerun.sh        (EPOCHS=2 for a smoke test)
# =============================================================================
#SBATCH --job-name=hmcda
#SBATCH --cpus-per-task=40
#SBATCH --mem=96gb
#SBATCH --output=logs/hmcda_rerun_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large

set -euo pipefail
P="/raid/home/${USER}/scripts/microRNA_project"
cd "$P"
mkdir -p logs results/comparison
R="/raid/home/${USER}/scripts/external_code/HMCDA"
W="$P/results/hmcda_work"
mkdir -p "$W"
[ "$(git -C "$R" rev-parse --short HEAD)" = "4d9e8bb" ] || { echo "HMCDA not at 4d9e8bb"; exit 1; }
EPOCHS="${EPOCHS:-100}"
PAR="${PAR:-10}"
SMOKE=""; [ "$EPOCHS" -lt 100 ] && SMOKE="--smoke"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONNOUSERSITE=1
SPLITS="ex5 rot1 rot2 rot3 rot4 rot5"

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
set -u
F='Warning|warnings.warn|^\s*$'
for sp in $SPLITS; do
  set +u; conda activate hmcda_py36; set -u
  python -W ignore analysis/hmcda_rerun_mg.py --repo "$R" --work "$W" --split "$sp" --dump
  set +u; conda activate mirna_ms; set -u
  python analysis/hmcda_rerun.py --prep --split "$sp" --work "$W"
done
set +u; conda activate hmcda_py36; set -u
{
  for v in released dedup; do
    for s in 1 2 3 4 5; do echo "ex5 $v $s"; done
    for r in 1 2 3 4 5; do echo "rot$r $v 1"; done
  done
} | xargs -P "$PAR" -L 1 bash -c 'python -W ignore analysis/hmcda_rerun_mg.py --repo "'"$R"'" --work "'"$W"'" \
      --split "$0" --variant "$1" --seed "$2" --epochs "'"$EPOCHS"'" 2>&1 | grep --line-buffered -v -E "'"$F"'" \
      | grep --line-buffered -E "AUROC|epoch 0 |Error|Trace|error"'
set +u; conda activate mirna_ms; set -u
python analysis/hmcda_rerun.py --report --work "$W" $SMOKE 2>&1 | grep -v -E "UserWarning|warnings.warn"
echo "Done: $(date)"
