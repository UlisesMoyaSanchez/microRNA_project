#!/bin/bash
# =============================================================================
# slurm_higldp_rerun.sh -- HiGLDP's own released code, as released and corrected step by step
# (analysis/higldp_rerun.py + analysis/higldp_rerun_run.py; BITACORA 2026-11-16).
#
# HiGLDP (github.com/St0neWang/HiGLDP @ 9808ba8, identical to Zenodo 10.5281/zenodo.18522355; no licence) is
# not vendored: it is cloned into /raid/home/$USER/scripts/external_code/HiGLDP. Its code needs Python 3.10,
# PyTorch 1.13.1, PyG 2.3.1 and TensorFlow 2.10 (README): conda env higldp_py310. protocheck and the matched
# sampler run in mirna_ms.
#
# Usage: sbatch training/slurm_higldp_rerun.sh        (EPOCHS=5 SEEDS=123 FOLDS=1 for a smoke test)
# =============================================================================
#SBATCH --job-name=higldp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96gb
#SBATCH --output=logs/higldp_rerun_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large

set -euo pipefail
P="/raid/home/${USER}/scripts/microRNA_project"
cd "$P"
mkdir -p logs results/comparison
R="/raid/home/${USER}/scripts/external_code/HiGLDP"
W="$P/results/higldp_work"
mkdir -p "$W"
[ "$(git -C "$R" rev-parse --short HEAD)" = "9808ba8" ] || { echo "HiGLDP not at 9808ba8"; exit 1; }
EPOCHS="${EPOCHS:-200}"
SEEDS="${SEEDS:-123 1 2}"
FOLDS="${FOLDS:-1 2 3 4 5}"
PAR="${PAR:-1}"   # 4 in parallel ran out of GPU memory (smoke job 4907)
SMOKE=""; [ "$EPOCHS" -lt 200 ] && SMOKE="--smoke"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONNOUSERSITE=1 TF_CPP_MIN_LOG_LEVEL=3 CUDA_DEVICE_ORDER=PCI_BUS_ID

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms; set -u
python analysis/higldp_rerun.py --prep --repo "$R" --work "$W"
set +u; conda activate higldp_py310; set -u
python analysis/higldp_rerun_run.py --repo "$R" --work "$W" --check-adj
F='Warning|warnings.warn|deprecat|^\s*$|tensorflow|Instructions for updating|^\*+$'
{
  for f in $FOLDS; do for s in $SEEDS; do for st in released ported foldfeat validation; do
    echo "$f $s $st"; done; done; done
} | xargs -P "$PAR" -L 1 bash -c 'python -W ignore analysis/higldp_rerun_run.py --repo "'"$R"'" --work "'"$W"'" \
      --fold "$0" --seed "$1" --step "$2" --epochs "'"$EPOCHS"'" 2>&1 | grep --line-buffered -v -i -E "'"$F"'" \
      | grep --line-buffered -E "fold |epoch 199|Error|Trace|error"'
set +u; conda activate mirna_ms; set -u
python analysis/higldp_rerun.py --report --repo "$R" --work "$W" $SMOKE 2>&1 | grep -v -E "UserWarning|warnings.warn"
echo "Done: $(date)"
