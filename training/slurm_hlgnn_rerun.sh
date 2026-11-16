#!/bin/bash
# =============================================================================
# slurm_hlgnn_rerun.sh -- HLGNN-MDA's own released code, as released and against degree-matched negatives
# (analysis/hlgnn_rerun.py; BITACORA 2026-11-16).
#
# HLGNN-MDA (github.com/LiangYu-Xidian/HLGNN-MDA @ e3ac824, no licence) is not vendored: it is cloned into
# /raid/home/$USER/scripts/external_code/HLGNN-MDA and its pytorch_DGCNN C++ library built with make.
#
# Usage: TRAIN_NEG=uniform HOP=1 sbatch --export=ALL training/slurm_hlgnn_rerun.sh
# =============================================================================
#SBATCH --job-name=hlgnn
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --output=logs/hlgnn_rerun_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large

set -euo pipefail
cd "/raid/home/${USER}/scripts/microRNA_project"
mkdir -p logs results/comparison
R="/raid/home/${USER}/scripts/external_code/HLGNN-MDA"
[ "$(git -C "$R" rev-parse --short HEAD)" = "e3ac824" ] || { echo "HLGNN-MDA not at e3ac824"; exit 1; }
[ -f "$R/pytorch_DGCNN/lib/build/dll/libgnn.so" ] || { echo "libgnn.so not built"; exit 1; }

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms
set -u
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8

python analysis/hlgnn_rerun.py --repo "$R" --hop "${HOP:-1}" --train-neg "${TRAIN_NEG:-uniform}" \
    --epochs "${EPOCHS:-50}" ${SEEDS:+--seeds $SEEDS} 2>&1 | grep --line-buffered -v -E "UserWarning|warnings.warn|it/s\]|batch/s\]|^\s*$"
echo "Done: $(date)"
