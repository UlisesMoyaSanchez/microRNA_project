#!/bin/bash
# =============================================================================
# slurm_heldout_grid.sh — Attribute the 0.9836 -> 0.6268 collapse to its two causes.
#
# The retrain changed the split AND the negatives at once. This scores the retrained
# checkpoint on the same held-out edges under BOTH negative samplers, filling in the
# missing cell of the 2x2 so the drop can be split into "cost of an honest split" and
# "cost of honest negatives". Inference only.
#
# Results saved to: results/comparison/heldout_grid.json
#
# Usage:
#   sbatch --export=ALL,CONFIG=configs/config_v2_edgesplit.yaml,\
# CKPT=checkpoints_v2_edgesplit/best_model.pt training/slurm_heldout_grid.sh
# =============================================================================
#SBATCH --job-name=mirna_grid
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --output=logs/grid_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx_large

set -euo pipefail

PROJECT_DIR="/raid/home/${USER}/scripts/microRNA_project"
cd "${PROJECT_DIR}"
mkdir -p logs results/comparison

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms
set -u

CONFIG="${CONFIG:-configs/config_v2_edgesplit.yaml}"
CKPT="${CKPT:-checkpoints_v2_edgesplit/best_model.pt}"
SPLIT="${SPLIT:-val}"

echo "========================================"
echo "Job:        ${SLURM_JOB_ID:-local}"
echo "Node:       $(hostname)"
echo "Date:       $(date)"
echo "Config:     ${CONFIG}"
echo "Checkpoint: ${CKPT}"
echo "Split:      ${SPLIT}"
echo "========================================"

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT}"
    exit 1
fi

echo ""
python training/eval_heldout_grid.py \
    --config "${CONFIG}" \
    --checkpoint "${CKPT}" \
    --split "${SPLIT}"

CKPT_TAG="$(basename "$(dirname "${CKPT}")")"

echo ""
echo "========================================"
echo "Results:"
cat "results/comparison/heldout_grid_${CKPT_TAG}_${SPLIT}.json"
echo ""
echo "Done: $(date)"
echo "========================================"
