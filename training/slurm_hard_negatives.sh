#!/bin/bash
# =============================================================================
# slurm_hard_negatives.sh — Separate regulatory specificity from popularity bias.
#
# Crosses two negative samplers (uniform / gene-degree-matched) with two scorers
# (a model-free gene-degree heuristic / HGT V2), all on the same positive pairs,
# with the scored pairs masked from message passing. Inference only.
#
# Results saved to: results/comparison/hard_negatives.json
#
# Usage:
#   sbatch training/slurm_hard_negatives.sh
# =============================================================================
#SBATCH --job-name=mirna_hardneg
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --output=logs/hardneg_%j.out
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

CONFIG="${CONFIG:-configs/config_v2.yaml}"
CKPT="${CKPT:-checkpoints_v2/best_model.pt}"

echo "========================================"
echo "Job:        ${SLURM_JOB_ID:-local}"
echo "Node:       $(hostname)"
echo "Date:       $(date)"
echo "Config:     ${CONFIG}"
echo "Checkpoint: ${CKPT}"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT}"
    exit 1
fi

echo ""
python training/eval_hard_negatives.py \
    --config "${CONFIG}" \
    --checkpoint "${CKPT}"

echo ""
echo "========================================"
echo "Results:"
cat results/comparison/hard_negatives.json
echo ""
echo "Done: $(date)"
echo "========================================"
