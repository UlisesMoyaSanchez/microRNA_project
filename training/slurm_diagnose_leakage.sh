#!/bin/bash
# =============================================================================
# slurm_diagnose_leakage.sh — Quantify the message-passing shortcut in the
# reported link-prediction AUROC. Inference only; no training, no checkpoint
# is written.
#
# Scores the same (miRNA, gene) pairs with the same V2 checkpoint under three
# encoder views of the graph:
#   (a) intact       — reproduces the original AUROC (~0.984)
#   (c) self-masked  — the scored pairs removed from message passing
#   (b) relation-off — the whole miRNA<->gene relation removed
#
# Results saved to: results/comparison/leakage_diagnostic.json
#
# Usage:
#   sbatch training/slurm_diagnose_leakage.sh
#   sbatch --export=ALL,CONFIG=configs/config_v2.yaml,CKPT=checkpoints_v2/best_model.pt \
#          training/slurm_diagnose_leakage.sh
# =============================================================================
#SBATCH --job-name=mirna_leak_diag
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --output=logs/leak_diag_%j.out
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
echo "Running leakage diagnostic..."
python training/diagnose_leakage.py \
    --config "${CONFIG}" \
    --checkpoint "${CKPT}"

echo ""
echo "========================================"
echo "Results:"
cat results/comparison/leakage_diagnostic.json
echo ""
echo "Done: $(date)"
echo "========================================"
