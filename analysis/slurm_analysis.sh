#!/bin/bash
# =============================================================================
# slurm_analysis.sh — Run model interpretation and figure generation.
#
# Usage: sbatch analysis/slurm_analysis.sh
# Dependency: slurm_train.sh must have completed (best_model.pt must exist).
#
# Requires 1 GPU for gradient saliency computation.
# =============================================================================
#SBATCH --job-name=mirna_analysis
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --output=logs/analysis_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx_large

set -euo pipefail

PROJECT_DIR="/raid/home/${USER}/scripts/microRNA_project"
cd "${PROJECT_DIR}"
mkdir -p logs results/interpretation results/figures

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms
set -u

echo "========================================"
echo "Job:    ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "========================================"

echo ""
echo "[1/3] Evaluating model on test set..."
python training/evaluate.py --config configs/config_v2.yaml

echo ""
echo "[2/3] Running model interpretation (saliency + edge scoring + enrichment)..."
python analysis/interpret.py \
    --config configs/config_v2.yaml \
    --top_k 30 \
    --top_genes 50

echo ""
echo "[3/3] Generating paper figures..."
python analysis/visualize.py --config configs/config_v2.yaml

echo ""
echo "========================================"
echo "Analysis complete: $(date)"
echo "Figures: ${PROJECT_DIR}/results/figures/"
ls -lh "${PROJECT_DIR}/results/figures/"
echo "========================================"
