#!/bin/bash
# =============================================================================
# slurm_celltype_baseline.sh — Is 0.9916 a real result or a reconstruction of its own
# label?
#
# Scores two no-graph controls (nearest-centroid, logistic regression) on the same
# X_pca cell features and the same held-out cell split the 0.9916 headline number came
# from. No message passing, no checkpoint. Also reports what fraction of the stored
# cell_type label is literally argmax(marker score) — the mechanism that generated the
# label in the first place.
#
# If either control lands near 0.9916, the graph/transformer is not earning its
# complexity on this task, mirroring the topology-baseline finding for link prediction.
#
# Results saved to: results/comparison/celltype_baseline_<config-stem>_<split>.json
#
# Usage:
#   sbatch --export=ALL,CONFIG=configs/config_v2_edgesplit.yaml,SPLIT=test \
#       training/slurm_celltype_baseline.sh
# =============================================================================
#SBATCH --job-name=mirna_celltype_baseline
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --output=logs/celltype_baseline_%j.out
#SBATCH --nodelist=dgxa100jal
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
SPLIT="${SPLIT:-val}"
CONFIG_STEM="$(basename "${CONFIG}" .yaml)"

echo "========================================"
echo "Job:    ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "Config: ${CONFIG}"
echo "Split:  ${SPLIT}"
echo "========================================"
echo ""

python training/eval_celltype_baseline.py \
    --config "${CONFIG}" \
    --split "${SPLIT}"

echo ""
echo "========================================"
echo "Results:"
cat "results/comparison/celltype_baseline_${CONFIG_STEM}_${SPLIT}.json"
echo ""
echo "Done: $(date)"
echo "========================================"
