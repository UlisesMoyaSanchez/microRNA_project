#!/bin/bash
# =============================================================================
# slurm_verify_graph.sh — Independent recompute-and-compare check that a built
# heterograph's edges match what the config claims. Read-only, CPU, no GPU.
# Loads the graph + the processed AnnData, so it needs the same RAM as the build.
#
# Usage: CONFIG=configs/config_v3fixed_edgesplit.yaml sbatch analysis/slurm_verify_graph.sh
# =============================================================================
#SBATCH --job-name=verify_graph
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --output=logs/verify_graph_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large

set -euo pipefail

PROJECT_DIR="/raid/home/${USER}/scripts/microRNA_project"
cd "${PROJECT_DIR}"
mkdir -p logs

CONFIG="${CONFIG:-configs/config_v3fixed_edgesplit.yaml}"

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms
set -u

echo "========================================"
echo "Job:    ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "Config: ${CONFIG}"
echo "Git:    $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "========================================"

python analysis/verify_graph_construction.py --config "${CONFIG}"

echo "========================================"
echo "Verification complete: $(date)"
echo "========================================"
