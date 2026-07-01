#!/bin/bash
# =============================================================================
# slurm_build_graph.sh — Build the heterogeneous PyG graph.
# Requires high RAM (PCC correlation matrix for 1000 genes).
# No GPU needed.
#
# Usage: sbatch data/03_build_graph/slurm_build_graph.sh
# Dependency: slurm_preprocess.sh must have completed.
# =============================================================================
#SBATCH --job-name=mirna_build_graph
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --output=logs/build_graph_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large

set -euo pipefail

PROJECT_DIR="/raid/home/${USER}/scripts/microRNA_project"
cd "${PROJECT_DIR}"
mkdir -p logs data/graphs

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms
set -u

echo "========================================"
echo "Job:    ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "========================================"

python data/03_build_graph/build_heterograph.py --config configs/config.yaml

echo ""
echo "========================================"
echo "Graph build complete: $(date)"
echo "Output: ${PROJECT_DIR}/data/graphs/"
ls -lh "${PROJECT_DIR}/data/graphs/"
echo "========================================"
