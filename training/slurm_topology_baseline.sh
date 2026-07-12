#!/bin/bash
# =============================================================================
# slurm_topology_baseline.sh — Is 0.62 the model's achievement or the task's ceiling?
#
# Scores model-free topology heuristics (gene degree, preferential attachment,
# common neighbours, Adamic-Adar) on the SAME held-out miRNA->gene edges the
# retrained HGT was evaluated on, against both negative samplers. No learning,
# no checkpoint — CPU-only maths, but a GPU node makes the matmuls instant.
#
# If the best heuristic lands near the HGT's 0.6268, the model has learned nothing
# beyond trivial graph structure and ~0.62 is the information ceiling of the task
# as posed. That is the finding Path A rests on.
#
# Results saved to: results/comparison/topology_baseline.json
#
# Usage:
#   sbatch --export=ALL,CONFIG=configs/config_v2_edgesplit.yaml \
#       training/slurm_topology_baseline.sh
# =============================================================================
#SBATCH --job-name=mirna_topo
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --output=logs/topology_%j.out
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

echo "========================================"
echo "Job:    ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "Config: ${CONFIG}"
echo "Split:  ${SPLIT}"
echo "========================================"
echo ""

python training/eval_topology_baseline.py \
    --config "${CONFIG}" \
    --split "${SPLIT}"

echo ""
echo "========================================"
echo "Results:"
cat results/comparison/topology_baseline.json
echo ""
echo "Done: $(date)"
echo "========================================"
