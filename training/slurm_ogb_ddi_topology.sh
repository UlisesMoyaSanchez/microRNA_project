#!/bin/bash
# =============================================================================
# slurm_ogb_ddi_topology.sh — Model-free topology baseline audit on ogbl-ddi
# (4,267 nodes, dense-matrix path). Part of the "light" OGB audit extension —
# no GNN training, compares against OGB's own published leaderboard.
#
# Run data/01_download/slurm_inspect_ogb.sh first and review
# results/comparison/ogb_protocol_verification.json before this.
#
# Usage: sbatch training/slurm_ogb_ddi_topology.sh
#    or: sbatch --export=ALL,SPLIT=valid training/slurm_ogb_ddi_topology.sh
# =============================================================================
#SBATCH --job-name=ogb_ddi_topo
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --output=logs/ogb_ddi_topo_%j.out
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

SPLIT="${SPLIT:-test}"

echo "========================================"
echo "Job:    ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "Split:  ${SPLIT}"
echo "Python: $(python --version)"
echo "========================================"

python training/eval_ogb_topology_baseline.py \
    --config configs/config_ogb_ddi.yaml \
    --split "${SPLIT}"

echo ""
echo "========================================"
echo "ogbl-ddi topology baseline complete: $(date)"
echo "========================================"
