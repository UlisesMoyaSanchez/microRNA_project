#!/bin/bash
# =============================================================================
# slurm_ogb_ppa_topology.sh — Model-free topology baseline audit on ogbl-ppa
# (576,289 nodes, ~30.3M training edges — OGB's largest link-prediction dataset,
# sparse/pairwise scoring path, never materializes an N x N matrix). Part of the
# "light" OGB audit extension — no GNN training, compares against OGB's own
# published leaderboard.
#
# mem=192gb (well above this repo's usual 64gb eval-job default): OGB's own
# dataset processing for ppa is memory-heavy during construction, and chunked
# pairwise scoring holds several sparse (chunk, n) row-blocks in memory at once.
# No rush on this job, so generous headroom beats an overnight OOM. Peak RSS is
# logged at the end (see eval_ogb_topology_baseline.py) to right-size --mem on
# future runs.
#
# Run slurm_ogb_ddi_topology.sh FIRST — same code path, much faster feedback
# loop to catch bugs before spending a large-memory ppa allocation on them.
#
# Usage: sbatch training/slurm_ogb_ppa_topology.sh
#    or: sbatch --export=ALL,SPLIT=valid training/slurm_ogb_ppa_topology.sh
# =============================================================================
#SBATCH --job-name=ogb_ppa_topo
#SBATCH --cpus-per-task=16
#SBATCH --mem=192gb
#SBATCH --output=logs/ogb_ppa_topo_%j.out
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
    --config configs/config_ogb_ppa.yaml \
    --split "${SPLIT}"

echo ""
echo "========================================"
echo "ogbl-ppa topology baseline complete: $(date)"
echo "========================================"
