#!/bin/bash
# =============================================================================
# slurm_download_ogb.sh — Download ogbl-ddi and ogbl-ppa (Open Graph Benchmark).
# Normally a no-op after slurm_inspect_ogb.sh has already run once (OGB's
# PygLinkPropPredDataset skips re-downloading if the target files exist), but
# kept as its own documented step per this repo's download_*.py convention.
#
# Usage: sbatch data/01_download/slurm_download_ogb.sh
# =============================================================================
#SBATCH --job-name=ogb_download
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --output=logs/ogb_download_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large

set -euo pipefail

PROJECT_DIR="/raid/home/${USER}/scripts/microRNA_project"
cd "${PROJECT_DIR}"
mkdir -p logs data/raw/ogb

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms
set -u

echo "========================================"
echo "Job:    ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "Python: $(python --version)"
echo "========================================"

python data/01_download/download_ogb.py --dataset both --out_dir data/raw/ogb

echo ""
echo "========================================"
echo "OGB download complete: $(date)"
du -sh data/raw/ogb/*
echo "========================================"
