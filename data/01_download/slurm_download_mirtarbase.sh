#!/bin/bash
# =============================================================================
# slurm_download_mirtarbase.sh — Download the REAL miRTarBase (release 10.0)
# human MTIs for the interaction-source audit (goal.md §3.2.4).
#
# Separate from slurm_download.sh on purpose: that script covers the original
# miRDB/CellxGene/GEO pull and has those stages commented out as done. This one
# fetches an additional, independent interaction source and writes to distinct
# filenames, so the existing miRDB graph is never touched.
#
# Usage: sbatch data/01_download/slurm_download_mirtarbase.sh
# =============================================================================
#SBATCH --job-name=mirtarbase_dl
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --output=logs/download_mirtarbase_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large

set -euo pipefail

PROJECT_DIR="/raid/home/${USER}/scripts/microRNA_project"
cd "${PROJECT_DIR}"
mkdir -p logs data/raw

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

python data/01_download/download_mirtarbase.py \
    --source mirtarbase \
    --config configs/config_mirtarbase_edgesplit.yaml

echo ""
echo "========================================"
echo "miRTarBase download complete: $(date)"
ls -lh data/raw/hsa_MTI_r10.0.csv data/raw/mirtarbase_real_hsa10_all.tsv
echo "========================================"
