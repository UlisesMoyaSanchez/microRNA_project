#!/bin/bash
# =============================================================================
# slurm_preprocess.sh — Preprocess scRNA-seq and bulk miRNA data.
# High CPU + RAM, no GPU required.
#
# Usage: sbatch data/02_preprocess/slurm_preprocess.sh
# Dependency: run after slurm_download.sh completes.
# =============================================================================
#SBATCH --job-name=mirna_preprocess
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large

set -euo pipefail

PROJECT_DIR="/raid/home/${USER}/scripts/microRNA_project"
cd "${PROJECT_DIR}"
mkdir -p logs data/processed

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms
set -u

echo "========================================"
echo "Job:    ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "CPUs:   ${SLURM_CPUS_PER_TASK}"
echo "Mem:    ${SLURM_MEM_PER_NODE:-128gb}"
echo "========================================"

echo ""
echo "[1/2] Preprocessing scRNA-seq (QC, HVG, PCA, Leiden, cell-type annotation)..."
python data/02_preprocess/preprocess_scrna.py --config configs/config.yaml

echo ""
echo "[2/2] Preprocessing bulk miRNA expression (GEO SOFT parsing, normalization)..."
python data/02_preprocess/preprocess_mirna.py --config configs/config.yaml

echo ""
echo "========================================"
echo "Preprocessing complete: $(date)"
echo "Output: ${PROJECT_DIR}/data/processed/"
ls -lh "${PROJECT_DIR}/data/processed/"
echo "========================================"
