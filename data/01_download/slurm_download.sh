#!/bin/bash
# =============================================================================
# slurm_download.sh — Download all raw data (CellxGene, GEO, miRTarBase).
# No GPU required; high I/O node.
#
# Usage: sbatch data/01_download/slurm_download.sh
# =============================================================================
#SBATCH --job-name=mirna_download
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --output=logs/download_%j.out
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

echo ""
echo "[1/3] Downloading CellxGene scRNA-seq (MS + controls)..."
#python data/01_download/download_cellxgene.py --config configs/config.yaml
echo "  [SKIP] CellxGene already downloaded."

echo ""
echo "[2/3] Downloading GEO miRNA datasets..."
python data/01_download/download_geo_mirna.py --config configs/config.yaml

echo ""
echo "[3/3] Downloading miRTarBase interactions..."
#python data/01_download/download_mirtarbase.py --config configs/config.yaml
echo "  [SKIP] miRDB already downloaded (mirtarbase_hsa.tsv)."

echo ""
echo "========================================"
echo "All downloads complete: $(date)"
echo "Output: ${PROJECT_DIR}/data/raw/"
du -sh "${PROJECT_DIR}/data/raw/"
echo "========================================"
