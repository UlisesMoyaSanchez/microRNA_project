#!/bin/bash
# =============================================================================
# slurm_independent_eval.sh — Evaluate V2 model on an independent dataset.
#
# Tests generalization to data NOT seen during training.
# Uses evaluate_independent.py to:
#   1. Preprocess the independent AnnData (aligned to training gene set)
#   2. Build a new HeteroData for these cells
#   3. Load V2 checkpoint and measure AUROC, AUPRC, cell-type accuracy
#
# Results saved to: results/comparison/independent_eval.json
#                   results/comparison/independent_eval_report.txt
#
# Usage:
#   # With a specific h5ad file:
#   sbatch --export=ALL,H5AD=/path/to/dataset.h5ad training/slurm_independent_eval.sh
#
#   # Example with GEO data already downloaded:
#   sbatch --export=ALL,H5AD=data/raw/geo/GSE289530/GSE289530.h5ad \
#          training/slurm_independent_eval.sh
# =============================================================================
#SBATCH --job-name=mirna_indep_eval
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --output=logs/indep_eval_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx_large

set -euo pipefail

PROJECT_DIR="/raid/home/${USER}/scripts/microRNA_project"
cd "${PROJECT_DIR}"
mkdir -p logs results/comparison

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms
set -u

CONFIG="${CONFIG:-configs/config_v2.yaml}"
# Override H5AD via: sbatch --export=ALL,H5AD=/path/to/file.h5ad
H5AD="${H5AD:-data/raw/geo/GSE289530/GSE289530.h5ad}"
CONDITION_COL="${CONDITION_COL:-condition}"
CELLTYPE_COL="${CELLTYPE_COL:-cell_type}"

echo "========================================"
echo "Job:         ${SLURM_JOB_ID:-local}"
echo "Node:        $(hostname)"
echo "Date:        $(date)"
echo "Config:      ${CONFIG}"
echo "Dataset:     ${H5AD}"
echo "Cond col:    ${CONDITION_COL}"
echo "CT col:      ${CELLTYPE_COL}"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ ! -f "${H5AD}" ]]; then
    echo "ERROR: h5ad file not found: ${H5AD}"
    echo "Please set the H5AD environment variable to an existing .h5ad file."
    echo "Example: sbatch --export=ALL,H5AD=/path/to/file.h5ad training/slurm_independent_eval.sh"
    exit 1
fi

echo ""
echo "Running independent evaluation..."
python training/evaluate_independent.py \
    --config "${CONFIG}" \
    --h5ad "${H5AD}" \
    --condition_col "${CONDITION_COL}" \
    --celltype_col "${CELLTYPE_COL}"

echo ""
echo "========================================"
echo "Results:"
cat results/comparison/independent_eval.json
echo ""
echo "Done: $(date)"
echo "========================================"
