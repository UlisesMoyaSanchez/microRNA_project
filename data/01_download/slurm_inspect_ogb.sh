#!/bin/bash
# =============================================================================
# slurm_inspect_ogb.sh — Step 1 of the OGB light-audit extension (ogbl-ddi,
# ogbl-ppa): download both datasets via OGB's own API and verify the split
# protocol assumptions in inspect_ogb_split.py before any scoring code is
# written. Read results/comparison/ogb_protocol_verification.json by hand
# after this completes.
#
# Usage: sbatch data/01_download/slurm_inspect_ogb.sh
# =============================================================================
#SBATCH --job-name=ogb_inspect
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --output=logs/ogb_inspect_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large

set -euo pipefail

PROJECT_DIR="/raid/home/${USER}/scripts/microRNA_project"
cd "${PROJECT_DIR}"
mkdir -p logs data/raw/ogb results/comparison

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

echo "Checking outbound access to OGB's data host..."
curl -sI --max-time 15 https://snap.stanford.edu/ogb/data/linkproppred/ddi.zip \
    | head -n1 || echo "WARNING: curl check failed — the Python download below will surface the real error."

python data/01_download/inspect_ogb_split.py --dataset both

echo ""
echo "========================================"
echo "OGB inspection complete: $(date)"
echo "Review results/comparison/ogb_protocol_verification.json by hand before"
echo "writing training/eval_ogb_topology_baseline.py."
echo "========================================"
