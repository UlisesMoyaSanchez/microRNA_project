#!/bin/bash
# =============================================================================
# slurm_validate_lp_audit.sh -- lp_audit validation on the paper's graphs.
#   1. pytest tests/test_lp_audit.py (parity with the torch scorers, on the DGX env)
#   2. analysis/validate_lp_audit.py      (5 HMDD/CoupleMDA graphs, 2x2 protocol grid)
#   3. analysis/validate_lp_audit_ogb.py  (ogbl-ddi, homogeneous mode)
#
# Usage: sbatch training/slurm_validate_lp_audit.sh
# =============================================================================
#SBATCH --job-name=lp_audit_validate
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --output=logs/lp_audit_validate_%j.out
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

echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Date: $(date)"
echo "Git: $(git rev-parse --short HEAD)  Python: $(python --version)"

python -m pytest tests/test_lp_audit.py -q
python analysis/validate_lp_audit.py
python analysis/validate_lp_audit_ogb.py

echo "lp_audit validation complete: $(date)"
