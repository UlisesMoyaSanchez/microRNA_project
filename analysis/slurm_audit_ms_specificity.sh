#!/bin/bash
# =============================================================================
# slurm_audit_ms_specificity.sh — Evidence for EVALUATION_AUDIT.md Contribution 4:
# the MS label never enters the graph.
#
# Read-only. Writes exactly one file: results/comparison/ms_specificity_audit.json
# No GPU — it loads the graph on CPU and computes summary statistics.
#
# MUST RUN BEFORE the Track B wiring fix: it documents the pre-fix state, and the
# graph_sha256 it records is what lets a reviewer tell the two states apart once
# the graph is rebuilt.
#
# Usage: sbatch analysis/slurm_audit_ms_specificity.sh
#        CONFIG=configs/config_v2.yaml sbatch analysis/slurm_audit_ms_specificity.sh
# =============================================================================
#SBATCH --job-name=ms_audit
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --output=logs/audit_ms_specificity_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large

set -euo pipefail

PROJECT_DIR="/raid/home/${USER}/scripts/microRNA_project"
cd "${PROJECT_DIR}"
mkdir -p logs results/comparison

# Config is overridable rather than hardcoded — the bug in slurm_build_graph.sh
# that would silently audit the wrong graph.
CONFIG="${CONFIG:-configs/config_v2.yaml}"

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms
set -u

echo "========================================"
echo "Job:    ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "Config: ${CONFIG}"
echo "Git:    $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "========================================"

python analysis/audit_ms_specificity.py --config "${CONFIG}"

echo ""
echo "========================================"
echo "MS-specificity audit complete: $(date)"
ls -lh "${PROJECT_DIR}/results/comparison/ms_specificity_audit.json"
echo "========================================"
