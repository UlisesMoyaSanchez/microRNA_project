#!/bin/bash
# =============================================================================
# slurm_hmdd_survey_topology.sh — HMDD topology-baseline audit: does a
# no-learning heuristic close the gap to a surveyed paper's own reported AUROC,
# on that paper's own data?
#
# Parameterized wrapper over training/eval_hmdd_survey_topology_baseline.py,
# dispatching to configs/config_hmdd_survey_${PAPER}.yaml. One SBATCH file
# covers every paper in the HMDD-cluster audit (data/01_download/
# download_hmdd_survey_canonical5430.py and its Stage 2/3 siblings) — unlike the
# OGB extension's ddi/ppa split, every dataset here is small (<2,000 nodes), so
# one resource profile is enough.
#
# Prerequisite: data/01_download/inspect_hmdd_survey_sources.py must have been
# run and passed (all_checks_passed=true in
# results/comparison/hmdd_survey_protocol_verification.json) before trusting
# this job's output.
#
# Usage:
#   sbatch --export=ALL,PAPER=mgcnss training/slurm_hmdd_survey_topology.sh
#   sbatch --export=ALL,PAPER=nimgsa training/slurm_hmdd_survey_topology.sh
#   sbatch --export=ALL,PAPER=hlgnn_mda training/slurm_hmdd_survey_topology.sh
# =============================================================================
#SBATCH --job-name=hmdd_survey_topo
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --output=logs/hmdd_survey_topo_%j.out
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

if [[ -z "${PAPER:-}" ]]; then
    echo "error: set PAPER (mgcnss|nimgsa|hlgnn_mda|...) via --export=ALL,PAPER=<name>" >&2
    exit 1
fi
CONFIG="configs/config_hmdd_survey_${PAPER}.yaml"

echo "========================================"
echo "Job:    ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "Paper:  ${PAPER}"
echo "Config: ${CONFIG}"
echo "========================================"
echo ""

python data/01_download/inspect_hmdd_survey_sources.py

python training/eval_hmdd_survey_topology_baseline.py --config "${CONFIG}"

echo ""
echo "========================================"
echo "Results:"
cat "results/comparison/hmdd_survey_topology_baseline_${PAPER}.json"
echo ""
echo "Done: $(date)"
echo "========================================"
