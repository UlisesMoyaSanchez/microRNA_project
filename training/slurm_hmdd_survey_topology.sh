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
#   sbatch --export=ALL,PAPER=meahne,REGIME=seen training/slurm_hmdd_survey_topology.sh
#
# REGIME selects which edges the heuristics may see (default held_out, the corrected
# protocol and the one the published artifacts were produced under). REGIME=seen is the
# conventional protocol -- together the two make up the 2x2 protocol grid this project
# measures on its own graph. Output goes to a _seen-suffixed JSON, so a seen run can
# never overwrite a held_out artifact.
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
REGIME="${REGIME:-held_out}"
if [[ "${REGIME}" != "held_out" && "${REGIME}" != "seen" ]]; then
    echo "error: REGIME must be held_out or seen, got '${REGIME}'" >&2
    exit 1
fi
# Explicit if, not `[[ ]] && VAR=`: under `set -e` that idiom's exit status depends on a
# bash subtlety, and this script runs unattended.
if [[ "${REGIME}" == "seen" ]]; then
    SUFFIX="_seen"
else
    SUFFIX=""
fi

echo "========================================"
echo "Job:    ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "Paper:  ${PAPER}"
echo "Regime: ${REGIME}"
echo "Config: ${CONFIG}"
echo "========================================"
echo ""

python data/01_download/inspect_hmdd_survey_sources.py

python training/eval_hmdd_survey_topology_baseline.py --config "${CONFIG}" \
    --edge-regime "${REGIME}"

echo ""
echo "========================================"
echo "Results:"
cat "results/comparison/hmdd_survey_topology_baseline_${PAPER}${SUFFIX}.json"
echo ""
echo "Done: $(date)"
echo "========================================"
