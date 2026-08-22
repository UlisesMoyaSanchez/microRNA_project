#!/bin/bash
# =============================================================================
# slurm_survey_bipartite.sh — train this project's model on a surveyed paper's own
# miRNA-disease graph, one cell of the trained protocol grid per job.
#
# Answers the question the model-free grid (Table 6) cannot: does a TRAINED model move
# across the protocol grid the way the no-learning floor does? MEAHNE and DiGAMN are the
# two ends of the observed gap range, so the answer cannot be an artifact of one end.
#
# Prerequisite: data/03_build_graph/build_survey_bipartite_graph.py --paper <name>
# must have been run (writes data/graphs_survey/<name>/hetero_graph.pt).
#
# Usage:
#   sbatch --export=ALL,PAPER=meahne,REGIME=held_out,SEED=42 training/slurm_survey_bipartite.sh
#   sbatch --export=ALL,PAPER=digamn,REGIME=seen,SEED=7     training/slurm_survey_bipartite.sh
# =============================================================================
#SBATCH --job-name=survey_bipartite
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --output=logs/survey_bipartite_%j.out
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

for v in PAPER REGIME SEED; do
    if [[ -z "${!v:-}" ]]; then
        echo "error: set ${v} via --export=ALL,PAPER=...,REGIME=...,SEED=..." >&2
        exit 1
    fi
done
if [[ "${REGIME}" != "held_out" && "${REGIME}" != "seen" ]]; then
    echo "error: REGIME must be held_out or seen, got '${REGIME}'" >&2
    exit 1
fi
CONFIG="configs/config_survey_${PAPER}_${REGIME}.yaml"

echo "========================================"
echo "Job:    ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "Paper:  ${PAPER}"
echo "Regime: ${REGIME}"
echo "Seed:   ${SEED}"
echo "Config: ${CONFIG}"
echo "========================================"
echo ""

python training/train_survey_bipartite.py --config "${CONFIG}" --seed "${SEED}"

echo ""
echo "Done: $(date)"
