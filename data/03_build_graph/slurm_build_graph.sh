#!/bin/bash
# =============================================================================
# slurm_build_graph.sh — Build the heterogeneous PyG graph.
# Requires high RAM (PCC correlation matrix for 1000 genes).
# No GPU needed.
#
# Usage: sbatch data/03_build_graph/slurm_build_graph.sh
#        CONFIG=configs/config_mirtarbase_edgesplit.yaml sbatch data/03_build_graph/slurm_build_graph.sh
#        FORCE=1 CONFIG=... sbatch data/03_build_graph/slurm_build_graph.sh   # rebuild in place
# Dependency: slurm_preprocess.sh must have completed.
#
# The config used to be hardcoded to configs/config.yaml, so submitting this for any
# other interaction source built (or skipped) the miRDB graph and exited 0 — a green
# job that produced nothing the caller asked for.
# =============================================================================
#SBATCH --job-name=mirna_build_graph
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --output=logs/build_graph_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large

set -euo pipefail

PROJECT_DIR="/raid/home/${USER}/scripts/microRNA_project"
cd "${PROJECT_DIR}"
mkdir -p logs

CONFIG="${CONFIG:-configs/config.yaml}"
FORCE="${FORCE:-0}"

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms
set -u

# Derive the output dir FROM the config rather than assuming data/graphs — that
# assumption is what let a miRTarBase build write nothing and still report success.
GRAPHS_DIR=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['data']['graphs_dir'])" "${CONFIG}")
mkdir -p "${GRAPHS_DIR}"

echo "========================================"
echo "Job:    ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "Config: ${CONFIG}"
echo "Graphs: ${GRAPHS_DIR}"
echo "Force:  ${FORCE}"
echo "========================================"

FORCE_FLAG=""
[ "${FORCE}" = "1" ] && FORCE_FLAG="--force"

python data/03_build_graph/build_heterograph.py --config "${CONFIG}" ${FORCE_FLAG}

echo ""
echo "========================================"
echo "Graph build complete: $(date)"
echo "Output: ${GRAPHS_DIR}"
ls -lh "${GRAPHS_DIR}"
echo "========================================"
