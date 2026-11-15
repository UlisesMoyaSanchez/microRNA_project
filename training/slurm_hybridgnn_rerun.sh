#!/bin/bash
# =============================================================================
# slurm_hybridgnn_rerun.sh -- HybridGNN's own released code, as released and corrected step by
# step (analysis/hybridgnn_rerun.py; BITACORA 2026-11-14).
#
# HybridGNN's repository has no licence, so it is not vendored: it is cloned once, outside the
# repo, pinned to commit f441278, and its cv_mirna_split.py is imported unmodified.
#
# Usage: DATASET=hmdd_v4|canonical sbatch --export=ALL training/slurm_hybridgnn_rerun.sh
#        (SEEDS / EPOCHS also via the environment; SCRIPT=hybridgnn_leak_factorial.py for the
#        three-leak factorial)
# =============================================================================
#SBATCH --job-name=hybridgnn_rerun
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --output=logs/hybridgnn_rerun_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx_large

set -euo pipefail
cd "/raid/home/${USER}/scripts/microRNA_project"
mkdir -p logs results/comparison

REPO_DIR="/raid/home/${USER}/scripts/external_code/HybridGNN-miRNA-disease"
REPO_COMMIT=f441278
if [ ! -d "${REPO_DIR}" ]; then
    git clone -q https://github.com/mbasharatahmad/HybridGNN-miRNA-disease.git "${REPO_DIR}"
fi
# Check out only if needed: two jobs sharing the clone must not both run git (index.lock, job 3726).
if [ "$(git -C "${REPO_DIR}" rev-parse --short HEAD)" != "${REPO_COMMIT}" ]; then
    git -C "${REPO_DIR}" checkout -q "${REPO_COMMIT}"
fi
echo "HybridGNN at $(git -C "${REPO_DIR}" rev-parse --short HEAD), dataset ${DATASET:?set DATASET}"

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms
set -u

python "analysis/${SCRIPT:-hybridgnn_rerun.py}" --repo "${REPO_DIR}" --dataset "${DATASET}" \
    --seeds ${SEEDS:-42 1 2 3 4} --epochs "${EPOCHS:-200}" 2>&1 | grep -v -E "UserWarning|warnings.warn|_register_pytree"
echo "Done: $(date)"
