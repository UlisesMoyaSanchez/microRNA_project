#!/bin/bash
# =============================================================================
# slurm_nimgsa_rerun.sh -- NIMGSA's own model under its published protocol and under
# progressively corrected ones (analysis/nimgsa_rerun.py; BITACORA 2026-11-14).
#
# NIMGSA's code has no licence, so it is not vendored: it is cloned once, outside the repo,
# pinned to commit 5bf10a9, and only its models.py is imported.
#
# Usage: sbatch training/slurm_nimgsa_rerun.sh        (SEEDS / EPOCHS via the environment)
# =============================================================================
#SBATCH --job-name=nimgsa_rerun
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --output=logs/nimgsa_rerun_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx_large

set -euo pipefail
cd "/raid/home/${USER}/scripts/microRNA_project"
mkdir -p logs results/comparison

NIMGSA_DIR="/raid/home/${USER}/scripts/external_code/NIMGSA"
NIMGSA_COMMIT=5bf10a9
if [ ! -d "${NIMGSA_DIR}" ]; then
    git clone -q https://github.com/zhanglabNKU/NIMGSA.git "${NIMGSA_DIR}"
fi
git -C "${NIMGSA_DIR}" checkout -q "${NIMGSA_COMMIT}"
echo "NIMGSA at $(git -C "${NIMGSA_DIR}" rev-parse --short HEAD)"

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms
set -u

python analysis/nimgsa_rerun.py --nimgsa-dir "${NIMGSA_DIR}" \
    --seeds ${SEEDS:-1 2 3 4 5} --epochs "${EPOCHS:-300}"
echo "Done: $(date)"
