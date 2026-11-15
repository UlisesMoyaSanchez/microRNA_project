#!/bin/bash
# =============================================================================
# slurm_hgdti_rerun.sh -- HGDTI's own released code, as released and corrected step by step
# (analysis/hgdti_rerun.py; BITACORA 2026-11-14).
#
# HGDTI's materials (Google Drive folder from the paper) have no licence and are not vendored:
# they live in /raid/home/$USER/scripts/external_code/HGDTI, with src/tools.py reconstructed from
# the released tools.cpython-37.pyc (decompyle3; the authors released no source for it).
#
# Usage: SEEDS=10 sbatch --export=ALL training/slurm_hgdti_rerun.sh   (one job per seed)
# =============================================================================
#SBATCH --job-name=hgdti_rerun
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --output=logs/hgdti_rerun_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx_large

set -euo pipefail
cd "/raid/home/${USER}/scripts/microRNA_project"
mkdir -p logs results/comparison
REPO_DIR="/raid/home/${USER}/scripts/external_code/HGDTI"
PYC_SHA=da4e0532b9494fb4
[ "$(sha256sum "${REPO_DIR}/src/__pycache__/tools.cpython-37.pyc" | cut -c1-16)" = "${PYC_SHA}" ] \
    || { echo "tools.cpython-37.pyc does not match ${PYC_SHA}"; exit 1; }
[ -f "${REPO_DIR}/src/tools.py" ] || { echo "missing reconstructed src/tools.py"; exit 1; }

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms
set -u

python analysis/hgdti_rerun.py --repo "${REPO_DIR}" --seeds ${SEEDS:-10} --epochs "${EPOCHS:-1000}" 2>&1 \
    | grep --line-buffered -v -E "Epoch: |^folder |UserWarning|warnings.warn|_register_pytree"
echo "Done: $(date)"
