#!/bin/bash
# =============================================================================
# slurm_mgcnss_rerun.sh -- MGCNSS's own released code, as released and corrected step by step
# (analysis/mgcnss_rerun.py; BITACORA 2026-11-16).
#
# MGCNSS (github.com/15136943622/MGCNSS @ f8a87e6, no licence) is not vendored: it is cloned into
# /raid/home/$USER/scripts/external_code/MGCNSS. Runs in mirna_ms (plain PyTorch; src/Utils.py, shipped only
# as bytecode, is provided by the driver).
#
# Usage: sbatch training/slurm_mgcnss_rerun.sh        (EPOCHS=20 SEEDS=1 for a smoke test)
# =============================================================================
#SBATCH --job-name=mgcnss
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --output=logs/mgcnss_rerun_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large

set -euo pipefail
P="/raid/home/${USER}/scripts/microRNA_project"
cd "$P"
mkdir -p logs results/comparison
R="/raid/home/${USER}/scripts/external_code/MGCNSS"
[ "$(git -C "$R" rev-parse --short HEAD)" = "f8a87e6" ] || { echo "MGCNSS not at f8a87e6"; exit 1; }
set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms
set -u
python analysis/mgcnss_rerun.py --repo "$R" --seeds ${SEEDS:-1 2 3} --epochs "${EPOCHS:-2000}" 2>&1 \
  | grep --line-buffered -v -E "^stdv|^layer:|Warning|warnings.warn"
echo "Done: $(date)"
