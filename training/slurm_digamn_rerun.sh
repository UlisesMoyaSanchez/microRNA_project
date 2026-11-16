#!/bin/bash
# =============================================================================
# slurm_digamn_rerun.sh -- DiGAMN's own released code, as released and corrected step by step
# (analysis/digamn_rerun.py; BITACORA 2026-11-16).
#
# DiGAMN (github.com/yinboliu-git/GAMN @ 6695f4e, no licence) is cloned into
# /raid/home/$USER/scripts/external_code/GAMN. It needs DGL: the CPU build in external_code/pydeps_dgl (see
# slurm_couplemda_rerun.sh), so this runs on CPU.
#
# Usage: sbatch --export=ALL,SEEDS="521",EPOCHS=5000 training/slurm_digamn_rerun.sh
# =============================================================================
#SBATCH --job-name=digamn
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --output=logs/digamn_rerun_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large

set -euo pipefail
cd "/raid/home/${USER}/scripts/microRNA_project"
mkdir -p logs results/comparison
EXT="/raid/home/${USER}/scripts/external_code"
[ "$(git -C "${EXT}/GAMN" rev-parse --short HEAD)" = "6695f4e" ] || { echo "GAMN not at 6695f4e"; exit 1; }

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms
set -u
export PYTHONNOUSERSITE=1 PYTHONPATH="${EXT}/pydeps_dgl" CUDA_VISIBLE_DEVICES="" DGLBACKEND=pytorch
export OMP_NUM_THREADS=32 PYTHONWARNINGS=ignore

python analysis/digamn_rerun.py --repo "${EXT}/GAMN" --epochs "${EPOCHS:-5000}" --seeds ${SEEDS:-521 1 2} 2>&1 \
    | grep --line-buffered -v -E "UserWarning|warnings.warn|DeprecationWarning"
echo "Done: $(date)"
