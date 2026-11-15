#!/bin/bash
# =============================================================================
# slurm_ngcn_rerun.sh -- NGCN's own released code, as released and corrected step by step
# (analysis/ngcn_rerun.py; BITACORA 2026-11-15).
#
# NGCN (github.com/Junyue28/NGCN @ 7924dbf: NGCN.rar + data.rar, no licence) is not vendored: the two
# archives are extracted into /raid/home/$USER/scripts/external_code/NGCN (data/ inside the code folder).
# DTINet (github.com/luoyunan/DTINet, GPL-3) supplies the DTI-free control features.
#
# Usage: RATIO=1 sbatch --export=ALL training/slurm_ngcn_rerun.sh   (RATIO=10 for the 1:10 setting)
# =============================================================================
#SBATCH --job-name=ngcn_rerun
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --output=logs/ngcn_rerun_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx_large

set -euo pipefail
cd "/raid/home/${USER}/scripts/microRNA_project"
mkdir -p logs results/comparison
EXT="/raid/home/${USER}/scripts/external_code"
[ -f "${EXT}/NGCN/main.py" ] && [ -f "${EXT}/NGCN/data/mat_drug_protein.txt" ] || { echo "NGCN not extracted"; exit 1; }

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms
set -u

python analysis/ngcn_rerun.py --repo "${EXT}/NGCN" --dtinet "${EXT}/DTINet" --ratio "${RATIO:-1}" \
    --epochs "${EPOCHS:-3000}" 2>&1 | grep --line-buffered -v -E "UserWarning|warnings.warn|_register_pytree"
echo "Done: $(date)"
