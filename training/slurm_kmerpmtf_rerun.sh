#!/bin/bash
# =============================================================================
# slurm_kmerpmtf_rerun.sh -- kmerPMTF's own released code, as released and corrected step by step
# (analysis/kmerpmtf_rerun.py; BITACORA 2026-11-15).
#
# kmerPMTF (github.com/Weihankk/kmerPMTF @ c4f9a3e, no licence) is not vendored: it is cloned into
# /raid/home/$USER/scripts/external_code/kmerPMTF, with its figshare data (24916023, CC BY 4.0) in data/.
#
# Usage: SPECIES=Prunus_persica sbatch --export=ALL training/slurm_kmerpmtf_rerun.sh
#        (EPOCHS, SEEDS, FOLDS optional)
# =============================================================================
#SBATCH --job-name=kmerpmtf
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --output=logs/kmerpmtf_rerun_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx_large

set -euo pipefail
cd "/raid/home/${USER}/scripts/microRNA_project"
mkdir -p logs results/comparison
EXT="/raid/home/${USER}/scripts/external_code/kmerPMTF"
[ "$(git -C "${EXT}" rev-parse --short HEAD)" = "c4f9a3e" ] || { echo "kmerPMTF not at c4f9a3e"; exit 1; }

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms
set -u

python analysis/kmerpmtf_rerun.py --repo "${EXT}" --data "${EXT}/data" --species "${SPECIES}" \
    --epochs "${EPOCHS:-3000}" --seeds ${SEEDS:-123 1 2} --folds "${FOLDS:-5}" 2>&1 \
    | grep --line-buffered -v -E "UserWarning|warnings.warn|_register_pytree"
echo "Done: $(date)"
