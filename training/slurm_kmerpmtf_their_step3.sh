#!/bin/bash
# slurm_kmerpmtf_their_step3.sh -- kmerPMTF's own step3_train.py, unmodified, on the rerun's preprocessed
# inputs (analysis/kmerpmtf_their_step3.py). Usage: SPECIES=Prunus_persica K=2 sbatch --export=ALL ...
#SBATCH --job-name=kmer_s3
#SBATCH --cpus-per-task=8
#SBATCH --mem=200gb
#SBATCH --output=logs/kmerpmtf_their_step3_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx_large

set -euo pipefail
cd "/raid/home/${USER}/scripts/microRNA_project"
EXT="/raid/home/${USER}/scripts/external_code/kmerPMTF"
set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms
set -u
python analysis/kmerpmtf_their_step3.py --repo "${EXT}" --data "${EXT}/data" --species "${SPECIES}" \
    --chebconvk "${K:-2}" --epoch "${EPOCHS:-3000}" 2>&1 | grep --line-buffered -v -E "UserWarning|warnings.warn"
echo "Done: $(date)"
