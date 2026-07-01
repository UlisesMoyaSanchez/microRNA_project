#!/bin/bash
# =============================================================================
# launch_one.sh - ejecutar main.py con SLURM.
#
# Usage: sbatch launch_one.sh
# =============================================================================
#SBATCH --job-name=cpa_main
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --output=main_%j_%x.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx_large

set -euo pipefail

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate torch
set -u

cd "/raid/home/${USER}/scripts/complex_product_attention"

echo "========================================"
echo "Job:      ${SLURM_JOB_ID:-local}"
echo "Node:     $(hostname)"
echo "Date:     $(date)"
echo "Script:   main.py"
echo "========================================"

python main.py