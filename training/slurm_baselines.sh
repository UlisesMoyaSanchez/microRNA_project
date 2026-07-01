#!/bin/bash
# =============================================================================
# slurm_baselines.sh — Train & evaluate all baseline and ablation models.
#
# Runs run_baselines.py which covers:
#   random       — floor reference (no training)
#   mlp          — no graph structure
#   homo_gcn     — homogeneous GCN (no type-awareness)
#   ablation_no_mirna   — HGT without miRNA→gene edges
#   ablation_no_coexpr  — HGT without gene co-expression edges
#
# Results saved to: results/comparison/comparison_table.tsv
#
# Usage:
#   sbatch training/slurm_baselines.sh
#   To skip training and only evaluate saved checkpoints:
#   sbatch --export=ALL,SKIP_TRAINING=1 training/slurm_baselines.sh
# =============================================================================
#SBATCH --job-name=mirna_baselines
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --output=logs/baselines_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx_large

set -euo pipefail

PROJECT_DIR="/raid/home/${USER}/scripts/microRNA_project"
cd "${PROJECT_DIR}"
mkdir -p logs results/comparison

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms
set -u

CONFIG="${CONFIG:-configs/config_v2.yaml}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"

echo "========================================"
echo "Job:    ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "Config: ${CONFIG}"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

EXTRA_ARGS=""
if [[ "${SKIP_TRAINING}" == "1" ]]; then
    EXTRA_ARGS="--skip_training"
    echo "Mode: EVALUATE ONLY (skip training)"
else
    echo "Mode: TRAIN + EVALUATE"
fi

echo ""
echo "Running baseline & ablation experiments..."
python training/run_baselines.py \
    --config "${CONFIG}" \
    ${EXTRA_ARGS}

echo ""
echo "========================================"
echo "Results:"
cat results/comparison/comparison_table.tsv
echo ""
echo "Done: $(date)"
echo "========================================"
