#!/bin/bash
# =============================================================================
# slurm_train.sh — Multi-GPU A100 training with torchrun (DDP / NCCL).
#
# Usage:         sbatch training/slurm_train.sh
# Resume:        sbatch training/slurm_train.sh --resume
# Dependency:    slurm_build_graph.sh must have completed.
#
# Adjust --gres=gpu:N to change the number of GPUs (1–8 on dgxa100jal).
# =============================================================================
#SBATCH --job-name=mirna_train
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --output=logs/train_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --gres=gpu:4
#SBATCH --partition=dgx_large

set -euo pipefail

PROJECT_DIR="/raid/home/${USER}/scripts/microRNA_project"
cd "${PROJECT_DIR}"
mkdir -p logs checkpoints

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms
set -u

# CONFIG can be overridden via:  sbatch --export=ALL,CONFIG=configs/config_v2.yaml training/slurm_train.sh
CONFIG="${CONFIG:-configs/config.yaml}"

# Pick up optional --resume flag passed via sbatch
EXTRA_ARGS="${@:-}"

NGPUS=$(echo "${SLURM_GPUS_ON_NODE:-${SLURM_JOB_GPUS:-4}}" | tr ',' '\n' | wc -l)
# Ensure NGPUS is an integer
NGPUS=$(python -c "print(int('${NGPUS}'.strip()))")

echo "========================================"
echo "Job:    ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "GPUs:   ${NGPUS}"
echo "CPUs:   ${SLURM_CPUS_PER_TASK}"
echo "========================================"
echo "Config: ${CONFIG}"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "Launching torchrun with ${NGPUS} processes..."
torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    training/train.py \
    --config "${CONFIG}" \
    ${EXTRA_ARGS}

echo ""
echo "========================================"
echo "Training complete: $(date)"
echo "Checkpoint: ${PROJECT_DIR}/checkpoints/best_model.pt"
echo "========================================"
