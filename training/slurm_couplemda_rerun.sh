#!/bin/bash
# =============================================================================
# slurm_couplemda_rerun.sh -- CoupleMDA's own released code, with its test negatives changed one way at a
# time (analysis/couplemda_rerun.py; BITACORA 2026-11-15).
#
# CoupleMDA (github.com/lizhj39/CoupleMDA @ c9128cb, no licence) is cloned into
# /raid/home/$USER/scripts/external_code/CoupleMDA. It needs DGL, which is not in mirna_ms: DGL 2.1.0 (CPU
# build; data.dgl.ai, which hosts the CUDA builds, is unreachable) and torchdata 0.7.1 are installed with
# pip --target into external_code/pydeps_dgl. PYTHONNOUSERSITE keeps ~/.local's `datasets` (which breaks
# on this pyarrow when torchdata imports it) out of the path.
#
# Usage: sbatch --export=ALL training/slurm_couplemda_rerun.sh   (EPOCHS, SEEDS optional)
# =============================================================================
#SBATCH --job-name=couplemda
#SBATCH --cpus-per-task=32
#SBATCH --mem=96gb
#SBATCH --output=logs/couplemda_rerun_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large

set -euo pipefail
cd "/raid/home/${USER}/scripts/microRNA_project"
mkdir -p logs results/comparison
EXT="/raid/home/${USER}/scripts/external_code"
[ "$(git -C "${EXT}/CoupleMDA" rev-parse --short HEAD)" = "c9128cb" ] || { echo "CoupleMDA not at c9128cb"; exit 1; }

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms
set -u
export PYTHONNOUSERSITE=1 PYTHONPATH="${EXT}/pydeps_dgl" CUDA_VISIBLE_DEVICES="" DGLBACKEND=pytorch
export OMP_NUM_THREADS=32

python analysis/couplemda_rerun.py --repo "${EXT}/CoupleMDA" --epochs "${EPOCHS:-100}" --seeds ${SEEDS:-5555 1 2} 2>&1 \
    | grep --line-buffered -v -E "Epoch [0-9]+ \| Iteration [0-9]+ \| Train_Loss [0-9.]+ \| lr|evaluating|it/s\]|s/it\]|UserWarning|warnings.warn"
echo "Done: $(date)"
