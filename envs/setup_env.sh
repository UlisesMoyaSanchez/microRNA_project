#!/bin/bash
# =============================================================================
# setup_env.sh — Create conda environment 'mirna_ms' with all project deps.
#
# Strategy: clone the existing 'torch' env (preserves PyTorch + CUDA setup),
# then add bioinformatics and graph-ML libraries on top.
#
# Usage:  sbatch envs/setup_env.sh
# =============================================================================
#SBATCH --job-name=mirna_env_setup
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --output=logs/setup_env_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large

set -euo pipefail

PROJECT_DIR="/raid/home/${USER}/scripts/microRNA_project"
ENV_NAME="mirna_ms"

mkdir -p "${PROJECT_DIR}/logs"

echo "========================================"
echo "Job:   ${SLURM_JOB_ID:-local}"
echo "Node:  $(hostname)"
echo "Date:  $(date)"
echo "Target env: ${ENV_NAME}"
echo "========================================"

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
set -u

# ── Clone base torch env if mirna_ms does not exist ──────────────────────────
if conda env list | grep -q "^${ENV_NAME}"; then
    echo "Environment '${ENV_NAME}' already exists — skipping clone."
else
    echo "[1/4] Cloning 'torch' → '${ENV_NAME}'..."
    conda create --name "${ENV_NAME}" --clone torch -y
fi

set +u
conda activate "${ENV_NAME}"
set -u

echo "Python: $(python --version)"
echo "Torch:  $(python -c 'import torch; print(torch.__version__)')"

# ── Detect PyTorch + CUDA version for PyG wheels ─────────────────────────────
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_TAG=$(python -c "
import torch
cv = torch.version.cuda
if cv:
    tag = 'cu' + cv.replace('.','')
    print(tag[:5])   # e.g. cu121
else:
    print('cpu')
")
echo "Torch version: ${TORCH_VER}  CUDA tag: ${CUDA_TAG}"

# ── Install PyTorch Geometric ────────────────────────────────────────────────
echo "[2/4] Installing PyTorch Geometric..."
pip install torch_geometric

PYG_WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"
echo "  PyG wheel index: ${PYG_WHEEL_URL}"
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f "${PYG_WHEEL_URL}" || \
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    --no-index -f "${PYG_WHEEL_URL}" || \
echo "  WARNING: pyg_lib extras failed — torch_geometric still usable without them."

# ── Install bioinformatics libraries ─────────────────────────────────────────
echo "[3/4] Installing bioinformatics libraries..."
pip install \
    scanpy \
    anndata \
    "cellxgene-census>=1.9.0" \
    biopython \
    gseapy \
    leidenalg \
    igraph

# pyarrow_hotfix (required by tiledbsoma/somacore) fails to import when
# pyarrow >= 14.0, because pa.unregister_extension_type("arrow.py_extension_type")
# no longer works in newer pyarrow. Pin to <14 to fix.
echo "  Pinning pyarrow < 14 (pyarrow_hotfix compatibility)..."
pip install "pyarrow>=12.0,<14.0"

# ── Install utilities ────────────────────────────────────────────────────────
echo "[4/4] Installing utilities..."
pip install \
    pandas \
    openpyxl \
    scipy \
    matplotlib \
    seaborn \
    "umap-learn>=0.5" \
    networkx \
    tqdm \
    pyyaml

# ── Smoke test ───────────────────────────────────────────────────────────────
echo "========================================"
echo "Smoke test..."
python - <<'EOF'
import torch
import torch_geometric
import scanpy
import anndata
import cellxgene_census
import gseapy
print(f"  torch            {torch.__version__}")
print(f"  torch_geometric  {torch_geometric.__version__}")
print(f"  scanpy           {scanpy.__version__}")
print(f"  anndata          {anndata.__version__}")
print(f"  cellxgene_census OK")
print(f"  gseapy           {gseapy.__version__}")
print("All imports OK.")
EOF

echo "========================================"
echo "Environment '${ENV_NAME}' is ready: $(date)"
echo "Activate with: conda activate ${ENV_NAME}"
echo "========================================"
