#!/bin/bash
# Re-run GO/KEGG enrichment only, using already-generated top_circuits_by_celltype.tsv
#SBATCH --job-name=mirna_enrichment
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --output=logs/enrichment_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large

set -euo pipefail

PROJECT_DIR="/raid/home/${USER}/scripts/microRNA_project"
cd "${PROJECT_DIR}"
mkdir -p logs results/interpretation/enrichment

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms
set -u

echo "========================================"
echo "Job:    ${SLURM_JOB_ID:-local}"
echo "Date:   $(date)"
echo "========================================"

python - <<'PYEOF'
import os, sys, logging
import pandas as pd

sys.path.insert(0, "/raid/home/umoya/scripts/microRNA_project")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

try:
    import gseapy as gp
except ImportError:
    log.error("gseapy not installed")
    sys.exit(1)

circuits_path = "results/interpretation/top_circuits_by_celltype.tsv"
out_dir       = "results/interpretation/enrichment"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(circuits_path, sep="\t")
cell_types = df["cell_type"].unique()

gene_sets = {
    "GO_Biological_Process_2021": "GO",
    "KEGG_2021_Human":            "KEGG",
}

for cell_type in sorted(cell_types):
    safe_name = cell_type.replace(" ", "_").replace("/", "_")
    top_genes = df[df["cell_type"] == cell_type]["target_gene"].head(50).tolist()
    if len(top_genes) < 5:
        log.info(f"Skipping {cell_type}: too few genes ({len(top_genes)})")
        continue
    log.info(f"Enrichment for {cell_type} ({len(top_genes)} genes)...")
    for gene_set, tag in gene_sets.items():
        try:
            enr = gp.enrichr(
                gene_list=top_genes,
                gene_sets=gene_set,
                organism="human",
                outdir=None,
                verbose=False,
            )
            out_file = os.path.join(out_dir, f"{safe_name}_{tag}.tsv")
            enr.results.sort_values("Adjusted P-value").to_csv(out_file, sep="\t", index=False)
            sig = (enr.results["Adjusted P-value"] < 0.05).sum()
            log.info(f"  Saved {out_file}  ({sig} significant terms)")
        except Exception as exc:
            log.warning(f"  {cell_type}/{gene_set}: {exc}")

log.info("Enrichment complete.")
PYEOF

echo "========================================"
echo "Done: $(date)"
ls -lh results/interpretation/enrichment/
echo "========================================"
