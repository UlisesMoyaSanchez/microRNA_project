#!/bin/bash
# =============================================================================
# slurm_survey_floor_seeds.sh -- the model-free floor of the five survey graphs, averaged
# over several splits instead of one.
#
# The published floors (hmdd_survey_topology_baseline_<paper>[_seen].json) are one split at
# the config seed (42). The independent audit (BITACORA 2026-11-02) measured the corrected
# floor 0.011-0.019 lower as a mean of five splits. This reruns the same script at the
# trained grid's seeds (analysis/aggregate_trained_grid.py::SEEDS) in both edge regimes, to
# `..._s<seed>.json`; seed 42 must reproduce the published file exactly.
#
# Usage: sbatch training/slurm_survey_floor_seeds.sh
# =============================================================================
#SBATCH --job-name=survey_floor_seeds
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --output=logs/survey_floor_seeds_%j.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large

set -euo pipefail
cd "/raid/home/${USER}/scripts/microRNA_project"
mkdir -p logs results/comparison

set +u
source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate mirna_ms
set -u

SEEDS="${SEEDS:-42 7 123 2024}"
for paper in nimgsa cksnp_gnn digamn meahne couplemda; do
    for regime in held_out seen; do
        for s in ${SEEDS}; do
            echo "== ${paper} ${regime} seed=${s}"
            python training/eval_hmdd_survey_topology_baseline.py \
                --config "configs/config_hmdd_survey_${paper}.yaml" \
                --edge-regime "${regime}" --seed "${s}" 2>&1 | grep -E "Wrote|Error|Traceback" || true
        done
    done
done
echo "Done: $(date)"
