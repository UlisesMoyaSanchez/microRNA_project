---
name: pipeline-status
description: Checks where the miRNA-MS data/training pipeline stands on the DGX cluster (dgxum) — currently running/queued SLURM jobs plus which pipeline stage (download/preprocess/build_graph/train/evaluate/analysis) has completed, based on expected output artifacts and logs. Use when checking on a submitted job or figuring out what stage the pipeline is at, instead of manually ssh-ing and tailing logs.
---

# Pipeline Status

All real work for this project happens on the DGX cluster over SSH
(`ssh dgxum`), project path `/raid/home/umoya/scripts/microRNA_project` — not
locally. This skill checks that machine.

## Steps

1. **Check the SLURM queue:**
   ```
   ssh dgxum "squeue -u umoya"
   ```
   Report any running or pending jobs (job name, state, time).

2. **Check pipeline-stage artifacts, in order**, to determine what's already
   complete (run one combined `ssh` call):
   - Download: `data/raw/cellxgene_ms.h5ad`, `data/raw/cellxgene_ctrl.h5ad`,
     `data/raw/mirtarbase_hsa.tsv`, `data/raw/geo/GSE289530/`
   - Preprocess: `data/processed/scrna_processed.h5ad`,
     `data/processed/mirna_expr.tsv`, `data/processed/cell_type_labels.txt`
   - Build graph: `data/graphs/hetero_graph.pt`, `data/graphs/index_maps.pkl`
   - Train: `checkpoints*/best_model.pt` (check all `checkpoints*` dirs, not
     just the default one — configs write to variant-specific dirs)
   - Analysis: `results/REPORT.md`, `results/interpretation/*.tsv`,
     `results/figures/*.pdf`

   Use `ls -la --time-style=full-iso` (or `stat`) so you can report mtimes,
   not just existence — a stale artifact from days ago vs. one just written
   is a meaningfully different status.

3. **Tail the most relevant log** under `logs/` for whichever stage is
   currently running (per the queue) or most recently completed (per
   mtimes) — logs are named `<stage>_<jobid>.out` /
   `train_rank{N}.log` / `baselines.log`. Show the last ~20 lines.

4. **Report a stage-by-stage table**: done / running / not started, with the
   artifact mtime or the log tail as evidence — don't just say "looks done,"
   show what you checked.
