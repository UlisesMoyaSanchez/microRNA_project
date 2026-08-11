# Reproducing "Transductive Evaluation and Popularity-Biased Negative Sampling Substantially Inflate microRNA–Target Link Prediction"

This branch (`reproduce/paper-v1`) is a **squashed, curated snapshot** of the
project repository, containing exactly what is needed to re-run the paper's
experiments: the data pipeline, model code, training/evaluation scripts, the
configs used for every reported number, and the headline result artifacts
that feed the paper's tables and figures. It deliberately excludes the
manuscript drafts, slide decks, internal audit narratives, and superseded
result dumps that live on the project's main development branch — this is a
reproduction package, not the full research history.

## What the paper is about

GNN papers predicting microRNA–target interactions routinely report AUROC in
the 0.91–0.99 range. This paper audits *why*: on a real biomedical
heterogeneous graph (single-cell RNA-seq + miRDB v6.0 miRNA–target
interactions), a **conventional evaluation protocol** (message-passing edges
seen during training, uniform-random negatives) reports **AUROC 0.9867 ±
0.0011** (4 seeds, test set). Correcting two flaws simultaneously — holding
out edges from message passing, and sampling negatives that match the
degree distribution instead of uniformly — drops this to **AUROC 0.6276 ±
0.0070**, within 3.6 points of a one-line, no-learning topological heuristic
(Adamic–Adar, 0.5912). The inflation recurs across six independent model
architectures (including a graph-free MLP) and does not appear in a control
task (cell-type classification) run through the same pipeline. An external
audit on two public OGB link-prediction benchmarks (`ogbl-ddi`, `ogbl-ppa`)
reproduces OGB's own published leaderboard numbers exactly, validating the
heuristic-baseline implementation independently of this project's graph.

The microRNA–Multiple Sclerosis graph is the **worked example**, not the
paper's claim — the contribution is the evaluation-protocol audit and the
minimum reporting standard it argues for (an automated leak-free split check
plus at least one model-free baseline), not a disease-biology finding.

## Environment setup

```bash
python -m venv .venv && source .venv/bin/activate   # or your preferred env manager
pip install torch==2.1.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.1+cu118.html
pip install -r requirements.txt
```

`requirements.txt` is pinned to the exact versions used to produce the
results in this repo (Python 3.10.13, CUDA 11.8). See `envs/setup_env.sh`
for the original SLURM-based cluster bootstrap this was derived from, if you
are working on a similarly configured cluster instead of a fresh env.

## Pipeline

```
01_download → 02_preprocess → 03_build_graph → training → evaluation → analysis
```

```bash
sbatch data/01_download/slurm_download.sh          # scRNA-seq + bulk miRNA + miRDB
sbatch data/02_preprocess/slurm_preprocess.sh       # QC, HVG, PCA, Leiden, normalization
sbatch data/03_build_graph/slurm_build_graph.sh     # assembles data/graphs_v3fixed/hetero_graph.pt
sbatch training/slurm_train.sh                      # trains one config from configs/
sbatch analysis/slurm_analysis.sh                   # interpretation + figures
```

Each stage is also runnable directly with `python <script>.py --config
configs/<config>.yaml` for local debugging (see each script's `--help`); the
`slurm_*.sh` wrappers are what were actually used to produce the numbers in
the paper. Raw/processed/graph outputs are intentionally **not** committed
(`.gitignore` excludes `data/raw/`, `data/processed*/`, `data/graphs*/`,
`checkpoints*/`, `logs/`) — regenerate them by running the pipeline.

### Source datasets

| Input | Source | Role |
|---|---|---|
| scRNA-seq (MS + control) | [CellxGene Census](https://chanzuckerberg.github.io/cellxgene-census/) | cell node features, expression edges, cell-type labels |
| Bulk miRNA expression, [GSE289530](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE289530) | NCBI GEO | miRNA node universe |
| miRNA–target interactions | [miRDB v6.0](https://mirdb.org), score ≥ 80 | positive edges (file kept as `mirtarbase_hsa.tsv` for pipeline-naming compatibility — it is miRDB, not miRTarBase) |
| `ogbl-ddi`, `ogbl-ppa` | [Open Graph Benchmark](https://ogb.stanford.edu/) | independent external audit of the heuristic-baseline implementation, not part of the miRNA graph |

The disease label (MS vs. control) is data provenance for the worked
example, not a claim this paper makes — see the abstract above.

## Config lineage — which config produced which number

`configs/` only contains the lineage that feeds the paper (dead/superseded
config generations were pruned from this branch):

| Config group | Purpose |
|---|---|
| `config_v3fixed_edgesplit*.yaml` (4 seeds × uniform/degree-matched) | **Held-out** protocol: edges excluded from message passing (the "honest" condition) |
| `config_v3fixed_transductive*.yaml` (4 seeds × uniform/degree-matched) | **Seen** protocol: edges included in message passing (the "conventional" condition) |
| `config_v3fixed_baselines_*.yaml` (4 files) | Cross-architecture grid: 6 architectures × the 4 seen/held-out × uniform/matched cells |
| `config_mirtarbase_edgesplit*.yaml` | Alternate interaction-source arm (sensitivity check) |
| `config_ogb_ddi.yaml`, `config_ogb_ppa.yaml` | External OGB audit (model-free heuristics only, no training) |
| `config_ablation_no_coexpr.yaml`, `config_ablation_no_mirna.yaml` | Ablations |

All of the above build against `data/graphs_v3fixed/` (produced by
`data/03_build_graph/build_heterograph.py` on this branch — the graph
construction bugs present in earlier `v2`/`v3` config generations are fixed
here). Seeds used throughout: `{123, 777, 2024, 7}`.

## Paper artifact → source file map

Every number and figure in the paper traces to a committed artifact under
`results/comparison/` or `results/literature_survey.tsv`; nothing here was
hand-typed. `analysis/make_manuscript_tables.py` and
`analysis/make_manuscript_figures.py` read **only** these committed files
and raise (rather than silently skip) if one is missing — running them
against this branch is itself a reproduction check.

| Artifact | Feeds | Produced by |
|---|---|---|
| `multiseed_auroc_test_v3fixed.json`, `multiseed_auprc_test_v3fixed.json`, `multiseed_seen_edges_test_v3fixed.json` | Table 1 (headline 2×2 protocol grid), Fig. 1, Fig. 3 | `training/aggregate_seeds.py`, aggregating per-seed output of `training/eval_heldout_grid.py` |
| `topology_baseline_v3fixed_test.json` | Table 2 (model-free baselines), Fig. 1 | `training/eval_topology_baseline.py` |
| `celltype_baseline_config_v2_edgesplit_test.json` | Table 4 (cell-type no-graph control) | `training/eval_celltype_baseline.py` |
| `comparison_table_checkpoints_v3fixed_baselines_{edgesplit,edgesplit_uniform,transductive,transductive_uniform}.tsv` | Table S1 (6-architecture × 4-condition grid), Fig. 2 | `training/run_baselines.py` |
| `ogb_ddi_topology_baseline_{test,valid}.json`, `ogb_ppa_topology_baseline_{test,valid}.json`, `ogb_protocol_verification.json` | External OGB validation (Discussion) | `training/eval_ogb_topology_baseline.py` (see `training/slurm_ogb_ddi_topology.sh` / `slurm_ogb_ppa_topology.sh`) |
| `literature_survey.tsv` | Table 3 (7-paper pilot survey) | manually curated, not regenerated by a script |
| `results/figures/manuscript/fig{1,2,3}_*.pdf` | Figures 1–3 | `analysis/make_manuscript_figures.py` |

To regenerate the tables/figures from these artifacts:

```bash
python analysis/make_manuscript_tables.py
python analysis/make_manuscript_figures.py
```

Each script listed above documents its own CLI flags in its module
docstring — pass `--help` or read the top of the file for the exact
invocation used for a given cell of the grid.

## Repository structure

```
microRNA_project/
├── requirements.txt              # pinned deps (see Environment setup)
├── data/
│   ├── 01_download/               # CellxGene, GEO, miRDB download scripts
│   ├── 02_preprocess/             # QC, normalization, cell-type annotation
│   └── 03_build_graph/            # build_heterograph.py -> data/graphs_v3fixed/
├── models/                        # hetero_gnn.py, layers.py, losses.py, baselines.py
├── training/                      # train.py, evaluate*.py, run_baselines.py,
│                                   # aggregate_seeds.py, eval_*.py, slurm_*.sh
├── analysis/                      # interpret.py, visualize.py,
│                                   # make_manuscript_tables.py, make_manuscript_figures.py
├── configs/                       # config_v3fixed_*, config_mirtarbase_*, config_ogb_*,
│                                   # config_ablation_* (see Config lineage above)
├── envs/setup_env.sh              # original cluster env bootstrap (reference)
└── results/
    ├── comparison/                 # headline JSON/TSV artifacts (see artifact map above)
    ├── figures/manuscript/         # fig1-3 PDFs
    └── literature_survey.tsv       # Table 3 source
```

## Architecture

```
miRNA nodes ──(regulates)──► gene nodes ──(expressed_in)──► cell nodes
     ▲                           │
     └──(regulated_by)───────────┘
                                 └──(coexpressed_with)──► gene nodes
```

Three node types, five edge types. The model (`models/hetero_gnn.py`,
`miRNAGraphTransformer`) stacks 3 HGT layers (256 hidden channels, 8
attention heads) with two task heads: link prediction (BCE) and cell-type
classification (CrossEntropy). `models/baselines.py` holds the model-free
topological heuristics (Adamic–Adar, common neighbors, resource allocation,
gene-degree) and the alternate trained architectures used for the
cross-architecture grid.

### Cell-type markers (used for the cell-typing control task)

| Cell type | Key markers |
|---|---|
| T cell (general) | CD3D, CD3E, CD3G |
| CD4+ T cell | CD4, IL7R |
| CD8+ T cell | CD8A, CD8B |
| Th17 | IL17A, RORC, IL23R, CCR6 |
| Treg | FOXP3, IL2RA, CTLA4 |
| B cell | MS4A1, CD19, CD79A |
| NK cell | GNLY, NKG7, NCAM1 |
| Monocyte | CD14, LYZ, CST3 |
| Microglia | CX3CR1, P2RY12, TMEM119, SLC2A5 |
| Oligodendrocyte | MBP, PLP1, MOG, CNP |
| Astrocyte | GFAP, S100B, AQP4, ALDH1L1 |

## Correctness gates

`training/test_edge_split.py` checks that the held-out protocol actually
excludes evaluation edges from message passing (no leakage). Run it after
building the graph and before trusting any held-out number:

```bash
python training/test_edge_split.py
```
