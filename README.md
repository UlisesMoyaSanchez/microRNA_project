# miRNA-MS: Discovering microRNA Regulatory Networks in Multiple Sclerosis via Heterogeneous Graph Deep Learning

A computational pipeline that integrates single-cell RNA-seq transcriptomics, bulk miRNA expression, and validated miRNA–target interaction databases into a heterogeneous graph, then trains a **Heterogeneous Graph Transformer (HGT)** to identify cell-type-specific miRNA regulatory circuits implicated in Multiple Sclerosis (MS).

---

## Scientific Background

Multiple Sclerosis is a chronic autoimmune and neurodegenerative disease of the central nervous system. A key challenge in MS research is understanding how **microRNAs** (small non-coding RNAs, ~22 nt) orchestrate post-transcriptional gene regulation across different immune and glial cell populations. Classical bulk RNA-seq approaches average signals across heterogeneous cell populations, masking cell-type-specific regulatory programs.

This project addresses three problems simultaneously:
- **Multi-omics integration** — linking miRNA profiles (bulk GEO datasets) with single-cell transcriptomics (CellxGene Census)
- **Context-aware target prediction** — beyond sequence-based tools, the model learns which miRNA–gene interactions are active in each cell type
- **Cell-type annotation** — unsupervised Leiden clustering with supervised marker-gene scoring identifies 11 immune and CNS cell types

---

## Architecture

```
miRNA nodes ──(regulates)──► gene nodes ──(expressed_in)──► cell nodes
     ▲                           │
     └──(regulated_by)───────────┘
                                 └──(coexpressed_with)──► gene nodes
```

The graph has three node types and five edge types:

| Node type | Features | Count (approx.) |
|-----------|----------|-----------------|
| `miRNA` | Random init embedding (64-d, learned) | 2,578 (hsa probes in GSE289530, filtered to miRDB) |
| `gene` | Mean log-normalized expression (1-d) | ~3,000 HVGs |
| `cell` | PCA embedding (50-d) | ~111,000 |

| Edge type | Source |
|-----------|--------|
| `(miRNA, regulates, gene)` | miRDB v6.0 (score ≥ 80) |
| `(gene, regulated_by, miRNA)` | reverse of above |
| `(cell, expresses, gene)` | log-norm expression > 0.10 |
| `(gene, expressed_in, cell)` | reverse of above |
| `(gene, coexpressed_with, gene)` | Pearson correlation > 0.60 |

The model (`miRNAGraphTransformer`) stacks **3 HGT layers** (256 hidden channels, 8 attention heads) followed by two task heads:
- **TargetPredictor** — binary link prediction for (miRNA, gene) pairs (BCE loss)
- **CellTypeClassifier** — 11-class cell-type classification on cell nodes (CrossEntropy)

Training uses a combined loss: `L = L_link + 0.5·L_clf + 0.01·L_sparsity`, where the sparsity term (L1 on miRNA embeddings) encourages parsimonious regulatory circuits.

### What The Input Data Looks Like

The model is not trained from a single table. It integrates three aligned inputs:

| Input | Shape | Used for |
|------|-------|----------|
| scRNA-seq processed matrix | cells x genes | cell node features, expression edges, cell labels |
| bulk miRNA expression matrix | miRNAs x samples | miRNA node universe and expression context |
| miRNA-target interaction table | pairs (miRNA, target_gene) | positive link labels and graph edges |

Minimal example (illustrative):

```text
scRNA-seq (rows are cells):
cell_id    condition   cell_type       MS4A1   PTPRC   HLA-DRA
cell_001   MS          B_cell          1.24    0.00    2.17

bulk miRNA (rows are miRNAs):
miRNA             GSM_001   GSM_002   GSM_003
hsa-miR-140-5p     8.41      8.12      8.55
hsa-miR-146a-3p    6.02      6.11      5.94

miRNA-target interactions (positive edges):
mirna             target_gene          evidence
hsa-miR-140-5p    IL7R                 miRDB_v6_prediction
hsa-miR-146a-3p   TRAF6                miRDB_v6_prediction
```

### Is This A Supervised Learning Problem?

Yes. Training is multi-task and supervised in two ways:

- Cell classification supervision: labels are `cell_type` indices built during scRNA preprocessing from marker-based annotation.
- Link prediction supervision: positive labels are the miRNA-gene interaction pairs loaded from `data/raw/mirtarbase_hsa.tsv`.

Important naming note: in this repository, `mirtarbase_hsa.tsv` is currently populated from **miRDB v6.0** high-confidence predictions (score threshold in config), and the filename is kept for pipeline compatibility.

---

## Repository Structure

```
microRNA_project/
├── configs/
│   └── config.yaml                  # Central configuration (paths, hyperparameters)
├── data/
│   ├── 01_download/
│   │   ├── download_cellxgene.py    # CellxGene Census scRNA-seq download
│   │   ├── download_geo_mirna.py    # GEO bulk miRNA dataset download
│   │   ├── download_mirtarbase.py   # miRDB v6.0 download + RefSeq→symbol mapping
│   │   └── slurm_download.sh
│   ├── 02_preprocess/
│   │   ├── preprocess_scrna.py      # QC, HVG, PCA, Leiden, cell-type annotation
│   │   ├── preprocess_mirna.py      # GEO SOFT parsing, normalization, miRNA filtering
│   │   └── slurm_preprocess.sh
│   ├── 03_build_graph/
│   │   ├── build_heterograph.py     # Assemble PyG HeteroData object
│   │   └── slurm_build_graph.sh
│   └── raw/                         # Downloaded raw files (not committed)
├── models/
│   ├── hetero_gnn.py                # miRNAGraphTransformer (main model)
│   ├── layers.py                    # NodeProjector, HGTLayer
│   └── losses.py                    # CombinedLoss (link + classification + sparsity)
├── training/
│   ├── train.py                     # Multi-GPU DDP training loop
│   ├── evaluate.py                  # Validation/test metrics
│   └── slurm_train.sh
├── analysis/
│   ├── interpret.py                 # Gradient saliency + edge attention ranking
│   ├── visualize.py                 # UMAP, circuit plots
│   └── slurm_analysis.sh
├── envs/
│   └── setup_env.sh                 # Conda environment setup
└── logs/                            # Slurm output logs
```

---

## Data

### Source Datasets

| File | Source | Description | Status |
|------|--------|-------------|--------|
| `data/raw/cellxgene_ms.h5ad` | [CellxGene Census](https://chanzuckerberg.github.io/cellxgene-census/) v2025-11-08 | scRNA-seq: 11,245 MS patient cells × 61,497 genes (blood + brain + CSF) | ✅ |
| `data/raw/cellxgene_ctrl.h5ad` | CellxGene Census v2025-11-08 | scRNA-seq: 100,000 healthy control cells × 61,497 genes | ✅ |
| `data/raw/mirtarbase_hsa.tsv` | [miRDB v6.0](https://mirdb.org) (filtered) | 412,771 human interactions · 2,638 miRNAs → 16,722 genes · score ≥ 80 | ✅ |
| `data/raw/geo/GSE289530/` | [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE289530) | Bulk miRNA — CD14⁺ monocytes, CD8⁺ T cells, neutrophils; MS vs. HC · 166 samples · Affymetrix GPL19117 | ✅ |

### Processed Files (generated by pipeline)

| File | Description |
|------|-------------|
| `data/processed/scrna_processed.h5ad` | QC-filtered, normalized, PCA + UMAP + Leiden, cell-type annotated AnnData |
| `data/processed/mirna_expr.tsv` | Merged bulk miRNA expression matrix (rows = miRNAs, cols = samples) |
| `data/processed/mirna_meta.tsv` | Sample metadata (condition, GEO accession) |
| `data/processed/cell_type_labels.txt` | Ordered list of cell-type class names |
| `data/graphs/hetero_graph.pt` | `torch_geometric.data.HeteroData` object (the full graph) |
| `data/graphs/index_maps.pkl` | `gene2idx`, `mirna2idx`, `cell2idx`, `cell_type_labels` dicts |

### miRNA–Target Interactions: miRDB v6.0

The interaction database uses **miRDB v6.0** (miRTarBase was switched out due to server unavailability from this cluster). miRDB provides computational predictions scored 0–100 based on seed-match quality and target site accessibility. Only interactions with **score ≥ 80** are retained:

- 812,240 raw human predictions parsed from `miRDB_v6.0_prediction_result.txt.gz`
- RefSeq mRNA accessions converted to HGNC gene symbols via [mygene.info](https://mygene.info) REST API (batch POST `/v3/query`, 1,000 accessions/request)
- 38,298 / 38,380 accessions successfully mapped (99.8%)
- **412,771 deduplicated (miRNA, gene) pairs** across 2,638 miRNAs and 16,722 target genes
- Stored as `mirtarbase_hsa.tsv` (filename kept for pipeline compatibility)

### scRNA-seq Data Filters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `min_genes` | 200 | Remove low-quality / empty droplets |
| `min_cells` | 3 | Remove unexpressed genes |
| `max_pct_mito` | 20% | Remove dying or damaged cells |
| `n_top_genes` | 3,000 | Highly variable genes (HVG) for downstream graph |
| Leiden resolution | 0.8 | Cluster granularity for cell-type annotation |
| PCA components | 50 | Input to KNN neighbor graph and cell node features |

### GEO Bulk miRNA Datasets

| Accession | Tissue / Cell type | Comparison | Platform | Samples |
|-----------|-------------------|------------|----------|---------|
| [GSE289530](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE289530) | CD14⁺ monocytes, CD8⁺ T cells, neutrophils (PBMC) | MS vs. HC | Affymetrix GPL19117 (miRNA-4) | 166 |

Probe IDs in the series matrix use Affymetrix `MIMAT*_st` accessions. The pipeline maps them to standard `hsa-miR-*` names by parsing the `*_family.soft.gz` platform annotation table (`!platform_table_begin` section), then restricts to Homo sapiens probes. This yields **2,578 human miRNA probes** per sample.

> **Why these datasets were replaced**: the originally configured accessions (GSE107742, GSE119453, GSE41995) were found to be incorrect — GSE107742 is a bovine endometrial array, GSE119453 is a human hematopoiesis GWAS, and GSE41995 had no parseable series matrix. GSE289530 is the most comprehensive recent MS miRNA dataset (published April 2026).

---

## Pipeline

```
01_download → 02_preprocess → 03_build_graph → training → analysis
```

Each stage has a corresponding Slurm script:

```bash
sbatch data/01_download/slurm_download.sh       # ~10 min
sbatch data/02_preprocess/slurm_preprocess.sh   # ~60–90 min (16 CPUs, 128 GB RAM)
sbatch data/03_build_graph/slurm_build_graph.sh # ~30 min
sbatch training/slurm_train.sh                  # 4× A100 GPUs
sbatch analysis/slurm_analysis.sh
```

### Stage 1 — Download

- **CellxGene**: queries the Census API with disease/tissue/organism filters; subsamples controls to 100,000 cells to stay within memory
- **GEO**: downloads `_family.soft.gz` + `_series_matrix.txt.gz` for three accessions with 3-attempt retry logic
- **miRDB**: downloads prediction file, filters to human miRNAs (`hsa-` prefix) at score ≥ threshold, maps RefSeq → HGNC symbols via mygene.info

### Stage 2 — Preprocess

**scRNA-seq** (`preprocess_scrna.py`):
1. Load CellxGene h5ad files; **remap integer var_names to gene symbols** using `var['feature_name']` (CellxGene Census encodes gene symbols in `feature_name`, not the default integer index)
2. Merge MS + control AnnData (outer join on genes; `condition` label added)
3. QC filtering (min_genes, min_cells, max mitochondrial %)
4. CPM normalization → log1p → freeze all 61,497 gene counts as `.raw`
5. HVG selection (3,000 genes, batch-aware across `condition`)
6. Scale → PCA (50 PCs) → KNN neighbors → Leiden clustering → UMAP
7. Cell-type annotation: `sc.tl.score_genes` against marker-gene sets scored on `.raw.var_names` (all genes, not just HVGs); per-cell argmax assigns one of 11 labels

**Bulk miRNA** (`preprocess_mirna.py`):
1. Parse GEO series matrix (`*_series_matrix.txt.gz`); accessions missing their matrix file are skipped gracefully
2. **Probe ID mapping** (Affymetrix arrays): read `*_family.soft.gz` platform table, build `MIMAT*_st → hsa-miR-*` map for Homo sapiens probes; translate and filter probe IDs before any further processing
3. Harmonize remaining probe names to lowercase `hsa-miR-XXX` format (strip `_st`/`_x_st` Affymetrix suffixes)
4. log2(x+1) + quantile normalization per accession
5. Merge across accessions (union of miRNAs, fill NaN with 0)
6. Filter to miRNAs present in `mirtarbase_hsa.tsv` (arm-suffix stripping applied on both sides for robust matching)

### Stage 3 — Build Graph

`build_heterograph.py` assembles the PyG `HeteroData` object:
- Builds `gene2idx`, `mirna2idx`, `cell2idx` index maps; miRNAs restricted to those targeting at least one HVG in the scRNA dataset
- **Node features**: gene (mean log-norm expression, 1-d), cell (PCA 50-d), miRNA (random init 64-d)
- **Cell labels**: cell-type class index from Leiden/marker annotation
- **Edges**: miRDB interactions, sparse expression edges (threshold 0.10), PCC co-expression edges (threshold 0.60, top 1,000 variable genes)
- Outputs `hetero_graph.pt` + `index_maps.pkl`

### Stage 4 — Training

Multi-GPU DDP training via `torchrun --nproc_per_node=4`:
- Full graph on CPU; NeighborLoader samples mini-batches per GPU (neighbors: [10, 5] per HGT layer)
- Negative sampling for link prediction (equal number of random negatives per positive edge)
- Adam optimizer, lr = 1×10⁻³, weight_decay = 1×10⁻⁵
- Early stopping (patience = 20 epochs without validation loss improvement)
- Best model checkpoint saved to `checkpoints/`

### Stage 5 — Analysis & Interpretation

- **Gradient saliency**: `|∂L_celltype / ∂miRNA_embedding|` per cell type — ranks miRNAs by influence on classification
- **Edge attention ranking**: scores all (miRNA, gene) pairs via the TargetPredictor head to identify top predicted regulatory circuits
- **Pathway enrichment**: GO Biological Process + KEGG via `gseapy` on top target genes per miRNA per cell type
- Outputs: `results/interpretation/mirna_saliency_by_celltype.tsv`, `top_circuits_by_celltype.tsv`, `enrichment/<celltype>_GO.tsv`

---

## Cell-Type Markers

The following marker-gene sets are used for Leiden cluster annotation:

| Cell type | Key markers |
|-----------|-------------|
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

---

## Environment Setup

```bash
bash envs/setup_env.sh
conda activate mirna_ms
```

Key dependencies: `scanpy`, `anndata`, `torch`, `torch_geometric`, `cellxgene_census`, `pandas`, `requests`, `gseapy`, `openpyxl`

---

## Configuration

All parameters are centralized in `configs/config.yaml`:

| Section | Key parameters |
|---------|---------------|
| `data.cellxgene` | tissue filters, HVG count, QC thresholds, cell caps |
| `data.mirna` | miRDB URL, score threshold (80), min interactions per miRNA (5) |
| `data.geo` | `accessions:` dict mapping GEO ID → description |
| `data.graph` | co-expression threshold (0.60), expression edge threshold (0.10) |
| `model` | hidden_channels (256), num_heads (8), num_layers (3), dropout (0.2), mirna_init_dim (64) |
| `training` | batch_size, lr, num_epochs (200), patience (20), loss weights |
| `cell_type_markers` | marker gene lists for 11 cell types |