# Research Goal & Publication Roadmap — miRNA-MS Project

This is a planning document, not manuscript prose. It exists to track what this
project needs before it can go from "Master's thesis defense" to a
journal/conference submission, and to keep a running list of venue options.
Update it as gaps close.

---

## 1. Research Aims

**Working hypothesis:** integrating single-cell RNA-seq, bulk miRNA expression,
and miRNA→gene interaction priors within a heterogeneous graph transformer (HGT)
lets the model learn **cell-type-specific miRNA regulatory signatures** (e.g. in
Th17 cells or reactive microglia) and prioritize regulatory circuits that are
invisible to bulk transcriptomics or sequence-only tools (e.g. TargetScan).

Four guiding sub-questions (from the thesis defense deck,
`presentation/beamer_ms_project/sections/01_context.tex`):
1. **Classification** — can the model correctly recognize 11 cell types?
2. **Link prediction** — can it distinguish which miRNA–gene pairs are
   plausible in this context?
3. **Interpretability** — which miRNAs and circuits emerge as priority
   biological candidates?
4. **Translation** — which results are stable enough to guide external
   validation?

**General objective:** a reproducible pipeline that turns open data into a
heterogeneous graph and an HGT model that discovers miRNA regulatory circuits
in MS — not just a cell-type classifier.

---

## 2. Current Status & Key Results

Best model: `miRNAGraphTransformer` V2 (`checkpoints_v2/best_model.pt`,
epoch 191), 4 HGT layers / 512 channels / 8 heads.

| Model | val_loss | AUROC | AUPRC | cell_acc | cell_f1 |
|---|:--:|:--:|:--:|:--:|:--:|
| **HGT V2** | **0.1541** | **0.9836** | **0.9731** | **0.9961** | **0.9935** |
| Random | 1.4245 | – | – | 0.0909 | 0.0799 |
| MLP | 0.0992 | – | – | 0.9254 | 0.8883 |
| Homogeneous GCN | 0.1411 | – | – | 0.8945 | 0.8631 |
| Ablation: no miRNA edges | 0.0159 | – | – | 0.9895 | 0.9821 |
| Ablation: no co-expression edges | 0.0118 | – | – | 0.9933 | 0.9889 |

Test-set range across reruns: AUROC 0.978–0.979, AUPRC 0.967–0.973,
accuracy ~0.994 (variation <0.002, from random test-loader sampling only —
not yet a seed/split stability study, see gaps below).

Key finding to lead with: **ablations keep cell-classification accuracy high
but cannot match V2's link-prediction performance** — evidence that cell
typing is the "easy" task and regulatory-circuit inference is the harder,
more valuable one that only the full HGT solves.

Biological candidates already defensible:
- **hsa-miR-146a-3p** ranks in Th17's top-10 salient miRNAs — known NF-κB
  regulator, a positive biological check on the model.
- **hsa-miR-140-5p** ranks #2 in oligodendrocytes (saliency 0.66) — published
  role in oligodendrocyte differentiation.
- **hsa-miR-23a-3p → CCL7** is the single highest-confidence predicted edge
  (score 0.986); CCL7/MCP-3 is a chemokine implicated in monocyte recruitment
  to the CNS in MS.
- Pan-cell-type candidates: hsa-miR-140-5p and hsa-miR-4659b-3p appear in the
  top-10 of 6 different cell types each.

Deliverables already produced: full data pipeline (download → preprocess →
build graph), baseline + ablation + V2–V4 model variants, saliency/circuit/
enrichment interpretation outputs, exported figures (`results/figures/`).

---

## 3. Gaps to Close Before Submission

Ordered roughly by importance / reviewer visibility:

- [ ] **Data provenance in the manuscript text.** The interaction table is
      miRDB v6.0 predictions, not miRTarBase (miRTarBase's server was
      unavailable during data collection). This is already disclosed inside
      the repo (`README.md`, `results/REPORT.md`) but the on-disk filename
      (`data/raw/mirtarbase_hsa.tsv`) still says miRTarBase "for pipeline
      compatibility." Before writing the Methods section: cite miRDB
      correctly throughout, and consider renaming the file (or adding a
      `data/raw/README` note) so anyone auditing the repo isn't misled.
- [ ] **External validation** of the top circuits (miR-23a-3p→CCL7,
      miR-140-5p in oligodendrocytes) against literature or an experimental
      interaction database — currently the model's own predictions are the
      only evidence.
- [ ] **MS-vs-control differential analysis** — saliency/circuits have not
      yet been split and compared by condition; this is explicitly called out
      as missing in `results/REPORT.md` and the outlook slide.
- [ ] **Seed / train-test-split stability analysis** — current metric
      variation (<0.002) only reflects test-loader resampling on a single
      trained checkpoint, not multiple training seeds.
- [ ] **Head-on framing of the classification-vs-link-prediction gap** — the
      ablations show cell typing is comparatively easy; the manuscript needs
      to state this directly rather than let it read as an implicit caveat,
      so reviewers don't mistake high `cell_acc` alone for the paper's
      contribution.
- [ ] **Training convergence** — V2 had not plateaued at 200 epochs
      (val_loss still decreasing); consider extended training with a cosine
      LR schedule before finalizing numbers for submission.
- [ ] **Author list, affiliations, funding/ethics statements** — none of this
      exists yet in any project file; needs a decision (advisor, committee,
      collaborators) before drafting.

---

## 4. Candidate Venues

Not decided yet — options to evaluate, roughly tiered by fit for a
methods + disease-application computational biology paper coming out of
thesis-level work:

**Journals**
- *BMC Bioinformatics* — open access, methods-friendly, common venue for
  GNN-on-omics work; realistic first choice.
- *NAR Genomics and Bioinformatics* — open access, good fit for a
  data-integration + interpretable-ML biology story.
- *Scientific Reports* — broad scope, solid fallback if reviewers want more
  biological validation than currently available.
- *RNA Biology* — miRNA-specific readership, worth it if the biological
  (rather than methods) framing is emphasized.

**Conferences / workshops**
- *PSB (Pacific Symposium on Biocomputing)* — good fit for interpretable
  computational biology work at this maturity level.
- *ISMB/ECCB proceedings track* — higher bar, but strong fit if the
  external-validation gap is closed first.
- ML-for-biology workshops (e.g. MLCB, LMRL at NeurIPS) — lower barrier,
  useful for getting the methods contribution in front of an ML audience
  before a full journal submission.

Decide after the "gaps to close" checklist above is substantially done —
external validation status in particular will determine whether this reads
as a methods paper (workshop/Bioinformatics-style) or a biology-discovery
paper (RNA Biology/Scientific Reports-style).

---

## 5. Suggested Next Steps

- [ ] Resolve the miRDB/miRTarBase naming + citation issue (repo + future
      manuscript text).
- [ ] Run MS-vs-control differential saliency analysis.
- [ ] Identify an external validation source for the top 2–3 circuits
      (miR-23a-3p→CCL7, miR-140-5p/oligodendrocytes, miR-146a-3p/Th17) and
      check them against it.
- [ ] Run multi-seed training to report stability, not just single-run
      metrics.
- [ ] Extend V2 training past 200 epochs with a cosine LR schedule and
      confirm whether AUROC improves past 0.99.
- [ ] Decide authorship/affiliations.
- [ ] Revisit the venue list above once external validation results exist —
      that's the deciding factor between the methods-venue and
      biology-venue tiers.
