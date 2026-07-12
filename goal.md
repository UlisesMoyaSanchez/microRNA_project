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

> **READ THIS FIRST (2026-07-11).** The headline link-prediction result did not
> survive a correct evaluation. AUROC fell from **0.9836 → 0.6268** once the
> miRNA→gene edges were genuinely held out. The original framing of this project
> is no longer supportable. See §2.1 and §2.2. The cell-type classification
> result is unaffected and remains genuine.

### 2.1 The decision gate: what an honest protocol gives

Three DGX runs, in order. Each one narrows the question.

| Protocol | AUROC | Job |
|---|:--:|---|
| Published: all edges scorable, uniform negatives | **0.9836** | 5594 |
| Same checkpoint, but scored pair masked from message passing | 0.9766 | 5593 |
| Same checkpoint, degree-matched (hard) negatives — edges *still seen* | 0.8828 | 5595/5596 |
| **Retrained on a real held-out edge split + hard negatives** | **0.6268** | **5603** |

What each step established:

- **Not message-passing leakage.** Masking the scored pair out of the encoder
  costs 0.0087 AUROC. The model was *not* reading the answer off the graph.
- **Not (only) popularity bias.** Against degree-matched negatives the old
  checkpoint held 0.8828 while a model-free gene-popularity heuristic collapsed
  to chance (0.5150). Uniform negatives were inflating the number by ~9 points.
- **It was weight-level memorization.** This is what neither diagnostic could
  rule out, and it is what the retrain exposed. Once the pairs are genuinely
  unseen — absent from supervision *and* from the encoder's input, in both
  directions — the model retains only **0.6268**. Above chance (0.5), but not a
  useful predictor of novel regulatory interactions.

Training curve (job 5603) — the two tasks separate cleanly:

```
Epoch 001  train 1.1605  val 1.1136  AUROC 0.5324  cell_acc 0.7046
Epoch 016  train 0.3279  val 1.9532  AUROC 0.6268  cell_acc 0.9307   <- peak AUROC
Epoch 026  train 0.2087  val 2.7011  AUROC 0.6197  cell_acc 0.9556   <- early stop
```

Training loss falls while validation loss rises **monotonically from epoch 1** —
immediate overfitting on the link task. Meanwhile cell accuracy climbs to 0.9556
and is still climbing when the run stops. **Cell typing is real; the regulatory
link head was an artifact of the evaluation protocol.**

### 2.2 Why 0.62 may be a ceiling, not a bug

Worth settling before spending more GPU time. **miRDB edges are derived from
seed-sequence complementarity, and the graph contains no sequence information at
all.** Gene features are 1-D (mean expression); miRNA features are a learnable
embedding with no biological content. For a *held-out* pair, the only available
route to a prediction is topology — co-targeting structure among the remaining
edges. It is entirely plausible that ~0.62 is near the information ceiling of the
task *as currently posed*, and that no architecture change moves it.

**Decisive test (cheap, pending):** a topology-only heuristic baseline
(common-neighbour / co-targeting Jaccard, no learning) on the same held-out
edges. If it also lands near 0.62, the HGT adds nothing over trivial graph
structure and the ceiling is in the data, not the model.

### 2.3 What still stands

- **Cell-type classification**, evaluated on a real cell-level split: genuine
  and unaffected by any of the above.
- **The pipeline and the diagnostic infrastructure**: download → preprocess →
  graph → train → interpret, plus `training/splits.py`,
  `training/diagnose_leakage.py`, `training/eval_hard_negatives.py`,
  `training/test_edge_split.py`.

### 2.4 What no longer stands

- **The comparison-table claim that V2 is the *only* model doing link
  prediction.** That was the `HeteroData.get()` bug producing `nan`, not a
  result. With the bug fixed, `homo_gcn` reaches 0.9170 and
  `ablation_no_coexpr` 0.9374 (transductive). V2's edge is real but *moderate*.
- **Every biological candidate below.** `hsa-miR-23a-3p → CCL7` (0.986),
  `hsa-miR-146a-3p` in Th17, `hsa-miR-140-5p` in oligodendrocytes, and all of
  `top_circuits_by_celltype.tsv` are ranked by the link head whose genuine
  generalization is 0.62. They are **not** validated findings and must not be
  presented as such. Whether they survive re-derivation from an honestly-trained
  model is an open question — and if 0.62 is the ceiling, they may not be
  recoverable at all.

---

## 2.5 Decision: which paper is this?

Three paths were considered (2026-07-11). **Path A is chosen**, gated on the
topology-only baseline in §2.2.

### Path A — Methods-critique paper *(CHOSEN)*

**The finding is the result:** *transductive evaluation inflates miRNA-target
link prediction from 0.62 to 0.98, and uniform negatives inflate it further.*

- We have the full decomposition and the instrumentation to prove it: the
  message-passing leakage diagnostic, the popularity-bias 2×2 (hard negatives ×
  degree heuristic), and the honest retrain.
- This flaw is widespread in GNN-for-biology link-prediction papers. A clean,
  quantified, reproducible demonstration on a real biomedical graph — plus the
  corrected protocol (`training/splits.py`) as a reusable contribution — is a
  genuine paper.
- **It is not the paper we set out to write, but it is defensible, and nobody
  can take it apart.** That is worth more than a headline that collapses under
  review.
- Cost: low. The experiments are largely done.

### Path B — Fix the model (rejected for now)

The honest lever is **features, not architecture**: add miRNA seed sequence and
gene 3′UTR sequence so held-out pairs have something to generalize *from*;
enrich gene features beyond 1-D mean expression. Cheaper intermediate attempts
(lower `disjoint_train_ratio`, shrink the model, regularize harder, early-stop
on AUROC) are worth one run but would not plausibly take 0.62 near 0.9.

**Rejected because** it is a substantially different project, and §2.2 suggests
the ceiling may be in the data. Revisit only if the topology baseline shows the
HGT genuinely beats trivial structure.

### Path C — Drop link prediction, keep cell-type + interpretability (rejected)

Clean and honest, but weak: without a trustworthy link head the miRNA arm
contributes little, and cell-type classification on scRNA-seq is not novel.

---

## 3. Gaps to Close Before Submission

Ordered roughly by importance / reviewer visibility:

- [~] **No edge-level train/test split for link prediction (found 2026-07-01;
      code fixed 2026-07-11, retrain pending).** The old split partitioned
      `cell` nodes only, so the miRNA→gene edges scored in `evaluate()` were
      the *same full edge set* supervised during training (`pos_edge_full`).
      The ~0.98 AUROC measured reconstruction of already-seen interactions in
      a new cell context, not generalization to held-out pairs.

      **Diagnosed (job 5593, `training/diagnose_leakage.py`).** Two distinct
      leaks were possible; only one was real:
      - *Message passing:* masking the scored pair out of the encoder costs
        almost nothing — AUROC 0.9853 → 0.9766 (−0.009). The model was **not**
        reading the answer off the edge it was asked to predict.
      - *Relation removal:* dropping miRNA↔gene entirely collapses it to
        0.5551. The signal lives in the interaction topology, which is what an
        HGT is supposed to exploit — not a bug.
      - What the diagnostic **cannot** rule out is weight-level memorization,
        since the checkpoint still trained on those pairs as targets. Only a
        held-out edge set can.

      **Fixed in `training/splits.py`** (branch `edge-split-hard-negatives`):
      `RandomLinkSplit` with `rev_edge_types`, so the reverse relation
      `(gene, regulated_by, miRNA)` is stripped in lockstep — otherwise a
      held-out edge stays reachable in one hop and the split is worthless.
      44,186 positives → 24,745 message-passing / 10,605 train-supervision /
      4,418 val / 4,418 test, verified leak-free by `training/test_edge_split.py`.

      **CLOSED (job 5603): held-out AUROC = 0.6268, not 0.98.** The leak was
      weight-level memorization after all — exactly what the diagnostic warned
      it could not rule out. See §2.1. This gap is resolved *as a gap*, but the
      answer invalidates the original paper (§2.5).
- [~] **Negative sampling is uniform-random, not "hard negatives"
      (measured 2026-07-11, job 5595; training fix pending).** Confirmed and
      quantified with `training/eval_hard_negatives.py`, crossing two negative
      samplers with two scorers on the same pairs:

      | scorer | uniform neg | degree-matched neg |
      |---|---|---|
      | gene-degree heuristic (model-free) | 0.7760 | 0.5150 |
      | HGT V2 | 0.9758 | **0.8828** |

      Read: uniform negatives inflate the metric by ~9 AUROC points, and a
      model-free "guess the popular gene" heuristic gets 0.776 against them —
      so the published number is weak evidence of specificity. But the HGT
      holds 0.8828 when negatives are popularity-matched, while the heuristic
      falls to chance — so at the time this read as "specificity, not popularity
      bias." Degree-matched negatives are now used in training as well
      (`splits.LinkSampler`), shared with the eval script so the reported metric
      describes the trained objective.

      **CLOSED, with a correction (job 5603).** The 0.8828 did *not* survive.
      It was measured on a checkpoint that had those pairs as training targets,
      so it still carried memorization. On genuinely held-out edges the model
      gets **0.6268**. Uniform negatives were inflating the number, but that was
      the *smaller* of the two problems.
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

## 5. Next Steps — executing Path A

**The gate closed (job 5603): held-out AUROC = 0.6268.** See §2.1. The project
is now a **methods-critique paper** (§2.5, Path A). The steps below serve that
paper, not the original one.

### 5.1 Close the argument (the paper's core evidence)

- [ ] **Topology-only baseline** — the decisive experiment (§2.2). A
      no-learning heuristic (common-neighbour / co-targeting Jaccard) on the
      same held-out edges. Tells us whether 0.62 is the task's information
      ceiling or whether the HGT genuinely beats trivial structure. **This
      determines how the paper is framed, so it runs first.**
- [ ] **Decompose the 0.98 → 0.62 drop.** The retrain changed *two* things at
      once (held-out edges *and* hard negatives), so the drop is currently not
      attributable. Need the missing cell: held-out edges × *uniform* negatives.
      A reviewer will ask for this 2×2, and it is the paper's central table.
- [ ] **Rebuild `results/comparison/`** under the corrected protocol, with
      held-out and transductive AUROC as separate columns, for every baseline
      and ablation. `sbatch --export=ALL,CONFIG=configs/config_v2_edgesplit.yaml
      training/slurm_baselines.sh`.
- [ ] **Multi-seed** (3–5 seeds) on the corrected protocol. Now worth doing —
      the design is final, and a methods paper *must* show the effect is not
      seed noise.

### 5.2 Bugs found in job 5603 — fix before any further training

- [ ] **Early stopping selects on `val_loss`, which peaked at epoch 1.** So
      `checkpoints_v2_edgesplit/best_model.pt` is an epoch-1 model with AUROC
      0.5324 (chance) — unusable. Must select on `val_auroc`. Does not change
      the conclusion (peak was 0.6268), but the saved artifact is wrong.
- [ ] **DDP deadlock at early stop.** The `break` in `train.py` sits inside
      `if rank() == 0:`, so only rank 0 leaves the loop while ranks 1–3 hang at
      `dist.barrier()` → SIGABRT. Harmless in 5603 (training had finished) but
      it will bite any run that early-stops.

### 5.3 Still relevant to Path A

- [ ] Resolve the miRDB/miRTarBase naming + citation issue. **Now load-bearing:**
      the paper's argument depends on what the edges *are* (sequence-derived
      predictions), so their provenance must be exact.
- [ ] Decide authorship/affiliations.
- [ ] Re-pick the venue for a methods/evaluation paper (§4's methods tier), not
      a biology-discovery one.

### 5.4 Dropped or suspended

- [~] Re-derive circuit rankings from an honestly-trained model — only
      meaningful if the topology baseline shows the model beats trivial
      structure. Otherwise there is nothing to re-derive.
- [x] ~~External validation of top circuits~~ — **suspended.** They derive from
      a link head with 0.62 generalization. Validating them against the
      literature now would risk validating an artifact.
- [x] ~~MS-vs-control differential saliency~~ — **suspended**, same reason.
- [x] ~~Extend V2 past 200 epochs to push AUROC past 0.99~~ — **dropped.** That
      target was the transductive number, which we now know is not real.
