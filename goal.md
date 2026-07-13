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

> **Canonical results document: [`results/EVALUATION_AUDIT.md`](results/EVALUATION_AUDIT.md)**
> — the full evidence base, every number traceable to a job ID and a JSON artifact.
> Spanish summary for collaborators: [`results/RESUMEN_AUDITORIA.md`](results/RESUMEN_AUDITORIA.md)
>
> This section is a summary only. **Numbers live in the audit document, not here**, so the
> two cannot drift apart.

**The headline link-prediction result did not survive a correct evaluation.**
All figures below are on the untouched **test** split.

| | AUROC |
|---|:--:|
| Published (edges seen, uniform negatives) | **0.9836** |
| **Honest (held-out edges, matched negatives)** | **0.6271** |
| Best *no-learning* heuristic on the same edges (`adamic_adar`) | 0.5912 |
| Cell-type classification (unaffected, genuine) | **0.9916** |

Three facts, in ascending order of severity:

1. **The transformer beats a two-line 1999 formula by 3.6 AUROC points.**
2. **Under the published protocol, a scorer that ignores the miRNA entirely (`gene_degree`,
   0.8712) *beats* the trained model (0.8056)** — by 6.6 points. The deep model was worse
   than counting how many miRNAs already target the gene.
3. **A model trained with uniform negatives *becomes* the popularity heuristic.** Tested
   against matched negatives it scores 0.5118 — chance, indistinguishable from
   `gene_degree`'s 0.5126. It learned the shortcut and nothing else. **Uniform negatives do
   not merely flatter a model; they select for one that has learned nothing transferable.**

**Attribution** (training and evaluation negatives always consistent): an honest split costs
**−0.178**, honest negatives cost **−0.101**, together **−0.357** — *super-additive*, so
fixing only one of the two understates the damage.

**A fourth finding, architectural.** `TargetPredictor` (`models/layers.py:77`) takes
`[miRNA_emb ‖ gene_emb]` — **no cell input** — and `analysis/interpret.py:301` scores each
pair *once, globally*, then filters that single ranking per cell type by miRNA saliency. So
**`miR-23a-3p→CCL7` has the identical score in every cell type.** The project's central
premise — cell-type-specific regulation — was never implemented and could not be by this
architecture. No retraining fixes it.

**The new baselines.** Any future link-prediction work on this graph must beat
**`adamic_adar` = 0.5912** (matched) and **`gene_degree` = 0.8712** (uniform). **Not the HGT.**

**Selection bias, for the record.** Reporting the model-selected `val_auroc` (0.6467) rather
than the test number (0.6271) would have inflated our own result by +0.020 — in a paper
about optimistically-biased evaluation. Caught before publication, not after.

## 2.5 Decision: which paper is this?

Three paths were considered (2026-07-11). **Path A is chosen.** Its gating condition —
the topology-only baseline — has since been run and reported (§2).

### Path A — Methods-critique paper *(CHOSEN)*

**The finding is the result:** *transductive evaluation inflates miRNA-target
link prediction from 0.63 to 0.98; uniform negatives inflate it further; and a model
trained with uniform negatives learns nothing but gene popularity.*

- We have the full decomposition and the instrumentation to prove it: the
  message-passing leakage diagnostic, the popularity-bias 2×2 (hard negatives ×
  degree heuristic), and the honest retrain.
- **The premise is now evidenced** (`results/LITERATURE_SURVEY.md`, pilot n=7) — but it
  is **not** the premise we assumed. The field does *not* routinely leak test edges
  (2/7 strip them correctly; our own split was *worse* than the norm). What is universal
  is that **0/7 papers report any model-free baseline**, while unlabeled pairs are treated
  as uniform negatives in the majority. Published AUROCs sit at 0.91–0.99 — and under that
  same protocol a one-line popularity heuristic reaches 0.8712 on our graph. **A field with
  no model-free control cannot know whether its 0.97 is a result or a popularity effect.**
  That claim is supported and requires accusing nobody of leakage.
- **It is not the paper we set out to write, but it is defensible, and nobody
  can take it apart.** That is worth more than a headline that collapses under
  review.
- Cost: low. The experiments are largely done.

### Path B — Fix the model (REJECTED — the condition was tested and failed)

The honest lever is **features, not architecture**: add miRNA seed sequence and
gene 3′UTR sequence so held-out pairs have something to generalize *from*;
enrich gene features beyond 1-D mean expression.

**The stated condition for revisiting this was:** *"only if the topology baseline
shows the HGT genuinely beats trivial structure."* **That test has now run (job
5604), and the answer does not clear the bar:**

- On the test set the HGT beats the best model-free heuristic by **3.6 points**
  (0.6271 vs 0.5912). Non-zero, but a 4-layer/512-channel transformer buying 3.6 AUROC
  points over a two-line 1999 formula is not evidence the architecture is doing real work.
- Worse, under uniform negatives the model (0.8056) is **beaten by a heuristic that
  ignores the miRNA entirely** (0.8712) — by 6.6 points.

So the marginal value of the model over trivial topology is small and, under the
published protocol, *negative*. Combined with the fact that the graph carries no
sequence signal, there is no plausible route from 0.63 to a usable number without
rebuilding the feature space — a different project. **Path B is closed** as a rescue for
*this* paper; its ingredients are the improvement roadmap in `EVALUATION_AUDIT.md`.

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

      **CLOSED (job 5605): held-out test AUROC = 0.6271, not 0.98.** The leak was
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

      **CLOSED, with a correction.** The 0.8828 did *not* survive. It was measured
      on a checkpoint that had those pairs as training targets, so it still carried
      memorization. On genuinely held-out test edges the model gets **0.6271**.
      Uniform negatives were inflating the number (−0.101), but the missing split
      cost more (−0.178), and together they cost −0.357. See §2 and
      `results/EVALUATION_AUDIT.md`.
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
## 5. Next Steps — the publication plan

The audit is sound. **The paper is not finished.** Five items, in priority order.
Full detail in [`results/EVALUATION_AUDIT.md`](results/EVALUATION_AUDIT.md).

### 5.1 Done

- [x] **Test-set numbers (2026-07-13).** Was the most urgent gap: the headline had been a
      model-*selected* `val_auroc`. Reporting it would have inflated our own result by
      +0.020 in a paper about optimistically-biased evaluation. Now 0.6271 (test).

- [x] **Multi-seed, held-out row (2026-07-13).** 4 seeds {123, 777, 2024, 7} × 2 training
      samplers, all scored on the untouched **test** split
      (`training/aggregate_seeds.py`, `results/comparison/multiseed_auroc_test.json`).
      **Every headline claim survives, and the spread is small** — the single-seed numbers
      were representative, not lucky.

      Held-out test AUROC, mean ± std over n=4:

      | trained with | eval: uniform neg | eval: degree-matched neg |
      |---|---|---|
      | degree-matched (hard) | 0.5594 ± 0.0073 | **0.6262 ± 0.0071** |
      | uniform | **0.8056 ± 0.0063** | 0.5395 ± 0.0249 |
      | *gene-degree heuristic (n=1)* | *0.8712* | *0.5126* |

      Three things are now quantified rather than asserted:
      1. The honest headline **0.6262 ± 0.0071** (single-seed was 0.6271 — inside 0.2 σ).
      2. **Uniform negatives buy a model that cannot beat counting edges.** Trained *and*
         evaluated on uniform negatives it scores 0.8056 ± 0.0063 — which **loses to the
         one-line gene-degree heuristic (0.8712) by −0.066**. Swap in degree-matched
         negatives at eval and the same model **collapses to 0.5395 ± 0.0249, i.e. chance.**
         It learned popularity and nothing else. AUPRC agrees (0.7948 ± 0.0089 → 0.5466).
      3. **Hard negatives are what make the model non-trivial:** trained hard and evaluated
         hard, it beats the heuristic **+0.1136** (0.6262 vs 0.5126). *A lower number that
         means something, versus a higher number that doesn't* — the paper in one row.

### 5.2 Blocking submission

- [ ] **Multi-seed, the *seen-edges* row — BLOCKED, needs a decision.** The held-out row is
      done (§5.1), but the top row of the 2×2 — the published **0.9836 / 0.8828** — is a
      pair of **constants hardcoded at `training/eval_heldout_grid.py:166`** from the
      original single-seed transductive run. It is *not* recomputed per seed, so every
      attribution that subtracts from it (*"cost of an honest split = +0.4282"*) still
      carries **n=1 and no error bar.**

      **It cannot simply be re-run.** `train.py:271` now builds the edge split
      *unconditionally* (`if _pos is not None`), with `hard_negatives` defaulting to `True`
      and no flag to disable either. `config_v2.yaml` sets neither key — so training it
      today reproduces `config_v2_edgesplit.yaml`, **not** the transductive protocol. The
      leaky code path was deleted in `8a12ce3` and no longer exists. Choose one:
      - **(a) Re-add the leak behind an explicit `edge_split: false` flag**, retrain 4 seeds,
        get a real error bar on the inflation magnitude. Honest, and the flag is arguably a
        *feature* of a paper about this exact bug — but it means reintroducing a bug on purpose.
      - **(b) Report the attribution as single-seed and say so**, with the ± only on the
        held-out row. Cheap and defensible; the inflation is ~0.43, far larger than any
        plausible seed noise (σ ≈ 0.007–0.025 everywhere we *can* measure it).
      - Recommend **(b)** — the effect dwarfs the variance — unless a reviewer demands (a).
- [ ] **Make the finding about the *protocol*, not about our model.** We have so far only
      shown that *our* HGT was evaluated badly — a reviewer will say exactly that. Run
      `random`, `mlp`, `homo_gcn`, `ablation_no_coexpr`, `hgt_v2` through **both**
      protocols. `run_baselines.py` already emits two of the three cells; add
      held-out × uniform via `LinkSampler(hard=False)`. If the inflation appears for
      *every* architecture, the claim becomes structural rather than anecdotal.
- [~] **Support the premise — PILOT DONE (n=7), needs expansion.**
      `results/LITERATURE_SURVEY.md`. It **corrected our claim**: the leak is *not*
      universal (2/7 strip test edges correctly), but **0/7 report a model-free baseline**
      and 3/7 methods sections do not even permit the reader to tell whether held-out edges
      reached the encoder. Expand to 20–30 papers, add a second independent rater for the
      "unclear" calls, and record the supporting quote per cell.
- [ ] **Generalize past our own graph.** One dataset + one interaction database is not a
      claim about a field. Repeat the audit on a second, independent interaction source:
      **miRTarBase** (experimentally validated — independent *in kind*, not just in content;
      its server was down during original collection, so **retry it**), with TargetScan as a
      weaker fallback (also sequence-derived — say so if we must use it).

### 5.3 Housekeeping for the manuscript

- [ ] Data provenance stated exactly: **miRDB v6.0 predictions, score ≥ 80** — not
      experimentally validated, not miRTarBase. This is now *load-bearing*: the paper's
      ceiling argument rests on the edges being sequence-derived while the graph carries no
      sequence.
- [ ] Authorship / affiliations.
- [ ] Venue: an **evaluation/methods** venue, not a biology-discovery one.

### 5.4 Suspended or dropped

- [x] ~~External validation of top circuits~~ — **suspended.** They are ranked by a head
      3.6 points above a no-learning heuristic, and (per §2) were never cell-type-specific
      in the first place.
- [x] ~~MS-vs-control differential saliency~~ — **suspended**, same reason.
- [x] ~~Push AUROC past 0.99~~ — **dropped.** That target was the transductive number.
- [~] Re-derive circuits from an honestly-trained model — only meaningful if the model
      beats trivial structure by enough to matter. It does not.

### 5.5 After submission — the improvement track (a different paper)

Ordered by expected value; see `EVALUATION_AUDIT.md` for the reasoning.

1. **Fix the task, not the model.** Predicting *miRDB's own sequence-based predictions*
   from a graph with **no sequence information** is close to unlearnable by construction.
   Predict **experimentally validated** interactions (miRTarBase) *using* miRDB as a prior
   feature — then the graph contributes what sequence cannot: **context**.
2. **Condition the link head on cell type** — `score(m, g, c)`. What the project always
   claimed, and what `TargetPredictor` cannot do.
3. **Add sequence features** (seed, 3′UTR) — but **circular if the target is miRDB**. Only
   meaningful with (1).
4. **Control the overfitting** (train loss 0.038 vs val loss 6.9 on 10,605 supervision
   edges): `disjoint_train_ratio` → 0.0, fewer layers, stronger weight decay. Worth a few
   points, not a transformation.
