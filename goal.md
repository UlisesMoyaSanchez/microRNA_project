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

> **READ THIS FIRST (updated 2026-07-13).** The headline link-prediction result did
> not survive a correct evaluation. AUROC fell from **0.9836 → 0.6467** once the
> miRNA→gene edges were genuinely held out — and a **no-learning heuristic scores
> 0.5911** on the same edges. The original framing of this project is no longer
> supportable. See §2.1–§2.2. Cell-type classification is unaffected and remains
> genuine (**0.9950**).
>
> **Full evidence base: [`results/EVALUATION_AUDIT.md`](results/EVALUATION_AUDIT.md)**
> · Spanish summary for collaborators: [`results/RESUMEN_AUDITORIA.md`](results/RESUMEN_AUDITORIA.md)

### 2.1 The decision gate: what an honest protocol gives

Four DGX runs, in order. Each one narrows the question.

| Protocol | AUROC | Job |
|---|:--:|---|
| Published: all edges scorable, uniform negatives | **0.9836** | 5594 |
| Same checkpoint, but scored pair masked from message passing | 0.9766 | 5593 |
| Same checkpoint, degree-matched (hard) negatives — edges *still seen* | 0.8828 | 5595/5596 |
| **Retrained on a real held-out edge split + hard negatives** | **0.6467** | **5605** |

*(Job 5603 was the first attempt at this run and reported 0.6268; it early-stopped on
`val_loss`, which peaks at epoch 1. Job 5605 selects on `val_auroc`, converged at epoch
144, and supersedes it. Both agree on the conclusion.)*

What each step established:

- **Not message-passing leakage.** Masking the scored pair out of the encoder
  costs 0.0087 AUROC. The model was *not* reading the answer off the graph.
- **Not (only) popularity bias.** Against degree-matched negatives the old
  checkpoint held 0.8828 while a model-free gene-popularity heuristic collapsed
  to chance (0.5150). Uniform negatives were inflating the number by ~9 points.
- **It was weight-level memorization.** This is what neither diagnostic could
  rule out, and it is what the retrain exposed. Once the pairs are genuinely
  unseen — absent from supervision *and* from the encoder's input, in both
  directions — the model retains only **0.6467**. Above chance (0.5), but not a
  useful predictor of novel regulatory interactions.

Training curve (job 5605) — the two tasks separate cleanly:

```
Epoch 001  train 1.1605  val 1.1135  AUROC 0.5343  cell_acc 0.7049
Epoch 119  ...                       AUROC 0.6467  ...              <- best (saved)
Epoch 144  train 0.0384  val 6.9147  AUROC 0.6395  cell_acc 0.9950  <- early stop
```

Training loss falls to 0.038 while validation loss climbs to 6.9 — **the link head
overfits from epoch 1**. Meanwhile cell accuracy rises monotonically to **0.9950**.
**Cell typing is real; the regulatory link head was an artifact of the protocol.**

### 2.2 It is close to a ceiling — and the protocol was doing the work (job 5604)

**miRDB edges are derived from seed-sequence complementarity, and the graph contains
no sequence information at all.** Gene features are 1-D (mean expression); miRNA
features are a learnable embedding with no biological content. For a *held-out* pair
there is no sequence signal to generalize from — only topology.

The decisive test has now been run. `training/eval_topology_baseline.py` scores
**model-free** heuristics on the *same* 4,418 held-out edges, from training edges only:

| Scorer | Uniform neg | Degree-matched neg |
|---|:--:|:--:|
| `gene_degree` — **ignores the miRNA entirely** | **0.8723** | 0.5123 |
| `pref_attach` | 0.8371 | 0.5071 |
| `common_neigh` | 0.8582 | 0.5836 |
| `adamic_adar` — best heuristic | 0.8616 | **0.5911** |
| **HGT V2, retrained** | — | **0.6467** |

**(a) The HGT beats trivial topology by 5.6 points** (0.6467 vs 0.5911). Real, but a
4-layer/512-channel transformer buying ~5 AUROC points over a two-line formula from
1999 is not regulatory inference — it is slightly-better-than-trivial graph completion.
**This rules out Path B:** no architecture change bridges 0.65 → 0.9 without a sequence
signal in the graph.

**(b) The strongest number in the project: under uniform negatives, a scorer that
ignores the miRNA entirely reaches 0.8723.** Gene popularity alone.

| Protocol | Popularity heuristic | Deep model |
|---|:--:|:--:|
| **Sloppy** (uniform negatives) | **0.8723** | 0.9836 |
| **Honest** (held-out + matched neg.) | **0.5123** | 0.6467 |

Under the published protocol, the gap between a one-line heuristic and a graph
transformer is ~11 points. **The evaluation protocol, not the model, was doing most of
the work.** That is a model-free control no reviewer can argue with, and it is what
Path A rests on.

### 2.2b Attribution — which sin cost what? (jobs 5605 + 5607)

Both trained *and* evaluated with the same negative distribution, so nothing is confounded:

| | Uniform neg | Degree-matched neg |
|---|:--:|:--:|
| **Edges seen** (published) | **0.9836** | 0.8828 |
| **Edges held out** (honest) | 0.8132 | **0.6467** |

- Honest **split** alone: **−0.170**
- Honest **negatives** alone: **−0.101**
- Both: **−0.337** — *super-additive* (−0.170 + −0.101 = −0.271). Fixing only one of the
  two problems substantially understates the damage.

**The sharpest fact in the audit.** On the same held-out edges under the same sloppy
protocol (uniform negatives):

| Held-out edges + uniform negatives | AUROC |
|---|:--:|
| `gene_degree` — no learning, no model, **ignores the miRNA** | **0.8723** |
| HGT V2 trained end-to-end with uniform negatives (job 5607) | **0.8132** |

**The one-line popularity heuristic beats the graph transformer by 6 points.** Under the
published evaluation protocol the deep model was not merely unnecessary — it was *worse*
than counting how many miRNAs already target the gene.

*(An earlier attempt, job 5606, scored the matched-negative checkpoint against uniform
negatives and got 0.5533 — lower than against matched negatives, which is nonsense on its
face. That is a **train/eval mismatch, not a difficulty measure**: a model trained on
matched negatives learns to ignore gene degree and then cannot exploit it. Not reported as
a result; it is why job 5607 exists.)*

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

### Path B — Fix the model (REJECTED — the condition was tested and failed)

The honest lever is **features, not architecture**: add miRNA seed sequence and
gene 3′UTR sequence so held-out pairs have something to generalize *from*;
enrich gene features beyond 1-D mean expression.

**The stated condition for revisiting this was:** *"only if the topology baseline
shows the HGT genuinely beats trivial structure."* **That test has now run (job
5604), and the answer does not clear the bar:**

- The HGT beats the best model-free heuristic by **5.6 points** (0.6467 vs 0.5911).
  Non-zero, but a 4-layer/512-channel transformer buying ~5 AUROC points over a
  two-line 1999 formula is not evidence that the architecture is doing real work.
- Worse, under uniform negatives the model (0.8132) is **beaten by a heuristic that
  ignores the miRNA entirely** (0.8723) — see §2.2b.

So the marginal value of the model over trivial topology is small and, under the
sloppy protocol, *negative*. Combined with the fact that the graph carries no
sequence signal, there is no plausible route from 0.65 to a usable number without
rebuilding the feature space — which is a different project. **Path B is closed.**

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

      **CLOSED (job 5605): held-out AUROC = 0.6467, not 0.98.** The leak was
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
      gets **0.6467**. Uniform negatives were inflating the number (−0.101), but
      the missing split cost more (−0.170), and together they cost −0.337. See
      §2.2b.
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

**The gate closed (job 5605): held-out AUROC = 0.6467.** See §2.1. The project
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
