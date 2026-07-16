# Evaluation Audit — miRNA-MS Project

**Status:** canonical results document. Supersedes `results/archive_pre_audit/REPORT.md`
and `results/archive_pre_audit/EXEC_SUMMARY.md`, which report numbers we now know are artifacts.
**Last updated:** 2026-07-13
**Spanish summary for clinical collaborators:** [`RESUMEN_AUDITORIA.md`](RESUMEN_AUDITORIA.md)

> **All headline numbers below are on the untouched TEST split** (4,418 edges never used for
> training or model selection). Validation numbers appear only where explicitly labelled
> *(model selection)*. This matters more here than in most papers: a manuscript arguing that
> the field reports optimistically-biased link-prediction numbers cannot itself report a
> model-selected validation number. Selecting on `val_auroc` and reporting it cost us
> **+0.020** of illusory AUROC (0.6467 val → 0.6271 test).

> **A note on the word "original".** Throughout this document, *the original protocol* means
> the one used in the **thesis-defense version** of this work: miRNA→gene edges seen during
> training, uniform random negatives. **Nothing from this project has been published.** The
> 0.9836 was never in print — it was corrected *before* submission, not after. Where this
> document says *published*, it refers to **other groups' papers**.

---

## The headline

The original link-prediction result — **AUROC 0.9836** — does not survive a correct
evaluation. Retrained under a verified leak-free edge split with popularity-matched
negatives, the same architecture scores **AUROC 0.6271 on the test set**.

A **no-learning heuristic** (Adamic–Adar) scores **0.5912** on the identical edges. The
four-layer, 512-channel heterogeneous graph transformer buys **3.6 AUROC points over trivial
graph structure**.

**Cell-type classification is unaffected and genuine: test accuracy 0.9916.** The two tasks
separated cleanly. Cell typing is real. The regulatory link head was an artifact of how it
was measured.

---

## The complete test-set table

All scorers, both negative samplers, on the same 4,418 held-out test edges. Training edges
only (35,350) are visible to any scorer.

| Scorer | Uniform negatives | Degree-matched negatives |
|---|:--:|:--:|
| `gene_degree` — no learning, **ignores the miRNA** | **0.8712** | 0.5126 |
| `pref_attach` — no learning | 0.8381 | 0.5074 |
| `common_neigh` — no learning | 0.8597 | 0.5840 |
| `adamic_adar` — no learning, best heuristic | 0.8630 | **0.5912** |
| **HGT trained with uniform negatives** | **0.8056** | **0.5118** |
| **HGT trained with matched negatives** | *0.5554* † | **0.6271** |

† *Not a valid measurement — see "the mismatch trap" below.*

### Three facts, in ascending order of how bad they are

**1. The deep model barely beats a formula from 1999.** 0.6271 vs 0.5912 — **+3.6 points**
for a 175 MB transformer over two lines of arithmetic.

**2. Under the original protocol, the model is beaten by a scorer that ignores the miRNA.**
`gene_degree` reaches **0.8712**; the HGT trained under that same protocol reaches **0.8056**.
The deep model is not merely unnecessary — it is **6.6 points worse than counting how many
miRNAs already target the gene**.

**3. A model trained with uniform negatives *becomes* the popularity heuristic.** Evaluate it
against matched negatives and it scores **0.5118** — chance, and statistically
indistinguishable from `gene_degree`'s **0.5126**. It learned the popularity shortcut and
**nothing else**. This is the mechanism behind the inflation, caught in the act: uniform
negatives do not merely flatter a model, **they select for a model that has learned nothing
transferable.**

---

## Attributing the collapse

Jobs 5605 (matched) and 5607 (uniform), each **trained and evaluated with the same negative
distribution**, so nothing is confounded:

| | Uniform negatives | Degree-matched negatives |
|---|:--:|:--:|
| **Edges seen in training** (original) | **0.9836** | 0.8828 |
| **Edges held out** (honest, test) | 0.8056 | **0.6271** |

| Effect | Cost |
|---|:--:|
| An honest **split** alone (negatives held uniform) | **−0.178** |
| Honest **negatives** alone (edges held seen) | **−0.101** |
| Both — original → honest | **−0.357** |

The effects are **super-additive**: −0.178 + −0.101 = −0.279, but the true total is −0.357.
**Fixing only one of the two problems substantially understates the damage** — a paper that
holds out edges but keeps uniform negatives still reports an inflated number.

### The mismatch trap (a methodological warning worth publishing on its own)

An earlier attempt scored the *matched-negative* checkpoint against *uniform* negatives and
got **0.5554** — **lower** than the 0.6271 it gets against the harder negatives, which is
absurd on its face.

That figure is a **train/eval mismatch, not a difficulty measurement.** A model trained on
degree-matched negatives learns to *ignore* gene degree, and can then no longer exploit the
easy degree signal that uniform negatives hand it. **Attribution requires the negative
distribution to be identical at training and evaluation time.** Any paper that changes its
negative sampler only at evaluation is measuring an artifact. We nearly did.

---

## What was asked, and what came back

Four experiments, each narrowing the question. The order is the method.

### 1. Was the model reading the answer off the graph? — **No.**
`training/diagnose_leakage.py` · job **5593**

| Encoder view | AUROC |
|---|:--:|
| (a) Graph intact — as originally evaluated | 0.9853 |
| (c) **Scored pair masked out of message passing** | **0.9766** (−0.009) |
| (b) miRNA↔gene relation removed entirely | 0.5551 (−0.430) |

**Establishes:** masking the scored edge costs essentially nothing. Row (b) collapsing to
chance is expected and *healthy* — the signal lives in the interaction topology, which is what
a graph model should exploit.
**Cannot rule out:** weight-level memorization. Only a retrain can.

### 2. Was it specificity, or popularity bias? — **Partly popularity.**
`training/eval_hard_negatives.py` · jobs **5595/5596**

| Scorer | Uniform neg | Matched neg |
|---|:--:|:--:|
| Gene-degree heuristic | 0.7760 | 0.5150 |
| HGT V2 | 0.9758 | 0.8828 |

**Establishes:** uniform negatives inflate by ~9 points.
**Cannot rule out:** the same thing — 0.8828 was still measured on a checkpoint trained on
those pairs. *This is the trap the whole audit exists to escape.*

### 3. Does it survive a real held-out split? — **No. This is the finding.**
`training/splits.py` · job **5605**

Split verified leak-free by `training/test_edge_split.py` (all eight checks at zero):
```
44,186 positives = 24,745 message-passing + 10,605 train-supervision + 4,418 val + 4,418 test
```
Held-out edges are absent from supervision **and** from the encoder's input, **in both
directions** — the reverse relation `(gene, regulated_by, miRNA)` is stripped in lockstep via
`RandomLinkSplit(rev_edge_types=...)`. Without that, a held-out edge stays reachable in one
hop and the split is worthless.

Converged at 144 epochs, selected on `val_auroc`. The link head **overfits from epoch 1**:
training loss falls to 0.038 while validation loss climbs to 6.9. Cell accuracy rises
monotonically to 0.9950 (val) / 0.9916 (test).

**Establishes:** the leak was weight-level memorization — precisely what neither diagnostic
above could exclude.

### 4. Is 0.63 the model's achievement, or the task's ceiling? — **Near the ceiling.**
`training/eval_topology_baseline.py` · job **5604**

See the complete table above. The HGT beats the best model-free heuristic by 3.6 points.

**Why there is a ceiling.** miRDB edges are determined by **seed-sequence complementarity**,
and the graph contains **no sequence information whatsoever** — gene features are 1-D (mean
log-normalized expression), miRNA features are a learnable embedding with no biological
content. For a pair the model has never seen, **there is no sequence signal to generalize
from**, only topology. The task as posed is close to unlearnable by construction.

---

## Contribution 3 — the model cannot express the claim it was built for

**Architectural, not statistical.** No retraining fixes this.

`TargetPredictor` (`models/layers.py:77`) takes `[miRNA_emb ‖ gene_emb]` and returns a scalar.
**It has no cell input.** And `analysis/interpret.py:301` scores every miRNA→gene pair
**once, globally**, then builds `top_circuits_by_celltype.tsv` by filtering that *single global
ranking* through each cell type's top-saliency miRNAs (`interpret.py:314-318`).

Therefore **`hsa-miR-23a-3p → CCL7` carries the identical score in every cell type.** The
"cell-type-specific regulatory circuits" were never cell-type-specific: the specificity lives
entirely in the *miRNA saliency filter*, never in the *edge score*.

The project's central premise — cell-type-specific miRNA regulation — was **not implemented**,
and could not have been by this architecture. This generalizes well beyond us: **claims of
cell-type- or context-specific interaction prediction should be checked against whether the
scoring head takes context as input at all.**

---

## The contributions (what the paper claims)

1. **Quantified inflation, decomposed.** 0.9836 → 0.6271. Split costs −0.178, negatives cost
   −0.101, together −0.357 — *super-additive*, so fixing one understates the damage.
2. **A model-free control that indicts the protocol, not the model.** `gene_degree` (0.8712)
   beats the trained transformer (0.8056) under the original protocol. And a uniform-negative
   model, tested against matched negatives, *is* the popularity heuristic (0.5118 ≈ 0.5126).
3. **Context-specific claims may not be architecturally supported** (above).
4. **A corrected, reusable protocol**: `training/splits.py`, `training/test_edge_split.py`,
   and three model-free controls.

### The new baselines

Any future link-prediction work on this graph must beat **`adamic_adar` = 0.5912** (matched
negatives) and **`gene_degree` = 0.8712** (uniform negatives). **Not the HGT.** Reporting a
number without these two controls is reporting nothing.

---

## How the results could be improved

In order of expected value. **These are a different paper — do not start before the methods
paper is submitted.**

1. **Fix the task, not the model.** We are predicting *miRDB's own sequence-based predictions*
   from a graph with **no sequence information**. That is close to unlearnable by construction,
   which is why 0.63 sits so near the topology ceiling. The right task is to predict
   **experimentally validated** interactions (miRTarBase) *using* miRDB as a prior feature.
   Then the graph contributes what sequence cannot: **context**.
2. **Condition the link head on cell type** — `score(m, g, c)` instead of `score(m, g)`. This
   is what the project always claimed and what `TargetPredictor` cannot do. The constructive
   counterpart to Contribution 3.
3. **Add sequence features** (miRNA seed, gene 3′UTR). Necessary for generalization to unseen
   pairs — but **circular if the target is miRDB**, which *is* a seed-match model. Only
   meaningful together with (1).
4. **Control the overfitting.** The link head overfits from epoch 1 on only 10,605 supervision
   edges. Cheap probes: `disjoint_train_ratio` 0.3 → 0.0 (35,350 edges), fewer layers, smaller
   hidden dim, stronger weight decay. Expect a few points, not a transformation.

---

## What no longer stands

- **The regulatory circuits are not findings.** They are ranked by a head 3.6 points above a
  no-learning heuristic — and, per Contribution 3, they were never cell-type-specific anyway.
  **External validation against the literature is suspended**: checking them would risk
  validating an artifact.
- **"V2 is the only model that does link prediction."** That was the `HeteroData.get()` bug
  returning `None` for the tuple edge-type key. Fixed, `homo_gcn` reaches 0.9170 and
  `ablation_no_coexpr` 0.9374 (transductive); `random` lands at 0.5126 — the smoke test.
- **`val_loss` as a cross-model column.** A model without a link head optimizes a strictly
  smaller objective, which is why `ablation_no_mirna` showed the "best" loss while being the
  worst model.

## What does stand

- **Cell-type classification: 0.9916 (test)**, on a real cell-level split.
- **The audit instrumentation**, now the project's main asset: `training/splits.py`,
  `training/test_edge_split.py`, `training/diagnose_leakage.py`,
  `training/eval_hard_negatives.py`, `training/eval_topology_baseline.py`,
  `training/eval_heldout_grid.py`.

---

## Reproduction

DGX (`ssh dgxum`, `/raid/home/umoya/scripts/microRNA_project`); PyG is not installed locally.

```bash
# Gate: if the split leaks, nothing below means anything. CPU-only.
python training/test_edge_split.py --config configs/config_v2_edgesplit.yaml

sbatch --export=ALL,CONFIG=configs/config_v2.yaml,CHECKPOINT=checkpoints_v2/best_model.pt \
    training/slurm_diagnose_leakage.sh                                    # 5593
sbatch --export=ALL,CONFIG=configs/config_v2.yaml,CKPT=checkpoints_v2/best_model.pt \
    training/slurm_hard_negatives.sh                                      # 5595/5596
sbatch --export=ALL,CONFIG=configs/config_v2_edgesplit.yaml training/slurm_train.sh          # 5605
sbatch --export=ALL,CONFIG=configs/config_v2_edgesplit_uniform.yaml training/slurm_train.sh  # 5607

# Test-set evaluation (the numbers reported above)
sbatch --export=ALL,CONFIG=configs/config_v2_edgesplit.yaml,SPLIT=test \
    training/slurm_topology_baseline.sh                                   # 5611/5612
sbatch --export=ALL,CONFIG=configs/config_v2_edgesplit.yaml,\
CKPT=checkpoints_v2_edgesplit/best_model.pt,SPLIT=test training/slurm_heldout_grid.sh
sbatch --export=ALL,CONFIG=configs/config_v2_edgesplit_uniform.yaml,\
CKPT=checkpoints_v2_edgesplit_uniform/best_model.pt,SPLIT=test training/slurm_heldout_grid.sh
```

Every number traces to a job ID and a JSON artifact under `results/comparison/`.

---

## Not yet publishable — what is still missing

The audit is sound. The **paper** is not finished. Four gaps, in priority order:

1. **~~Test-set numbers~~ — DONE (2026-07-13).** Was the most urgent: the headline was a
   model-selected validation number.
2. **~~Multi-seed~~ — HELD-OUT ROW DONE (2026-07-13); seen-edges row BLOCKED.**
   4 seeds {123, 777, 2024, 7} × 2 training samplers, scored on the untouched **test** split
   (`training/aggregate_seeds.py` → [`multiseed_auroc_test.json`](comparison/multiseed_auroc_test.json)).
   **Every claim in this audit survives, with small spread** — the single-seed numbers were
   representative, not lucky.

   Held-out test AUROC, mean ± std, n=4:

   | trained with | eval: uniform neg | eval: degree-matched neg |
   |---|---|---|
   | degree-matched (hard) | 0.5594 ± 0.0073 | **0.6262 ± 0.0071** |
   | uniform | **0.8056 ± 0.0063** | 0.5395 ± 0.0249 |
   | *gene-degree heuristic (n=1)* | *0.8712* | *0.5126* |

   The honest headline is **0.6262 ± 0.0071** (single-seed 0.6271 sat inside 0.2 σ). And the
   central claim is now measured, not asserted: a model **trained with uniform negatives**
   reaches 0.8056 ± 0.0063 under uniform evaluation — **below the 0.8712 one-line gene-degree
   heuristic** — and **falls to 0.5395 ± 0.0249 (chance)** the moment the negatives are
   degree-matched. *It learned gene popularity and nothing else.* Trained with hard negatives
   instead, it clears the heuristic by **+0.1136**. AUPRC agrees throughout.

   **Still blocked — the seen-edges row has no error bar, and cannot get one by re-running.**
   The original **0.9836 / 0.8828** are **single-seed constants**, not recomputed per seed, so
   *"cost of an honest split = +0.4282"* remains n=1. (They were hardcoded in
   `eval_heldout_grid.py:166` until 2026-07-16 and now live in `evaluation.reference_seen_edges`
   in the `config_v2_edgesplit*.yaml` files — because a constant baked into the *script* was
   inherited by runs on other graphs, which then emitted an attribution against a graph they
   had nothing to do with. **The move did not make them reproducible**; this row is still n=1.)
   Worse,
   `train.py:271` now builds the edge split **unconditionally** (`hard_negatives` defaults to
   `True`, no off switch), and `config_v2.yaml` sets neither key — so retraining it today
   silently reproduces the *edge-split* run, not the transductive one. The leaky path was
   removed in `8a12ce3`. Either re-add the leak behind an explicit `edge_split: false` flag and
   retrain 4 seeds, or **report the attribution as single-seed and say so** — recommended, since
   the ~0.43 inflation dwarfs the σ ≈ 0.007–0.025 we measure everywhere else.
3. **The finding must be about the *protocol*, not about our model.** Right now we have only
   shown that *our* HGT was evaluated badly. Run `random`, `mlp`, `homo_gcn`,
   `ablation_no_coexpr` and `hgt_v2` through **both** protocols. If the inflation appears for
   *every* architecture, the claim becomes structural rather than anecdotal.
4. **~~Support the premise~~ — PILOT DONE (n=7):** [`LITERATURE_SURVEY.md`](LITERATURE_SURVEY.md).
   It **corrected the claim we were about to make.** The field does *not* routinely leak test
   edges — 2/7 strip them correctly, and our own original split (cells only) was **worse than
   the norm**. What *is* universal: **0/7 papers report any model-free baseline**, unlabeled
   pairs are treated as uniform negatives in the majority, and 3/7 methods sections do not let
   the reader determine whether held-out edges reached the encoder at all.

   **This is the motivation section, and it is stronger than the strawman would have been:**
   published AUROCs in this literature sit at **0.91–0.99**; under that same protocol, on a
   real biomedical graph, a one-line popularity heuristic reaches **0.8712** — inside that band
   — and beats a trained graph transformer. *A field that never reports a model-free control
   cannot know whether its 0.97 is a result or a popularity effect.*
   **Still needed:** expand to 20–30 papers, two independent raters for the "unclear" calls.
5. **Generalize past our own graph.** One dataset and one interaction database is not a claim
   about a field. Repeat the audit on a second, independent interaction source — **miRTarBase**
   (experimentally validated, and independent *in kind* from miRDB), with TargetScan as a
   weaker fallback.
