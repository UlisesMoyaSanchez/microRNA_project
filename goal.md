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

> **Graph lineage — UPDATED 2026-07-27, re-measurement now complete.** Every number in this
> section is now measured on **`data/graphs_v3fixed/`** (independently verified, §3.2/D4).
> The full lineage — both the honest endpoint (`checkpoints_v3fixed_edgesplit[_uniform]`) and
> the seen-edges reference (`checkpoints_v3fixed_transductive[_uniform]`) — has been trained
> **and** evaluated with 4 seeds each (jobs 5743–5768 training; 5808–5844 evaluation;
> `training/aggregate_seeds.py` for the held-out row,
> `results/comparison/multiseed_seen_edges_test_v3fixed.json` for the seen-edges row, since
> the latter's `"seen_edges"` key doesn't fit `aggregate_seeds.py`'s `"held_out"` reader).
> The topology floor was regenerated on this graph too
> (`results/comparison/topology_baseline_v3fixed_test.json`) and is numerically **identical**
> to the pre-fix value — expected, since the graph fixes touched co-expression and
> cell→gene edges, not miRNA→gene topology. Everything below is now single-graph.
> The pre-fix (v2) files are untouched and still readable for comparison
> (`results/comparison/{topology_baseline,multiseed_auroc,multiseed_auprc}_test.json`,
> no `_v3fixed` suffix).

> **Nothing from this project has been published.** *The original protocol* below means the
> one used in the **thesis-defense version**: edges seen in training, uniform negatives. The
> 0.9836 was never in print — it was corrected *before* submission, not after. Where these
> documents say *published*, they refer to **other groups' papers**.

| | AUROC |
|---|:--:|
| Original (edges seen, uniform negatives) — pre-fix n=1 | 0.9836 |
| Edges seen, uniform negatives, **v3fixed graph, n=4 seeds** | **0.9867 ± 0.0011** |
| **Honest (held-out edges, matched negatives), v3fixed graph, n=4 seeds** | **0.6276 ± 0.0070** |
| Best *no-learning* heuristic on the same edges (`adamic_adar`), v3fixed graph | 0.5912 |
| Cell-type classification (unaffected, genuine) | **0.9916** |

Per-seed values: [123: 0.6304, 777: 0.6291, 2024: 0.6335, 7: 0.6174] AUROC
(`results/comparison/multiseed_auroc_test_v3fixed.json`). This replicates the pre-fix
single-seed 0.6271 almost exactly, now with an error bar. The AUPRC analog is
**0.6598 ± 0.0049** (`multiseed_auprc_test_v3fixed.json`). The "uniform negatives become the
popularity shortcut" finding also replicates: a model trained with uniform negatives scores
**0.5352 ± 0.0202** against matched negatives (pre-fix single-seed: 0.5118) — still
indistinguishable from chance / `gene_degree` (0.5126, itself unchanged from pre-fix, as
expected for a purely topological heuristic).

Three facts, in ascending order of severity:

1. **The transformer beats a two-line 1999 formula by 3.6 AUROC points.**
2. **Under the original protocol, a scorer that ignores the miRNA entirely (`gene_degree`,
   0.8712) *beats* the trained model (0.8056)** — by 6.6 points. The deep model was worse
   than counting how many miRNAs already target the gene.
3. **A model trained with uniform negatives *becomes* the popularity heuristic.** Tested
   against matched negatives it scores 0.5118 — chance, indistinguishable from
   `gene_degree`'s 0.5126. It learned the shortcut and nothing else. **Uniform negatives do
   not merely flatter a model; they select for one that has learned nothing transferable.**

**Attribution** (training and evaluation negatives always consistent): an honest split costs
**−0.178**, honest negatives cost **−0.101**, together **−0.357** — *super-additive*, so
fixing only one of the two understates the damage. *(Pre-fix, n=1, preserved as the original
record — see `multiseed_auroc_test.json`, no `_v3fixed` suffix.)*

**Same attribution, v3fixed graph, n=4 seeds (2026-07-27):** cost of honest negatives alone
**−0.0619** (0.9867 → 0.9248, both n=4 means), cost of an honest split alone **−0.4225**
(0.9867 → 0.5642), total **−0.3591** (0.9867 → 0.6276) — the total lands within 0.002 of the
pre-fix −0.357, a strong replication, but the *decomposition* shifts: the split now accounts
for more of the damage and negatives less, because "cost of split alone" holds negatives at
*uniform* while the honest-split model was trained with *matched* negatives — a train/eval
negative mismatch cell, not a clean ablation. Read the total as the robust number; treat the
two-term split with the same caution the pre-fix figure already carried.
(`results/comparison/multiseed_auroc_test_v3fixed.json`, printed by
`training/aggregate_seeds.py --checkpoint-prefix checkpoints_v3fixed --split test`.)

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
original protocol, *negative*. Combined with the fact that the graph carries no
sequence signal, there is no plausible route from 0.63 to a usable number without
rebuilding the feature space — a different project. **Path B is closed** as a rescue for
*this* paper; its ingredients are the improvement roadmap in `EVALUATION_AUDIT.md`.

### Path C — Drop link prediction, keep cell-type + interpretability (rejected)

Clean and honest, but weak: without a trustworthy link head the miRNA arm
contributes little, and cell-type classification on scRNA-seq is not novel.

---

## 3. Gaps to Close Before Submission — the publication plan

The audit is sound. **The paper is not finished.** This is the **single** live checklist:
an earlier duplicate list was deleted on 2026-07-14 because it had drifted out of sync
(it still marked multi-seed as pending and external validation as open, when the first is
done and the second is suspended). Full detail in
[`results/EVALUATION_AUDIT.md`](results/EVALUATION_AUDIT.md).

> **State as of 2026-07-27 — no measurement gap remains for the core claim.** The graph fix
> is re-measured multi-seed (§2), and the cross-architecture check came back structural
> (§3.2, jobs 5849-5852). What still blocks submission is **three decisions** (D2, D3, D5 —
> all paragraph-level, none requiring compute), **scope** (expand the literature survey;
> the pre-registered miRTarBase arms, whose data is already on disk), and **manuscript
> housekeeping** (§3.3: retitle per D1, provenance corrections, authorship, venue).

### 3.1 Done

- [x] **Test-set numbers (2026-07-13).** Was the most urgent gap: the headline had been a
      model-*selected* `val_auroc`. Reporting it would have inflated our own result by
      +0.020 in a paper about optimistically-biased evaluation. Now 0.6271 (test).

- [x] **Multi-seed, held-out row (2026-07-13, pre-fix graph).** 4 seeds {123, 777, 2024, 7} ×
      2 training samplers, all scored on the untouched **test** split
      (`training/aggregate_seeds.py`, `results/comparison/multiseed_auroc_test.json`).
      **Every headline claim survives, and the spread is small** — the single-seed numbers
      were representative, not lucky. *Replicated on the fixed graph 2026-07-27 — see §2's
      updated headline table; the honest AUROC moves from 0.6262 ± 0.0071 here to
      0.6276 ± 0.0070 there, same conclusion.*

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

### 3.2 Blocking submission

- [ ] **The MS framing is not supported by the pipeline — D1 and D4 ANSWERED, D2/D3/D5 OPEN.**
      Evidence is in hand and reproducible: `analysis/audit_ms_specificity.py` →
      `results/comparison/ms_specificity_audit.json` (SLURM job **5716**, clean tree at
      `b790659`, graph `c5d98d15…`) — the **pre-fix** record, deliberately preserved. The
      audit only establishes facts; the remaining decisions are still yours.
      **D1 was answered 2026-07-27 (option (a), below), which unblocks the title and
      abstract** — the manuscript is no longer gated on the word "MS". D4 was taken
      *fix-first* on 2026-07-17 and is tracked below; it is engineering, and it did not
      answer D1.

      **The facts, none of them disputed.** MS enters this pipeline exactly twice — the
      cellxgene `disease` filter (which cells were downloaded) and `batch_key="condition"` in
      HVG selection (`preprocess_scrna.py:85`, which shapes the 3,000-gene vocabulary). It is
      never a node feature, never a label, never an input to any head. Specifically:
      `torch.randn` replay matches the shipped miRNA features with **max|diff| = 0.0** — they
      are reproducible from a seed alone and therefore provably carry no data; `mirna_expr.tsv`
      (166 samples, the only modality with an MS/HC contrast) has **zero readers**;
      `condition` is `{Control: 166}` from a silent default while the truth recoverable from
      `Sample_description` is **{Control: 78, MS: 51, T1D: 37}**; and `TargetPredictor` takes
      no condition input, so it could not have used the label anyway.

      - **D1 — ANSWERED 2026-07-27: option (a), retitle and de-scope.** The paper does **not**
        claim MS. It is about link-prediction evaluation on a biomedical graph; **MS is dataset
        provenance** — the cells came from MS patients and controls, and nothing downstream
        knows which is which — stated in one sentence and not load-bearing anywhere else.
        This unblocks the title and abstract.

        *What (a) settles:* the framing, the title, and the scope of every claim. No result
        may be described as MS-specific, MS-associated, or disease-relevant; the word appears
        in the data-provenance sentence and in `data/` paths, nowhere else.

        *What (a) does NOT settle:* D2, D3 and D5 below remain open. (a) chooses not to
        *claim* MS; it does not decide whether the MS-blindness itself gets written up (D2),
        whether the T1D mislabelling is disclosed (D3), or whether the collaborator summary
        carries that disclosure (D5). Those are still yours, and they are now being decided
        for a paper that no longer claims MS — which makes D2's "so what else didn't you
        check?" objection weaker, since nothing rests on the variable any more.

        *The options as weighed, preserved:*
        - **(a) Retitle and de-scope** — CHOSEN. Fastest path to submission and the only one
          the pipeline actually supports.
        - **(b) Keep the framing and wire MS in** (Track B) before submitting — rejected.
          §2.5 already **rejected Path B** as a rescue for this paper and moved its
          ingredients to §3.5 as a different paper; choosing (b) reopens a closed decision
          and delays submission by months.
        - **(c) De-scope the headline, add Track B as a supplementary sensitivity row** —
          *"we gave the miRNAs real biology and the inflation did not move."* Strictly the
          strongest argument, and consistent with §2.5 — but it requires doing Track B first,
          which is the delay (b) was rejected for. **Not foreclosed:** if Track B is ever
          built (§3.5), (c) is (a) plus one supplementary row, and the paper can absorb it
          without re-framing.
      - **D2 — Is this a contribution or a limitation?** As a contribution it becomes a new
        `## Contribution 4` section in `EVALUATION_AUDIT.md` (after the Contribution 3 prose
        at `:170-187`), renumbering the list item 4 ("a corrected, reusable protocol") → 5. The
        case for contribution: Contribution 3 already asks *"does the scoring head take context
        as input?"*; this generalizes it to a three-question checklist — is the context variable
        **present in the artifact**, does the **head take it**, does the **evaluation vary with
        it** — and we fail all three on the variable the project is named after. The case
        against: it invites "so what else didn't you check?"
        *Reframed by D1 (2026-07-27):* the "so what else" objection is now weaker — with MS
        de-scoped to provenance, **no claim in the paper rests on the variable**, so writing
        up the checklist costs nothing defensively. It is a question of whether the
        generalizable lesson is worth the section, not of exposure.
      - **D3 — Do we publish the T1D mislabelling?** It is an error we found in our own repo:
        37 diabetics filed as healthy controls by a default return. Publishing it costs nothing
        factually (no reported number moves — `EVALUATION_AUDIT.md:164-165` already documents
        the miRNA features as content-free) and a methods-critique paper that audits its own
        provenance has standing that one which doesn't, lacks. But it is an admission, and it
        is your call whether it reads as rigour or as sloppiness.
      - **D4 — ANSWERED 2026-07-17: FIX, fix-first. Graph rebuilt and verified; re-measurement
        NOW DONE (2026-07-27).** The decision reversed the default recommendation below,
        which is left intact as the record of what was weighed. Progress, all on
        `graphs_v3fixed` (sha `84329f70…`, manifest `data/graphs_v3fixed/graph_manifest.json`):
        - [x] Both bugs fixed (`7ac3319`); `config_fingerprint` now hashes the build code, so
              a graph-changing edit can no longer silently reuse a cached graph.
        - [x] Graph rebuilt — job **5717**, 2026-07-17.
        - [x] Transductive protocol restored behind `training.edge_split` (`ca2e7f3`), the leak
              reintroduced deliberately and only behind that flag. This also takes option **(a)**
              on the *seen-edges row* item below, which was previously open.
        - [x] Leaky endpoint retrained on the fixed graph — job **5718**, best `val_auroc`
              **0.9946**. **This is a control, not a result**: `train.py:315-336` assigns
              `val_sup = train_sup = all_pos`, so it is a reconstruction score on memorized
              edges against uniform negatives. Never report it in a results table unlabelled.
        - [x] Graph independently verified — job **5728**, 2026-07-18, `9dc22ed`, **all three
              edge types PASS**. Co-expression endpoints are **all** within the top-1000 most
              variable genes, and the pre-fix counterfactual yields **1,130** edges vs the
              emitted **828** — so the fix demonstrably moved the artifact rather than
              reproducing it. `expresses` 16,682,321 (spot-check min 0.1023 > 0.10 threshold);
              `regulates` 44,186.
        - [x] **Honest endpoint retrained — 4 seeds, 2026-07-21 (jobs 5743, 5755-5760, plus
              two interleaved runs logged only in `logs/train_rank0.log`).**
              `checkpoints_v3fixed_edgesplit{,_uniform}_s{123,777,2024,7}/` all exist.
              Evaluated 2026-07-27 (jobs 5808-5815, rerun as 5828-5844 after the
              `reference_seen_edges` update below so `attribution` would be populated) —
              honest test AUROC **0.6276 ± 0.0070** (n=4), replicating the pre-fix single-seed
              0.6271. See §2 for the full table.
        - [x] **Topology baseline regenerated** on the fixed graph — job 5827, 2026-07-27,
              written to `results/comparison/topology_baseline_v3fixed_test.json` (a *new*
              path, not the old `topology_baseline_test.json`: `eval_topology_baseline.py
              --out` defaults to a graph-agnostic filename that would have overwritten the
              preserved pre-fix record). `adamic_adar` degree-matched = **0.5912**, bit-for-bit
              identical to the pre-fix value — expected, since the graph fixes touched
              co-expression and cell→gene edges, not miRNA→gene topology, and a useful sanity
              check that nothing else moved.
        - [x] **`ms_specificity_audit_v3fixed.json` is already the post-patch record —
              goal.md's earlier claim otherwise was itself stale, corrected 2026-07-27.**
              The file on disk is dated 2026-07-21 12:19, which is *after* patch `9dc22ed`
              (2026-07-18 15:09) landed, and `coexpr_gene_selection.selection_used` reads
              `"most_variable"` — the correct, post-fix verdict. No re-run needed; whoever
              wrote it after the patch already superseded this checklist item without
              updating the checklist. Left here as a record so a future reader doesn't
              re-run a job that's already done.
              *This is the paper's own thesis landing on our own repo, and §3.2/D2 should
              decide whether it belongs in the manuscript as a second instance of the pattern.*

        *Original framing, preserved:* **fix, or document and freeze?** Co-expression edges
        use `X[:, :1000]` — the **first** 1,000 genes in `var` order, not the most variable as
        the docstring claims (measured overlap: **jaccard 0.294**). And the `cell→gene`
        threshold `0.10` is applied to **z-scaled** data (`preprocess_scrna.py:87`), so it means
        0.1 σ above each gene's mean, not the log-norm level the config implies — all 15,978,902
        `expresses` edges mean something other than documented. **Fixing either one changes the
        graph and invalidates every number in the paper.** Default recommendation:
        **document as known deviations, fix only in the Track B branch.** Fixing them quietly
        before submission would be the worst of the three options.
      - **D5 — Does `RESUMEN_AUDITORIA.md` get this?** It is declared the collaborator-facing
        summary. A cohort documented as MS-vs-HC that silently contains diabetics is exactly
        what a clinical collaborator should hear directly — and they may know something about
        GSE289530 that we don't. Recommend yes, in the same commit as the English text so the
        two cannot drift (§3 already deleted one duplicate checklist for drifting).

      **Ordering constraint, already satisfied:** Track A documents the *pre-fix* state, so the
      audit ran and was committed before any Track B change. `graph_sha256` in the JSON is what
      identifies that state once the graph is rebuilt. Tag the commit `audit-ms-specificity`
      once D1–D5 are answered and the prose lands.

      **Also fix regardless of D1–D5** (§3.3): the `data.geo.accessions` description of
      GSE289530 asserts *"CD14+ monocytes, CD8+ T cells, neutrophils; MS vs HC"*. Both halves
      are false — the samples are generic lymphocytes/monocytes/neutrophils and the cohort
      includes T1D. Same species of error as the `mirtarbase_hsa.tsv`-holds-miRDB filename.

- [x] **Multi-seed, the *seen-edges* row — DONE on `graphs_v3fixed`, 2026-07-27.**
      `ca2e7f3` restored the transductive protocol behind `training.edge_split`; the 4-seed
      fan-out then ran 2026-07-21 (jobs 5761-5768, training) and was evaluated 2026-07-27
      (jobs 5816-5823, `eval_heldout_grid.py` → `results/comparison/seen_grid_*_test.json`,
      one per seed per training condition). Hand-aggregated (not
      `training/aggregate_seeds.py` — key mismatch, see §2) into
      `results/comparison/multiseed_seen_edges_test_v3fixed.json`:
      - trained uniform / eval uniform: **0.9867 ± 0.0011** (n=4) — the row that matches the
        original protocol's semantics (uniform negatives) and now replaces the old n=1
        0.9836.
      - trained uniform / eval degree-matched: **0.9248 ± 0.0045** (n=4), replaces the old
        n=1 0.8828.
      - trained degree-matched / eval uniform: 0.9697 ± 0.0051; trained degree-matched / eval
        degree-matched: 0.9683 ± 0.0070 (no v2 analog — a new cell the original protocol
        never measured, since it only ever used uniform negatives).
      These two means (0.9867 / 0.9248) are now declared in `evaluation.reference_seen_edges`
      in all 10 `config_v3fixed_edgesplit*.yaml` files (base + 4 seeds × 2 negative
      conditions), so `attribution` is no longer `null` for this graph — see §2's updated
      attribution paragraph. **Unlike the old pair, these ARE reproducible**: rerun
      `training/slurm_heldout_grid.sh` against the 8 `checkpoints_v3fixed_transductive*_s*`
      checkpoints and re-aggregate.

      *Historical record, pre-fix graph, still open and no longer actionable on this graph:*
      the original **0.9836 / 0.8828** are a pair of **single-seed constants** from the
      original transductive run on the pre-fix graph. They were never recomputed per seed and
      never will be — the leaky path that produced them was deleted in `8a12ce3`. They remain
      n=1 in `config_v2_edgesplit*.yaml`, preserved as-is for historical comparison only.

      *Location changed 2026-07-16, nothing else did:* they were hardcoded in
      `training/eval_heldout_grid.py:166` and are now declared in
      `evaluation.reference_seen_edges` in the 10 `config_v2_edgesplit*.yaml` files — because
      hardcoding them in the script meant any *other* graph's run inherited miRDB's constants
      and emitted a meaningless attribution. **This did not make them reproducible.** They are
      still n=1, still not re-derivable, and this item is still open; the numbers merely now
      travel with the graph that owns them, and a config without them gets `attribution: null`.

      **RESOLVED 2026-07-17/2026-07-27 — option (a) was taken, not (b).** This paragraph
      originally posed a choice: `train.py:271` built the edge split unconditionally with no
      flag to disable it, since the leaky code path had been deleted in `8a12ce3`. The
      recommendation below was (b) — report single-seed and say so. Instead, `ca2e7f3`
      **re-added the leak behind an explicit `training.edge_split: false` flag**, and it was
      exercised for real: 4 seeds trained 2026-07-21 (jobs 5761-5768), evaluated 2026-07-27
      (jobs 5816-5823). The error bar this paragraph said we could skip now exists — see the
      seen-edges numbers earlier in this item. (b) would have been fine; (a) is what actually
      happened and it cost one extra training sweep, not a research-grade effort.
- [x] **Make the finding about the *protocol*, not about our model — DONE 2026-07-27, and it
      came back STRUCTURAL.** This was the last experimental gap for the core claim; what
      remains in §3.2 is decisions (D2/D3/D5) and scope (lit survey, miRTarBase), not
      measurement. Six architectures × 2 protocols × 2 negative samplers on `graphs_v3fixed`,
      one job per cell — jobs **5849-5852**, tables at
      `results/comparison/comparison_table_checkpoints_v3fixed_baselines_*.tsv`, full write-up
      as **experiment 5** in `EVALUATION_AUDIT.md`.

      **Every trained architecture inflates by 0.24-0.31** between the original protocol
      (seen edges + uniform negatives) and the honest one (held-out + degree-matched):
      `hgt_v2` +0.3142, `ablation_no_coexpr` +0.2943, `homo_gcn` +0.2941, and a **graph-free
      `mlp` +0.2420**. The untrained `random` control does not move (−0.0043), which is what
      makes the rest trustworthy. The inflation therefore is not a property of the
      transformer, of heterogeneous message passing, or even of *having a graph* — it is a
      property of the protocol. `gene_degree` (0.8712) also beats **all four** trained models
      under the original protocol, generalizing §2's claim 2 beyond the HGT.

      Two things worth carrying into the manuscript beyond the headline:
      - The **super-additivity replicates per architecture** for every graph model, and the
        `mlp` is an instructive exception: with no graph its only signal is gene degree, so
        the negatives axis absorbs nearly all the damage (−0.2628) and its seen+matched cell
        is **0.4994, exact chance** — allowed to memorize and still no better than a coin flip.
      - **`homo_gcn` (0.6236) beats `hgt_v2` (0.6080)** on the honest protocol. The
        heterogeneous transformer loses to a plain GCN, which strengthens the
        "architecture earns nothing" argument with a direct architectural comparison.

      Caveats, both recorded in `EVALUATION_AUDIT.md` and in each config header: **n=1**
      (seed 42, by design — the effect is 0.24-0.31 vs seed spread 0.005-0.02; report it as
      n=1, never with a ±), and `hgt_v2` reads **0.6080 here vs §2's 0.6276** because
      `run_baselines.py` uses its own single-GPU loop rather than `train.py`'s DDP one and
      sits ~0.02-0.06 lower throughout. The table compares architectures under identical
      conditions; it does not restate the headline.

      **Getting there required fixing three defects in `run_baselines.py`, one of them a real
      finding about this repo** (all are the paper's own bug class — a default standing in for
      a computation that never ran):
      1. **Model selection on `val_loss`.** `train.py:389-394` already documents that
         criterion saving a **chance-level checkpoint (0.5324)** for a model reaching 0.6268,
         because the link head overfits from epoch 1. `train.py` was fixed; `run_baselines.py`
         kept the broken rule. Caught mid-run from a symptom (early stop at epoch 29 vs. 124);
         the first grid was cancelled, its checkpoints deleted, and the whole grid re-run.
         **The uncorrected table would have been wrong in the direction that flatters our own
         thesis** — which is precisely the failure mode this paper is about, found in our own
         audit tooling.
      2. Hardcoded `checkpoints_v2/best_model.pt` as the reference row — the *pre-fix* graph —
         plus a cache filename carrying no graph identity, so a v3fixed run would have
         *loaded* pre-fix numbers rather than recomputing them. Now declared via
         `evaluation.reference_checkpoint`; a config without it gets no row, never a borrowed one.
      3. Fixed output-table and per-architecture checkpoint paths, so the four cells
         overwrote each other's artifacts. Now keyed to the config's `checkpoint_dir` stem.
- [x] **Support the premise — EXPANDED TO n=21 papers (2026-08-12), spot-checked (2026-08-13),
      DOUBLE-RATED (2026-09-01; the second-rater item is now closed).**
      `results/LITERATURE_SURVEY.md`. It **corrected our claim twice**: the leak is *not*
      universal (7/21 strip test edges correctly, only 2/21 clearly do not); and the blind
      second pass overturned "0/22 report a model-free baseline" — **6 of 21 do**, against this
      subfield's classical untrained methods (SPM, HNM, TCRWMDA, BNPMDA, LLCMDA, RWR), which
      the original criterion's enumerated list had silently excluded. The surviving, stronger
      claim: those six clear the untrained comparator by only **1.7-7.4 AUROC points and none
      remarks on it**. 12/21 (57%) methods sections still do not permit the reader to
      tell whether held-out edges reached the encoder. Target of 20-30 papers met with 15 new,
      open-access, PMC-verified papers; supporting quote recorded per cell in the TSV.
      2026-08-13: re-verified 14 "unclear" cells against primary sources -- 11/14 held up as a
      genuine reporting gap (all 14 checked papers had public GitHub code, so the info exists
      but isn't disclosed in prose), 3/14 were extraction errors and are now corrected.
      Remaining: a second independent rater for the "unclear" calls -- reported as the
      manuscript's fifth Discussion limitation rather than resolved.
- [x] **Quantify the inflation on surveyed papers' own data — ALL 7 PAPERS DONE (2026-08-13).**
      `results/HMDD_TOPOLOGY_AUDIT.md`. Instead of only inferring inflation from our own graph,
      computed our model-free topology baseline directly on the HMDD data behind 7 miRNA-disease
      survey papers (MGCNSS, NIMGSA, HLGNN-MDA, DiGAMN, CKSNP-GNN, MEAHNE, CoupleMDA; 3 dropped
      with documented reasons). Mean gap: trained models beat the topology floor by 4.7 AUROC
      points on average (0.951 vs 0.905) -- but 2/7 (MEAHNE, CoupleMDA, the two sparsest graphs)
      show the trained model adding little to nothing over popularity, MEAHNE's heuristic
      outright *beating* its trained model (0.9848 vs 0.9520) -- same signature as our own
      graph. Sanity-checked via degree-matched negatives collapsing to ~0.55-0.58 on all seven
      (rules out a leak/bug). Bonus: filled in 4 "not extracted" headline-AUROC cells in the
      original literature survey while tracing each paper's protocol. Remaining: manuscript
      integration.
- [~] **Generalize past our own graph — DATA IN HAND, PIPELINE UNBLOCKED (2026-07-16).
      Next gate: pre-register the subsample, then build.**
      One dataset + one interaction database is not a claim about a field. The second,
      independent source is downloaded: **miRTarBase release 10.0**, human, experimentally
      validated → `data/raw/mirtarbase_real_hsa10_all.tsv` (**1,730,376** pairs, 2,983 miRNAs,
      16,974 genes). Configs: `config_mirtarbase_edgesplit{,_uniform}.yaml`. TargetScan
      remains the weaker fallback (also sequence-derived — say so if we must use it).

      **The premise holds:** only **22.7%** of miRDB's edges appear in miRTarBase, so the
      sources are independent in content as well as in kind.

      **But the comparison is confounded as currently configured.** 99.5% of miRTarBase
      (1,722,514 / 1,730,376) is the **`Functional MTI (Weak)`** tier — qPCR / microarray /
      CLIP-seq. The strong tier (reporter assay + western blot) is only **7,862** edges over
      929 miRNAs. Strong+weak, which both configs currently use, is **3.6× denser than miRDB**
      (density 0.034 vs 0.009; median degree/miRNA **345 vs 99**).

      That collides with §3.1's finding that uniform negatives lose to edge-counting (`100dac6`):
      a denser, higher-degree graph makes degree shortcuts *more* exploitable, not less. If the
      audit looks worse on miRTarBase, we cannot separate *"the inflation reappears on an
      independent source"* from *"this graph is simply denser."* Choose one before building:
      - **(a) strong-only** — gold standard, but 929 miRNAs likely too few for the same architecture.
      - **(b) strong+weak, report density and degree as covariates** — keeps n, states the confound.
      - **(c) subsample miRTarBase to match miRDB's density/degree distribution** — the cleanest
        controlled contrast; costs an arbitrary sampling choice that must be pre-registered here.

      **Decision taken 2026-07-16: run BOTH arms as a sensitivity analysis** — (b) at real
      scale, plus (c) density-matched. (a) is rejected. Arm 1 shows the effect where the data
      actually is; arm 2 shows it is not an artifact of density. If they disagree, that
      disagreement is itself the finding.

      ---

      #### PRE-REGISTRATION — committed 2026-07-16, before any miRTarBase graph was built

      Nothing below has been run. No miRTarBase model exists, no subsample TSV exists. The
      commit timestamp of this block is the evidence, and it is the only thing that separates
      a sensitivity analysis from running two arms and keeping the flattering one. **If any
      part of this turns out to be unworkable, amend it in a commit of its own that says so —
      do not silently edit it once numbers exist.**

      **What is fixed across both arms.** Model, optimizer, splits and protocol are identical
      to `config_v2_edgesplit.yaml`. **Nothing is tuned on miRTarBase** — any hyperparameter
      change makes the source contrast uninterpretable. Both arms use `mirna_features: random`
      (Track B's features must not enter, or the contrast confounds source with features).

      **Arm 1 — strong+weak.** As downloaded: 155,120 edges / 2,920 miRNAs post-HVG-filter,
      density 0.0177, median degree/miRNA 33. Confounded by density *by construction*. Stated,
      not hidden.

      **Arm 2 — density-matched.** Subsample miRTarBase to the built miRDB graph's **per-gene
      in-degree**, gene by gene (identity-matched, not merely distribution-matched).

      *Why per-gene in-degree, and not something else.* The confound is that a denser graph
      makes the degree shortcut more exploitable. The shortcut is `gene_degree` — 0.8712,
      which **ignores the miRNA entirely** — and it is also what `sample_degree_matched_negatives`
      bins on (`splits.py:194-199`). Matching gene in-degree *per gene* makes the two graphs
      indistinguishable **to the confounder itself**: `gene_degree` assigns a near-identical
      score vector on both, so any AUROC difference is attributable to *which pairs are
      positive*, not to how popular the genes are. Distribution-matching would only equalize
      the histogram. Joint (gene × miRNA) matching is **not attempted**: realizing two degree
      marginals simultaneously as a subgraph of a *given* bipartite graph is generally
      infeasible, and chasing it is exactly the arbitrary tuning this block exists to prevent.
      The cost is stated plainly: this makes gene degree a *fixed covariate by construction*
      rather than a free variable — which is what a matched control is for.

      *Algorithm (fixed now).* Reference degrees and the 3,000-gene vocabulary come from
      `data/graphs/index_maps.pkl`; miRDB degrees are recomputed from `mirtarbase_hsa.tsv`.
      **Matching happens POST-HVG-filter** — the 155,120 and 44,186 figures already are, and
      matching pre-filter would match the wrong distribution. Gate: `assert reference total ==
      44,186` before sampling, so the reference reproduces the paper's own headline first.
      1. `t[g]` := gene `g`'s in-degree in the miRDB graph.
      2. Process genes in **ascending** order of available incident miRTarBase edges (scarce
         genes commit first, so a greedy quota cannot strand them). Ties broken by gene name.
      3. For each `g`, sample `k = min(t[g], |E_g|)` edges without replacement, weighted toward
         miRNAs under their miRDB out-degree quota. `numpy.random.default_rng(seed)` only —
         never the global RNG.
      4. **Deficits are NOT redistributed.** Where miRTarBase has fewer edges on a gene than
         miRDB does, that gene under-shoots and stays under-shot. Topping up elsewhere to hit
         44,186 would break the identity match to make a total look tidy.
      5. Seed **101** for the headline; **202/303** as a sampling-variance check on the
         held-out×matched cell only.

      *Pre-registered expectations — recorded so they cannot be reframed as results.*
      - Total edges **will undershoot 44,186**. Expected, not a failure. Do not tune to hit it.
      - Median miRNA out-degree will land near miRDB's 11 but is **reported as a covariate,
        not matched**. Do not tune it.
      - miRNA node count will fall from 2,920 toward miRDB's 2,460 as a *consequence* of
        per-gene matching. That is a sanity signal, **not a target**.

      **Criterion — "the inflation reproduces" on an arm iff ALL THREE hold**, each on the
      held-out test split. (The transductive row is deliberately absent: it cannot be measured
      on miRTarBase — `train.py:271` builds the edge split unconditionally and the leaky path
      was deleted in `8a12ce3` — so every criterion here is held-out only, and no attribution
      against miRDB's 0.9836 may be computed. `eval_heldout_grid.py` now enforces that.)
      - **(a)** uniform-trained × uniform-eval **loses to** `gene_degree`(uniform): margin ≤ 0.
        *miRDB: −0.0656.*
      - **(b)** uniform-trained × matched-eval sits within **0.03** of `gene_degree`(matched) —
        i.e. it learned popularity and nothing transferable. *miRDB: |0.5395 − 0.5126| = 0.027.*
        The threshold is calibrated to miRDB's own result and is stated as such, not chosen later.
      - **(c)** hard-trained × matched-eval **beats** `gene_degree`(matched): margin > 0.
        *miRDB: +0.1136.*
      **FAILS TO REPRODUCE** if any of (a)–(c) breaks.

      **Pre-registered reading of the four outcomes:**

      | arm 1 (strong+weak) | arm 2 (density-matched) | what we will write |
      |---|---|---|
      | reproduces | reproduces | The inflation **generalizes**; density is not the driver. Strongest outcome. |
      | reproduces | does not | The effect is **density-dependent**. Weaken the claim to "on graphs of miRDB-like density". This is why arm 2 exists. |
      | does not | reproduces | Density **masks** the effect at high density (plausible: at 0.0177 nearly every gene is popular, so `gene_degree` loses resolution). Report; do not spin. |
      | does not | does not | The effect is **miRDB-specific**. The generalization claim does not survive and §2.5 Path A needs re-scoping. **We commit now to reporting this.** |

      **Tie-break, fixed in advance:** if the arms disagree, **arm 2 is primary** (arm 1 is
      confounded by density) and the disagreement is itself reported.
      **Expected direction, stated in advance:** inflation should be *larger* on arm 1 than
      arm 2, because a denser, higher-degree graph makes the degree shortcut more exploitable.
      **If it is not, that is informative and gets reported as such.**
      **Both arms are reported regardless of outcome.**

      **Provenance caveat (verified 2026-07-16):** the miRTarBase site is headed *"Release 11.0"*
      (cite: miRTarBase 2025, NAR) but serves every file from `/files/10.0/`; `/files/11.0/` 404s
      and nothing states 11.0 content was published. We pin and report **10.0** — the only release
      whose bytes we can verify. Do not cite this as 11.0.

      **Blockers cleared 2026-07-16 — five ways this arm would have produced a green job and a
      wrong number.** Every one of them is the bug class this paper is about: a default standing
      in for a computation that never ran. None was hypothetical; each was verified against the
      live cluster before being fixed.
      1. `slurm_build_graph.sh` hardcoded `--config configs/config.yaml` and `mkdir -p data/graphs`.
         Submitting it for miRTarBase would have built (or skipped) the **miRDB** graph and
         exited 0. Now takes `CONFIG=`/`FORCE=` and derives the output dir from the config.
      2. `build_heterograph.py` early-returned on any existing graph file. Change the source,
         re-run, and the old graph came back with every log green — then you train on it. Now
         writes a `graph_manifest.json` carrying a **config fingerprint** and refuses to reuse a
         graph whose fingerprint differs; `--force` rebuilds. Verified the miRDB and miRTarBase
         fingerprints differ (`f5de0079` vs `4ea194e3`). Graphs predating manifests — including
         the one every published number comes from — are still reused, with a loud warning that
         they cannot be verified. **Path A's graph is untouched: sha256 `c5d98d15…`, unchanged
         and matching the hash recorded in `ms_specificity_audit.json`.**
      3. `eval_heldout_grid.py` hardcoded miRDB's `0.9836 / 0.8828` as the reference row and
         wrote it into **every** JSON. Run against miRTarBase it would have emitted an
         attribution of miRTarBase's held-out numbers against miRDB's constants — well-formed,
         plausible, meaningless. The reference now lives in `evaluation.reference_seen_edges` in
         the config of the graph that **owns** those numbers (the 10 `config_v2_edgesplit*`
         files); a config without it gets `attribution: null` and an explicit note, never a
         borrowed constant.
      4. `aggregate_seeds.py` hardcoded the `checkpoints_v2_*` stems and the miRDB
         `topology_baseline_<split>.json`, so a miRTarBase sweep would have been invisible to it
         or compared against miRDB's `gene_degree`. Now `--checkpoint-prefix` and
         `--topology-baseline`. It also refuses `n=1`, which used to print a confident
         `+/- 0.0000` that is indistinguishable from a real zero-variance result.
      5. `data/processed_mirtarbase/` did not exist, though the config points `processed_dir`
         there for a 4.5 GB `scrna_processed.h5ad`. Symlinked to `../processed/`, not copied —
         see `data/processed_mirtarbase/README.md`. The arms' `cellxgene` and `graph` config
         blocks are byte-identical, so the scRNA side is the same computation; two real copies
         could drift and turn "miRDB vs miRTarBase" into "one gene vocabulary vs another" while
         every filename still said interaction source.

### 3.3 Housekeeping for the manuscript

- [ ] Data provenance stated exactly: **miRDB v6.0 predictions, score ≥ 80** — not
      experimentally validated, not miRTarBase. This is now *load-bearing*: the paper's
      ceiling argument rests on the edges being sequence-derived while the graph carries no
      sequence.

      Concretely: the interaction table is miRDB (miRTarBase's server was unavailable
      during data collection). This is disclosed inside the repo (`README.md`), but the
      on-disk filename `data/raw/mirtarbase_hsa.tsv` still says miRTarBase "for pipeline
      compatibility." Before writing Methods: cite miRDB correctly throughout, and either
      rename the file or add a `data/raw/README` note so anyone auditing the repo isn't
      misled.
- [ ] **State the classification-vs-link-prediction gap head-on.** The ablations show cell
      typing is comparatively easy (0.9916) while the link head barely clears trivial
      topology. The manuscript must say this directly rather than let it read as an implicit
      caveat — a reviewer must not be able to mistake a high `cell_acc` for the paper's
      contribution.
- [ ] **Retitle, per D1 (answered 2026-07-27, option (a)).** The paper is about
      link-prediction evaluation on a biomedical graph; **MS is dataset provenance**. Concretely:
      - The title and abstract must not claim MS, multiple sclerosis, or disease relevance.
      - MS appears in exactly one place in the prose — the data-provenance sentence naming
        the cellxgene cohort — and nowhere else. `data/` paths and config names may keep it;
        they are filenames, not claims.
      - No result may be described as MS-specific, MS-associated, or disease-relevant. §3.4
        already suspended the two items that would have (circuit validation, MS-vs-control
        saliency); this closes the framing they lived in.
      - Sweep the existing prose for inherited MS framing before submission — `README.md`,
        `results/EVALUATION_AUDIT.md`, `results/RESUMEN_AUDITORIA.md` and the beamer deck all
        predate this decision.
- [ ] Authorship / affiliations.
- [ ] Venue: an **evaluation/methods** venue, not a biology-discovery one. Consistent with
      D1 — a de-scoped methods paper has no business at a biology-discovery venue anyway.

### 3.4 Suspended or dropped

- [x] ~~External validation of top circuits~~ — **suspended.** They are ranked by a head
      3.6 points above a no-learning heuristic, and (per §2) were never cell-type-specific
      in the first place.
- [x] ~~MS-vs-control differential saliency~~ — **suspended**, same reason.
- [x] ~~Push AUROC past 0.99~~ — **dropped.** That target was the transductive number.
- [~] Re-derive circuits from an honestly-trained model — only meaningful if the model
      beats trivial structure by enough to matter. It does not.

### 3.5 After submission — the improvement track (a different paper)

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
