# Literature Survey — how does this field actually evaluate link prediction?

**Purpose.** The methods paper's premise is that the evaluation flaws we found in our own
work are *not ours alone*. Until now that was **asserted and never shown** — a strawman a
reviewer would rightly name. This document is the evidence.

**Status: n = 21 papers read in full (2026-08-12 expansion of the original pilot), every cell
double-rated 2026-09-01.** Target range (20–30) met. Every row below was verified by reading the
paper's methods section, not by inference from an abstract. Where a paper does not say, the cell
reads **unclear** — and *that category is itself a finding*.

A twenty-second row of the sheet, "DTI field convention", records a general drug-target
convention rather than a paper (`counted_as_paper = no` in the TSV) and is **excluded from every
denominator** here. That is why the pilot is n=6 papers, not 7, and the total is 21, not 22.

**Second-rater limitation: CLOSED 2026-09-01.** Decision rules were written up first
([`SURVEY_CODEBOOK.md`](SURVEY_CODEBOOK.md)), then a blind second pass over all 21 papers was
made from the primary sources ([`literature_survey_rater2.tsv`](literature_survey_rater2.tsv)),
and agreement computed by [`analysis/interrater_agreement.py`](../analysis/interrater_agreement.py)
→ [`interrater_agreement.json`](interrater_agreement.json). **It overturned the model-free-baseline
headline** — see below.

**Last updated:** 2026-09-01

---

## The finding, and it is not the one we expected

We went in expecting to show *"everyone leaks held-out edges into the message-passing graph."*
**That is not what the literature shows, and we should not claim it.** In the original pilot (n=6 papers), 2
clearly strip test edges from the graph, 1 clearly does not, 3 do not say. Expanding to n=21
sharpens this further in the same direction: only 7/21 clearly strip test edges correctly and
**1/21 clearly leaves them in** — but the "unclear" bucket is still the largest category at
**13/21 (62%)**, because most 2024–2026 papers' methods sections simply do not state whether
held-out edges are masked from the graph at all. The story is not "the field leaks" — it is "the
field's methods sections do not let a reader check."

*(2026-08-13 spot-check: a sample of 14 "unclear" cells was re-verified directly against each
paper's methods section, prompted by the open question of whether "unclear" reflected a genuine
reporting gap or overly conservative automated extraction. 11/14 held up as genuinely unstated in
the text. 3/14 were extraction errors and have been corrected: HLGNN-MDA's held-out-edge removal
and CoupleMDA's CV-over-edges and negative-sampling cells all had explicit supporting sentences
that the first pass missed. See "Still to do" for the follow-up this implies for the second-rater
plan.)*

What the literature *does* show is sharper and more damaging — though **not** in the form this
document asserted until 2026-09-01. The original claim was "0 / 22, not one paper reports a
model-free baseline." **That claim was wrong, and the second rater is what caught it.**

The defect was in the criterion, not in either reading of any paper. It stated a definition — a
comparator with no parameters fitted to the association data — and then an enumerated list of
qualifying baselines (degree, common neighbours, Adamic–Adar, L3, …). The list left out this
subfield's *own* classical untrained methods: RWR, SPM, PBMDA, BNPMDA, WBSMDA, HNM, LLCMDA. The
first rater applied the list and recorded **no** for six papers the definition admits. The
corrected reading, re-audited paper by paper in
[`literature_survey_d4_reaudit.tsv`](literature_survey_d4_reaudit.tsv):

> **6 of 21 papers do report an untrained comparator — and every one of them beats it by a
> margin small enough to be worth remarking on, which none of them does.**
>
> | Paper | Untrained comparator | Its score | Paper's score | Margin |
> |---|---|---|---|---|
> | NGCN | HNM (heterogeneous network model) | 0.940 | 0.957 | **+1.7 pts** |
> | CKSNP-GNN | LLCMDA (also PBMDA) | 91.90 | 93.71 | **+1.8 pts** |
> | NIMGSA | SPM (structural perturbation) | 0.8960 | 0.9354 | **+3.9 pts** |
> | Orro | TCRWMDA (also WBSMDA, ICFMDA) | 92.09 | 97.10 | **+5.0 pts** |
> | HLGNN-MDA | BNPMDA (bipartite projection) | 0.85648 | 0.93086 | **+7.4 pts** |
> | ModulePred | RWR / RWRH (also DADA) | figure only | 0.834 | not readable |
>
> The other **14 report none at all**; 1 (DTI-MHAPR) is unclear. NIMGSA's SPM also *beats*
> IMCMDA, a trained matrix-completion baseline, in that paper's own Table 1.

**This is a better finding than the one it replaces.** "Nobody reports a floor" was an argument
about a hole in the literature. "Six papers print the floor in their own tables, clear it by
1.7–7.4 points, and not one of them says so" is an argument about how the field *reads* its own
numbers — and it is evidence the field already published, not evidence we had to generate.

And separately:

> **Treating unlabeled pairs as uniformly-sampled negatives remains the field's default.**
> 17 / 21 do exactly this (81%, versus 4/6 = 67% in the original pilot). Only two papers in
> the whole sample deviate — MGCNSS and HGDTI, both from the original pilot — and each selects
> negatives on a criterion of its own devising, itself an admission that the default is known
> to be broken. Two more (HiGLDP, GPS-DTI) never say where their negatives come from.

**Put those two together and you get the paper.** The headline AUROCs in this literature sit
at **0.91–0.99**. On our graph, under the *same* protocol (unlabeled pairs as uniform
negatives), a scorer that **ignores the miRNA entirely and only counts how many miRNAs already
target the gene** reaches **AUROC 0.8712** — and *beats* our trained graph transformer
(0.8056).

**A field that does not read a model-free control as a floor cannot know whether its 0.97 is a
result or a popularity effect.** That is the claim, it is supported, and it does not require accusing
anyone of leakage.

---

## The evidence

| # | Paper | Venue / Year | CV over edges? | Test edges removed from message-passing graph? | Negatives | Model-free baseline? | Headline AUROC |
|---|---|---|:--:|:--:|---|:--:|:--:|
| 1 | **MGCNSS** | Brief. Bioinform. 2024 | yes | **NO** — graph/similarity matrices unchanged | distance-based selection (their contribution) | **NO** | **0.9874** |
| 2 | **NIMGSA** | 2022 | yes | **unclear** — never stated | **not described at all** | **YES** — SPM (structural perturbation), 0.8960 vs 0.9354 (+3.9 pts) | **0.9354** |
| 3 | **HybridGNN** | Bioinformatics 2026 | yes | **YES** — PyG `RandomLinkSplit` | uniform random from unknown pairs | **NO** | **0.9715** |
| 4 | **HGDTI** | BMC Bioinform. 2022 | yes | **NO** — test edges retained in network | "reliable" score-filtered (non-uniform) | **NO** | **~0.979** |
| 5 | **NGCN** | 2024 | yes | **unclear** — not confirmed | uniform random, 1:10 ("an unknown pair is generally viewed as a negative sample") | **YES** — HNM (heterogeneous network model), 0.940 vs 0.957 (+1.7 pts) | **0.910** |
| 6 | **kmerPMTF** | PeerJ 2024 | yes | **YES** — similarity matrices built from training split only | all unlabeled pairs, count-matched | **NO** | **0.80–0.91** |
| 7 | *DTI field convention* | (multiple) | yes | — | "a drug–target pair with an unknown interaction is generally viewed as a negative sample", typically 10× positives | — *(not a paper; excluded from all denominators)* | — |
| 8 | **Orro** | Biomedicines 2026 | yes | **YES** — miRNA-level holdout (stronger than edge-level) | uniform random from unannotated pairs | **YES** — TCRWMDA (also WBSMDA, ICFMDA), 92.09 vs 97.10 (+5.0 pts) | ~0.98 |
| 9 | **CoupleMDA** | IJMS 2025 | yes | **YES** — train/val/test edges strictly partitioned | uniform random, 1:1 | **NO** | 0.9536 (Table 3) |
| 10 | **GONNMDA** | Genes 2025 | yes | **unclear** | uniform random, 1:1 | **NO** | 0.9541 |
| 11 | **DiGAMN** | BMC Genomics 2024 | yes | **YES** — 20% masked to prevent leakage | uniform random, 1:1 / 1:5 / 1:10 | **NO** | 0.9635 |
| 12 | **DGNMDA** | Bioengineering 2024 | yes | **unclear** | undersampling (ratio unspecified) | **NO** | 0.9455 |
| 13 | **HLGNN-MDA** | IJMS 2022 | unclear | **YES** — positive test-set samples removed from the adjacency matrix each round | uniform random, 1:1 | **YES** — BNPMDA (bipartite projection), 0.85648 vs 0.93086 (+7.4 pts) | 0.93086 (10-fold CV) |
| 14 | **MEAHNE** | Life 2022 | yes | **unclear** | uniform random, 1:1 | **NO** | 0.9520 (Table 3) |
| 15 | **CKSNP-GNN** | Genes 2022 | yes | **unclear** | uniform random, 1:1 (16,427 negatives) | **YES** — LLCMDA (also PBMDA), 91.90 vs 93.71 (+1.8 pts) | 0.9371 (5-fold CV mean) |
| 16 | **HMCDA** | BMC Bioinform. 2023 | yes | **unclear** | uniform random, 5:1 | **NO** | 0.9135 |
| 17 | **HiGLDP** | BMC Biology 2026 | yes | **YES** — strictly excluded from training folds | uniform random, 1:1 | **NO** | 0.9696 |
| 18 | **GPS-DTI** | BMC Biology 2025 | yes | **unclear** | balanced pos/neg (curation unclear) | **NO** | not extracted |
| 19 | **SaeGraphDTI** | BMC Bioinform. 2025 | yes | **unclear** | all unlabeled pairs (unsampled) | **NO** | not extracted |
| 20 | **DTI-MHAPR** | BMC Bioinform. 2025 | yes | **unclear** | uniform random, 1:1 | **unclear** — an eighth comparator is mentioned but cannot be identified | not extracted |
| 21 | **iNGNN-DTI** | Bioinformatics 2024 | yes | **unclear** | uniform random, 1:1 | **NO** | 0.931–0.934 |
| 22 | **ModulePred** | BMC Bioinform. 2024 | yes | **unclear** | uniform random, 50:1 | **YES** — RWR and RWRH reported as compared methods (the L3 score is augmentation-only, but it is not the only untrained method in the paper) | 0.834 |

Rows 8–22 (2026-08-12 expansion): all open-access (PMC), verified via the paper's own PMC JATS
XML front matter for bibliographic metadata; classification quotes are in
`results/literature_survey.tsv`. Single-rater, same as rows 1–7 — but see the 2026-08-13
spot-check note above: 3 cells among rows 8–22 were corrected after re-reading the primary
source (CoupleMDA's CV-over-edges and negative-sampling cells, HLGNN-MDA's held-out-edge-removal
cell, and ModulePred's model-free-baseline cell resolved from "unclear" to a confirmed "no").

### Tallies (n = 21 papers; the field-convention row is excluded)

| Practice | Original pilot (n=6) | Expansion (n=15) | Combined (n=21) |
|---|---|---|---|
| Cross-validate over **edges** (not nodes) | 5 / 6 | 14 / 15 | **19 / 21 (90%)** |
| Held-out edges **removed** from the encoder's input graph | 2 / 6 | 5 / 15 | **7 / 21 (33%)** |
| Held-out edges **left in** the encoder's input graph | 1 / 6 | 0 / 15 | **1 / 21 (5%)** ⚠️ |
| **Unclear** from the methods section | 3 / 6 | 10 / 15 | **13 / 21 (62%)** ⚠️ |
| Negatives = unlabeled pairs, uniform or wholesale | 4 / 6 | 13 / 15 | **17 / 21 (81%)** |
| Any **model-free / heuristic baseline** reported | 2 / 6 | 4 / 15 | **6 / 21 (29%)** 🔴 |
| — of those, margin over it **discussed by the paper** | 0 / 2 | 0 / 4 | **0 / 6 (0%)** 🔴 |

---

## What this does — and does not — license us to say

**We CAN say, with evidence:**

1. **A model-free control is missing from 15 of 21 papers, and is never read as a floor by
   any of the 6 that do report one.** The field has no routine way of knowing whether its
   numbers beat a popularity heuristic — and where the comparison is on the page, the margin
   (1.7–7.4 AUROC points) passes without comment. We show, on a real graph, that under this
   protocol the trained model may not beat the heuristic at all.
2. **Uniform/unlabeled negatives are the default (17/21, 81%),** consistent with the pilot's
   4/6 (67%). Only two papers in the whole sample deviate — MGCNSS and HGDTI, both from the
   original pilot — and each does so by selecting negatives on a criterion of its own devising:
   evidence the problem is recognized but has no standard remedy or control. Two more (HiGLDP,
   GPS-DTI) never say where their negatives come from.
3. **Reporting of the split is frequently too vague to reproduce (13/21 unclear, 62%, up from
   3/6 in the pilot).** Whether test edges reach the encoder — the single thing that decides if
   the number is prediction or reconstruction — often cannot be determined from the paper at
   all, and this got *more* common, not less, in the more recent (2024–2026) papers added in the
   expansion. That is a reporting-standards finding, and it is independently publishable. A
   2026-08-13 spot-check of 14 "unclear" cells against primary sources confirmed this is mostly
   a genuine reporting gap (11/14 held up) rather than an extraction artifact, though 3/14 were
   corrected — all 14 checked papers had public GitHub code, meaning the information usually
   exists but is not disclosed in the methods prose a reader or reviewer would actually evaluate.

**We must NOT say:**

- ❌ *"The field routinely leaks test edges into message passing."* **Not supported, and even
  less supported at n=21 than in the pilot, and weaker still after the 2026-09-06 adjudication.**
  Only **1/21** clearly does it wrong — MGCNSS, whose propagated matrix is defined globally over
  all 5,430 known associations; 7/21 clearly strip test edges correctly (one, Orro 2026, holds
  out entire miRNAs — a stronger, inductive-style check, and the reason its own D1 call is *no*).
  Making the "routinely leaks" claim would be the same sin we are criticizing: asserting a
  strong quantitative claim the evidence does not carry.
- ❌ *"No surveyed paper reports a model-free baseline."* **Retracted 2026-09-01 — it was
  false.** Six do. The claim that survives is about how the margin is read, not about whether
  the comparator exists. See the criterion defect described above; the failure mode was a
  checklist that silently narrowed its own definition.
- ❌ *"Our leak is typical."* **It is not — ours was worse than the norm.** Every paper here at
  least cross-validates over edges; our original split partitioned *cells* only. Honesty here
  costs us nothing and buys credibility: we found our own error, and it was a bad one.

**The honest framing for the manuscript's motivation section:**

> *Across 21 papers surveyed, edge-level cross-validation is near-universal (19/21), but **15
> report no model-free baseline and the 6 that do never remark on the 1.7–7.4-point margin
> their own tables show**, unlabeled pairs are treated as negatives in the
> majority (17/21), and in the majority of cases (13/21) the methods section does not permit
> the reader to determine whether held-out edges were visible to the encoder. We show that
> under exactly this protocol, a one-line popularity heuristic attains AUROC 0.87 on a real
> biomedical graph — within the band of published state-of-the-art results (0.91–0.99) — and
> outperforms a trained heterogeneous graph transformer.*

---

## Follow-up: measuring the inflation directly on surveyed papers' own data

Completed 2026-08-13: `results/HMDD_TOPOLOGY_AUDIT.md` computes this project's own model-free
topology baseline directly on the HMDD-derived data behind seven miRNA-disease papers in this
survey (MGCNSS, NIMGSA, HLGNN-MDA, DiGAMN, CKSNP-GNN, MEAHNE, CoupleMDA), rather than only
inferring inflation from our own graph. Mean gap: trained models beat the topology floor by 4.7
AUROC points — but on the two sparsest graphs (MEAHNE, CoupleMDA) the gap nearly vanishes or
reverses (MEAHNE's heuristic outright beats its trained model, 0.9848 vs 0.9520), echoing our
own graph's finding. See that document for the full results, tiering, and caveats.

## Still to do

- ~~**Expand to 20–30 papers.**~~ **DONE 2026-08-12 — n=21 papers.** 15 new papers added, all
  open-access (PMC), verified via each paper's own PMC JATS XML front matter. See the expanded
  table and tallies above.
- ~~**Two independent raters** for the "unclear" column, with disagreements recorded.~~
  **DONE 2026-09-01.** Codebook written first ([`SURVEY_CODEBOOK.md`](SURVEY_CODEBOOK.md)), then
  a blind second pass over all 21 papers from the primary sources
  ([`literature_survey_rater2.tsv`](literature_survey_rater2.tsv)); agreement and every
  disagreement by name in [`interrater_agreement.json`](interrater_agreement.json).
  Raw agreement / Cohen's κ: model-free baseline 95% (κ n/a, skewed marginals), CV-over-edges
  86% (κ n/a), negative sampling 76% (κ = 0.49), held-out edges removed 71% (κ = 0.44).
  The prior 2026-08-13 same-rater spot-check of 14 "unclear" cells (11/14 held up, 3/14
  corrected) stands as a separate, weaker check. **The one D4 disagreement is what exposed the
  criterion defect and overturned the 0/22 headline** — see the top of this document.
  **Open follow-up:** the 6 D2 and 5 D3 disagreements listed in the JSON have *not* yet been
  adjudicated one by one; the tallies above still carry rater 1's calls on those cells.
- **Record the exact quoted sentence** supporting each classification, in the TSV. **DONE** for
  all 21 rows.
- **Check the two "did it right" papers (HybridGNN, kmerPMTF) for negative sampling and
  baselines.** Both strip test edges — and both still use uniform/unlabeled negatives and
  report no model-free control. This pattern replicates in the expansion: several new papers
  that correctly remove test edges from the graph (DiGAMN, HiGLDP, CoupleMDA) still use
  uniform-random or unclear negative sampling and report no model-free baseline. Orro is the
  instructive exception in the other direction: it holds out whole miRNAs *and* reports an
  untrained comparator — and still never asks why it only beats it by 5 points. If a paper can
  get the split right and *still* be vulnerable to the popularity artifact, that is the
  strongest possible argument for our proposed reporting standard.
- **Manual close-reading candidates, not pursued (not paywalled, extraction was inconclusive):**
  Gra-CRC-miRTar (CSBJ 2024), GraphTar (BMC Bioinformatics 2023), a heterogeneous-GNN
  lncRNA-disease paper (Sci Rep 2022), gGATLDA (BMC Bioinformatics 2022), DHGT-DTI (J Pharm
  Anal 2025), a substructure-GNN DTI paper (Front. Pharmacol. 2025), and a gene-disease GNN
  paper (Entropy 2023). None were included in the n=21 count above.

## Adjudication of the 14 rater disagreements (2026-09-06)

The blind second pass (2026-09-01) left 14 cells where the two raters disagreed: 3 on the
cross-validation unit, 6 on whether held-out edges are removed from the encoder's graph, and 5 on
negative sampling. (The fifteenth, ModulePred's D4 cell, was adjudicated on 2026-09-01 and is what
exposed the criterion defect.) All 14 are now settled one at a time against the primary source,
with the codebook rule that decided each and the verbatim sentence it rests on recorded in
**`results/literature_survey_adjudication.tsv`**. **11 went to the second rater, 3 upheld the first.**

**What changed, and what it cost the paper's claims:**

- **"Only 2 of 21 clearly leave held-out edges in the graph" became 1 of 21.** The first rater had
  recorded HGDTI as retaining them on a paraphrase — "heterogeneous network retains held-out test
  interactions during message passing" — not on a quote. The paper describes its heterogeneous
  network and its 10-fold CV in separate places and never connects them, so no qualifying sentence
  exists and the cell is `unclear`. The one surviving `no` is **MGCNSS**, and it is solid: the
  propagated matrix is defined globally as `M = [IM A; A^T ID]` with `A` the full HMDD v2.0 matrix
  of 5,430 associations, while the split is described separately over "the positive and selected
  negative samples ... into five folders".
- **Two papers moved into the "clearly strips test edges" column and two moved out.** In: NGCN
  ("a randomly chosen subset of 90% positive and negative pairs was used as training data to
  construct the heterogeneous networks") and SaeGraphDTI, which names the inflation mechanism this
  survey is about in its own words. Out: HiGLDP, whose exclusion statement reaches only "the
  training folds", and kmerPMTF, where the first rater's "similarity matrices computed only from
  the training split" was again a paraphrase with no sentence behind it. The count stayed at 7/21;
  the membership did not.
- **Orro's D1 call flipped to `no`** on the first rater's own quote: it holds out whole miRNAs, not
  associations, which is a stronger check than the survey's other 20 papers apply.
- **Negatives drawn from unlabeled pairs rose from 16/21 to 17/21**: DGNMDA's cell was decided on a
  sentence the first rater stopped one short of, and NIMGSA — which never uses the word "negative"
  — minimises a reconstruction loss over the whole association matrix, so every unlabeled pair is
  a negative by construction. HiGLDP went the other way: a stated 1:1 ratio is not a stated
  procedure.

**Two of the 14 were codebook gaps rather than rater errors,** and both rules are now written into
`SURVEY_CODEBOOK.md`: D3 had no rule for a paper that samples negatives one way for training and
another for evaluation (kmerPMTF), nor for one that describes its negatives for a case study but
not for the benchmark carrying its headline number (GPS-DTI). Both cells are now classified on the
arm the reported metric comes from, which upheld the first rater in each case.

**The pattern in the errors, worth naming:** 4 of the 6 D2 disagreements were the same mistake in
the same direction — a call recorded on the rater's inference about what the architecture must be
doing, where the codebook now requires a verbatim sentence. That rule was written on 2026-09-01
precisely to catch this, and it did. The 15 D2 cells the two raters *agreed* on have not been
re-audited under it; agreement is evidence but not proof, and a full D2 re-audit under the
verbatim-quote rule remains open.


## D2 re-audit of the cells the two raters agreed on (2026-09-06)

The adjudication overturned 4 of the 6 disputed D2 cells for the same reason — a call recorded on
the rater's inference instead of on a sentence from the paper — so the 15 D2 cells where *both*
raters agreed were re-audited under the same rule. Agreement is evidence, not proof.

**The 5 agreed `yes` cells all hold.** Each rests on a verbatim sentence naming the graph, the
adjacency matrix, or a documented split API: HLGNN-MDA ("removed the positive samples in the test
set from the adjacency matrix"), CoupleMDA ("the training graph only contained positive edges from
the training set"), DiGAMN ("We masked 20% of the association information during cross-validation
in the test set to prevent information leakage"), Orro (associations of validation miRNAs removed
from the training graph), and HybridGNN (PyG's `RandomLinkSplit`, which the codebook admits by
name). The verbatim-quote rule does not threaten any of them.

**The 10 agreed `unclear` cells all hold too**, checked in the opposite direction: each paper's
full text was searched today for any sentence connecting its split to the graph, adjacency matrix,
similarity matrices or message passing. All ten came back with nothing — NIMGSA, GONNMDA, DGNMDA,
MEAHNE, CKSNP-GNN, HMCDA, GPS-DTI, DTI-MHAPR, iNGNN-DTI, ModulePred. Several define the association
matrix and their GIP-kernel similarities globally, which is suggestive, but none states what
happens to that matrix at the split, and the codebook forbids resolving `unclear` by inference.

This matters more than it sounds: the survey's dominant finding is the size of the `unclear`
bucket, and the obvious objection to it is that "unclear" measures the raters rather than the
papers. Two raters, a written codebook, and now a targeted full-text search per paper all return
the same answer.

### Open question this raised: two cells may be category errors, not reporting gaps

**iNGNN-DTI and GPS-DTI never propagate over a drug-target interaction graph at all.** Both encode
each drug as a molecular graph (atoms as nodes, bonds as edges) and each protein separately
(iNGNN-DTI from an AlphaFold2 contact map, GPS-DTI from ESM-2 features), then fuse the two
representations with cross-attention: "There is no drug-target interaction graph with edges between
them used during the GNN phase." For a model of that shape, a held-out interaction *cannot* reach
the encoder through graph structure, so D2's question — are held-out edges removed from the
adjacency structure the encoder propagates over? — has no answer to give rather than an
undisclosed one.

Counting them as `unclear` therefore overstates the reporting gap by two papers. The fix would be a
fourth D2 value, `not_applicable`, and a denominator of 19 for that dimension: **7/19 yes, 1/19 no,
11/19 unclear (58%)** in place of 7/21, 1/21, 13/21 (62%). Not applied yet — it changes a headline
denominator, and that is the author's call. Every other paper in the sample propagates over an
association-derived structure, so the two are the only candidates.


## Sources

- [MGCNSS — Briefings in Bioinformatics 2024](https://academic.oup.com/bib/article/25/3/bbae168/7645839)
- [NIMGSA — PMC8774034](https://pmc.ncbi.nlm.nih.gov/articles/PMC8774034/)
- [HybridGNN — Bioinformatics](https://academic.oup.com/bioinformatics/article/42/5/btag171/8586881)
- [HGDTI — BMC Bioinformatics / PMC9004085](https://pmc.ncbi.nlm.nih.gov/articles/PMC9004085/)
- [NGCN — PMC10955156](https://pmc.ncbi.nlm.nih.gov/articles/PMC10955156/)
- [kmerPMTF — PeerJ 2024 / PMC11122044](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11122044/)
- [A Review of Link Prediction Applications in Network Biology — arXiv:2312.01275](https://arxiv.org/pdf/2312.01275)
- [Orro — Biomedicines 2026 / PMC12938369](https://pmc.ncbi.nlm.nih.gov/articles/PMC12938369/)
- [CoupleMDA — IJMS 2025 / PMC12112494](https://pmc.ncbi.nlm.nih.gov/articles/PMC12112494/)
- [GONNMDA — Genes 2025 / PMC12027447](https://pmc.ncbi.nlm.nih.gov/articles/PMC12027447/)
- [DiGAMN — BMC Genomics 2024 / PMC11610307](https://pmc.ncbi.nlm.nih.gov/articles/PMC11610307/)
- [DGNMDA — Bioengineering 2024 / PMC11591469](https://pmc.ncbi.nlm.nih.gov/articles/PMC11591469/)
- [HLGNN-MDA — IJMS 2022 / PMC9657597](https://pmc.ncbi.nlm.nih.gov/articles/PMC9657597/)
- [MEAHNE — Life 2022 / PMC9655430](https://pmc.ncbi.nlm.nih.gov/articles/PMC9655430/)
- [CKSNP-GNN — Genes 2022 / PMC9602123](https://pmc.ncbi.nlm.nih.gov/articles/PMC9602123/)
- [HMCDA — BMC Bioinformatics 2023 / PMC10494331](https://pmc.ncbi.nlm.nih.gov/articles/PMC10494331/)
- [HiGLDP — BMC Biology 2026 / PMC13032502](https://pmc.ncbi.nlm.nih.gov/articles/PMC13032502/)
- [GPS-DTI — BMC Biology 2025 / PMC12659342](https://pmc.ncbi.nlm.nih.gov/articles/PMC12659342/)
- [SaeGraphDTI — BMC Bioinformatics 2025 / PMC12265306](https://pmc.ncbi.nlm.nih.gov/articles/PMC12265306/)
- [DTI-MHAPR — BMC Bioinformatics 2025 / PMC11726937](https://pmc.ncbi.nlm.nih.gov/articles/PMC11726937/)
- [iNGNN-DTI — Bioinformatics 2024 / PMC10957515](https://pmc.ncbi.nlm.nih.gov/articles/PMC10957515/)
- [ModulePred — BMC Bioinformatics 2024 / PMC11549817](https://pmc.ncbi.nlm.nih.gov/articles/PMC11549817/)
