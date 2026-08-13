# Literature Survey — how does this field actually evaluate link prediction?

**Purpose.** The methods paper's premise is that the evaluation flaws we found in our own
work are *not ours alone*. Until now that was **asserted and never shown** — a strawman a
reviewer would rightly name. This document is the evidence.

**Status: n = 22 papers read in full (2026-08-12 expansion of the original 7).** Target range
(20–30) met. Every row below was verified by reading the paper's methods section, not by
inference from an abstract. Where a paper does not say, the cell reads **unclear** — and *that
category is itself a finding*. **Remaining limitation: single rater.** All 22 classifications,
including the original 7, were made by one reader; the "unclear" calls have not been
cross-checked by a second independent rater. See "Still to do."

**Last updated:** 2026-08-13

---

## The finding, and it is not the one we expected

We went in expecting to show *"everyone leaks held-out edges into the message-passing graph."*
**That is not what the literature shows, and we should not claim it.** In the original n=7, 2
clearly strip test edges from the graph, 2 clearly do not, 3 do not say. Expanding to n=22
sharpens this further in the same direction: only 7/22 clearly strip test edges correctly and
2/22 clearly leave them in — but the "unclear" bucket is still the largest category at
**12/22 (55%)**, because most 2024–2026 papers' methods sections simply do not state whether
held-out edges are masked from the graph at all. The story is not "the field leaks" — it is "the
field's methods sections do not let a reader check."

*(2026-08-13 spot-check: a sample of 14 "unclear" cells was re-verified directly against each
paper's methods section, prompted by the open question of whether "unclear" reflected a genuine
reporting gap or overly conservative automated extraction. 11/14 held up as genuinely unstated in
the text. 3/14 were extraction errors and have been corrected: HLGNN-MDA's held-out-edge removal
and CoupleMDA's CV-over-edges and negative-sampling cells all had explicit supporting sentences
that the first pass missed. See "Still to do" for the follow-up this implies for the second-rater
plan.)*

What the literature *does* show is sharper, more universal, and more damaging, and it gets
*stronger*, not weaker, at n=22:

> **Not one of the 22 papers surveyed reports a single model-free baseline.**
> **0 / 22.** No gene/node degree. No common neighbours. No Adamic–Adar. Not even random. One
> paper (ModulePred) computes an L3 topological score, but only for network data augmentation —
> confirmed absent from every results/comparison table, so even the closest near-exception does
> not count. Every comparison is a learned method against other learned methods.

And separately:

> **Treating unlabeled pairs as uniformly-sampled negatives remains the field's default.**
> 16 / 22 do exactly this (73%, versus 4/7 = 57% in the original pilot). Among the original
> seven, two of the three that deviate from uniform sampling made "better negative selection"
> their *headline contribution* — itself an admission that the default is known to be broken.

**Put those two together and you get the paper.** The headline AUROCs in this literature sit
at **0.91–0.99**. On our graph, under the *same* protocol (unlabeled pairs as uniform
negatives), a scorer that **ignores the miRNA entirely and only counts how many miRNAs already
target the gene** reaches **AUROC 0.8712** — and *beats* our trained graph transformer
(0.8056).

**A field that never reports a model-free control cannot know whether its 0.97 is a result or
a popularity effect.** That is the claim, it is supported, and it does not require accusing
anyone of leakage.

---

## The evidence

| # | Paper | Venue / Year | CV over edges? | Test edges removed from message-passing graph? | Negatives | Model-free baseline? | Headline AUROC |
|---|---|---|:--:|:--:|---|:--:|:--:|
| 1 | **MGCNSS** | Brief. Bioinform. 2024 | yes | **NO** — graph/similarity matrices unchanged | distance-based selection (their contribution) | **NO** | **0.9874** |
| 2 | **NIMGSA** | 2022 | yes | **unclear** — never stated | **not described at all** | **NO** | **0.9354** |
| 3 | **HybridGNN** | Bioinformatics 2026 | yes | **YES** — PyG `RandomLinkSplit` | uniform random from unknown pairs | **NO** | **0.9715** |
| 4 | **HGDTI** | BMC Bioinform. 2022 | yes | **NO** — test edges retained in network | "reliable" score-filtered (non-uniform) | **NO** | **~0.979** |
| 5 | **NGCN** | 2024 | yes | **unclear** — not confirmed | uniform random, 1:10 ("an unknown pair is generally viewed as a negative sample") | **NO** | **0.910** |
| 6 | **kmerPMTF** | PeerJ 2024 | yes | **YES** — similarity matrices built from training split only | all unlabeled pairs, count-matched | **NO** | **0.80–0.91** |
| 7 | *DTI field convention* | (multiple) | yes | — | "a drug–target pair with an unknown interaction is generally viewed as a negative sample", typically 10× positives | **NO** | — |
| 8 | **Orro** | Biomedicines 2026 | yes | **YES** — miRNA-level holdout (stronger than edge-level) | uniform random from unannotated pairs | **NO** | ~0.98 |
| 9 | **CoupleMDA** | IJMS 2025 | yes | **YES** — train/val/test edges strictly partitioned | uniform random, 1:1 | **NO** | 0.9536 (Table 3) |
| 10 | **GONNMDA** | Genes 2025 | yes | **unclear** | uniform random, 1:1 | **NO** | 0.9541 |
| 11 | **DiGAMN** | BMC Genomics 2024 | yes | **YES** — 20% masked to prevent leakage | uniform random, 1:1 / 1:5 / 1:10 | **NO** | 0.9635 |
| 12 | **DGNMDA** | Bioengineering 2024 | yes | **unclear** | undersampling (ratio unspecified) | **NO** | 0.9455 |
| 13 | **HLGNN-MDA** | IJMS 2022 | unclear | **YES** — positive test-set samples removed from the adjacency matrix each round | uniform random, 1:1 | **NO** | 0.93086 (10-fold CV) |
| 14 | **MEAHNE** | Life 2022 | yes | **unclear** | uniform random, 1:1 | **NO** | 0.9520 (Table 3) |
| 15 | **CKSNP-GNN** | Genes 2022 | yes | **unclear** | uniform random, 1:1 (16,427 negatives) | **NO** | 0.9371 (5-fold CV mean) |
| 16 | **HMCDA** | BMC Bioinform. 2023 | yes | **unclear** | uniform random, 5:1 | **NO** | 0.9135 |
| 17 | **HiGLDP** | BMC Biology 2026 | yes | **YES** — strictly excluded from training folds | uniform random, 1:1 | **NO** | 0.9696 |
| 18 | **GPS-DTI** | BMC Biology 2025 | yes | **unclear** | balanced pos/neg (curation unclear) | **NO** | not extracted |
| 19 | **SaeGraphDTI** | BMC Bioinform. 2025 | yes | **unclear** | all unlabeled pairs (unsampled) | **NO** | not extracted |
| 20 | **DTI-MHAPR** | BMC Bioinform. 2025 | yes | **unclear** | uniform random, 1:1 | **NO** | not extracted |
| 21 | **iNGNN-DTI** | Bioinformatics 2024 | yes | **unclear** | uniform random, 1:1 | **NO** | 0.931–0.934 |
| 22 | **ModulePred** | BMC Bioinform. 2024 | yes | **unclear** | uniform random, 50:1 | **NO** — L3 score confirmed used only for network augmentation, absent from every comparison table | 0.834 |

Rows 8–22 (2026-08-12 expansion): all open-access (PMC), verified via the paper's own PMC JATS
XML front matter for bibliographic metadata; classification quotes are in
`results/literature_survey.tsv`. Single-rater, same as rows 1–7 — but see the 2026-08-13
spot-check note above: 3 cells among rows 8–22 were corrected after re-reading the primary
source (CoupleMDA's CV-over-edges and negative-sampling cells, HLGNN-MDA's held-out-edge-removal
cell, and ModulePred's model-free-baseline cell resolved from "unclear" to a confirmed "no").

### Tallies (n = 22)

| Practice | Original pilot (n=7) | Expansion (n=15) | Combined (n=22) |
|---|---|---|---|
| Cross-validate over **edges** (not nodes) | 7 / 7 | 14 / 15 | **21 / 22 (95%)** |
| Held-out edges **removed** from the encoder's input graph | 2 / 7 | 5 / 15 | **7 / 22 (32%)** |
| Held-out edges **left in** the encoder's input graph | 2 / 7 | 0 / 15 | **2 / 22 (9%)** ⚠️ |
| **Unclear** from the methods section | 3 / 7 | 9 / 15 (+1 NA) | **12 / 22 (55%)** ⚠️ |
| Negatives = unlabeled pairs, uniform | 4 / 7 | 12 / 15 | **16 / 22 (73%)** |
| Any **model-free / heuristic baseline** reported | 0 / 7 | 0 / 15 | **0 / 22 (0%)** 🔴 |

---

## What this does — and does not — license us to say

**We CAN say, with evidence:**

1. **No model-free control is reported anywhere in this sample (0/22).** The field has no
   routine way of knowing whether its numbers beat a popularity heuristic — and we show, on a
   real graph, that under its own default protocol they may not. This held with zero exceptions
   across a 3x larger, independently sourced sample than the original pilot.
2. **Uniform/unlabeled negatives are the default (16/22, 73%),** consistent with the pilot's
   4/7 (57%). Among the original seven, the papers that deviate do so as their *headline
   contribution* — evidence the problem is recognized but has no standard remedy or control.
3. **Reporting of the split is frequently too vague to reproduce (12/22 unclear, 55%, up from
   3/7 in the pilot).** Whether test edges reach the encoder — the single thing that decides if
   the number is prediction or reconstruction — often cannot be determined from the paper at
   all, and this got *more* common, not less, in the more recent (2024–2026) papers added in the
   expansion. That is a reporting-standards finding, and it is independently publishable. A
   2026-08-13 spot-check of 14 "unclear" cells against primary sources confirmed this is mostly
   a genuine reporting gap (11/14 held up) rather than an extraction artifact, though 3/14 were
   corrected — all 14 checked papers had public GitHub code, meaning the information usually
   exists but is not disclosed in the methods prose a reader or reviewer would actually evaluate.

**We must NOT say:**

- ❌ *"The field routinely leaks test edges into message passing."* **Not supported, and even
  less supported at n=22 than at n=7.** Only 2/22 clearly do it wrong; 7/22 clearly strip test
  edges correctly (one, Orro 2026, holds out entire miRNAs — a stronger, inductive-style check).
  Making the "routinely leaks" claim would be the same sin we are criticizing: asserting a
  strong quantitative claim the evidence does not carry.
- ❌ *"Our leak is typical."* **It is not — ours was worse than the norm.** Every paper here at
  least cross-validates over edges; our original split partitioned *cells* only. Honesty here
  costs us nothing and buys credibility: we found our own error, and it was a bad one.

**The honest framing for the manuscript's motivation section:**

> *Across 22 papers surveyed, edge-level cross-validation is near-universal (21/22), but **no
> paper reports a model-free baseline**, unlabeled pairs are treated as negatives in the
> majority (16/22), and in the majority of cases (12/22) the methods section does not permit
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

- ~~**Expand to 20–30 papers.**~~ **DONE 2026-08-12 — n=22.** 15 new papers added, all
  open-access (PMC), verified via each paper's own PMC JATS XML front matter. See the expanded
  table and tallies above.
- **Two independent raters** for the "unclear" column, with disagreements recorded. **Still
  open, but partially de-risked.** A 2026-08-13 spot-check re-verified 14 of the 15 "unclear"
  cells among rows 8-22 directly against each paper's methods section (not a second rater, but a
  second read by the same reader, prompted by the question of whether "unclear" was a genuine
  reporting gap or an extraction artifact). Result: 11/14 held up, 3/14 were corrected (see the
  note at the top of this document). This is evidence the "unclear" calls are not rubber-stamped,
  but it is not a substitute for an independent second rater — the correction rate (3/14, ~21%)
  is high enough that an independent pass over the full n=22 could still move the tallies further.
  This remains the survey's primary open item and is reported as such in the manuscript's
  Discussion (fifth limitation).
- **Record the exact quoted sentence** supporting each classification, in the TSV. **DONE** for
  all 22 rows.
- **Check the two "did it right" papers (HybridGNN, kmerPMTF) for negative sampling and
  baselines.** Both strip test edges — and both still use uniform/unlabeled negatives and
  report no model-free control. This pattern replicates in the expansion: several new papers
  that correctly remove test edges from the graph (Orro, DiGAMN, HiGLDP, CoupleMDA) still use
  uniform-random or unclear negative sampling and report no model-free baseline. If a paper can
  get the split right and *still* be vulnerable to the popularity artifact, that is the
  strongest possible argument for our proposed reporting standard.
- **Manual close-reading candidates, not pursued (not paywalled, extraction was inconclusive):**
  Gra-CRC-miRTar (CSBJ 2024), GraphTar (BMC Bioinformatics 2023), a heterogeneous-GNN
  lncRNA-disease paper (Sci Rep 2022), gGATLDA (BMC Bioinformatics 2022), DHGT-DTI (J Pharm
  Anal 2025), a substructure-GNN DTI paper (Front. Pharmacol. 2025), and a gene-disease GNN
  paper (Entropy 2023). None were included in the n=22 count above.

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
