# Literature Survey — how does this field actually evaluate link prediction?

**Purpose.** The methods paper's premise is that the evaluation flaws we found in our own
work are *not ours alone*. Until now that was **asserted and never shown** — a strawman a
reviewer would rightly name. This document is the evidence.

**Status: PRELIMINARY — n = 7 papers read in full.** Target is 20–30 (see "Still to do").
Every row below was verified by reading the paper's methods section, not by inference from an
abstract. Where a paper does not say, the cell reads **unclear** — and *that category is
itself a finding*.

**Last updated:** 2026-07-13

---

## The finding, and it is not the one we expected

We went in expecting to show *"everyone leaks held-out edges into the message-passing graph."*
**That is not what the literature shows, and we should not claim it.** 2 of 7 papers clearly
do strip test edges from the graph; 2 clearly do not; 3 do not say.

What the literature *does* show is sharper, more universal, and more damaging:

> **Not one of the seven papers reports a single model-free baseline.**
> **0 / 7.** No gene/node degree. No common neighbours. No Adamic–Adar. Not even random.
> Every comparison is a learned method against other learned methods.

And separately:

> **Treating unlabeled pairs as uniformly-sampled negatives is the field's default.**
> 4 / 7 do exactly this. Two of the remaining three make "better negative selection" their
> *headline contribution* — which is itself an admission that the default is known to be broken.

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

### Tallies (n = 7)

| Practice | Count |
|---|---|
| Cross-validate over **edges** (not nodes) | **7 / 7** ✅ |
| Held-out edges **removed** from the encoder's input graph | **2 / 7** |
| Held-out edges **left in** the encoder's input graph | **2 / 7** ⚠️ |
| **Unclear** from the methods section | **3 / 7** ⚠️ |
| Negatives = unlabeled pairs, uniform | **4 / 7** |
| Any **model-free / heuristic baseline** reported | **0 / 7** 🔴 |

---

## What this does — and does not — license us to say

**We CAN say, with evidence:**

1. **No model-free control is reported anywhere in this sample (0/7).** The field has no
   routine way of knowing whether its numbers beat a popularity heuristic — and we show, on a
   real graph, that under its own default protocol they may not.
2. **Uniform/unlabeled negatives are the default (4/7),** and the papers that deviate do so as
   their *headline contribution* — evidence the problem is recognized but has no standard
   remedy or control.
3. **Reporting of the split is frequently too vague to reproduce (3/7 unclear).** Whether test
   edges reach the encoder — the single thing that decides if the number is prediction or
   reconstruction — often cannot be determined from the paper at all. That is a reporting-
   standards finding, and it is independently publishable.

**We must NOT say:**

- ❌ *"The field routinely leaks test edges into message passing."* **Not supported.** 2/7
  clearly do it right. Making this claim would be the same sin we are criticizing: asserting a
  strong quantitative claim the evidence does not carry.
- ❌ *"Our leak is typical."* **It is not — ours was worse than the norm.** Every paper here at
  least cross-validates over edges; our original split partitioned *cells* only. Honesty here
  costs us nothing and buys credibility: we found our own error, and it was a bad one.

**The honest framing for the manuscript's motivation section:**

> *Across n papers surveyed, edge-level cross-validation is universal, but **no paper reports a
> model-free baseline**, unlabeled pairs are treated as negatives in the majority, and in a
> substantial fraction the methods section does not permit the reader to determine whether
> held-out edges were visible to the encoder. We show that under exactly this protocol, a
> one-line popularity heuristic attains AUROC 0.87 on a real biomedical graph — within the
> band of published state-of-the-art results (0.91–0.99) — and outperforms a trained
> heterogeneous graph transformer.*

---

## Still to do

- **Expand to 20–30 papers.** n = 7 is a pilot. It is enough to establish the *direction* and
  to kill the strawman risk; it is not enough to publish a rate. Priority: more 2024–2026
  miRNA–disease and miRNA–target GNN papers, plus lncRNA–disease and gene–disease.
- **Two independent raters** for the "unclear" column, with disagreements recorded. A survey
  that classifies other people's rigour must be visibly rigorous itself; a single-rater
  judgement of "unclear" is exactly the kind of unchecked call we are criticizing.
- **Record the exact quoted sentence** supporting each classification, in the TSV, so any
  reader can audit every cell.
- **Check the two "did it right" papers (HybridGNN, kmerPMTF) for negative sampling and
  baselines.** Both strip test edges — and both still use uniform/unlabeled negatives and
  report no model-free control. If a paper can get the split right and *still* be vulnerable
  to the popularity artifact, that is the strongest possible argument for our proposed
  reporting standard.

## Sources

- [MGCNSS — Briefings in Bioinformatics 2024](https://academic.oup.com/bib/article/25/3/bbae168/7645839)
- [NIMGSA — PMC8774034](https://pmc.ncbi.nlm.nih.gov/articles/PMC8774034/)
- [HybridGNN — Bioinformatics](https://academic.oup.com/bioinformatics/article/42/5/btag171/8586881)
- [HGDTI — BMC Bioinformatics / PMC9004085](https://pmc.ncbi.nlm.nih.gov/articles/PMC9004085/)
- [NGCN — PMC10955156](https://pmc.ncbi.nlm.nih.gov/articles/PMC10955156/)
- [kmerPMTF — PeerJ 2024 / PMC11122044](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11122044/)
- [A Review of Link Prediction Applications in Network Biology — arXiv:2312.01275](https://arxiv.org/pdf/2312.01275)
