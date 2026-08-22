# HMDD topology-baseline audit — quantifying AUROC inflation on the survey's own data

**Purpose.** The literature survey (`results/LITERATURE_SURVEY.md`) shows the field never
reports a model-free control, and — by inference from our own graph — asserts that published
AUROCs are probably inflated by the same popularity effect (0.8712 on our own miRNA-gene graph,
beating a trained heterogeneous graph transformer's 0.8056). This document measures that
inflation directly, on seven of the survey's own miRNA-disease papers' own data, instead of
only inferring it.

**Status: complete, all 7 in-scope papers (2026-08-13).** Methodology, tiering, and the
3-papers-dropped rationale (GONNMDA repo is a stub, Orro's data isn't public, DGNMDA's only
source is unreachable) are unchanged from the approved plan — see
`/home/umoya/.claude/plans/reactive-spinning-yeti.md`. All runs executed on the DGX via SLURM
(jobs 5884-5891), not locally.

Methodology: `training/eval_hmdd_survey_topology_baseline.py` reuses
`training.eval_topology_baseline.build_scorers()` **unmodified** — the same four no-learning
heuristics (miRNA/disease degree, preferential attachment, common-neighbors, Adamic-Adar)
already used on this project's own miRNA-gene graph, applied here to each paper's own
miRNA-disease matrix. Every run's protocol was independently re-verified against the downloaded
bytes by `data/01_download/inspect_hmdd_survey_sources.py` before being trusted
(`all_checks_passed: true`).

## Results

| Paper | Tier | Their reported AUROC | Our best topology-only AUROC (uniform neg.) | Gap | Negatives |
|---|:--:|:--:|:--:|:--:|---|
| **MGCNSS** | 1 — exact split | 0.9874 | 0.9136 | −0.074 | their own (distance-based) |
| **DiGAMN** | 2 — exact matrix, our split | 0.9635 | 0.8714 | −0.092 | uniform 1:1 (our approx.) |
| **CoupleMDA** | 1 (pos.) / 2 (neg.) | 0.9536 | 0.9406 | **−0.013** | uniform 1:1 (our approx.) |
| **MEAHNE** | 2 — exact matrix, our split | 0.9520 | **0.9848** | **+0.033** ⚠️ | uniform 1:1 (our approx.) |
| **CKSNP-GNN** | 2 — exact matrix, our split | 0.9371 | 0.8822 | −0.055 | uniform 1:1 (our approx.) |
| **NIMGSA** | 2 — exact matrix, our split | 0.9354 | 0.8680 | −0.067 | uniform 1:1 (our approx.) |
| **HLGNN-MDA** | 2 — exact matrix, our split | 0.93086 | 0.8680 | −0.063 | uniform 1:1 (our approx.) |
| **Mean** | | 0.951 | 0.905 | **−0.047** | |

Under **degree-matched** negatives (candidate diseases restricted to the same popularity bin as
the true disease), every heuristic on every paper collapses to 0.53-0.58 — confirming the
uniform-negative numbers above are a genuine popularity-driven ceiling, not a bug or a leak, on
all seven graphs. Full per-heuristic numbers and per-paper caveats:
`results/comparison/hmdd_survey_topology_baseline_{mgcnss,nimgsa,hlgnn_mda,digamn,cksnp_gnn,
meahne,couplemda}.json`. Verification gate: `results/comparison/
hmdd_survey_protocol_verification.json`.

Bonus: extracting each paper's own headline number for this audit filled in four "not
extracted" cells left open in the original literature survey (HLGNN-MDA, CKSNP-GNN, MEAHNE,
CoupleMDA) — `results/literature_survey.tsv` and `LITERATURE_SURVEY.md` updated accordingly.

## Seven papers, five distinct graphs (found 2026-08-22)

**MGCNSS, NIMGSA and HLGNN-MDA evaluate on the same matrix.** Their three `matrix.csv` files
differ only in float formatting (`1,0,0` vs `1.000000000000000000e+00`), so their md5s differ,
but the parsed tensors are `torch.equal` — 495 x 383, 5,430 associations, the canonical
HMDD v3.2 benchmark. This was known at download time and is stated in
`data/01_download/download_hmdd_survey_canonical5430.py` ("One fetch of the canonical matrix
serves all three"); what was missing is that it never reached the manuscript, whose Table 3 and
prose read as seven independent graphs.

| Graph | Shape | Positives | Density | Papers |
|---|---|---:|---:|---|
| canonical-5430 | 495 x 383 | 5,430 | 0.02864 | **MGCNSS, NIMGSA, HLGNN-MDA** |
| CKSNP-GNN | 901 x 877 | 16,427 | 0.02079 | CKSNP-GNN |
| DiGAMN | 917 x 792 | 14,550 | 0.02003 | DiGAMN |
| CoupleMDA | 2,090 x 1,754 | 15,032 | 0.00410 | CoupleMDA |
| MEAHNE | 1,296 x 11,783 | 17,972 | 0.00118 | MEAHNE |

Consequences, all now propagated to `manuscript/jbi/main.tex` and
`manuscript/jbi/tables/table5_hmdd_survey_audit.tex`:

- NIMGSA and HLGNN-MDA produce **bit-identical** model-free numbers in every cell (same graph,
  same generated split, same seed). MGCNSS differs (0.9136 vs 0.8680) only because it is scored
  on the paper's own bundled Tier-1 split.
- The mean gap is now reported **both ways**: -0.047 per paper (7), -0.039 per distinct graph (5).
  The per-paper mean is weighted 3x toward one benchmark.
- "Ten independently-sourced graphs" was inaccurate: ten cases, **eight distinct graphs**.

That much of this subfield validates on one shared dataset is itself a finding about its
evaluation practice, and is reported as such rather than quietly corrected.

**Also corrected in the same pass:** the Abstract claimed the gap "reverses on the two sparsest
graphs, where the heuristic beats the paper's own trained model." It reverses on **one** —
MEAHNE (+0.033). On CoupleMDA the heuristic still loses, by 1.3 points. Results §4.5 already
stated this correctly; the Abstract and the Discussion's sixth limitation did not.

## What this does — and does not — license us to say

**We CAN say:** across seven externally-sourced miRNA-disease papers — five distinct graphs, see
the section above — a no-learning topology heuristic averages 4.7 AUROC points below the papers'
own trained-model numbers (3.9 averaging per graph) — a real, quantified gap, not our own graph's
finding by inference alone. **But the gap is highly
graph-dependent**, ranging from −9.2 points (DiGAMN) to **+3.3 points** (MEAHNE, where the
heuristic outright beats the trained model) and −1.3 points (CoupleMDA, essentially tied). Two
of seven papers — the two with the sparsest graphs relative to their node counts (MEAHNE:
17,972 edges over 1,296×11,783 nodes; CoupleMDA: ~15,000 edges over 2,090×1,754 nodes) — show
the same pattern as our own miRNA-gene graph: **the trained model provides little to no
measurable benefit over popularity alone.**

**We must NOT say:**
- "All surveyed models learned nothing beyond popularity" — five of seven still clearly beat
  the topology floor by 5.5-9.2 points. The claim is that the *floor is high and never
  reported*, not that every trained model is worthless.
- "MEAHNE's/CoupleMDA's models are bad" — we don't know their models are worse than others'; we
  know their **evaluation protocol** (uniform negatives on a sparse graph) cannot distinguish
  their trained model from a one-line popularity count, which is a claim about the *protocol*,
  not the *architecture*.
- That any Tier-2 number is a reproduction — six of seven papers' exact splits/negatives are not
  published; each config's `caveat` field states precisely what was and wasn't reconstructed
  exactly (see `configs/config_hmdd_survey_*.yaml`).
- That the graphs are directly comparable to each other — they differ in domain database,
  version, size, and density; only each row's own within-paper comparison (theirs vs. ours, same
  data) is apples-to-apples.

**The honest framing for the manuscript — REVISED 2026-08-22, the first version was wrong.**
The earlier framing claimed *sparser graphs make the popularity ceiling higher relative to what a
trained model can add*. It was tested and **refuted**; see the next section.

## Density hypothesis: tested and refuted; the real mechanism is dead candidate columns

`analysis/density_sweep.py`, 5 seeds per point. Artifacts: `results/comparison/
density_sweep_{canonical5430,cksnp_gnn,digamn,meahne,couplemda,padding}.json`.

**Between graphs, the correlation was never there.** Gap vs. density over the five distinct
graphs gives Spearman rho=+0.70, p=0.19 (per paper, n=7: +0.59, p=0.16). Right direction, no
significance, and DiGAMN breaks the ordering — dense, yet the largest gap of all.

**Within a graph, the effect runs BACKWARDS.** Subsampling positives from each graph
(100/75/50/25/10% retention) makes the model-free floor *fall*, monotonically, in all five:

| Graph | floor @100% | floor @10% | rho vs density |
|---|--:|--:|--:|
| canonical-5430 | 0.868 | 0.836 | +0.60 |
| CKSNP-GNN | 0.873 | 0.814 | +0.90 |
| DiGAMN | 0.882 | 0.830 | +1.00 |
| CoupleMDA | 0.944 | 0.899 | +1.00 |
| MEAHNE | 0.985 | 0.923 | +1.00 |

The degree-matched control falls too (0.56 → 0.51), so this is not a popularity effect either;
removing edges simply makes every degree estimate noisier. **Sparsity does not explain MEAHNE.**

**What does: the fraction of disease columns with degree zero.**

| Graph | Columns | Dead columns | Degree Gini | Floor |
|---|--:|--:|--:|--:|
| canonical-5430 | 383 | 0.0% | 0.712 | 0.8680 |
| CKSNP-GNN | 877 | 0.0% | 0.713 | 0.8822 |
| DiGAMN | 792 | 0.0% | 0.723 | 0.8714 |
| CoupleMDA | 1,754 | 55.7% | 0.874 | 0.9406 |
| MEAHNE | 11,783 | **92.4%** | 0.979 | **0.9848** |

Dead-column fraction vs. floor: rho=+0.89, p=0.041; density: rho=−0.90, p=0.037 (i.e. the
between-graph density correlation is real but is a *proxy* for this).

**Confirmed causally, not just correlationally.** Pad canonical-5430 — which has no dead columns
— with empty disease columns. Padding adds no edges, no features, nothing but candidate slots
that can never be positive:

| Empty columns added | Total columns | Dead | Model-free floor |
|--:|--:|--:|--:|
| 0 | 383 | 0.0% | 0.8591 ± 0.0141 |
| 500 | 883 | 56.6% | 0.9361 ± 0.0053 |
| 2,000 | 2,383 | 83.9% | 0.9701 ± 0.0047 |
| 5,000 | 5,383 | 92.9% | 0.9791 ± 0.0048 |
| 11,400 | 11,783 | 96.7% | 0.9817 ± 0.0056 |

At 92.9% dead columns — MEAHNE's own level — a graph with none of MEAHNE's data reaches 0.9791,
reproducing MEAHNE's 0.9848 almost exactly. **An AUROC reported under uniform negatives is partly
a measure of how a dataset padded its candidate space, and a padded dataset hands that inflation
to a trained model and a one-line heuristic alike.** This is a stronger, more mechanistic version
of this paper's own thesis than the density story it replaces, and it is a claim about dataset
construction plus protocol, not about anyone's architecture.

## Still to do

- Propagate the refutation into `manuscript/jbi/main.tex` §4.5, which still carries the old
  sparsity mechanism sentence.
- Dead-column fraction is not yet reported per graph in the manuscript's Table 3.
