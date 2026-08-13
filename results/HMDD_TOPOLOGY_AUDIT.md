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

## What this does — and does not — license us to say

**We CAN say:** across seven independently-sourced miRNA-disease datasets, a no-learning
topology heuristic averages 4.7 AUROC points below the papers' own trained-model numbers — a
real, quantified gap, not our own graph's finding by inference alone. **But the gap is highly
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

**The honest framing for the manuscript:** *sparser graphs make the popularity ceiling higher
relative to what a trained model can add* — this is a mechanism-level explanation for why the
gap varies as much as it does, and it is itself a useful, falsifiable claim a future paper could
test directly (does the gap shrink monotonically with edge density?).

## Still to do

- Fold into `manuscript/jbi/main.tex` — not done in this pass, results-first per the OGB
  extension's own sequencing.
- The graph-density-explains-the-gap observation above is a hypothesis, not yet tested; doing so
  would need edge density computed per paper and a correlation against the gap column.
