---
name: pre-submission-check
description: Cross-references goal.md's "Gaps to Close Before Submission" checklist against the current state of the repo and reports which items are actually resolved vs. still open, with evidence. Use before drafting a manuscript, or when asked "are we ready to write this up."
---

# Pre-Submission Check

`goal.md` (repo root) is the living roadmap toward a journal/conference
submission. It has a "Gaps to Close Before Submission" checklist. This skill
checks the repo for evidence on each item — don't take the checklist's
checkbox state at face value, verify against real files.

## Steps

1. Read `goal.md`'s "Gaps to Close Before Submission" section for the current
   list of items (it may have changed since this skill was written — always
   re-read it, don't rely on a cached list).

2. For each item, check for concrete evidence in the repo, for example:
   - **The seen-edges row has no error bar** (§3.2, the main blocker) — are
     `0.9836` / `0.8828` still hardcoded constants in
     `training/eval_heldout_grid.py`? Does `train.py` still build the edge
     split unconditionally with no `edge_split: false` flag? If both are still
     true, the attribution is n=1 and the item is open.
   - **Baselines through both protocols** (§3.2) — does `run_baselines.py`
     emit a held-out × uniform cell (via `LinkSampler(hard=False)`), and do
     `results/comparison/` artifacts show `random`, `mlp`, `homo_gcn`,
     `ablation_no_coexpr` and `hgt_v2` scored under *both* protocols? If the
     inflation is only demonstrated for the HGT, the claim is still anecdotal.
   - **Literature survey size** (§3.2) — `results/LITERATURE_SURVEY.md` /
     `literature_survey.tsv`: is it still the n=7 pilot, or expanded toward
     20–30 papers with a second rater on the "unclear" calls?
   - **A second interaction database** (§3.2) — is there any artifact built
     from miRTarBase (or TargetScan) rather than miRDB alone?
   - **miRDB/miRTarBase naming/citation** (§3.3) — is `data/raw/mirtarbase_hsa.tsv`
     still named that way despite holding miRDB v6.0 data? Does `README.md`
     still carry the disclosure note? (grep for "miRDB" and "pipeline
     compatibility")
   - **Author list / affiliations** (§3.3) — check for any manuscript draft
     file (not yet created as of the last review) containing this information.

   **Do NOT report these as open work — they are deliberately suspended**
   (`goal.md` §3.4), and re-raising them each run is exactly the drift this
   skill exists to prevent: external validation of the top circuits,
   MS-vs-control differential saliency, pushing AUROC past 0.99, and
   re-deriving circuits from an honestly-trained model. Multi-seed stability
   is **done** (n=4, §3.1) — do not report it as pending either.

3. Also flag drift: if any hardcoded-path or naming issues noted in `goal.md`
   have been fixed (or made worse) since it was written, say so explicitly —
   this list is meant to stay accurate, not just get re-confirmed. In
   particular, watch for **stale validation numbers**: the canonical figures
   are on the **test** split (0.6271 link AUROC, 0.9916 cell acc); a document
   still quoting the val numbers (0.6467 / 0.9950) has drifted.

4. Report a checklist with each item marked resolved / open / partial, the
   evidence for each, and — only if asked — a recommended priority order for
   what to close next.

Do not mark an item resolved just because it's old or because fixing it seems
likely; only mark it resolved if you found a specific file/artifact that
demonstrates it.
