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
   - **miRDB/miRTarBase naming/citation** — is `data/raw/mirtarbase_hsa.tsv`
     still named that way? Does `README.md` / `results/REPORT.md` still
     carry the disclosure note? (grep for "miRDB" and "pipeline
     compatibility")
   - **External validation** — does `results/` contain any file referencing
     an external database comparison for the top circuits (miR-23a-3p→CCL7,
     miR-140-5p/oligodendrocytes), or is validation still only internal
     model scores?
   - **MS-vs-control differential analysis** — search
     `results/interpretation/` and `analysis/` for any condition-split
     saliency output; if absent, it's still open.
   - **Seed/split stability** — check whether more than one seed's results
     appear in `results/comparison/` (currently only single-run metrics per
     the last review).
   - **Training convergence** — check the latest training log/config for
     whether epochs were extended past 200 or a cosine LR schedule was added.
   - **Author list / affiliations** — check for any manuscript draft file
     (not yet created as of the last review) containing this information.

3. Also flag drift: if any hardcoded-path or naming issues noted in `goal.md`
   have been fixed (or made worse) since it was written, say so explicitly —
   this list is meant to stay accurate, not just get re-confirmed.

4. Report a checklist with each item marked resolved / open / partial, the
   evidence for each, and — only if asked — a recommended priority order for
   what to close next.

Do not mark an item resolved just because it's old or because fixing it seems
likely; only mark it resolved if you found a specific file/artifact that
demonstrates it.
