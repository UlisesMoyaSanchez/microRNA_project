# BMC Bioinformatics draft

**Status (2026-07-29): outline + full abstract.** Body sections are bullet outlines
marked `[STAGE]`, not submission prose. Source of every number: `results/EVALUATION_AUDIT.md`
(canonical) and `results/PROTOCOLO_Y_CIFRAS_VIGENTES.md` (quick-reference). Do not hand-edit
a figure/number in `manuscript.tex` — if a cited number changes, pull it from the artifact
under `results/comparison/` again, the same rule the audit doc and the deck already enforce.

## Files

- `manuscript.tex` — the manuscript. **Single file, no `\input`** — BMC's own submission
  instructions require the whole thing as one `.tex` document (see the comment block at
  the top of the file).
- `references.bib` — the six external-precedent citations already vetted in
  `EVALUATION_AUDIT.md` §"External precedent for these pitfalls". Two entries are flagged
  `verify before submission` (full author lists not independently confirmed past the first
  author) — resolve before the file leaves draft status.
- `bmcart.cls`, `bmcart-biblio.sty`, `bmc-mathphys.bst` — **vendored, unmodified**, from
  BioMed Central's official LaTeX author template (`http://www.biomedcentral.com/authors/tex`,
  credited to Vytas Statulevicius / VTeX in the file header). License: LPPL — free to use
  as-is; if you ever need to modify the class itself, BMC's own header requires renaming it
  first. Fetched 2026-07-29 from a public GitHub mirror
  ([liubenyuan/latex-tools](https://github.com/liubenyuan/latex-tools/tree/master/bmc_article))
  because the official page sits behind a Springer login wall that this session's tools
  could not authenticate through — re-verify against BMC's current template zip
  (linked from the journal's submission-guidelines page) before final submission, in case
  the class has since been revised.

## Compiling

See [`COMPILE.md`](COMPILE.md) — four-step `pdflatex`/`bibtex` cycle, expected warnings,
and what to check if a step fails. Verified clean 2026-07-29, TeX Live 2021, 6 pages at
outline stage. `*.aux`/`*.log`/`*.bbl`/`*.blg` and the compiled PDF are gitignored; only
the source files are tracked.

## Before this leaves draft status

- Real author list + affiliation (currently one placeholder corresponding author).
- Funding / Acknowledgements / Ethics-statement `[STAGE]` blocks in the Declarations
  section.
- Expand every `[STAGE]`/outline bullet to prose.
- Resolve the two `verify before submission` citations in `references.bib`.
- Re-confirm the vendored template against BMC's current official zip (see above).
- Decide `\documentclass[twocolumn]{bmcart}` vs. the current single-column *Review* style
  before a submission-ready pass — single-column is easier to review, BMC's production
  system reflows to two columns regardless.
