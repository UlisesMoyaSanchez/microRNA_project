# How to compile `main.tex`

Elsevier's `elsarticle` class with the `elsarticle-num` numeric citation style needs a
full LaTeX + BibTeX cycle — one `pdflatex` pass alone will leave citations as `[?]` and
the references section empty. Run all four steps, in order, from this directory:

```bash
cd manuscript/jbi
pdflatex  -interaction=nonstopmode main.tex   # 1. first pass, writes main.aux
bibtex    main                                # 2. resolves \cite{} against references.bib
pdflatex  -interaction=nonstopmode main.tex   # 3. pulls the bibliography in
pdflatex  -interaction=nonstopmode main.tex   # 4. fixes cross-references/line numbers
```

Output: `main.pdf` in this same directory. Verified 2026-08-06 on TeX Live 2021 — exit
code 0 on all four steps, 18 pages at the current outline stage, no undefined references
or citations.

## One-liner

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Expected warnings (safe to ignore)

- `Overfull \hbox` / `Underfull \hbox` warnings — cosmetic line-breaking, common at
  outline stage with long `\texttt{}` runs; will mostly resolve once bullets become
  prose.
- `` `!h' float specifier changed to `!ht' `` — cosmetic, elsarticle's float placement
  policy.

## Why `cmap` and `lmodern` are loaded (do not remove)

Without them, `pdftotext`/copy-paste on the compiled PDF silently drops "fi"/"fl"
ligatures (e.g. "Inflation" → "Ination", "specifically" → "specically") even though the
PDF renders visually fine on screen — a known issue with plain Computer Modern under
`pdflatex`. `\usepackage{cmap}` plus `\usepackage{lmodern}` (added right after
`\documentclass`, before any other package) fixes text extraction; verified by
`pdftotext -layout main.pdf - | grep` against a list of common fi/fl words after every
change to this file.

## If it doesn't compile

- **`No file main.bbl` / undefined citations after pass 4** → step 2 (`bibtex`) didn't
  run or errored. Run it manually and read its output; a missing `references.bib` entry
  or a malformed BibTeX field is the usual cause.
- **`elsarticle.cls not found`** → you're not running from this directory, or the
  vendored `elsarticle.cls` / `elsarticle-num.bst` files aren't here. Both ship in this
  same folder (vendored, unmodified, from Elsevier's official LaTeX author template,
  LPPL license) — don't move `main.tex` without them.
- **Build artifacts** (`.aux`, `.log`, `.bbl`, `.blg`, `.out`, `.pdf`) should be
  gitignored the same way the BMC draft's are — safe to delete anytime and regenerate
  with the steps above; nothing here is hand-edited output.

## Format items not yet verified against JBI's live guide-for-authors

`https://www.sciencedirect.com/journal/journal-of-biomedical-informatics/publish/guide-for-authors`
returned HTTP 403 to automated fetches (direct and via Wayback Machine) on 2026-08-06.
Structure/limits used in `main.tex` come from search-result fragments plus standard
Elsevier `elsarticle` convention. Before submission, manually confirm on the live page:

- Exact keyword count cap (currently using 8; standard Elsevier default is often 6).
- Whether Highlights is mandatory and its exact character limit (used 85 chars/bullet,
  3–5 bullets — the general Elsevier default).
- Whether a graphical abstract is required (none included here).
- Any figure/table formatting rules beyond plain LaTeX `tabular`/`tikz`.
