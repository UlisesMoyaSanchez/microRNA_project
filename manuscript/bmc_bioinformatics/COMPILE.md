# How to compile `manuscript.tex`

BMC's `bmcart` class needs a full LaTeX + BibTeX cycle — one `pdflatex` pass alone will
leave citations as `[?]` and the references section empty. Run all four steps, in order,
from this directory:

```bash
cd manuscript/bmc_bioinformatics
pdflatex  -interaction=nonstopmode manuscript.tex   # 1. first pass, writes manuscript.aux
bibtex    manuscript                                # 2. resolves \cite{} against references.bib
pdflatex  -interaction=nonstopmode manuscript.tex   # 3. pulls the bibliography in
pdflatex  -interaction=nonstopmode manuscript.tex   # 4. fixes cross-references/page numbers
```

Output: `manuscript.pdf` in this same directory. Verified 2026-07-29 on TeX Live 2021 —
exit code 0 on all four steps, 6 pages at the current outline+abstract stage, no undefined
references or citations.

## One-liner

```bash
pdflatex manuscript.tex && bibtex manuscript && pdflatex manuscript.tex && pdflatex manuscript.tex
```

## Expected warnings (safe to ignore)

- `LaTeX Font Warning: Size substitutions...` and `Font shape ... not available` — cosmetic,
  from `bmcart`'s default font choices under plain Computer Modern. Does not affect content.
- `LaTeX Warning: Reference 'LastPage' ... undefined` **after pass 1 only** — expected before
  the bibliography exists; resolves itself by pass 3–4. If it's still there after all four
  steps, something upstream failed — check the `.log` file.

## If it doesn't compile

- **`No file manuscript.bbl` / undefined citations after pass 4** → step 2 (`bibtex`) didn't
  run or errored. Run it manually and read its output; a missing `references.bib` entry or a
  malformed BibTeX field is the usual cause.
- **`bmcart.cls not found`** → you're not running from this directory, or the vendored
  `bmcart.cls` / `bmcart-biblio.sty` / `bmc-mathphys.bst` files aren't here. All three ship
  in this same folder (see `README.md` for their provenance) — don't move `manuscript.tex`
  without them.
- **Build artifacts** (`.aux`, `.log`, `.bbl`, `.blg`, `.pdf`) are gitignored — safe to
  delete anytime and regenerate with the steps above; nothing here is hand-edited output.
