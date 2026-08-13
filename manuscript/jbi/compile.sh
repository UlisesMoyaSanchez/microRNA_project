#!/bin/bash
# Compile manuscript/jbi/main.tex to PDF. See COMPILE.md for the full explanation
# of why each step is needed (elsarticle-num requires a full pdflatex/bibtex cycle).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

for tool in pdflatex bibtex; do
    command -v "$tool" >/dev/null 2>&1 || { echo "error: $tool not found on PATH" >&2; exit 1; }
done

pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex

echo "Built manuscript/jbi/main.pdf"
