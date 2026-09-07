#!/usr/bin/env python3
"""Count the words in the JBI abstract, under one stated convention.

Two sessions counted this abstract and got 300 and 307 against a 300-word limit, from the
same unchanged text, because each stripped the LaTeX differently: `$\\pm$` left two bare `$`
tokens behind in one of them, and the structured labels were counted in one and added back by
hand in the other. This script fixes the convention so the number stops drifting.

The convention, chosen to match what a copy editor counts on the rendered page:

  * the structured labels (Objective:, Methods:, Results:, Conclusion:) ARE words -- they are
    printed and JBI's own template shows them inside the abstract;
  * a LaTeX command contributes only its printed argument (\\textbf{Methods:} -> Methods:);
  * math contributes its printed content, so `$\\pm$` is one word, not zero and not two;
  * `--` is an en dash inside a range, not a word boundary: 0.91--0.99 is one word, as it
    would be if typed 0.91-0.99;
  * a hyphenated compound is one word (model-free-baseline), the usual editorial rule.

Usage:  python analysis/abstract_wordcount.py [main.tex] [--limit 300]
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEX = ROOT / "manuscript" / "jbi" / "main.tex"


def extract(tex: str) -> str:
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if not m:
        sys.exit("no \\begin{abstract} ... \\end{abstract} block found")
    return m.group(1)


def to_plain_text(body: str) -> str:
    """Render the abstract source to the text a reader sees."""
    s = re.sub(r"(?<!\\)%.*", "", body)              # comments
    s = re.sub(r"\\(?:text(?:bf|it|rm|sf|tt)|emph)\{([^{}]*)\}", r"\1", s)  # keep the argument
    s = re.sub(r"\\pm\b", "+/-", s)                  # printed symbols become one token
    s = re.sub(r"\\times\b", "x", s)
    s = s.replace("$", " ")                          # math delimiters print nothing
    s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", s)     # other commands
    s = s.replace("\\ ", " ").replace("~", " ")
    s = re.sub(r"--+", "-", s)                       # en/em dash inside a range
    s = re.sub(r"[{}]", " ", s)
    return s


def count(text: str) -> list[str]:
    return [w for w in text.split() if re.search(r"[A-Za-z0-9]", w)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tex", nargs="?", default=str(DEFAULT_TEX))
    ap.add_argument("--limit", type=int, default=300, help="journal's word limit")
    ap.add_argument("--show", action="store_true", help="print the rendered text and every word")
    a = ap.parse_args()

    body = extract(Path(a.tex).read_text())
    plain = to_plain_text(body)
    words = count(plain)
    labels = [w for w in words if w.rstrip(":") in {"Objective", "Methods", "Results", "Conclusion"}]

    if a.show:
        print(" ".join(plain.split()), "\n")
        print(" | ".join(words), "\n")
    print(f"words (labels included) : {len(words)}")
    print(f"  of which labels       : {len(labels)}")
    print(f"words (labels excluded) : {len(words) - len(labels)}")
    over = len(words) - a.limit
    verdict = f"OVER by {over}" if over > 0 else f"within limit, {-over} to spare"
    print(f"limit {a.limit}: {verdict}")


if __name__ == "__main__":
    main()
