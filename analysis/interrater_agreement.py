#!/usr/bin/env python3
"""Inter-rater agreement between the survey's first rater and a blind second pass.

Rater 1 is `results/literature_survey_rater1_blind.tsv`: the first rater's calls as they
stood *before* the two raters were compared. Agreement has to be measured on that snapshot,
not on `results/literature_survey.tsv`, because adjudicating the disagreements edits the live
sheet -- the D4 re-audit moved six of its cells -- and scoring a rater against a sheet already
corrected by the comparison understates the disagreement the comparison found.
Rater 2 is a second pass made from the primary sources under `results/SURVEY_CODEBOOK.md`,
without sight of rater 1's calls.

Reports, per dimension: raw agreement, Cohen's kappa, the confusion matrix, and every
disagreeing paper by name, so each one can be adjudicated by hand rather than averaged away.

Usage:
    python analysis/interrater_agreement.py <rater2.tsv> [--json OUT]
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RATER1 = REPO / "results" / "literature_survey_rater1_blind.tsv"

# Rows that are not papers and so cannot be double-rated from a primary source.
NON_PAPER_ROWS = {"DTI field convention"}

DIMENSIONS = ["cv_over_edges", "testedges_removed_from_encoder_graph",
              "negative_sampling", "model_free_baseline"]

# Rater 1 recorded negatives as free text; the codebook's D3 is categorical. Mapping rater 1's
# text into those categories is itself a rating decision, so it is table-driven and printed in
# full, longest key first, rather than buried in a regex.
D3_MAP = [
    ("all unlabeled pairs treated as negative", "all_unlabeled"),
    ("all unlabeled pairs", "all_unlabeled"),
    ("unlabeled pairs as negatives", "all_unlabeled"),
    ("balanced dataset", "not_described"),
    ("undersampling", "not_described"),
    ("not described", "not_described"),
    ("distance-based selection", "non_uniform"),
    ("score-filtered", "non_uniform"),
    ("uniform random", "uniform_unlabeled"),
]


def map_d3(text):
    low = text.strip().lower()
    for needle, cat in D3_MAP:
        if needle in low:
            return cat
    return None


def load_rater1():
    calls, d3_audit = {}, []
    with RATER1.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            paper = row["paper"].strip()
            if paper in NON_PAPER_ROWS:
                continue
            d3 = map_d3(row["negative_sampling"])
            d3_audit.append((paper, row["negative_sampling"].strip(), d3))
            calls[paper] = {
                "cv_over_edges": row["cv_over_edges"].strip().lower(),
                "testedges_removed_from_encoder_graph":
                    row["testedges_removed_from_encoder_graph"].strip().lower(),
                "negative_sampling": d3,
                "model_free_baseline": row["model_free_baseline"].strip().lower(),
            }
    return calls, d3_audit


def load_rater2(path):
    calls = {}
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            paper = row["paper"].strip()
            calls[paper] = {
                "cv_over_edges": row["cv_over_edges"].strip().lower(),
                "testedges_removed_from_encoder_graph":
                    row["testedges_removed_from_encoder_graph"].strip().lower(),
                "negative_sampling": row["negative_sampling_category"].strip().lower(),
                "model_free_baseline": row["model_free_baseline"].strip().lower(),
                "_source": row.get("source_retrieved", "").strip().lower(),
            }
    return calls


def cohens_kappa(pairs):
    """Unweighted Cohen's kappa over (rater1, rater2) label pairs."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(a == b for a, b in pairs) / n
    c1, c2 = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(c1[k] * c2[k] for k in set(c1) | set(c2)) / (n * n)
    if expected == 1.0:
        # Both raters constant and identical: kappa is undefined, agreement is total.
        return float("nan")
    return (observed - expected) / (1 - expected)


def interpret(k):
    if k is None or k != k:
        return "undefined (no variation)"
    for thresh, label in ((0.81, "almost perfect"), (0.61, "substantial"),
                          (0.41, "moderate"), (0.21, "fair"), (0.0, "slight")):
        if k >= thresh:
            return label
    return "worse than chance"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rater2")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    r1, d3_audit = load_rater1()
    r2 = load_rater2(args.rater2)

    print("=" * 78)
    print("D3 mapping applied to rater 1's free text (audit this before trusting D3)")
    print("=" * 78)
    for paper, text, cat in d3_audit:
        flag = "  <-- UNMAPPED" if cat is None else ""
        print(f"  {paper:<26} {text[:44]:<46} -> {cat}{flag}")

    only1, only2 = sorted(set(r1) - set(r2)), sorted(set(r2) - set(r1))
    if only1:
        print(f"\nWARNING: in rater 1 only (name mismatch or unrated): {only1}")
    if only2:
        print(f"WARNING: in rater 2 only (name mismatch): {only2}")

    shared = sorted(set(r1) & set(r2))
    unreached = [p for p in shared if r2[p]["_source"] in {"paywalled", "failed"}]
    if unreached:
        print(f"\nExcluded, rater 2 could not reach the source: {unreached}")

    report = {"n_shared": len(shared), "n_excluded": len(unreached), "dimensions": {}}

    for dim in DIMENSIONS:
        pairs, disagreements = [], []
        for paper in shared:
            a, b = r1[paper][dim], r2[paper][dim]
            if b in {"not_rated", ""} or a is None or b is None:
                continue
            pairs.append((a, b))
            if a != b:
                disagreements.append((paper, a, b))

        k = cohens_kappa(pairs)
        agree = sum(a == b for a, b in pairs)
        n = len(pairs)
        print("\n" + "=" * 78)
        print(f"{dim}   (n={n})")
        print("=" * 78)
        if n:
            print(f"  raw agreement : {agree}/{n} = {agree / n:.1%}")
            kt = "n/a" if k is None or k != k else f"{k:.3f}"
            print(f"  Cohen's kappa : {kt}   ({interpret(k)})")
            # Kappa collapses toward 0 when one label dominates the marginals, even at high
            # agreement (the "kappa paradox"). PABAK removes that prevalence dependence and is
            # the honest companion figure for the skewed dimensions here.
            n_lab = len({a for a, _ in pairs} | {b for _, b in pairs})
            if n_lab > 1:
                pabak = (n_lab * (agree / n) - 1) / (n_lab - 1)
                print(f"  PABAK         : {pabak:.3f}   "
                      f"(prevalence-adjusted, {n_lab} observed categories)")

        matrix = defaultdict(Counter)
        for a, b in pairs:
            matrix[a][b] += 1
        labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
        if labels:
            header = "r1 \\ r2"
            print("\n  " + f"{header:<18}" + "".join(f"{l:>18}" for l in labels))
            for a in labels:
                print(f"  {a:<18}" + "".join(f"{matrix[a][b]:>18}" for b in labels))

        if disagreements:
            print(f"\n  Disagreements ({len(disagreements)}) — adjudicate each by hand:")
            for paper, a, b in disagreements:
                print(f"    {paper:<26} rater1={a:<18} rater2={b}")
        elif n:
            print("\n  No disagreements.")

        report["dimensions"][dim] = {
            "n": n, "agreement": agree / n if n else None,
            "kappa": None if k is None or k != k else round(k, 4),
            "disagreements": [{"paper": p, "rater1": a, "rater2": b}
                              for p, a, b in disagreements],
        }

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
