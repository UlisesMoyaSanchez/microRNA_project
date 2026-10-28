"""
hmdd_hidden_positives.py -- How many of the canonical HMDD matrix's negatives did a later
HMDD release record as associations, and how much of the degree-matched drop do they explain?

Measured on the one graph where it can be: the canonical 495 x 383 matrix (MGCNSS, NIMGSA,
HLGNN-MDA) is built from HMDD v2.0 (2014); HMDD v3.2 (2019) recorded many more associations. A negative that v3.2 lists as
an association is a hidden positive of the kind the label-noise objection invokes.

Steps:
  1. Names. Indices -> names from HLGNN-MDA's index files. Validated by requiring every
     matrix positive to reappear in HMDD v2.0, with precursor suffixes collapsed the way the
     matrix does (hsa-mir-125a-1 -> hsa-mir-125a; hsa-mir-21 untouched).
  2. Coverage. All 495 miRNAs match v3.2 by name; about half the diseases do (v3 renamed
     many). Only negatives on diseases present in v3.2 are judged, and v3.2 must recover the
     matrix's own positives on those diseases (recall check), or the comparison is void.
  3. Negatives. The canonical row of the protocol grid is regenerated exactly (same seed,
     same RNG call order as eval_hmdd_survey_topology_baseline.py's Tier-2 path), and the
     regenerated AUROCs must equal the committed artifact. Further seeds give the spread.
  4. Effect. On the judgeable pairs, each heuristic's held-out AUROC with uniform and with
     degree-matched negatives, before and after dropping the negatives v3.2 records as
     associations. The drop that survives the cleaning is the part hidden positives (as far
     as v3.2 knows them) cannot explain.

v3.2 is itself incomplete, so the measured rate is a lower bound on hidden positives. No
useful upper bound replaces it: the hidden positives v3.2 does record score above the
held-out positives on average (they do not tie with them), so an analytic bound that
assumes a tie is not conservative.

Prerequisites:
  python data/01_download/download_hmdd_survey_canonical5430.py
  python data/01_download/download_hmdd_versions.py

Usage:
  python analysis/hmdd_hidden_positives.py   # writes results/comparison/hmdd_hidden_positives.json
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.eval_hmdd_survey_topology_baseline import build_scoring_matrix, load_matrix
from training.eval_topology_baseline import build_scorers, uniform_negatives
from training.splits import degree_bins, gene_in_degree, sample_degree_matched_negatives

VERSIONS = Path("data/raw/hmdd_versions")
CONFIG = "configs/config_hmdd_survey_nimgsa.yaml"  # the canonical row of the protocol grid
ARTIFACT = Path("results/comparison/hmdd_survey_topology_baseline_nimgsa.json")
EXTRA_SEEDS = list(range(20))
MIN_RECALL = 0.98

_PRECURSOR = re.compile(r"^(hsa-(?:mir|let)-[0-9]+[a-z]*)-[0-9]+$")


def norm(s: str) -> str:
    return re.sub(r"[\s,]+", " ", str(s).strip().lower())


def mirna_key(s: str) -> str:
    return _PRECURSOR.sub(r"\1", norm(s))


def index_names(path: Path) -> list[str]:
    df = pd.read_excel(path, header=None)
    return [norm(x) for x in df.sort_values(0)[1]]


def hmdd_v2_pairs() -> set[tuple[str, str]]:
    with zipfile.ZipFile(VERSIONS / "hmdd2.zip") as z:
        df = pd.read_excel(io.BytesIO(z.read("alldata.xlsx")), header=None)
    df = df[pd.to_numeric(df[0], errors="coerce").notna()]  # drop any header row
    return {(mirna_key(m), norm(d)) for m, d in zip(df[1], df[2])}


def hmdd_v3_pairs() -> set[tuple[str, str]]:
    df = pd.read_csv(VERSIONS / "v3_alldata.txt", sep="\t", dtype=str, encoding="utf-8",
                     encoding_errors="replace")
    return {(mirna_key(m), norm(d)) for m, d in zip(df["mir"], df["disease"])}


def regenerate(A_full: torch.Tensor, seed: int, test_frac: float, ratio: int):
    """eval_hmdd_survey_topology_baseline.py, Mode C, held_out -- same calls, same order."""
    n_mirna, n_disease = A_full.shape
    gen = torch.Generator().manual_seed(seed)
    all_pos = A_full.nonzero().T
    n_pos = all_pos.shape[1]
    perm = torch.randperm(n_pos, generator=gen)
    n_test = max(1, int(round(n_pos * test_frac)))
    test_pos, train_pos = all_pos[:, perm[:n_test]], all_pos[:, perm[n_test:]]
    scorers = build_scorers(build_scoring_matrix(train_pos, all_pos, n_mirna, n_disease,
                                                 "held_out"))
    tiled = test_pos.repeat(1, ratio)
    unif = uniform_negatives(tiled, all_pos, n_mirna, n_disease, gen)
    bins = degree_bins(gene_in_degree(train_pos, n_disease))
    dm, _ = sample_degree_matched_negatives(tiled, all_pos, bins, n_disease, gen,
                                            torch.device("cpu"))
    return scorers, tiled, unif, dm


def aurocs(scorers, pos, neg) -> dict[str, float]:
    y = np.r_[np.ones(pos.shape[1]), np.zeros(neg.shape[1])]
    m, d = torch.cat([pos[0], neg[0]]).long(), torch.cat([pos[1], neg[1]]).long()
    return {k: float(roc_auc_score(y, M[m, d].numpy())) for k, M in scorers.items()}


def one_seed(A_full, seed, test_frac, ratio, mi, di, judge_d, v3) -> dict:
    scorers, pos, unif, dm = regenerate(A_full, seed, test_frac, ratio)
    keep = lambda pairs: pairs[:, torch.tensor([di[j] in judge_d for j in pairs[1].tolist()],
                                              dtype=torch.bool)]
    pos_j = keep(pos)
    out = {"seed": seed, "n_pos_judged": int(pos_j.shape[1])}
    for name, neg in (("uniform", unif), ("degree_matched", dm)):
        neg_j = keep(neg)
        hidden = torch.tensor([(mi[a], di[b]) in v3 for a, b in neg_j.T.tolist()],
                              dtype=torch.bool)
        all_ = aurocs(scorers, pos_j, neg_j)
        clean = aurocs(scorers, pos_j, neg_j[:, ~hidden])
        out[name] = {"n_neg_judged": int(neg_j.shape[1]),
                     "judged_fraction": float(neg_j.shape[1] / neg.shape[1]),
                     "hidden_positive_rate": float(hidden.float().mean()),
                     "best_auroc_as_scored": max(all_.values()),
                     "best_auroc_hidden_removed": max(clean.values()),
                     "per_heuristic_as_scored": all_, "per_heuristic_hidden_removed": clean}
    u, d = out["uniform"], out["degree_matched"]
    out["drop_as_scored"] = u["best_auroc_as_scored"] - d["best_auroc_as_scored"]
    out["drop_hidden_removed"] = u["best_auroc_hidden_removed"] - d["best_auroc_hidden_removed"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--out", default="results/comparison/hmdd_hidden_positives.json")
    args = p.parse_args()

    cfg = yaml.safe_load(open(CONFIG))
    dcfg, seed = cfg["data"], cfg["project"]["seed"]
    A_full = load_matrix(dcfg["matrix_path"])
    mi = index_names(VERSIONS / "canonical5430_mirna_index.xlsx")
    di = index_names(VERSIONS / "canonical5430_disease_index.xlsx")
    if (len(mi), len(di)) != tuple(A_full.shape):
        raise SystemExit(f"name lists {len(mi)}x{len(di)} != matrix {tuple(A_full.shape)}")
    r, c = A_full.nonzero().T.tolist()
    pos_names = {(mi[a], di[b]) for a, b in zip(r, c)}

    # 1. index -> name mapping
    v2 = hmdd_v2_pairs()
    missing_v2 = len(pos_names - v2)
    print(f"matrix positives (as names) {len(pos_names):,}; missing from HMDD v2.0: {missing_v2}")
    if missing_v2:
        raise SystemExit("index -> name mapping does not reproduce HMDD v2.0; stopping")

    # 2. coverage and recall in v3.2
    v3 = hmdd_v3_pairs()
    v3_m, v3_d = {a for a, _ in v3}, {b for _, b in v3}
    judge_d = {d for d in set(di) if d in v3_d}
    judged_pos = [q for q in pos_names if q[1] in judge_d and q[0] in v3_m]
    recall = sum(q in v3 for q in judged_pos) / len(judged_pos)
    print(f"v3.2: miRNAs {sum(m in v3_m for m in mi)}/{len(mi)}, diseases "
          f"{len(judge_d)}/{len(set(di))}; recall of matrix positives {recall:.3f}")
    if recall < MIN_RECALL:
        raise SystemExit(f"v3.2 recovers only {recall:.3f} of the matrix's own positives on the "
                         "judged diseases; a missing negative would not mean much")

    # 3. the exact canonical cell, checked against the artifact
    test_frac, ratio = float(dcfg["test_fraction"]), int(dcfg["negative_ratio"])
    scorers, pos, unif, dm = regenerate(A_full, seed, test_frac, ratio)
    art = json.load(open(ARTIFACT))["results"]
    for name, neg in (("uniform", unif), ("degree_matched", dm)):
        mine = max(aurocs(scorers, pos, neg).values())
        theirs = max(v["auroc"] for v in art[name].values())
        if abs(mine - theirs) > 1e-9:
            raise SystemExit(f"{name}: regenerated {mine} != artifact {theirs}")
    print(f"regenerated the protocol-grid cell exactly (seed {seed})")

    # 4. measure
    seeds = [seed] + [s for s in EXTRA_SEEDS if s != seed]
    runs = [one_seed(A_full, s, test_frac, ratio, mi, di, judge_d, v3) for s in seeds]
    for rr in runs:
        print(f"seed {rr['seed']:>2}  hidden rate uniform {rr['uniform']['hidden_positive_rate']:.3f}"
              f"  matched {rr['degree_matched']['hidden_positive_rate']:.3f}   drop "
              f"{rr['drop_as_scored']:.4f} -> {rr['drop_hidden_removed']:.4f} after removal")

    def summary(key):
        v = np.array([key(rr) for rr in runs])
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                "min": float(v.min()), "max": float(v.max())}

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "matrix": dcfg["matrix_path"], "later_release": "HMDD v3.2 (v3_alldata.txt)",
        "mapping_check": {"positives": len(pos_names), "missing_from_hmdd_v2": missing_v2},
        "coverage": {"mirnas_in_v3": sum(m in v3_m for m in mi), "n_mirnas": len(mi),
                     "diseases_in_v3": len(judge_d), "n_diseases": len(set(di)),
                     "recall_of_matrix_positives": recall},
        "protocol_grid_cell_reproduced": True,
        "seeds": seeds,
        "summary": {
            "hidden_rate_uniform": summary(lambda rr: rr["uniform"]["hidden_positive_rate"]),
            "hidden_rate_degree_matched":
                summary(lambda rr: rr["degree_matched"]["hidden_positive_rate"]),
            "drop_as_scored": summary(lambda rr: rr["drop_as_scored"]),
            "drop_hidden_removed": summary(lambda rr: rr["drop_hidden_removed"]),
        },
        "runs": runs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
