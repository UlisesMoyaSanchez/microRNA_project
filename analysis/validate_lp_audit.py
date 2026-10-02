"""
validate_lp_audit.py -- Does lp_audit flag the failure modes it claims to, on the paper's graphs?

Runs the tool on every distinct graph in analysis/density_sweep.py::GRAPHS under the
paper's 2x2 protocol grid. The ground truth is known by construction, not judged:

  edges seen     encoder_edges = every positive, held-out included   -> leak present
  edges held out encoder_edges = training positives only             -> no leak
  uniform neg    train and eval negatives both uniform               -> eval negatives do not
                                                                        mirror the positives
  degree-matched train and eval negatives both degree-matched        -> they do

so each cell has an expected status for edge_leakage and for negative_matching, and the
output is a confusion table (detections / misses / false alarms) per check. Train and eval
negatives always use the same sampler, so the train-vs-eval mismatch branch is exercised by
tests/test_lp_audit.py instead, not here.

Two honest limits, repeated in the output JSON:
  * dead_candidates is NOT independently validated: it re-derives the same dead-column
    fraction that analysis/graph_candidate_stats.py reports. Listed for completeness.
  * No model is trained, so the margin check is exercised only as a floor computation.
    The floor is cross-checked against the repo's own torch pipeline (density_sweep's
    retention=1.0 point, uniform negatives). Two criteria are reported side by side:
    the one fixed before running ("within 2 sd of the repo's per-split sd", which FAILED
    on digamn and couplemda by 0.011 and 0.002 AUROC) and a Welch z on the two means,
    added afterwards because the first compares a 4-split mean to a single-split sd. The
    switch is disclosed, not hidden: both numbers are in the output.

Usage:
  python analysis/validate_lp_audit.py            # writes results/comparison/lp_audit_validation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.density_sweep import GRAPHS, load_positives  # noqa: E402
from lp_audit import audit  # noqa: E402
from lp_audit.checks import (  # noqa: E402
    column_degree, degree_bins, degree_matched_negatives, uniform_negatives,
)

CELLS = {  # name -> (edges_seen, sampler)
    "conventional": (True, "uniform"),
    "seen_degree_matched": (True, "degree_matched"),
    "heldout_uniform": (False, "uniform"),
    "corrected": (False, "degree_matched"),
}
log = logging.getLogger("validate_lp_audit")


def run_cell(pos, n_rows, n_cols, test_fraction, seed, seen, sampler):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(pos.shape[1])
    n_test = max(1, int(round(pos.shape[1] * test_fraction)))
    held, train = pos[:, perm[:n_test]], pos[:, perm[n_test:]]
    if sampler == "uniform":
        draw = lambda p: uniform_negatives(p, pos, n_rows, n_cols, rng)  # noqa: E731
    else:
        bins = degree_bins(column_degree(train, n_cols))          # training edges only
        draw = lambda p: degree_matched_negatives(p, pos, n_rows, n_cols, bins, rng)[0]  # noqa: E731
    report = audit(n_rows, n_cols, train, held, encoder_edges=pos if seen else train,
                   train_neg=draw(train), eval_neg=draw(held), seed=seed)
    return {f.check: f for f in report.findings}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--out", default="results/comparison/lp_audit_validation.json")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    seeds = [42 + i for i in range(args.seeds)]

    rows = []
    for gname, spec in GRAPHS.items():
        pos_t, n_rows, n_cols = load_positives(spec)
        pos = pos_t.numpy().astype(np.int64)
        for cell, (seen, sampler) in CELLS.items():
            for seed in seeds:
                f = run_cell(pos, n_rows, n_cols, spec["test_fraction"], seed, seen, sampler)
                base = f["model_free_baseline"].evidence
                rows.append({
                    "graph": gname, "cell": cell, "seed": seed,
                    "edges_seen": seen, "sampler": sampler,
                    "status": {k: v.status for k, v in f.items()},
                    "heldout_in_encoder_fraction": f["edge_leakage"].evidence["fraction_heldout_in_encoder"],
                    "tv_eval_neg_vs_heldout_pos": f["negative_matching"].evidence["tv_eval_neg_vs_heldout_pos"],
                    "dead_column_fraction": f["dead_candidates"].evidence["dead_column_fraction"],
                    "floor_corrected": base["floor_corrected"]["floor"],
                    "floor_reference": base["reference_floor"],
                    "best_heuristic": base["floor_corrected"]["best"],
                })
            sel = [r for r in rows if r["graph"] == gname and r["cell"] == cell]
            log.info(f"{gname:<14}{cell:<22}leak={sel[0]['status']['edge_leakage']:<6}"
                     f"neg={sel[0]['status']['negative_matching']:<6}"
                     f"floor={np.mean([r['floor_reference'] for r in sel]):.4f}")

    # Confusion tables. Positive class = failure mode present by construction.
    def confusion(check, truth):
        tp = fn = fp = tn = 0
        for r in rows:
            fired = r["status"][check] in ("warn", "error")
            if truth(r):
                tp, fn = tp + fired, fn + (not fired)
            else:
                fp, tn = fp + fired, tn + (not fired)
        return {"detected": tp, "missed": fn, "false_alarms": fp, "correctly_silent": tn}

    summary = {
        "edge_leakage": confusion("edge_leakage", lambda r: r["edges_seen"]),
        "negative_matching": confusion("negative_matching", lambda r: r["sampler"] == "uniform"),
    }
    # Silence of the corrected arm. dead_candidates is a property of the GRAPH, not of the
    # protocol, so it is expected to fire on a dead-column graph in every arm; it is listed
    # separately instead of being counted as a false alarm.
    protocol_checks = ("edge_leakage", "negative_matching", "model_free_baseline")
    corrected_alarms = [
        {"graph": r["graph"], "seed": r["seed"],
         "checks": [k for k in protocol_checks if r["status"][k] in ("warn", "error")]}
        for r in rows if r["cell"] == "corrected"
        if any(r["status"][k] in ("warn", "error") for k in protocol_checks)]

    # Floor cross-check against the repo's torch pipeline.
    xcheck = []
    for gname in GRAPHS:
        sweep = json.load(open(f"results/comparison/density_sweep_{gname}.json"))
        ref = next(pt for pt in sweep["points"] if pt["retention"] == 1.0)["uniform"]
        theirs = np.array([r["best_uniform"] for r in sweep["runs"] if r["retention"] == 1.0])
        mine = np.array([r["floor_reference"] for r in rows
                         if r["graph"] == gname and r["cell"] == "heldout_uniform"])
        m = float(mine.mean())
        se = float(np.sqrt(mine.var(ddof=1) / len(mine) + theirs.var(ddof=1) / len(theirs)))
        z = (m - float(theirs.mean())) / se
        xcheck.append({"graph": gname, "lp_audit_floor": m, "repo_floor_mean": ref["mean"],
                       "repo_floor_std": ref["std"], "abs_diff": abs(m - ref["mean"]),
                       "within_2sd_of_repo_split_sd": bool(abs(m - ref["mean"]) <= 2 * max(ref["std"], 1e-4)),
                       "welch_z": z, "welch_within_2": bool(abs(z) <= 2)})

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "seeds": seeds,
        "n_audits": len(rows), "confusion": summary,
        "protocol_alarms_in_corrected_arm": corrected_alarms,
        "dead_candidates_by_graph": {
            g: {"fraction": float(np.mean([r["dead_column_fraction"] for r in rows if r["graph"] == g])),
                "flagged": rows[[r["graph"] for r in rows].index(g)]["status"]["dead_candidates"]}
            for g in GRAPHS},
        "floor_crosscheck_vs_density_sweep": xcheck,
        "limits": ["dead_candidates re-derives graph_candidate_stats.py and is not independently "
                   "validated", "no model is trained: the margin check is exercised as a floor "
                   "computation only", "OGB graphs are not covered here (dataset not on this "
                   "machine)"],
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    log.info(f"confusion: {json.dumps(summary)}")
    log.info(f"protocol alarms in the corrected arm: {corrected_alarms or 'none'}")
    for x in xcheck:
        log.info(f"floor x-check {x['graph']:<14} diff={x['abs_diff']:.4f}  "
                 f"within 2 repo-sd: {x['within_2sd_of_repo_split_sd']}  "
                 f"Welch z={x['welch_z']:+.2f}")


if __name__ == "__main__":
    main()
