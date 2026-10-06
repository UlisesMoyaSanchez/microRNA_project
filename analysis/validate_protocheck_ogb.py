"""
validate_protocheck_ogb.py -- protocheck on ogbl-ddi (homogeneous mode), DGX only.

Same 2x2 protocol grid and ground-truth-by-construction as validate_protocheck.py, on OGB's
official split: edges {seen, held_out} x negatives {official, degree_matched}. Held-out =
the validation positives; the encoder graph is dataset[0].edge_index (training edges).
'seen' adds the validation positives to the encoder's edges.

Cross-check, stronger than the HMDD one because nothing is random: in the (held_out,
official) cell the scorers, positives and negatives are all fixed, so protocheck's per-heuristic
AUROCs must reproduce results/comparison/ogb_ddi_topology_baseline_valid.json
(['results']['ogb_official']) to numerical precision.

ogbl-ppa is not run: 42M message-passing edges and 6M positives (the existing baseline needed
~31 GB RSS) are out of reach for a per-pair scorer in this pass.

Usage: sbatch training/slurm_validate_protocheck.sh
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

from protocheck import audit  # noqa: E402
from protocheck.checks import (  # noqa: E402
    column_degree, degree_bins, degree_matched_negatives, model_free_floor, uniform_negatives,
)

CELLS = {"conventional": (True, "official"), "seen_degree_matched": (True, "degree_matched"),
         "heldout_official": (False, "official"), "corrected": (False, "degree_matched")}
log = logging.getLogger("validate_protocheck_ogb")


def pairs_of(t) -> np.ndarray:
    a = np.asarray(t.cpu().numpy() if hasattr(t, "cpu") else t, dtype=np.int64)
    a = a.T if (a.ndim == 2 and a.shape[1] == 2 and a.shape[0] != 2) else a
    return a[:, a[0] != a[1]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data/raw/ogb")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/comparison/protocheck_validation_ogb_ddi.json")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from ogb.linkproppred import PygLinkPropPredDataset
    ds = PygLinkPropPredDataset(name="ogbl-ddi", root=args.root)
    n = int(ds[0].num_nodes)
    se = ds.get_edge_split()
    train = pairs_of(ds[0].edge_index)
    val_pos, test_pos = pairs_of(se["valid"]["edge"]), pairs_of(se["test"]["edge"])
    official_neg = pairs_of(se["valid"]["edge_neg"])
    log.info(f"n={n:,}  train edge pairs={train.shape[1]:,}  valid pos={val_pos.shape[1]:,}  "
             f"official neg={official_neg.shape[1]:,}")

    known = np.concatenate([train, val_pos, test_pos], axis=1)   # exclusion set for sampling
    rng = np.random.default_rng(args.seed)
    bins = degree_bins(column_degree(train, n, undirected=True))   # training edges only
    sample = lambda ref: degree_matched_negatives(ref, known, n, n, bins, rng, undirected=True)[0]  # noqa: E731

    rows = []
    for cell, (seen, neg_kind) in CELLS.items():
        eval_neg = official_neg if neg_kind == "official" else sample(val_pos)
        train_neg = (uniform_negatives(train, known, n, n, rng, undirected=True)
                     if neg_kind == "official" else sample(train))
        enc = np.concatenate([train, val_pos], axis=1) if seen else train
        rep = audit(n, n, train, val_pos, encoder_edges=enc, train_neg=train_neg,
                    eval_neg=eval_neg, mode="homogeneous", seed=args.seed)
        f = {x.check: x for x in rep.findings}
        base = f["model_free_baseline"].evidence
        rows.append({
            "cell": cell, "edges_seen": seen, "negatives": neg_kind,
            "status": {k: v.status for k, v in f.items()},
            "heldout_in_encoder_fraction": f["edge_leakage"].evidence["fraction_heldout_in_encoder"],
            "tv_eval_neg_vs_heldout_pos": f["negative_matching"].evidence["tv_eval_neg_vs_heldout_pos"],
            "dead_column_fraction": f["dead_candidates"].evidence["dead_column_fraction"],
            "floor_reference": base["reference_floor"],
            "heuristics_corrected": base["floor_corrected"]["heuristics"],
            "best_heuristic": base["floor_corrected"]["best"],
        })
        log.info(f"{cell:<22}leak={rows[-1]['status']['edge_leakage']:<6}"
                 f"neg={rows[-1]['status']['negative_matching']:<6}"
                 f"floor={rows[-1]['floor_reference']:.4f} ({rows[-1]['best_heuristic']})")

    repo = json.load(open("results/comparison/ogb_ddi_topology_baseline_valid.json"))
    ref = repo["results"]["ogb_official"]
    mine = next(r for r in rows if r["cell"] == "heldout_official")["heuristics_corrected"]
    parity = {h: {"protocheck": mine[h], "repo": ref[h]["auroc"], "abs_diff": abs(mine[h] - ref[h]["auroc"])}
              for h in ref}

    fired = lambda r, c: r["status"][c] in ("warn", "error")  # noqa: E731
    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "dataset": "ogbl-ddi", "seed": args.seed,
        "confusion": {
            "edge_leakage": {"detected": sum(fired(r, "edge_leakage") for r in rows if r["edges_seen"]),
                             "missed": sum(not fired(r, "edge_leakage") for r in rows if r["edges_seen"]),
                             "false_alarms": sum(fired(r, "edge_leakage") for r in rows if not r["edges_seen"]),
                             "correctly_silent": sum(not fired(r, "edge_leakage") for r in rows if not r["edges_seen"])},
            "negative_matching": {
                "detected": sum(fired(r, "negative_matching") for r in rows if r["negatives"] == "official"),
                "missed": sum(not fired(r, "negative_matching") for r in rows if r["negatives"] == "official"),
                "false_alarms": sum(fired(r, "negative_matching") for r in rows if r["negatives"] != "official"),
                "correctly_silent": sum(not fired(r, "negative_matching") for r in rows if r["negatives"] != "official")},
        },
        "parity_with_repo_official_cell": parity,
        "parity_max_abs_diff": max(v["abs_diff"] for v in parity.values()),
        "limits": ["one seed, four cells: a smoke-level confirmation, not a rate",
                   "OGB's official negatives are not labelled 'uniform' by OGB; whether the "
                   "negative_matching check fires on them is a measurement, not an assumption",
                   "ogbl-ppa not run", "dead_candidates is not independently validated"],
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    log.info(f"confusion: {json.dumps(out['confusion'])}")
    log.info(f"parity vs repo, max |AUROC diff| = {out['parity_max_abs_diff']:.2e}")


if __name__ == "__main__":
    main()
