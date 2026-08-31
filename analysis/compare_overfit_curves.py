"""
compare_overfit_curves.py — Does the train/val loss divergence documented on our own graph
(job 5605/5603, train.py) also show up on the five surveyed HMDD graphs (Table 7), or is it
specific to our graph?

Table 7 shows the trained-vs-model-free margin CLOSES on our own graph under the corrected
protocol but OPENS on all five external graphs. One candidate explanation is that our graph's
real node features let the model memorize training edges, and the corrected protocol strips
exactly that memorized signal. If the same loss/AUROC divergence signature (train_loss falling
while val_loss climbs, after val_auroc has already peaked) shows up on the external graphs too,
that is evidence AGAINST this explanation being specific to our graph.

Reads the per-epoch `history` arrays added to train.py and train_survey_bipartite.py's output
JSON (this run's diagnostic addition — the four-seed numbers already in Table 7 are untouched).

Usage:
  python analysis/compare_overfit_curves.py --dir /path/to/downloaded/jsons
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path


GRAPHS = {
    "own_graph": "history_own_graph.json",
    "canonical5430": "survey_trained_grid_canonical5430_held_out_s42.json",
    "cksnp_gnn": "survey_trained_grid_cksnp_gnn_held_out_s42.json",
    "digamn": "survey_trained_grid_digamn_held_out_s42.json",
    "meahne": "survey_trained_grid_meahne_held_out_s42.json",
    "couplemda": "survey_trained_grid_couplemda_held_out_s42.json",
}


def load_history(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    # own_graph's history.json is a bare list; the survey trainer nests it under "history".
    return data if isinstance(data, list) else data["history"]


def analyze(name: str, history: list[dict]) -> dict:
    peak_idx = max(range(len(history)), key=lambda i: history[i]["val_auroc"])
    peak = history[peak_idx]
    end = history[-1]
    # Divergence, qualitative and threshold-free (magnitude ratios are not comparable
    # across graphs -- each has a different patience/eval_every, so the post-peak window
    # length differs): after the AUROC peak, does val_loss end higher than it was at the
    # peak, while train_loss ends lower? That is exactly the signature EVALUATION_AUDIT.md
    # documented for our own graph ("training loss falls... validation loss climbs").
    val_loss_rises_after_peak = len(history) > peak_idx + 1 and end["val_loss"] > peak["val_loss"]
    train_loss_keeps_falling = len(history) > peak_idx + 1 and end["train_loss"] < peak["train_loss"]
    return {
        "graph": name,
        "n_points": len(history),
        "peak_epoch": peak["epoch"],
        "peak_val_auroc": round(peak["val_auroc"], 4),
        "val_loss_at_peak": round(peak["val_loss"], 4),
        "val_loss_at_end": round(history[-1]["val_loss"], 4),
        "train_loss_at_peak": round(peak["train_loss"], 4),
        "train_loss_at_end": round(history[-1]["train_loss"], 4),
        "diverges": bool(val_loss_rises_after_peak and train_loss_keeps_falling),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="Directory with the downloaded history JSONs")
    args = p.parse_args()
    d = Path(args.dir)

    rows = []
    for name, fname in GRAPHS.items():
        path = d / fname
        if not path.exists():
            print(f"  (skipping {name}: {fname} not found in {d})")
            continue
        rows.append(analyze(name, load_history(path)))

    header = (f"{'graph':<15} {'peak_ep':>7} {'peak_auroc':>10} "
              f"{'val_loss@peak':>13} {'val_loss@end':>12} "
              f"{'train_loss@peak':>15} {'train_loss@end':>14}  diverges")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['graph']:<15} {r['peak_epoch']:>7} {r['peak_val_auroc']:>10} "
              f"{r['val_loss_at_peak']:>13} {r['val_loss_at_end']:>12} "
              f"{r['train_loss_at_peak']:>15} {r['train_loss_at_end']:>14}  "
              f"{'YES' if r['diverges'] else 'no'}")

    n_diverge = sum(r["diverges"] for r in rows)
    print(f"\n{n_diverge}/{len(rows)} graphs show the loss/AUROC divergence signature.")


if __name__ == "__main__":
    main()
