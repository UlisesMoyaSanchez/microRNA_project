"""protocheck.audit -- run the four checks on one dataset and collect a Report."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from .checks import (
    SCOPE_NOTE, Finding, as_pairs, check_baseline_margin, check_dead_columns,
    check_leakage, check_negative_matching,
)

_MARK = {"ok": "[ ok ]", "info": "[info]", "warn": "[WARN]", "error": "[FAIL]",
         "unchecked": "[ -- ]"}


@dataclass
class Report:
    findings: list[Finding]
    dataset: dict

    @property
    def has_errors(self) -> bool:
        return any(f.status == "error" for f in self.findings)

    @property
    def has_warnings(self) -> bool:
        return any(f.status in ("warn", "error") for f in self.findings)

    def status(self, check: str) -> str:
        return next(f.status for f in self.findings if f.check == check)

    def to_dict(self) -> dict:
        return {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": self.dataset, "scope_note": SCOPE_NOTE,
                "findings": [asdict(f) for f in self.findings]}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def render(self) -> str:
        d = self.dataset
        lines = [f"protocheck  {d['n_rows']:,} x {d['n_cols']:,} candidates, mode={d['mode']}, "
                 f"train_pos={d['n_train_pos']:,}, heldout_pos={d['n_heldout_pos']:,}", ""]
        for f in self.findings:
            lines.append(f"{_MARK[f.status]} {f.check}: {f.message}")
        n = {s: sum(f.status == s for f in self.findings) for s in _MARK}
        lines += ["", f"{n['error']} failing, {n['warn']} warning, {n['unchecked']} unchecked.",
                  SCOPE_NOTE]
        return "\n".join(lines)


def audit(n_rows: int, n_cols: int, train_pos, heldout_pos, *, encoder_edges=None,
          encoder_edges_rev=None, train_neg=None, eval_neg=None, model_auroc=None,
          declared_samplers=None, mode: str = "bipartite", thresholds=None,
          seed: int = 0) -> Report:
    """Audit one evaluation set-up.

    train_pos / heldout_pos : (2, N) (row, col) pairs. Held-out are the positives the
                              reported metric was computed on.
    encoder_edges           : edges the model's encoder / message passing sees. Without it
                              the leakage check is reported as 'unchecked'.
    encoder_edges_rev       : reverse-relation edges as (col, row) pairs, if stored separately.
    train_neg / eval_neg    : negatives used for training and for the reported metric.
    model_auroc             : the AUROC you report, to compute the margin over the floor.
    declared_samplers       : {'train': 'uniform', 'eval': 'degree_matched'} -- the config
                              convention; compared as strings, not inspected.
    mode                    : 'bipartite' (rows and columns are different node types) or
                              'homogeneous' (one undirected node set; n_rows == n_cols).
    """
    train_pos, heldout_pos = as_pairs(train_pos), as_pairs(heldout_pos)
    undirected = mode == "homogeneous"
    findings = [
        check_leakage(n_cols, heldout_pos, encoder_edges=encoder_edges,
                      encoder_edges_rev=encoder_edges_rev, train_pos=train_pos,
                      eval_neg=eval_neg, undirected=undirected),
        check_negative_matching(n_rows, n_cols, train_pos, heldout_pos, train_neg=train_neg,
                                eval_neg=eval_neg, declared_samplers=declared_samplers,
                                thresholds=thresholds, undirected=undirected),
        check_baseline_margin(n_rows, n_cols, train_pos, heldout_pos, eval_neg,
                              encoder_edges=encoder_edges, model_auroc=model_auroc,
                              mode=mode, thresholds=thresholds, seed=seed),
        check_dead_columns(n_rows, n_cols, train_pos, heldout_pos, eval_neg=eval_neg,
                           thresholds=thresholds, undirected=undirected),
    ]
    return Report(findings, {"n_rows": int(n_rows), "n_cols": int(n_cols), "mode": mode,
                             "n_train_pos": int(train_pos.shape[1]),
                             "n_heldout_pos": int(heldout_pos.shape[1])})
