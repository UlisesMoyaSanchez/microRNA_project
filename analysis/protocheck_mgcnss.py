"""
protocheck_mgcnss.py -- protocheck on a published paper's own files: MGCNSS (Brief. Bioinform. 2024).

A worked example of the tool outside its seeded validation (analysis/validate_protocheck.py).
MGCNSS is the one surveyed paper whose repository ships everything the audit needs: its five
labeled CV folds (train7/test7_1, train_71..74/test71..74; fold 75 ships empty) and the
precomputed network its model reads (data/result.mat). src/Model.py::forward builds the
propagated adjacency from result.mat's raw_association, so that matrix is the encoder's
edge set.

Each fold is audited twice:
  split_only     the published split files alone -- what a reader of the paper can check.
                 The leakage check has no encoder edges and must come back 'unchecked'.
  with_encoder   plus raw_association as encoder_edges -- the model's actual input.

The reported AUROC (0.9874) is the paper's 5-fold mean, so the per-fold margin is
indicative only. What the audit cannot see, by its own scope: that the test negatives were
chosen by the paper's distance-based selector (the tool sees only their degree profile),
and that the public code keeps the epoch with the best test AUROC
(src/link_prediction_evaluate.py::predict_model) -- that is training code, not data.

Prerequisite: python data/01_download/download_hmdd_survey_canonical5430.py

Usage:
  python analysis/protocheck_mgcnss.py      # writes results/comparison/protocheck_mgcnss.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocheck import io as lio
from protocheck.audit import audit

DATA = Path("data/raw/hmdd_survey/mgcnss")
FOLDS = [("train7", "test7_1"), ("train_71", "test71"), ("train_72", "test72"),
         ("train_73", "test73"), ("train_74", "test74")]
N_MIRNA, N_DISEASE = 495, 383
HEADLINE_AUROC = 0.9874  # paper's Table 1, 5-fold mean ("AUC" there is roc_auc_score)
# sha256 of result.mat at the commit pinned in download_hmdd_survey_canonical5430.py.
RESULT_MAT_SHA256 = "1fb719cbff6c7cae7846004f46ecff8dd369babc4b025063a51f8bec9e5528af"


def encoder_edges(result_mat: Path) -> np.ndarray:
    digest = hashlib.sha256(result_mat.read_bytes()).hexdigest()
    if digest != RESULT_MAT_SHA256:
        raise ValueError(f"{result_mat} sha256 {digest} != pinned {RESULT_MAT_SHA256}; "
                         "re-run the download script or re-verify the pin")
    assoc = np.asarray(loadmat(result_mat)["raw_association"])
    if assoc.shape != (N_MIRNA, N_DISEASE):
        raise ValueError(f"raw_association has shape {assoc.shape}, expected "
                         f"({N_MIRNA}, {N_DISEASE})")
    rows, cols = np.nonzero(assoc)
    return np.vstack([rows, cols])  # 0-based (2, N), protocheck's convention


def summarise(report) -> dict:
    return {f.check: {"status": f.status, "message": f.message} for f in report.findings}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--out", default="results/comparison/protocheck_mgcnss.json")
    args = p.parse_args()

    enc = encoder_edges(DATA / "result.mat")
    print(f"encoder edges (raw_association): {enc.shape[1]:,}")

    folds = {}
    for train_name, test_name in FOLDS:
        tp, tn = lio.load_labeled_pairs(str(DATA / f"{train_name}.txt"), 1)
        hp, hn = lio.load_labeled_pairs(str(DATA / f"{test_name}.txt"), 1)
        common = dict(n_rows=N_MIRNA, n_cols=N_DISEASE, train_pos=tp, heldout_pos=hp,
                      train_neg=tn, eval_neg=hn, model_auroc=HEADLINE_AUROC)
        split_only = audit(**common)
        with_encoder = audit(**common, encoder_edges=enc)
        folds[test_name] = {"split_only": split_only.to_dict(),
                            "with_encoder": with_encoder.to_dict()}
        print(f"\n== {test_name}")
        for label, rep in (("split only", split_only), ("with encoder", with_encoder)):
            print(f"  {label:13s}", "  ".join(f"{c}={s['status']}"
                                               for c, s in summarise(rep).items()))

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "paper": "MGCNSS (tian2024mgcnss)",
        "headline_auroc": HEADLINE_AUROC,
        "encoder_edges_source": "result.mat raw_association (src/Model.py::forward)",
        "result_mat_sha256": RESULT_MAT_SHA256,
        "not_visible_to_protocheck": [
            "test negatives chosen by the paper's own distance-based selector",
            "epoch kept by best test AUROC in src/link_prediction_evaluate.py",
        ],
        "folds": folds,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
