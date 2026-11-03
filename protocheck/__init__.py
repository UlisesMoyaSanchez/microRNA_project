"""protocheck -- checks a link-prediction evaluation for leakage, mismatched negatives,
missing model-free baselines and dead candidate columns.

    from protocheck import audit
    report = audit(n_rows, n_cols, train_pos, heldout_pos, encoder_edges=..., eval_neg=...)
    print(report.render())

Depends on numpy, scipy and scikit-learn only.
"""

__version__ = "0.1.0"

from .audit import Report, audit
from .checks import DEFAULT_THRESHOLDS, SCOPE_NOTE, Finding

__all__ = ["audit", "Report", "Finding", "DEFAULT_THRESHOLDS", "SCOPE_NOTE", "__version__"]
