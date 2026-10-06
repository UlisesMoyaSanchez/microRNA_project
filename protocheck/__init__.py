"""protocheck -- warns about the evaluation errors described in the paper.

    from protocheck import audit
    report = audit(n_rows, n_cols, train_pos, heldout_pos, encoder_edges=..., eval_neg=...)
    print(report.render())

Depends on numpy, scipy and scikit-learn only.
"""

from .audit import Report, audit
from .checks import DEFAULT_THRESHOLDS, SCOPE_NOTE, Finding

__all__ = ["audit", "Report", "Finding", "DEFAULT_THRESHOLDS", "SCOPE_NOTE"]
