# protocheck

Warns about the evaluation errors described in the paper *The Protocol, Not the Model*,
given the data of a link-prediction experiment. Depends on numpy, scipy and scikit-learn
only (no torch, no PyG).

| Check | What it warns about | Needs |
|---|---|---|
| `edge_leakage` | held-out edges visible to the encoder (either direction), shared with training, or used as evaluation negatives | `encoder_edges` |
| `negative_matching` | train and eval negatives with different column-degree profiles; eval negatives that do not mirror the held-out positives; declared samplers that differ | `eval_neg` (+ `train_neg`) |
| `model_free_baseline` | computes the no-learning floor (degree, preferential attachment, common neighbours, Adamic-Adar) and, given your AUROC, the margin over it | held-out positives (+ `model_auroc`) |
| `dead_candidates` | candidate columns with no positive anywhere, and evaluation negatives that land on them | held-out + train positives |

A check whose input you did not supply is reported as `unchecked`, never as `ok`.
Thresholds are heuristics (`protocheck.DEFAULT_THRESHOLDS`), not validated cut-offs; read the
numbers in each finding rather than the status word.

## Use

```python
from protocheck import audit
report = audit(n_rows, n_cols, train_pos, heldout_pos,      # (2, N) arrays of (row, col)
               encoder_edges=train_pos, train_neg=train_neg, eval_neg=eval_neg,
               model_auroc=0.91)
print(report.render()); report.save("audit.json")
```

```bash
python -m protocheck --train-pos train_pos.npy --heldout-pos heldout_pos.npy \
    --encoder-edges encoder_edges.npy --eval-neg eval_neg.npy --train-neg train_neg.npy \
    --n-rows 495 --n-cols 383 --model-auroc 0.91 --json audit.json
```

Other input layouts (`python -m protocheck -h`): an HMDD-style 0/1 matrix plus the held-out
pairs (`--matrix M.csv --heldout-pos F --encoder-edges matrix`), (row, col, label) triples
(`--labeled-train/--labeled-test`), OGB `split_edge` (`protocheck.io.from_ogb_split_edge`) and
this repo's `EdgeSplit` (`protocheck.io.from_pyg_edge_split`). Homogeneous undirected graphs:
`--mode homogeneous`. Exit status 1 if any check fails, 2 on a usage error.

## Examples

`python -m protocheck.examples.make_examples` writes five synthetic datasets, each triggering
one failure mode (or none: `clean`). They are fictional: `model_auroc` is a made-up number.
`notebooks/protocheck_colab.ipynb` runs them and has a cell for your own files.

## What it cannot tell you

It inspects the pairs you supply. It cannot see leakage that does not pass through exact
edges (e.g. node features derived from held-out labels), cannot distinguish a correct
sampler with a different seed or parameter from a matched one, and does not test whether
any published number is inflated.

## Verification

`pytest tests/test_protocheck.py` checks that each example triggers its own check and no
other, and that the scorers reproduce `training/eval_topology_baseline.py::build_scorers`
and the OGB formulas. `python analysis/validate_protocheck.py` runs the tool on the paper's
graphs under the 2x2 protocol grid; read its docstring for what that does and does not show.
