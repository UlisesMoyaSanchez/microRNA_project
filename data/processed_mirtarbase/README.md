# `data/processed_mirtarbase/` — the miRTarBase arm's processed inputs

`scrna_processed.h5ad` here is a **symlink** to `../processed/scrna_processed.h5ad`:

```
scrna_processed.h5ad -> ../processed/scrna_processed.h5ad
```

## Why a symlink and not a copy

`config_mirtarbase_edgesplit.yaml` sets `processed_dir` to this directory, and
`build_heterograph.py` reads `{processed_dir}/scrna_processed.h5ad`. Without the
symlink the build simply fails — this directory did not exist at all until 2026-07-16.

Copying would work and would be wrong. The miRTarBase arm differs from the miRDB arm
in **exactly one field**, `data.mirna.interactions_file`; the `cellxgene` and `graph`
config blocks are byte-identical between the two (verified). The scRNA side is
therefore the same computation on the same inputs, and re-deriving it would only buy
4.5 GB and a chance to diverge.

That chance is the actual point. The gene vocabulary is the 3,000 HVGs from this file
(`preprocess_scrna.py:85-86`), and it is the *only* filter applied to an interaction
source (`build_heterograph.py:56`). If the two arms ever held different h5ads, the
comparison would silently stop being "miRDB vs miRTarBase" and become "one gene
vocabulary vs another" — while every log, filename and config still said interaction
source. The symlink makes divergence impossible rather than unlikely.

## Recreate

```bash
mkdir -p data/processed_mirtarbase
ln -sfn ../processed/scrna_processed.h5ad data/processed_mirtarbase/scrna_processed.h5ad
```

`mirna_expr.tsv` / `mirna_meta.tsv` are **not** symlinked: nothing reads them
(`results/comparison/ms_specificity_audit.json`, job 5716 — the GEO arm is orphaned).
If Track B wires them in, they must be regenerated per arm rather than linked, because
`preprocess_mirna.py` filters GEO probes against the arm's own interaction file.

Contents are gitignored; this README is not.
