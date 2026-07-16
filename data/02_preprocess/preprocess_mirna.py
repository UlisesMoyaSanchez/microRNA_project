"""
preprocess_mirna.py — Parse bulk miRNA expression from GEO SOFT files.

For each downloaded GEO accession:
  1. Parse the _series_matrix.txt.gz file (faster than full SOFT)
  2. Extract expression matrix + sample metadata
  3. Normalize (log2, quantile)
  4. Merge across accessions (union of miRNAs, intersection handled by NaN fill)
  5. Filter miRNAs to those present in miRTarBase

Output:
  data/processed/mirna_expr.tsv  (rows=miRNAs, cols=samples)
  data/processed/mirna_meta.tsv  (sample metadata: condition, accession)
"""

import os
import gzip
import argparse
import yaml
import re
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    return p.parse_args()


# ── Affymetrix GPL SOFT probe-to-miRNA mapping ───────────────────────────────

def load_soft_probe_map(soft_gz: str) -> dict[str, str]:
    """
    Parse the ^PLATFORM table in a GEO SOFT.gz file and return a dict:
      probe_id (e.g. 'MIMAT0000062_st') → miRNA name (e.g. 'hsa-let-7a-5p')
    Only human (Homo sapiens) miRNA probes are returned.
    """
    mapping: dict[str, str] = {}
    if not os.path.exists(soft_gz):
        return mapping
    opener = gzip.open if soft_gz.endswith(".gz") else open
    in_table = False
    with opener(soft_gz, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("!platform_table_begin"):
                in_table = True
                next(fh)  # skip column-header line
                continue
            if line.startswith("!platform_table_end"):
                in_table = False
                continue
            if not in_table:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            probe_id   = parts[0].strip()
            mirna_name = parts[3].strip()
            organism   = parts[5].strip()
            if probe_id.startswith("MIMAT") and "Homo sapiens" in organism and mirna_name:
                mapping[probe_id] = mirna_name
    return mapping


# ── GEO Series Matrix Parser ──────────────────────────────────────────────────

def parse_series_matrix(gz_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse a GEO series_matrix.txt.gz file.
    Returns (expr_df, meta_df).
    expr_df: rows=probes/miRNAs, cols=GSM sample IDs
    meta_df: rows=sample IDs, various metadata columns
    """
    meta_rows: dict[str, list] = {}
    expr_rows: list[list] = []
    sample_ids: list[str] = []
    in_table = False

    opener = gzip.open if gz_path.endswith(".gz") else open

    with opener(gz_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("!Series_"):
                continue
            if line.startswith("!Sample_"):
                key = line.split("\t")[0].lstrip("!")
                vals = line.split("\t")[1:]
                meta_rows.setdefault(key, []).extend(vals)
            elif line.startswith('"ID_REF"') or line.startswith("ID_REF"):
                sample_ids = line.split("\t")[1:]
                in_table = True
            elif line.startswith("!series_matrix_table_end"):
                in_table = False
            elif in_table and line:
                expr_rows.append(line.split("\t"))

    if not sample_ids or not expr_rows:
        raise ValueError(f"Could not parse expression table from {gz_path}")

    # Expression dataframe
    probe_ids = [r[0].strip('"') for r in expr_rows]
    values    = [r[1:] for r in expr_rows]
    expr_df   = pd.DataFrame(values, index=probe_ids, columns=sample_ids)
    expr_df   = expr_df.apply(pd.to_numeric, errors="coerce")

    # Metadata dataframe
    # Each meta_rows entry is a flat list — take first occurrence per sample
    sample_ids_clean = [s.strip('"') for s in sample_ids]
    meta_data: dict[str, list] = {}
    for key, vals in meta_rows.items():
        # pad/trim to match number of samples
        trimmed = [v.strip('"') for v in vals[: len(sample_ids_clean)]]
        if len(trimmed) == len(sample_ids_clean):
            meta_data[key] = trimmed
    meta_df = pd.DataFrame(meta_data, index=sample_ids_clean)

    return expr_df, meta_df


def infer_condition(meta_df: pd.DataFrame) -> pd.Series:
    """Heuristic: label samples as 'MS' or 'Control' based on metadata text."""
    text = meta_df.astype(str).apply(lambda col: col.str.lower()).sum(axis=1)
    ms_keywords    = ["multiple sclerosis", "_ms_", "_ms ", " ms ", "relapsing", "rrms"]
    ctrl_keywords  = ["healthy", "_hc_", "_hc ", " hc ", "control"]
    def _label(t: str) -> str:
        if any(k in t for k in ms_keywords):
            return "MS"
        if any(k in t for k in ctrl_keywords):
            return "Control"
        return "Control"   # default
    return text.apply(_label)


def normalize_expr(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Log2(x+1) normalization; quantile normalize across samples."""
    expr_df = np.log2(expr_df + 1)
    # Quantile normalization (reference = row means across samples)
    rank_mean = expr_df.stack().groupby(
        expr_df.rank(method="first").stack().astype(int)
    ).mean()
    expr_df = expr_df.rank(method="first").stack().astype(int).map(rank_mean).unstack()
    return expr_df


# Affymetrix miRNA-4 array (GPL19117) appends _st or _x_st to probe names.
_AFFY_ST = re.compile(r"[_-]x?st$", re.IGNORECASE)


def harmonize_mirna_names(index: pd.Index) -> pd.Index:
    """Standardize miRNA names to hsa-miR-XXX format (lowercase).

    Handles:
    - Affymetrix _st / _x_st suffix (GPL19117 miRNA-4 array)
    - Trailing asterisk (*) from old arm notation
    - Mixed dash/underscore/space separators
    """
    result = []
    for n in index:
        n = str(n).lower().strip()
        n = _AFFY_ST.sub("", n)   # strip _st / _x_st before normalizing
        n = re.sub(r"[-_\s]+", "-", n)
        n = n.strip("*")           # strip leading AND trailing asterisk
        result.append(n)
    return pd.Index(result)


def filter_to_mirtarbase(expr_df: pd.DataFrame, mirtarbase_tsv: str) -> pd.DataFrame:
    if not os.path.exists(mirtarbase_tsv):
        print(f"  WARNING: miRTarBase TSV not found ({mirtarbase_tsv}), skipping filter")
        return expr_df
    mt = pd.read_csv(mirtarbase_tsv, sep="\t")
    # Strip arm suffixes (-5p / -3p) from both sides before matching so that
    # GEO probes (e.g. hsa-miR-21) match miRDB names (e.g. hsa-miR-21-5p)
    _arm = re.compile(r"-[35]p$", re.IGNORECASE)
    known_mirnas = set(mt["mirna"].str.lower().apply(lambda x: _arm.sub("", x)).unique())
    probe_stripped = expr_df.index.str.lower().str.replace(_arm, "", regex=True)
    mask = probe_stripped.isin(known_mirnas)
    before = len(expr_df)
    expr_df = expr_df.loc[mask]
    print(f"  miRTarBase filter: {before} → {len(expr_df)} miRNAs")
    return expr_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    raw_dir   = cfg["data"]["raw_dir"]
    proc_dir  = cfg["data"]["processed_dir"]
    os.makedirs(proc_dir, exist_ok=True)

    accessions: dict = cfg["data"]["geo"]["accessions"]
    geo_dir = os.path.join(raw_dir, "geo")
    # Configurable interaction-source filename (see build_heterograph.py); defaults
    # to the legacy miRDB file so existing configs are unaffected.
    interactions_file = cfg["data"].get("mirna", {}).get("interactions_file", "mirtarbase_hsa.tsv")
    mirtarbase_tsv = os.path.join(raw_dir, interactions_file)

    all_expr:  list[pd.DataFrame] = []
    all_meta:  list[pd.DataFrame] = []

    for accession in accessions:
        matrix_gz = os.path.join(geo_dir, accession, f"{accession}_series_matrix.txt.gz")
        if not os.path.exists(matrix_gz):
            print(f"  SKIP {accession}: file not found ({matrix_gz})")
            continue

        print(f"\nParsing {accession}...")
        try:
            expr_df, meta_df = parse_series_matrix(matrix_gz)
        except Exception as exc:
            print(f"  ERROR parsing {accession}: {exc}")
            continue

        print(f"  Shape: {expr_df.shape[0]} probes × {expr_df.shape[1]} samples")

        # Map Affymetrix MIMAT probe IDs → hsa-miR names using SOFT file
        soft_gz = os.path.join(geo_dir, accession, f"{accession}_family.soft.gz")
        probe_map = load_soft_probe_map(soft_gz)
        if probe_map:
            print(f"  SOFT probe map: {len(probe_map):,} hsa probes loaded")
            expr_df.index = expr_df.index.map(lambda p: probe_map.get(p, p))
            # Keep only probes that mapped to an hsa-miRNA name
            hsa_mask = expr_df.index.str.startswith("hsa-", na=False)
            before_map = len(expr_df)
            expr_df = expr_df.loc[hsa_mask]
            print(f"  After hsa filter: {before_map} → {len(expr_df)} probes")

        expr_df.index = harmonize_mirna_names(expr_df.index)
        expr_df = normalize_expr(expr_df)

        meta_df["condition"] = infer_condition(meta_df).values
        meta_df["accession"] = accession

        all_expr.append(expr_df)
        all_meta.append(meta_df)

    if not all_expr:
        print("No GEO data parsed. Exiting.")
        return

    # Merge: union of miRNAs, concatenate samples
    print("\nMerging across accessions (union of miRNAs)...")
    merged_expr = pd.concat(all_expr, axis=1, join="outer").fillna(0.0)
    # Remove duplicate miRNA rows (keep mean)
    merged_expr = merged_expr.groupby(level=0).mean()

    merged_meta = pd.concat(all_meta, axis=0, join="outer")

    print(f"Merged expression: {merged_expr.shape[0]} miRNAs × {merged_expr.shape[1]} samples")

    # Filter to miRTarBase-validated miRNAs
    merged_expr = filter_to_mirtarbase(merged_expr, mirtarbase_tsv)

    # Save
    expr_out = os.path.join(proc_dir, "mirna_expr.tsv")
    meta_out = os.path.join(proc_dir, "mirna_meta.tsv")
    merged_expr.to_csv(expr_out, sep="\t")
    merged_meta.to_csv(meta_out, sep="\t")

    print(f"\nSaved: {expr_out}")
    print(f"Saved: {meta_out}")
    print("miRNA preprocessing complete.")


if __name__ == "__main__":
    main()
