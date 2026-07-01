"""
download_mirtarbase.py — Download and filter miRDB v6.0 miRNA-target interactions.

Source: miRDB v6.0 (https://mirdb.org)
  - Human miRNAs only (hsa- prefix)
  - High-confidence predictions (score >= mirdb_score_threshold in config.yaml)
  - Gene symbols resolved from RefSeq accessions via mygene.info REST API
  - Removes duplicates (miRNA, target_gene) pairs

Output: data/raw/mirtarbase_hsa.tsv  (columns: mirna, target_gene, evidence)
        (filename kept for compatibility with downstream pipeline)
"""

import os
import gzip
import argparse
import yaml
import pandas as pd
import requests


def download_mirdb(url: str, dest: str) -> None:
    print(f"Downloading miRDB from:\n  {url}")
    resp = requests.get(
        url,
        stream=True,
        timeout=180,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    resp.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=65536):
            fh.write(chunk)
    print(f"  Saved: {dest} ({os.path.getsize(dest) // 1024} KB)")


def parse_mirdb(gz_path: str, score_threshold: float) -> pd.DataFrame:
    """Parse miRDB prediction file.

    File format (no header, tab-separated):
        miRNA_name  RefSeq_accession  score
    """
    print(f"\nParsing: {gz_path}  (score >= {score_threshold})")
    rows = []
    with gzip.open(gz_path, "rt") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            mirna, accession, score_str = parts[0], parts[1], parts[2]
            if not mirna.startswith("hsa-"):
                continue
            try:
                if float(score_str) >= score_threshold:
                    rows.append((mirna, accession))
            except ValueError:
                continue
    df = pd.DataFrame(rows, columns=["mirna", "refseq_accession"])
    print(f"  Human interactions after score filter: {len(df):,}")
    print(f"  Unique miRNAs:            {df['mirna'].nunique():,}")
    print(f"  Unique RefSeq accessions: {df['refseq_accession'].nunique():,}")
    return df


def refseq_to_symbols(accessions: list, batch_size: int = 1000) -> dict:
    """Convert RefSeq mRNA accessions to HGNC gene symbols via mygene.info."""
    # Strip version numbers: NM_001234.5 -> NM_001234
    unique_clean = list({a.split(".")[0] for a in accessions})
    print(f"\nConverting {len(unique_clean):,} RefSeq accessions to gene symbols...")

    accession_to_symbol: dict = {}
    total_batches = (len(unique_clean) + batch_size - 1) // batch_size

    for i in range(0, len(unique_clean), batch_size):
        batch = unique_clean[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} accessions)...", end=" ", flush=True)
        try:
            resp = requests.post(
                "https://mygene.info/v3/query",
                json={
                    "q": batch,
                    "scopes": "refseq.rna",
                    "fields": "symbol",
                    "species": "human",
                },
                timeout=60,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            mapped = 0
            for hit in resp.json():
                if hit.get("notfound"):
                    continue
                symbol = hit.get("symbol")
                if symbol:
                    accession_to_symbol[hit.get("query", "")] = symbol
                    mapped += 1
            print(f"mapped {mapped}")
        except Exception as exc:
            print(f"WARNING: {exc}")

    print(f"  Total mapped: {len(accession_to_symbol):,} / {len(unique_clean):,}")
    return accession_to_symbol


def main() -> None:
    p = argparse.ArgumentParser(description="Download and preprocess miRDB v6.0")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    out_dir         = args.out_dir or cfg["data"]["raw_dir"]
    url             = cfg["data"]["mirna"]["mirdb_url"]
    score_threshold = float(cfg["data"]["mirna"]["mirdb_score_threshold"])

    os.makedirs(out_dir, exist_ok=True)

    gz_path  = os.path.join(out_dir, "mirdb_v6.gz")
    tsv_path = os.path.join(out_dir, "mirtarbase_hsa.tsv")  # name kept for pipeline compat

    if os.path.exists(tsv_path):
        with open(tsv_path) as _f:
            line_count = sum(1 for _ in _f)
        if line_count > 1:
            print(f"TSV already exists: {tsv_path} ({line_count - 1:,} interactions)")
            print("\nmiRDB download complete.")
            return
        print(f"TSV exists but is empty ({line_count} line); regenerating...")
        os.remove(tsv_path)

    if not os.path.exists(gz_path):
        download_mirdb(url, gz_path)
    else:
        print(f"miRDB gz already exists: {gz_path}")

    df = parse_mirdb(gz_path, score_threshold)

    accession_map = refseq_to_symbols(df["refseq_accession"].tolist())

    df["refseq_clean"] = df["refseq_accession"].str.split(".").str[0]
    df["target_gene"]  = df["refseq_clean"].map(accession_map)
    df = df.dropna(subset=["target_gene"])
    df["evidence"] = "miRDB_v6_prediction"

    df_out = (
        df[["mirna", "target_gene", "evidence"]]
        .drop_duplicates(subset=["mirna", "target_gene"])
        .reset_index(drop=True)
    )

    print(f"\nFinal interactions: {len(df_out):,}")
    print(f"  Unique miRNAs:       {df_out['mirna'].nunique():,}")
    print(f"  Unique target genes: {df_out['target_gene'].nunique():,}")

    df_out.to_csv(tsv_path, sep="\t", index=False)
    print(f"  Saved TSV: {tsv_path}")

    print("\nmiRDB download complete.")


if __name__ == "__main__":
    main()
