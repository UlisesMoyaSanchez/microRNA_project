"""
download_geo_mirna.py — Download curated miRNA GEO datasets for Multiple Sclerosis.

Curated accessions:
  GSE41995  — miRNA serum MS patients vs controls
  GSE107742 — miRNA PBMC MS patients vs controls
  GSE119453 — miRNA brain lesions MS

Each SOFT.gz file contains expression matrix + sample metadata.
Output: data/raw/geo/<ACCESSION>/<ACCESSION>_family.soft.gz
"""

import os
import ssl
import time
import argparse
import urllib.request
import yaml


GEO_FTP_SOFT = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}nnn/{acc}/soft/{acc}_family.soft.gz"
)
GEO_FTP_MATRIX = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}nnn/{acc}/matrix/{acc}_series_matrix.txt.gz"
)


def geo_stub(accession: str) -> str:
    """Return the GEO FTP subdirectory stub (eg. GSE41nnn)."""
    return accession[:-3]


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def download_file(url: str, dest: str, retries: int = 3) -> bool:
    for attempt in range(1, retries + 1):
        try:
            print(f"    [{attempt}/{retries}] GET {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=_ssl_context(), timeout=120) as resp:
                with open(dest, "wb") as fh:
                    fh.write(resp.read())
            print(f"    Saved: {dest} ({os.path.getsize(dest) // 1024} KB)")
            return True
        except Exception as exc:
            print(f"    WARNING attempt {attempt}: {exc}")
            if attempt < retries:
                time.sleep(10 * attempt)
    return False


def download_accession(accession: str, description: str, out_dir: str) -> None:
    acc_dir = os.path.join(out_dir, accession)
    os.makedirs(acc_dir, exist_ok=True)

    stub = geo_stub(accession)
    soft_dest = os.path.join(acc_dir, f"{accession}_family.soft.gz")
    matrix_dest = os.path.join(acc_dir, f"{accession}_series_matrix.txt.gz")

    print(f"\n  [{accession}] {description}")

    # Try SOFT file first (richer, includes GPL annotation)
    if not os.path.exists(soft_dest):
        url = GEO_FTP_SOFT.format(stub=stub, acc=accession)
        if not download_file(url, soft_dest):
            # Fallback: series matrix (lighter)
            url_m = GEO_FTP_MATRIX.format(stub=stub, acc=accession)
            print(f"    Falling back to series matrix: {url_m}")
            if not download_file(url_m, matrix_dest):
                print(f"    ERROR: Could not download {accession}.")
                print(f"    Manual download: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}")
    else:
        print(f"    Already exists: {soft_dest}")

    # Always download series matrix for easy pandas parsing
    if not os.path.exists(matrix_dest):
        url_m = GEO_FTP_MATRIX.format(stub=stub, acc=accession)
        download_file(url_m, matrix_dest)


def main() -> None:
    p = argparse.ArgumentParser(description="Download GEO miRNA datasets for MS")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    accessions: dict = cfg["data"]["geo"]["accessions"]
    out_dir = args.out_dir or os.path.join(cfg["data"]["raw_dir"], "geo")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Downloading {len(accessions)} GEO accessions → {out_dir}")
    for accession, description in accessions.items():
        download_accession(accession, description, out_dir)

    print("\nGEO download complete.")


if __name__ == "__main__":
    main()
