"""
download_mirtarbase.py — Download miRNA-target interactions from one of two sources.

  --source mirdb       (default; unchanged legacy behaviour)
    miRDB v6.0 (https://mirdb.org): sequence-based *predictions*, human only,
    score >= mirdb_score_threshold, RefSeq→symbol via mygene.info.
    Output filename: data/raw/mirtarbase_hsa.tsv (name kept for pipeline compat;
    the file holds miRDB, NOT miRTarBase — see README provenance note).

  --source mirtarbase  (the real thing)
    miRTarBase release 10.0 (https://awi.cuhk.edu.cn/miRTarBase): *experimentally
    validated* MTIs, independent IN KIND from miRDB (not sequence-derived).
    Downloads the human hsa_MTI.csv (~340 MB; heavily row-redundant, deduped to
    unique pairs here), keeps validated support types (strong + weak by default), and
    writes to the filename in cfg["data"]["mirna"]["interactions_file"] — a
    distinct file so the miRDB graph is never overwritten.

Both sources emit the SAME schema the rest of the pipeline consumes
(build_heterograph.py, preprocess_mirna.py):
    columns: mirna, target_gene, evidence   (tab-separated, with header)
    mirna       : mature name e.g. hsa-miR-21-5p
    target_gene : HGNC symbol (e.g. PTEN), NOT Entrez/Ensembl
    evidence    : provenance tag
"""

import os
import re
import gzip
import argparse
import yaml
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

# Human MTI file. The site's download page is headed "Release 11.0" (citation:
# miRTarBase 2025, NAR) but every file it links to — including this one — is
# served from the /files/10.0/ path, and nothing on the site states that 11.0
# content was published. 10.0 is therefore the only release we can actually
# verify, so it is what we download and what we tag. A /files/11.0/ path 404s.
MIRTARBASE_RELEASE = "10.0"
MIRTARBASE_HSA_URL = (
    f"https://awi.cuhk.edu.cn/miRTarBase/downloads/files/{MIRTARBASE_RELEASE}/hsa_MTI.csv"
)


def release_from_url(url: str) -> str:
    """Recover the release label from the download path so the evidence tag can
    never disagree with the bytes we fetched (a config override of
    mirtarbase_url retags automatically)."""
    m = re.search(r"/files/([^/]+)/", url)
    return m.group(1) if m else "unknown"


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


class _PlainRSAKeyExchangeAdapter(HTTPAdapter):
    """Allow the one legacy cipher the miRTarBase server insists on.

    awi.cuhk.edu.cn only offers TLS1.2 with AES128-GCM-SHA256 under *plain RSA*
    key exchange. Python's default cipher list offers ECDHE/DHE key exchange
    only, so the handshake fails with SSLV3_ALERT_HANDSHAKE_FAILURE even though
    curl and `openssl s_client` negotiate fine against the same host.

    Re-enabling this single AEAD cipher is narrower than the usual
    DEFAULT@SECLEVEL=1 workaround, which would also re-admit SHA1 and short keys
    process-wide. The cipher's only weakness is the lack of forward secrecy.
    Certificate verification and hostname checking stay fully ON.
    """

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = create_urllib3_context(ciphers="DEFAULT:AES128-GCM-SHA256")
        return super().init_poolmanager(*args, **kwargs)


def download_mirtarbase_csv(url: str, dest: str) -> None:
    print(f"Downloading miRTarBase from:\n  {url}")
    session = requests.Session()
    session.mount("https://", _PlainRSAKeyExchangeAdapter())
    with session.get(
        url,
        stream=True,
        timeout=180,
        headers={"User-Agent": "Mozilla/5.0"},
    ) as resp:
        resp.raise_for_status()
        expected = int(resp.headers.get("Content-Length", 0))
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    size = os.path.getsize(dest)
    # A silently truncated 340 MB body would surface downstream as "miRTarBase
    # has fewer interactions than miRDB" — a plausible-looking wrong result.
    if expected and size != expected:
        raise IOError(
            f"Truncated download: got {size:,} bytes, server declared {expected:,}. "
            f"Partial file left at {dest} for inspection; delete it and re-run."
        )
    print(f"  Saved: {dest} ({size / 1e6:.1f} MB)")


def _find_col(cols, *needles) -> str:
    """Case-insensitive column lookup by substring — miRTarBase has renamed
    columns across releases (e.g. 'Entrez Gene ID' vs 'Entrez ID')."""
    low = {c.lower(): c for c in cols}
    for needle in needles:
        for lc, orig in low.items():
            if needle in lc:
                return orig
    raise KeyError(f"No column matching any of {needles} in {list(cols)}")


def parse_mirtarbase(csv_path: str, keep_weak: bool = True,
                     release: str = MIRTARBASE_RELEASE) -> pd.DataFrame:
    """Parse the human hsa_MTI file into (mirna, target_gene, evidence).

    Keeps validated interactions only:
      - 'Functional MTI'          (strong: reporter assay / western blot)
      - 'Functional MTI (Weak)'   (weak: qPCR, microarray, HTS/CLIP)  [if keep_weak]
    'Non-Functional MTI' rows (validated *negatives*) are always dropped from the
    positive edge set.
    """
    print(f"\nParsing miRTarBase: {csv_path}  (keep_weak={keep_weak})")
    df = pd.read_csv(csv_path)
    c_mirna   = _find_col(df.columns, "mirna")
    c_gene    = _find_col(df.columns, "target gene")  # matches 'Target Gene', not the Entrez col
    c_support = _find_col(df.columns, "support type", "support")
    c_species = None
    try:
        c_species = _find_col(df.columns, "species (mirna)", "species")
    except KeyError:
        pass

    n0 = len(df)
    if c_species is not None:
        # Release 10.0 stores the 3-letter code ('hsa'), older releases spell out
        # 'Homo sapiens'. Accept either — matching only the latter silently drops
        # every row and yields an empty edge set.
        species = df[c_species].astype(str).str.strip()
        is_human = species.str.fullmatch("hsa", case=False, na=False) \
            | species.str.contains("homo sapiens", case=False, na=False)
        df = df[is_human]
        print(f"  Human rows ({c_species} in {{hsa, Homo sapiens}}): {len(df):,} / {n0:,}")

    support = df[c_support].astype(str)
    is_functional = support.str.contains("Functional MTI", case=False, na=False) \
        & ~support.str.contains("Non-Functional", case=False, na=False)
    if not keep_weak:
        is_functional &= ~support.str.contains("Weak", case=False, na=False)
    df = df[is_functional]

    out = pd.DataFrame({
        "mirna":       df[c_mirna].astype(str).str.strip(),
        "target_gene": df[c_gene].astype(str).str.strip(),
        "evidence":    f"miRTarBase_{release}:" + support.loc[df.index].str.strip(),
    })
    # Drop obvious junk (blank gene symbols, NaN-as-string)
    out = out[(out["target_gene"] != "") & (~out["target_gene"].str.lower().isin({"nan", "none"}))]
    out = out[out["mirna"].str.startswith("hsa-")]

    out = out.drop_duplicates(subset=["mirna", "target_gene"]).reset_index(drop=True)

    # An empty result means a filter stopped matching (miRTarBase has renamed
    # column *values* across releases before — 'hsa' vs 'Homo sapiens'). Fail
    # here rather than writing a 0-row TSV that the pipeline would happily build
    # an empty graph from.
    if out.empty:
        raise SystemExit(
            f"Parsed 0 validated human pairs from {n0:,} rows — a filter matched nothing. "
            f"Check the species and support-type values actually present in {csv_path}."
        )

    print(f"  Rows read: {n0:,}  →  validated human pairs (deduped): {len(out):,}")
    print(f"  Unique miRNAs:       {out['mirna'].nunique():,}")
    print(f"  Unique target genes: {out['target_gene'].nunique():,}")
    print(f"  Support-type breakdown:\n{out['evidence'].value_counts().to_string()}")
    return out


def run_mirdb(cfg: dict, out_dir: str) -> None:
    url             = cfg["data"]["mirna"]["mirdb_url"]
    score_threshold = float(cfg["data"]["mirna"]["mirdb_score_threshold"])

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


def run_mirtarbase(cfg: dict, out_dir: str, keep_weak: bool) -> None:
    url = cfg["data"]["mirna"].get("mirtarbase_url", MIRTARBASE_HSA_URL)
    release = release_from_url(url)
    out_name = cfg["data"]["mirna"].get("interactions_file")
    if not out_name or out_name == "mirtarbase_hsa.tsv":
        raise SystemExit(
            "Refusing to write real miRTarBase over the miRDB file. Set "
            "data.mirna.interactions_file to a distinct name (e.g. "
            "mirtarbase_real_hsa10_all.tsv) in the config."
        )

    csv_path = os.path.join(out_dir, f"hsa_MTI_r{release}.csv")
    tsv_path = os.path.join(out_dir, out_name)

    if not os.path.exists(csv_path):
        # Download to a temp path and rename only on success: the source CSV is
        # ~340 MB over an international link, and a truncated file left at
        # csv_path would be silently reused as "already exists" on the next run.
        part_path = csv_path + ".part"
        download_mirtarbase_csv(url, part_path)
        os.replace(part_path, csv_path)
    else:
        print(f"miRTarBase CSV already exists: {csv_path}")

    df_out = parse_mirtarbase(csv_path, keep_weak=keep_weak, release=release)
    df_out.to_csv(tsv_path, sep="\t", index=False)
    print(f"\n  Saved TSV: {tsv_path}")
    print("\nmiRTarBase download complete.")


def main() -> None:
    p = argparse.ArgumentParser(description="Download miRNA-target interactions (miRDB or miRTarBase)")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--source", choices=["mirdb", "mirtarbase"], default="mirdb",
                   help="Interaction source. 'mirdb' (default) preserves legacy behaviour.")
    p.add_argument("--strong_only", action="store_true",
                   help="miRTarBase only: keep 'Functional MTI' strong evidence only "
                        "(drop the weak tier). Default keeps strong + weak.")
    args = p.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    out_dir = args.out_dir or cfg["data"]["raw_dir"]
    os.makedirs(out_dir, exist_ok=True)

    if args.source == "mirdb":
        run_mirdb(cfg, out_dir)
    else:
        run_mirtarbase(cfg, out_dir, keep_weak=not args.strong_only)


if __name__ == "__main__":
    main()
