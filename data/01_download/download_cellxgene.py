"""
download_cellxgene.py — Download scRNA-seq data for MS from CellxGene Census.

Outputs:
  data/raw/cellxgene_ms.h5ad   — Multiple Sclerosis patients
  data/raw/cellxgene_ctrl.h5ad — Healthy controls (matched tissues)
"""

import os
import argparse
import yaml
import cellxgene_census


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download scRNA-seq from CellxGene Census")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--out_dir", default=None, help="Override raw_dir from config")
    return p.parse_args()


def build_obs_filter(disease_value: str, tissues: list[str]) -> str:
    tissue_clause = " or ".join(f'tissue_general == "{t}"' for t in tissues)
    return (
        f'(disease == "{disease_value}") '
        f'and ({tissue_clause}) '
        f'and is_primary_data == True'
    )


def _organism_key(census, display_name: str) -> str:
    """
    Resolve the internal census key for an organism display name.
    In census >= 2024, keys are snake_case (e.g. 'homo_sapiens');
    older builds used title case ('Homo sapiens').
    """
    available = list(census["census_data"].keys())
    # Try exact match first
    if display_name in available:
        return display_name
    # Try snake_case conversion
    snake = display_name.lower().replace(" ", "_")
    if snake in available:
        return snake
    raise KeyError(
        f"Organism '{display_name}' not found in census_data. "
        f"Available keys: {available}"
    )


def download_split(
    census,
    organism: str,
    obs_filter: str,
    out_path: str,
    max_obs: int | None,
) -> None:
    if os.path.exists(out_path):
        print(f"  Already exists — skipping: {out_path}")
        return

    print(f"  Filter: {obs_filter}")

    org_key = _organism_key(census, organism)
    print(f"  Organism key in census: '{org_key}'")

    if max_obs is not None:
        # Fetch only a subset of obs to avoid OOM
        obs_df = (
            census["census_data"][org_key]
            .obs.read(
                value_filter=obs_filter,
                column_names=["soma_joinid"],
            )
            .concat()
            .to_pandas()
        )
        if len(obs_df) > max_obs:
            print(f"  Subsampling {len(obs_df)} → {max_obs} cells")
            obs_df = obs_df.sample(n=max_obs, random_state=42)
        obs_coords = obs_df["soma_joinid"].tolist()
    else:
        obs_coords = None

    # Note: var_value_filter on 'feature_biotype' was removed — that column no
    # longer exists in census 2025-11-08. All features in the human census are
    # protein-coding genes; non-gene features are excluded during HVG selection.
    adata = cellxgene_census.get_anndata(
        census=census,
        organism=org_key,
        obs_value_filter=obs_filter,
        obs_coords=obs_coords,
    )
    print(f"  Shape: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    adata.write_h5ad(out_path)
    print(f"  Saved: {out_path}")


def main() -> None:
    args = parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    ccfg = cfg["data"]["cellxgene"]
    out_dir = args.out_dir or cfg["data"]["raw_dir"]
    os.makedirs(out_dir, exist_ok=True)

    organism = ccfg["organism"]
    tissues = ccfg["tissue"]

    census_version = "2025-11-08"  # pinned stable release
    print(f"Opening CellxGene Census (version={census_version})...")
    with cellxgene_census.open_soma(census_version=census_version) as census:

        print("\n[1/2] Downloading MS patients...")
        download_split(
            census=census,
            organism=organism,
            obs_filter=build_obs_filter(ccfg["disease"], tissues),
            out_path=os.path.join(out_dir, "cellxgene_ms.h5ad"),
            max_obs=ccfg.get("max_obs_ms"),
        )

        print("\n[2/2] Downloading healthy controls...")
        download_split(
            census=census,
            organism=organism,
            obs_filter=build_obs_filter("normal", tissues),
            out_path=os.path.join(out_dir, "cellxgene_ctrl.h5ad"),
            max_obs=ccfg.get("max_obs_ctrl"),
        )

    print("\nCellxGene download complete.")


if __name__ == "__main__":
    main()
