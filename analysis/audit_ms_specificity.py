"""
audit_ms_specificity.py — Evidence that the MS label never enters the graph.

Read-only. Regenerates every number behind EVALUATION_AUDIT.md "Contribution 4"
from the artifacts themselves, so none of them has to be typed into a document by
hand. (Hardcoded constants are what goal.md §3.2 is already blocked on — see
eval_heldout_grid.py:166. Do not add more.)

The claim under test: *the disease label is provenance, not signal.* The graph is
called an MS graph because of where the bytes came from, not because MS is
represented anywhere in it.

MUST RUN BEFORE the Track B wiring fix — it documents the pre-fix state, and the
graph_sha256 it records is what lets a reviewer tell the two apart afterwards.

Output: results/comparison/ms_specificity_audit.json   (the only file written)
"""

import os
import re
import sys
import json
import glob
import pickle
import hashlib
import argparse
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import yaml


# ── Provenance ────────────────────────────────────────────────────────────────

def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str, cwd: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def provenance(root: str, paths: dict) -> dict:
    out = {
        "git_sha":    git("rev-parse", "HEAD", cwd=root),
        "git_dirty":  bool(git("status", "--porcelain", cwd=root)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }
    # The load-bearing field: after Track B rebuilds the graph, this hash no
    # longer resolves to any live file, and the audit-ms-specificity tag is what
    # recovers the state these numbers describe.
    for name, p in paths.items():
        out[f"{name}_sha256"] = sha256_file(p) if os.path.exists(p) else None
    return out


# ── Checks ────────────────────────────────────────────────────────────────────

def check_mirna_features_are_noise(graph, mirna_init_dim: int) -> dict:
    """The killer check: are the miRNA features reproducible from a seed alone?

    build_heterograph.build_mirna_features does `torch.manual_seed(42);
    torch.randn(n, dim)`. If replaying that reproduces the stored features
    bit-for-bit, the graph's miRNA features provably carry zero data — no
    argument required.
    """
    stored = graph["miRNA"].x
    torch.manual_seed(42)
    replayed = torch.randn(stored.shape[0], mirna_init_dim)
    max_abs_diff = float((stored - replayed).abs().max()) if stored.shape == replayed.shape else None
    return {
        "n_mirna": int(stored.shape[0]),
        "feature_dim": int(stored.shape[1]),
        "reproduced_from_seed_42_alone": bool(max_abs_diff == 0.0),
        "max_abs_diff_vs_replay": max_abs_diff,
        "interpretation": (
            "miRNA node features are torch.randn(seed=42). They contain no biological "
            "information, MS-related or otherwise."
        ),
    }


def check_name_join(graph_mirna_names: list, geo_index: pd.Index) -> dict:
    """Why wiring the GEO arm naively would look like 'MS data does not help'."""
    B = set(map(str, graph_mirna_names))
    G = set(map(str, geo_index))
    exact = B & G
    casefold = {x.lower() for x in B} & {x.lower() for x in G}
    arm = re.compile(r"-[35]p$", re.IGNORECASE)
    stems = [arm.sub("", x.lower()) for x in G]
    return {
        "n_graph_mirna_nodes": len(B),
        "n_geo_mirnas": len(G),
        "exact_overlap": len(exact),
        "exact_overlap_pct": round(100 * len(exact) / len(B), 2),
        "casefold_overlap": len(casefold),
        "casefold_overlap_pct": round(100 * len(casefold) / len(B), 2),
        "n_graph_nodes_without_expression": len(B) - len(casefold),
        "exact_matches_are_all_let7": all("let-7" in x for x in exact),
        "geo_names_lowercase_mir": sum(1 for x in G if x.startswith("hsa-mir-")),
        "graph_names_camel_mir": sum(1 for x in B if x.startswith("hsa-miR-")),
        "arm_strip_collisions_in_geo": len(stems) - len(set(stems)),
        "interpretation": (
            "Self-inflicted: preprocess_mirna.harmonize_mirna_names lowercases names that "
            "the SOFT probe map had already resolved to proper mature form. Only let-7* "
            "survives an exact join because it has no 'mir' substring to mangle. "
            "Case-normalization alone recovers the join; arm-stripping must NOT be used for "
            "it — it merges distinct mature miRNAs (see collisions above)."
        ),
    }


def check_condition_labels(meta: pd.DataFrame, field: str = "Sample_description") -> dict:
    """The stored label is 100% wrong, and the truth is recoverable from the text.

    Parsing here is analysis, not a behaviour change: the artifact on disk is
    untouched. The fix belongs to Track B.
    """
    stored = meta["condition"].value_counts().to_dict() if "condition" in meta else {}
    out = {"stored_condition_value_counts": {str(k): int(v) for k, v in stored.items()}}
    if field not in meta.columns:
        out["error"] = f"{field} not in metadata"
        return out

    low = meta[field].astype(str).str.lower()
    grp = pd.Series("UNMATCHED", index=meta.index)
    # Order matters: a T1D row carries no control keyword, but make it explicit.
    grp[low.str.contains(r"rr-ms|multiple sclerosis|relapsing", regex=True)] = "MS"
    grp[low.str.contains(r"t1d|type 1 diabetes", regex=True)]                = "T1D"
    grp[low.str.contains(r"\bhc\b|healthy|healty", regex=True)]              = "Control"

    cell = pd.Series("other", index=meta.index)
    for c in ("lymphocyte", "monocyte", "neutrophil"):
        cell[low.str.contains(c)] = c
    sex = pd.Series("unknown", index=meta.index)
    sex[low.str.contains("female")] = "Female"
    sex[low.str.contains(r"male") & ~low.str.contains("female")] = "Male"

    out.update({
        "recovered_group_counts": {str(k): int(v) for k, v in grp.value_counts().items()},
        "n_unmatched": int((grp == "UNMATCHED").sum()),
        "recovered_celltype_x_group": pd.crosstab(cell, grp).to_dict(),
        "recovered_sex_x_group": pd.crosstab(sex, grp).to_dict(),
        "cohort_contains_t1d": bool((grp == "T1D").any()),
        "interpretation": (
            "infer_condition (preprocess_mirna.py:122-133) matches 'rrms' and ' ms ' but the "
            "data says 'RR-MS' (hyphen), so nothing matches and _label() returns 'Control' as "
            "a SILENT DEFAULT (line 132). Every sample, including the T1D patients, is "
            "labelled Control. The cohort is NOT MS-vs-HC as the config claims: it contains "
            "Type 1 Diabetes patients, who are currently indistinguishable from healthy "
            "controls. Note 'healty control' is misspelled in the source data."
        ),
    })
    return out


def check_cell_condition_absent(graph) -> dict:
    """visualize.py:109 guards on this attribute; it is never set."""
    return {
        "cell_has_condition_attr": bool(hasattr(graph["cell"], "condition")),
        "cell_node_attrs": sorted(graph["cell"].keys()),
        "interpretation": (
            "preprocess_scrna.py sets adata.obs['condition'], but build_heterograph never "
            "copies it onto the graph. The MS/control UMAP panel in visualize.py cannot "
            "render from the saved graph — it is dead code."
        ),
    }


def check_mirna_expression_vs_degree(graph_mirna_names, expr: pd.DataFrame,
                                     interactions_tsv: str, gene_vocab: set) -> dict:
    """Would MS features smuggle the degree shortcut back in?

    The paper's finding is that uniform negatives select for a gene-popularity
    heuristic. If miRNA expression correlated with miRNA out-degree, wiring it in
    would re-encode that shortcut as a feature and any 'improvement' would be the
    paper's own thesis firing at us.
    """
    from scipy.stats import spearmanr
    df = pd.read_csv(interactions_tsv, sep="\t")
    df = df[df["target_gene"].isin(gene_vocab)]
    deg = df.groupby("mirna").size()
    lut = {str(x).lower(): x for x in expr.index}
    mean_expr = expr.mean(axis=1)
    pairs = [(int(d), float(mean_expr[lut[str(m).lower()]]))
             for m, d in deg.items() if str(m).lower() in lut]
    if len(pairs) < 3:
        return {"error": "too few joinable miRNAs", "n": len(pairs)}
    D = np.array([p[0] for p in pairs]); E = np.array([p[1] for p in pairs])
    rho, p = spearmanr(D, E)
    return {
        "n_mirnas_with_degree_and_expression": len(pairs),
        "spearman_rho_degree_vs_mean_expression": round(float(rho), 4),
        "p_value": float(p),
        "abs_rho_exceeds_0_2": bool(abs(rho) > 0.2),
        "interpretation": (
            "Pre-registered threshold: |rho| > 0.2 would mean expression proxies out-degree, "
            "and no 'MS features help' claim could be made without a degree-matched control. "
            "Below it, expression is not popularity in disguise."
        ),
    }


def check_coexpr_gene_selection(adata_var: pd.DataFrame, top_n: int) -> dict:
    """build_coexpression_edges says 'top_n most variable'; it slices X[:, :top_n]."""
    n = min(top_n, len(adata_var))
    first_n = set(adata_var.index[:n])
    out = {"top_n": n, "n_genes": len(adata_var)}
    col = next((c for c in ("dispersions_norm", "dispersions", "variances_norm")
                if c in adata_var.columns), None)
    if col is None:
        out["error"] = f"no variance column in var; have {list(adata_var.columns)[:8]}"
        return out
    most_var = set(adata_var[col].sort_values(ascending=False).index[:n])
    inter = len(first_n & most_var)
    out.update({
        "variance_column_used": col,
        "overlap_first_n_vs_most_variable": inter,
        "jaccard": round(inter / len(first_n | most_var), 4),
        "interpretation": (
            "build_heterograph.py:151 slices X[:, :n_genes] — the first n columns in var "
            "order, which scanpy preserves from the original annotation, not by variance. "
            "The docstring at :149 ('top_n most variable') is false; co-expression edges live "
            "in an arbitrary subset of the gene nodes."
        ),
    })
    return out


def check_dead_config_keys(root: str, keys: list) -> dict:
    """Keys advertised as filters that no Python ever reads."""
    py = []
    for pat in ("**/*.py",):
        py.extend(glob.glob(os.path.join(root, pat), recursive=True))
    blobs = {}
    for f in py:
        if os.sep + ".git" + os.sep in f:
            continue
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                blobs[f] = fh.read()
        except OSError:
            pass
    out = {}
    for k in keys:
        readers = [os.path.relpath(f, root) for f, b in blobs.items() if k in b]
        readers = [r for r in readers if not r.startswith("analysis/audit_ms_specificity")]
        out[k] = {"read_by": readers, "is_dead": len(readers) == 0}
    out["interpretation"] = (
        "These keys are set in every config and documented as filters, but nothing reads "
        "them. They cannot thin the denser miRTarBase graph — which matters for the "
        "density-matched arm."
    )
    return out


def check_ms_entry_points(root: str) -> dict:
    """Where does MS actually enter the pipeline?"""
    def hits(rel: str, pat: str) -> list:
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            return []
        with open(p, encoding="utf-8", errors="ignore") as fh:
            return [i + 1 for i, ln in enumerate(fh) if re.search(pat, ln)]
    return {
        "preprocess_scrna_condition_lines": hits("data/02_preprocess/preprocess_scrna.py", r"condition"),
        "build_heterograph_condition_lines": hits("data/03_build_graph/build_heterograph.py", r"condition"),
        "build_heterograph_reads_mirna_expr": bool(
            hits("data/03_build_graph/build_heterograph.py", r"mirna_expr")),
        "mirna_expr_readers": [
            os.path.relpath(f, root)
            for f in glob.glob(os.path.join(root, "**/*.py"), recursive=True)
            if "mirna_expr" in open(f, encoding="utf-8", errors="ignore").read()
            and "preprocess_mirna" not in f and "audit_ms_specificity" not in f
        ],
        "interpretation": (
            "MS enters exactly twice, and neither is the graph: (1) which cells were "
            "downloaded (the cellxgene disease filter), and (2) batch_key='condition' in HVG "
            "selection, which shapes the gene vocabulary. It is never a node feature, never a "
            "label, and never an input to any head. The GEO miRNA arm — the only modality "
            "carrying an MS/HC contrast — is written to mirna_expr.tsv and never read."
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Audit: is anything in this graph actually MS-specific?")
    p.add_argument("--config", default="configs/config_v2.yaml")
    p.add_argument("--out", default="results/comparison/ms_specificity_audit.json")
    args = p.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    proc_dir   = cfg["data"]["processed_dir"]
    graphs_dir = cfg["data"]["graphs_dir"]
    raw_dir    = cfg["data"]["raw_dir"]
    graph_p    = os.path.join(graphs_dir, "hetero_graph.pt")
    maps_p     = os.path.join(graphs_dir, "index_maps.pkl")
    expr_p     = os.path.join(proc_dir, "mirna_expr.tsv")
    meta_p     = os.path.join(proc_dir, "mirna_meta.tsv")
    scrna_p    = os.path.join(proc_dir, "scrna_processed.h5ad")
    inter_p    = os.path.join(raw_dir, cfg["data"]["mirna"].get("interactions_file",
                                                                "mirtarbase_hsa.tsv"))

    print("=" * 70)
    print("MS-SPECIFICITY AUDIT — read-only; documents the PRE-FIX state")
    print("=" * 70)
    for label, path in [("graph", graph_p), ("index_maps", maps_p), ("mirna_expr", expr_p),
                        ("mirna_meta", meta_p), ("interactions", inter_p)]:
        print(f"  {label:14s}: {path}")

    print("\nLoading graph...")
    graph = torch.load(graph_p, weights_only=False)
    with open(maps_p, "rb") as fh:
        maps = pickle.load(fh)
    expr = pd.read_csv(expr_p, sep="\t", index_col=0)
    meta = pd.read_csv(meta_p, sep="\t", index_col=0)

    results = {
        "_provenance": provenance(root, {
            "graph": graph_p, "index_maps": maps_p,
            "mirna_expr": expr_p, "mirna_meta": meta_p, "interactions": inter_p,
        }),
        "_claim": (
            "The disease label is provenance, not signal: the graph is called an MS graph "
            "because of where the bytes came from, not because MS is represented in it."
        ),
        "graph_shape": {
            **{f"n_{nt}": int(graph[nt].num_nodes) for nt in graph.node_types},
            **{str(et): int(graph[et].edge_index.shape[1]) for et in graph.edge_types},
        },
    }

    print("\n[1/7] miRNA features: noise or data?")
    results["mirna_features_are_noise"] = check_mirna_features_are_noise(
        graph, cfg["model"]["mirna_init_dim"])

    print("[2/7] GEO <-> graph name join")
    results["name_join"] = check_name_join(list(maps["mirna2idx"].keys()), expr.index)

    print("[3/7] condition labels")
    results["condition_labels"] = check_condition_labels(meta)

    print("[4/7] cell.condition present?")
    results["cell_condition"] = check_cell_condition_absent(graph)

    print("[5/7] would expression re-encode the degree shortcut?")
    results["expression_vs_degree"] = check_mirna_expression_vs_degree(
        list(maps["mirna2idx"].keys()), expr, inter_p, set(maps["gene2idx"].keys()))

    print("[6/7] co-expression gene selection")
    if os.path.exists(scrna_p):
        import anndata as ad
        adata = ad.read_h5ad(scrna_p, backed="r")   # var only; do not load X
        results["coexpr_gene_selection"] = check_coexpr_gene_selection(
            adata.var, cfg["data"]["graph"]["coexpression_top_n_genes"])
        adata.file.close()
    else:
        results["coexpr_gene_selection"] = {"error": f"missing {scrna_p}"}

    print("[7/7] dead config keys + MS entry points")
    results["dead_config_keys"] = check_dead_config_keys(
        root, ["min_mirna_gene_interactions", "min_interactions"])
    results["ms_entry_points"] = check_ms_entry_points(root)

    os.makedirs(os.path.dirname(os.path.join(root, args.out)), exist_ok=True)
    out_p = os.path.join(root, args.out)
    with open(out_p, "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINDINGS")
    print("=" * 70)
    f = results["mirna_features_are_noise"]
    print(f"  miRNA features reproducible from seed 42 alone : {f['reproduced_from_seed_42_alone']}"
          f"  (max|diff| = {f['max_abs_diff_vs_replay']})")
    j = results["name_join"]
    print(f"  GEO<->graph join: exact {j['exact_overlap']} / casefold {j['casefold_overlap']}"
          f" of {j['n_graph_mirna_nodes']} nodes")
    c = results["condition_labels"]
    print(f"  stored condition : {c['stored_condition_value_counts']}")
    print(f"  recovered truth  : {c.get('recovered_group_counts')}  (unmatched={c.get('n_unmatched')})")
    print(f"  cohort has T1D   : {c.get('cohort_contains_t1d')}")
    d = results["expression_vs_degree"]
    print(f"  rho(expr, degree): {d.get('spearman_rho_degree_vs_mean_expression')}"
          f"  (>0.2 would forbid an improvement claim: {d.get('abs_rho_exceeds_0_2')})")
    print(f"  mirna_expr.tsv readers outside preprocess: {results['ms_entry_points']['mirna_expr_readers']}")
    print(f"\n  Saved: {out_p}")
    print("=" * 70)


if __name__ == "__main__":
    main()
