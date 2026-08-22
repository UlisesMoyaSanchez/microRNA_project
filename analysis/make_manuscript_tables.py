"""
make_manuscript_tables.py — the Results tables for the BMC Bioinformatics manuscript,
written as standalone LaTeX fragments (\\input{} from manuscript.tex), never hand-typed.

Reads ONLY committed artifacts under results/comparison/ and results/literature_survey.tsv
— same rule EVALUATION_AUDIT.md and make_slide_figures.py hold their own numbers to. If a
source artifact is missing, the corresponding table generator raises with the exact path
and the script/job that produces it, rather than silently skipping or guessing.

Outputs (manuscript/bmc_bioinformatics/tables/):
  table1_headline_grid.tex        — 2x2 protocol grid, mean +/- std, AUROC & AUPRC
  table2_model_free_baselines.tex — 4 no-learning heuristics + trained-model rows
  table3_literature_survey.tex    — 7-paper pilot survey summary
  table4_celltype_control.tex     — cell-type no-graph controls vs. HGT
  tableS1_architecture_grid.tex   — full 6-architecture x 4-condition AUROC grid

Usage:  python analysis/make_manuscript_tables.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = ROOT / "results" / "comparison"
OUT = ROOT / "manuscript" / "bmc_bioinformatics" / "tables"

PRETTY = {
    "hgt_v2":             "HGT (project model)",
    "homo_gcn":           "Homogeneous GCN",
    "ablation_no_coexpr": "HGT, no co-expression edges",
    "mlp":                "MLP (no graph)",
    "ablation_no_mirna":  "HGT, no miRNA input (smoke test)",
    "random":             "Untrained (control)",
}
ORDER_FULL = ["hgt_v2", "homo_gcn", "ablation_no_coexpr", "mlp", "ablation_no_mirna", "random"]


def tex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def mean_std(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


# ── Table 1 — headline 2x2 protocol grid, multiseed mean +/- std ──────────────────────
def table1_headline_grid() -> str:
    with open(COMP / "multiseed_seen_edges_test_v3fixed.json") as fh:
        seen = json.load(fh)["cells"]["uniform"]
    with open(COMP / "multiseed_auroc_test_v3fixed.json") as fh:
        held_auroc = json.load(fh)["cells"]
    with open(COMP / "multiseed_auprc_test_v3fixed.json") as fh:
        held_auprc = json.load(fh)["cells"]

    rows = [
        ("Edges seen in training (conventional)", "uniform-random negatives",
         seen["uniform"]["auroc"], seen["uniform"]["auprc"]),
        ("Edges seen in training (conventional)", "degree-matched negatives",
         seen["degree_matched"]["auroc"], seen["degree_matched"]["auprc"]),
        ("Edges held out (honest)", "uniform-random negatives",
         held_auroc["uniform"]["uniform"], held_auprc["uniform"]["uniform"]),
        ("Edges held out (honest)", "degree-matched negatives",
         held_auroc["degree_matched"]["degree_matched"], held_auprc["degree_matched"]["degree_matched"]),
    ]

    lines = [
        r"\begin{table}[h!]",
        r"\caption{Headline protocol grid: mean $\pm$ std AUROC and AUPRC over 4 seeds "
        r"(\{123, 777, 2024, 7\}), test set, \texttt{graphs\_v3fixed}. Rows cross the edge "
        r"split (seen vs.\ held out); columns cross the negative-sampling regime. The "
        r"``edges seen'' row reports the uniform-negative-trained model (matching the "
        r"conventional protocol's semantics) under both evaluation regimes; the ``edges "
        r"held out'' row reports each model evaluated with the negative type it was "
        r"trained on. See Methods for the train/eval-negative-matching convention.}",
        r"\label{tab:headline_grid}",
        r"\small",
        r"\begin{tabular}{llcc}",
        r"\hline",
        r"Split & Negatives & AUROC & AUPRC \\",
        r"\hline",
    ]
    for split, neg, auroc, auprc in rows:
        lines.append(
            f"{split} & {neg} & {mean_std(auroc['mean'], auroc['std'])} & "
            f"{mean_std(auprc['mean'], auprc['std'])} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ── Table 2 — model-free baselines vs. trained model, both negative regimes ───────────
def table2_model_free_baselines() -> str:
    with open(COMP / "topology_baseline_v3fixed_test.json") as fh:
        topo = json.load(fh)["results"]
    with open(COMP / "multiseed_auroc_test_v3fixed.json") as fh:
        multi = json.load(fh)["cells"]

    heuristic_order = ["gene_degree", "pref_attach", "common_neigh", "adamic_adar"]
    heuristic_names = {
        "gene_degree":  "Gene in-degree (ignores the miRNA)",
        "pref_attach":  "Preferential attachment",
        "common_neigh": "Common neighbours",
        "adamic_adar":  "Adamic--Adar (best heuristic)",
    }

    lines = [
        r"\begin{table}[h!]",
        r"\caption{Model-free (no-learning) heuristics vs.\ the trained model, held-out "
        r"test edges, \texttt{graphs\_v3fixed}. Heuristics carry no training seed (n=1, "
        r"identical to the pre-fix graph). Trained-model rows are mean $\pm$ std over 4 "
        r"seeds. The uniform-trained model evaluated against degree-matched negatives (the "
        r"``mismatch trap'\,'; $\dagger$) is a train/eval negative-distribution mismatch, "
        r"not a difficulty measurement -- see Methods.}",
        r"\label{tab:model_free_baselines}",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Scorer & Uniform negatives & Degree-matched negatives \\",
        r"\hline",
    ]
    for h in heuristic_order:
        u, m = topo["uniform"][h]["auroc"], topo["degree_matched"][h]["auroc"]
        lines.append(f"{heuristic_names[h]} & {u:.4f} & {m:.4f} \\\\")
    lines.append(r"\hline")

    trained_uniform = multi["uniform"]
    trained_matched = multi["degree_matched"]
    lines.append(
        "HGT, trained with uniform negatives & "
        f"{mean_std(trained_uniform['uniform']['mean'], trained_uniform['uniform']['std'])} & "
        f"{mean_std(trained_uniform['degree_matched']['mean'], trained_uniform['degree_matched']['std'])}"
        r" $\dagger$ \\"
    )
    lines.append(
        "HGT, trained with degree-matched negatives & "
        f"{mean_std(trained_matched['uniform']['mean'], trained_matched['uniform']['std'])}"
        r" $\dagger$ & "
        f"{mean_std(trained_matched['degree_matched']['mean'], trained_matched['degree_matched']['std'])}"
        r" \\"
    )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ── Table 3 — literature survey summary (pilot, n=7) ──────────────────────────────────
def table3_literature_survey() -> str:
    with open(ROOT / "results" / "literature_survey.tsv") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    lines = [
        r"\begin{table}[h!]",
        r"\caption{Pilot literature survey (n=7): each paper's methods section was read in "
        r"full and classified on whether held-out edges are removed from the "
        r"message-passing / similarity-computation graph, how negative pairs are chosen, "
        r"and whether any model-free baseline is reported. Full evidence quotes and source "
        r"URLs: \texttt{results/literature\_survey.tsv}.}",
        r"\label{tab:literature_survey}",
        r"\footnotesize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllcl}",
        r"\hline",
        r"Paper & Venue/year & Task & Held-out edges & Negative sampling \\",
        r" & & & removed from graph? & \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{tex_escape(row['paper'])} & {tex_escape(row['venue_year'])} & "
            f"{tex_escape(row['task'])} & {tex_escape(row['testedges_removed_from_encoder_graph'])} & "
            f"{tex_escape(row['negative_sampling'])} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", "}", ""]
    n_baseline = sum(1 for r in rows if r["model_free_baseline"].strip().lower() == "yes")
    lines.append(
        f"\\medskip\\par Model-free baseline reported: {n_baseline}/{len(rows)} papers."
    )
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ── Table 4 — cell-type classification, no-graph controls vs. HGT ─────────────────────
def table4_celltype_control() -> str:
    path = COMP / "celltype_baseline_config_v2_edgesplit_test.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "This artifact (job 5853, 2026-07-29) was produced by "
            "training/eval_celltype_baseline.py on dgxum and has not been synced to this "
            "machine (confirmed: not tracked in git, unlike the rest of results/comparison/). "
            "Sync it -- e.g. `scp dgxum:/raid/home/umoya/scripts/microRNA_project/"
            "results/comparison/celltype_baseline_config_v2_edgesplit_test.json "
            "results/comparison/` -- then rerun this script. Known values (from "
            "results/EVALUATION_AUDIT.md, \"The cell-type control\", not regenerated here): "
            "nearest_centroid acc=0.4654 f1_macro=0.4612; "
            "logistic_regression acc=0.6692 f1_macro=0.5737; "
            "HGT (reference, degree-matched) acc=0.9916, f1_macro not recorded."
        )
    with open(path) as fh:
        d = json.load(fh)

    lines = [
        r"\begin{table}[h!]",
        r"\caption{Cell-type classification: no-graph controls (nearest-centroid, "
        r"logistic regression, both on \texttt{X\_pca} alone, no message passing) vs.\ the "
        r"trained HGT, same split/config/seed, test set. Source: "
        f"{tex_escape(d.get('config', 'config_v2_edgesplit.yaml'))}" r", "
        f"n={d.get('n_eval', 'N/A')} held-out cells.}}",
        r"\label{tab:celltype_control}",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Classifier & Accuracy & Macro-F1 \\",
        r"\hline",
    ]
    nc, lr = d["results"]["nearest_centroid"], d["results"]["logistic_regression"]
    lines.append(f"Nearest-centroid (no-graph) & {nc['acc']:.4f} & {nc['f1_macro']:.4f} \\\\")
    lines.append(f"Logistic regression (no-graph) & {lr['acc']:.4f} & {lr['f1_macro']:.4f} \\\\")
    ref = d["reference"]
    f1 = ref.get("hgt_cell_f1")
    f1_str = f"{f1:.4f}" if f1 is not None else "--"
    lines.append(
        f"HGT (reference, degree-matched) & {ref['hgt_cell_acc_degree_matched']:.4f} & {f1_str} \\\\"
    )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ── Supplementary Table S1 — full six-architecture x four-condition AUROC grid ────────
def table_s1_architecture_grid() -> str:
    cells = {
        "seen_uniform":  ("transductive_uniform", "auroc_uniform"),
        "seen_matched":  ("transductive",         "auroc_matched"),
        "held_uniform":  ("edgesplit_uniform",    "auroc_uniform"),
        "held_matched":  ("edgesplit",            "auroc_matched"),
    }
    grid: dict[str, dict[str, float]] = {}
    for cell, (suffix, col) in cells.items():
        path = COMP / f"comparison_table_checkpoints_v3fixed_baselines_{suffix}.tsv"
        with open(path) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                val = row[col]
                grid.setdefault(row["model"], {})[cell] = None if val == "nan" else float(val)

    lines = [
        r"\begin{table}[h!]",
        r"\caption{Full six-architecture $\times$ four-condition "
        r"AUROC grid backing Figure~\ref{fig:architecture_inflation} (single seed, 42, "
        r"jobs 5849--5852, \texttt{graphs\_v3fixed}). "
        r"\texttt{ablation\_no\_mirna} has no link-prediction head by construction and is "
        r"retained only as a smoke test (cell classification accuracy unaffected). "
        r"Inflation = seen+uniform $-$ held+matched.}",
        r"\label{tab:architecture_grid}",
        r"\footnotesize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Architecture & Seen+uniform & Seen+matched & Held+uniform & Held+matched & Inflation \\",
        r"\hline",
    ]
    for m in ORDER_FULL:
        row = grid[m]
        vals = [row[c] for c in ("seen_uniform", "seen_matched", "held_uniform", "held_matched")]
        if all(v is not None for v in vals):
            infl = vals[0] - vals[3]
            cells_str = " & ".join(f"{v:.4f}" for v in vals)
            lines.append(f"{PRETTY[m]} & {cells_str} & {infl:+.4f} \\\\")
        else:
            lines.append(f"{PRETTY[m]} & \\multicolumn{{4}}{{c}}{{no link head}} & n/a \\\\")
    lines += [r"\hline", r"\end{tabular}", "}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def table6_protocol_grid() -> str:
    """The model-free 2x2 protocol grid, measured with the same four scorers on our own
    graph and on every distinct graph behind the seven surveyed papers.

    One row per distinct GRAPH, not per paper: MGCNSS, NIMGSA and HLGNN-MDA share a
    byte-identical canonical HMDD matrix, so it appears twice only because MGCNSS is
    scored on the paper's own bundled split and the other two on a generated one.
    """
    def load(path):
        with open(COMP / path) as fh:
            return json.load(fh)["results"]

    def best(res, regime):
        return max(v["auroc"] for v in res[regime].values()) if regime in res else None

    stats = {r["graph"]: r for r in
             json.load(open(COMP / "graph_candidate_stats.json"))["graphs"]}

    hmdd = "hmdd_survey_topology_baseline_%s%s.json"
    rows = [
        ("Own graph (primary case)", "own_graph", "--",
         load("topology_baseline_test_seen.json"), load("topology_baseline_test.json")),
        ("Canonical HMDD, paper's split", "canonical5430", "MGCNSS",
         load(hmdd % ("mgcnss", "_seen")), load(hmdd % ("mgcnss", ""))),
        ("Canonical HMDD, generated split", "canonical5430", "NIMGSA, HLGNN-MDA",
         load(hmdd % ("nimgsa", "_seen")), load(hmdd % ("nimgsa", ""))),
        ("DiGAMN", "digamn", "DiGAMN",
         load(hmdd % ("digamn", "_seen")), load(hmdd % ("digamn", ""))),
        ("CKSNP-GNN", "cksnp_gnn", "CKSNP-GNN",
         load(hmdd % ("cksnp_gnn", "_seen")), load(hmdd % ("cksnp_gnn", ""))),
        ("CoupleMDA", "couplemda", "CoupleMDA",
         load(hmdd % ("couplemda", "_seen")), load(hmdd % ("couplemda", ""))),
        ("MEAHNE", "meahne", "MEAHNE",
         load(hmdd % ("meahne", "_seen")), load(hmdd % ("meahne", ""))),
    ]

    lines = [
        r"\begin{table}[h!]",
        r"\caption{The model-free protocol grid, measured with the same four unmodified "
        r"topology scorers (Table~\ref{tab:model_free_baselines}) on our own graph and on "
        r"every distinct graph behind the seven surveyed papers. Rows are graphs, not "
        r"papers: MGCNSS, NIMGSA and HLGNN-MDA share a byte-identical canonical HMDD "
        r"matrix, which appears twice only because MGCNSS is scored on its own bundled "
        r"split and the other two on a generated one. Within a row the split and the "
        r"negatives are identical across all four cells -- only the adjacency the scorers "
        r"may see changes -- so every cell scores exactly the same pairs and ``Cost'' "
        r"(conventional corner minus corrected corner) is attributable to the protocol "
        r"alone. MGCNSS carries the paper's own negatives, so it has no negative-sampling "
        r"axis. ``Dead'' is the fraction of candidate columns with degree zero and Gini "
        r"the inequality of the column-degree distribution "
        r"(\texttt{results/comparison/graph\_candidate\_stats.json}).}",
        r"\label{tab:protocol_grid}",
        r"\footnotesize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccccc}",
        r"\hline",
        r" & & & & \multicolumn{2}{c}{Uniform negatives} & "
        r"\multicolumn{2}{c}{Degree-matched} & \\",
        r"\cline{5-6}\cline{7-8}",
        r"Graph & Paper(s) & Dead & Gini & seen & held out & seen & held out & Cost \\",
        r"\hline",
    ]

    for label, key, papers, seen, held in rows:
        st = stats[key]
        if "paper_split" in held:
            us, uh = best(seen, "paper_split"), best(held, "paper_split")
            ds = dh = None
        else:
            us, uh = best(seen, "uniform"), best(held, "uniform")
            ds, dh = best(seen, "degree_matched"), best(held, "degree_matched")
        cost = us - (dh if dh is not None else uh)
        cell = lambda v: "--" if v is None else f"{v:.4f}"
        lines.append(
            f"{tex_escape(label)} & {tex_escape(papers)} & "
            f"{st['dead_column_fraction'] * 100:.1f}\\% & "
            f"{st['column_degree_gini']:.3f} & "
            f"{cell(us)} & {cell(uh)} & {cell(ds)} & {cell(dh)} & "
            f"$+{cost:.3f}$ \\\\"
        )

    lines += [r"\hline", r"\end{tabular}", r"}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    generators = [
        ("table1_headline_grid.tex", table1_headline_grid),
        ("table2_model_free_baselines.tex", table2_model_free_baselines),
        ("table3_literature_survey.tex", table3_literature_survey),
        ("table4_celltype_control.tex", table4_celltype_control),
        ("table6_protocol_grid.tex", table6_protocol_grid),
        ("tableS1_architecture_grid.tex", table_s1_architecture_grid),
    ]
    failures = []
    for name, fn in generators:
        try:
            content = fn()
        except FileNotFoundError as e:
            print(f"SKIPPED {name}:\n  {e}\n")
            failures.append(name)
            continue
        (OUT / name).write_text(content)
        print(f"wrote {(OUT / name).relative_to(ROOT)}")

    if failures:
        print(f"\n{len(failures)} table(s) not generated (see above): {', '.join(failures)}")


if __name__ == "__main__":
    main()
