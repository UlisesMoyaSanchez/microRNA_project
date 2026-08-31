"""
make_manuscript_figures.py — the four Results figures for the BMC Bioinformatics
manuscript, in English, sized for bmcart's single-column layout (\\textwidth = 127mm).

Reads ONLY committed artifacts under results/comparison/, same rule
EVALUATION_AUDIT.md and make_slide_figures.py hold their own numbers to. Nothing here is
typed in by hand except axis labels and captions.

Figure 1 fixes a sourcing bug found while adapting the deck: make_slide_figures.py's
fig_grid() plots HGT's numbers from the n=1 six-architecture baseline grid (0.9222 /
0.6080, jobs 5849-5852), not the 4-seed headline numbers (0.9867 +/- 0.0011 / 0.6276 +/-
0.0070) the abstract and Results text actually report. This script sources the headline
grid from the multiseed artifacts instead — see results/EVALUATION_AUDIT.md, "Attributing
the collapse" (fixed-graph table).

Outputs (vector PDF; PNG twin only for on-screen review):
  results/figures/manuscript/fig1_protocol_grid.pdf
  results/figures/manuscript/fig2_architecture_inflation.pdf
  results/figures/manuscript/fig3_negative_sampler_collapse.pdf
  results/figures/manuscript/fig4_cross_graph_collapse.pdf

Usage:  python analysis/make_manuscript_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# Reuse the validated palette and small helpers from the deck figure script rather than
# re-deriving/re-validating a second palette for the same document family.
from make_slide_figures import BLUE, ORANGE, RAMP, INK, INK2, GRID, SURF, _despine

ROOT = Path(__file__).resolve().parents[1]
COMP = ROOT / "results" / "comparison"
FIGS = ROOT / "results" / "figures" / "manuscript"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": GRID,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "figure.facecolor": SURF,
    "axes.facecolor": SURF,
    "savefig.facecolor": SURF,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

# bmcart single-column \textwidth is 127mm ~= 5.0in; keep figures at or under that.
COL_WIDTH_IN = 5.0

PRETTY = {
    "hgt_v2":             "HGT (project model)",
    "homo_gcn":           "Homogeneous GCN",
    "ablation_no_coexpr": "HGT, no co-expression edges",
    "mlp":                "MLP (no graph)",
    "random":             "Untrained (control)",
}
ORDER = ["hgt_v2", "homo_gcn", "ablation_no_coexpr", "mlp", "random"]

# Shorter labels for the bar-chart x-axis (fig3), where five multi-word labels collide
# at single-column width; the full names stay in PRETTY for fig2's y-axis and the caption.
SHORT = {
    "hgt_v2":             "HGT",
    "homo_gcn":           "GCN",
    "ablation_no_coexpr": "HGT, no\nco-expr.",
    "mlp":                "MLP\n(no graph)",
    "random":             "Untrained\n(control)",
}


# ── Figure 1 — the headline 2x2 protocol grid, from the MULTISEED artifacts ───────────
def load_headline_grid() -> dict[str, dict[str, tuple[float, float]]]:
    """Reproduces EVALUATION_AUDIT.md's fixed-graph "Attributing the collapse" table:
    seen edges use the uniform-trained condition (the original protocol's semantics,
    from multiseed_seen_edges_test_v3fixed.json); held-out edges use the diagonal
    trained/eval-matched convention (from multiseed_auroc/auprc_test_v3fixed.json).
    Each value is (mean, std) AUROC.
    """
    with open(COMP / "multiseed_seen_edges_test_v3fixed.json") as fh:
        seen = json.load(fh)["cells"]["uniform"]
    with open(COMP / "multiseed_auroc_test_v3fixed.json") as fh:
        held = json.load(fh)["cells"]

    def ms(d: dict) -> tuple[float, float]:
        return d["mean"], d["std"]

    return {
        "seen_uniform":  ms(seen["uniform"]["auroc"]),
        "seen_matched":  ms(seen["degree_matched"]["auroc"]),
        "held_uniform":  ms(held["uniform"]["uniform"]),
        "held_matched":  ms(held["degree_matched"]["degree_matched"]),
    }


def fig1_protocol_grid(grid: dict[str, tuple[float, float]], out: Path) -> None:
    """Job: magnitude (+ spread) across two crossed factors -> heatmap, one hue light->dark."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    cmap = LinearSegmentedColormap.from_list("blues", RAMP)
    norm = Normalize(vmin=0.45, vmax=1.0)

    cells = [[grid["seen_uniform"], grid["seen_matched"]],
             [grid["held_uniform"], grid["held_matched"]]]

    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, 2.75))
    for i in range(2):
        for j in range(2):
            mean, std = cells[i][j]
            ax.add_patch(Rectangle((j + 0.01, i + 0.01), 0.98, 0.98,
                                   facecolor=cmap(norm(mean)), edgecolor=SURF, lw=2))
            ax.text(j + 0.5, i + 0.42, f"{mean:.4f}", ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color=SURF if norm(mean) > 0.55 else INK)
            ax.text(j + 0.5, i + 0.68, f"$\\pm${std:.4f}", ha="center", va="center",
                    fontsize=8.5, color=SURF if norm(mean) > 0.55 else INK2)

    ax.set_xlim(0, 2.55); ax.set_ylim(2, 0)
    ax.set_xticks([0.5, 1.5]); ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["uniform-random\nnegatives", "degree-matched\nnegatives"], fontsize=8.5)
    ax.set_yticklabels(["edges SEEN\n(conventional)", "edges HELD OUT\n(corrected)"], fontsize=8.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.annotate("", xy=(1.5, 1.5), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color=ORANGE,
                                shrinkA=28, shrinkB=28))
    delta = grid["seen_uniform"][0] - grid["held_matched"][0]
    ax.text(2.05, 1.0, f"-{delta:.3f}", color=ORANGE, fontsize=14,
            fontweight="bold", ha="left", va="center")
    ax.text(2.05, 1.28, "cost of the\ncorrected\nprotocol", color=INK2, fontsize=8,
            ha="left", va="center")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ── Figure 2 — inflation per architecture (slope chart), n=1 six-architecture grid ────
CELLS = {
    "seen_uniform":  ("transductive_uniform", "auroc_uniform"),
    "seen_matched":  ("transductive",         "auroc_matched"),
    "held_uniform":  ("edgesplit_uniform",    "auroc_uniform"),
    "held_matched":  ("edgesplit",            "auroc_matched"),
}


def load_architecture_grid() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for cell, (suffix, col) in CELLS.items():
        path = COMP / f"comparison_table_checkpoints_v3fixed_baselines_{suffix}.tsv"
        with open(path) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                val = row[col]
                out.setdefault(row["model"], {})[cell] = float("nan") if val == "nan" else float(val)
    return out


def fig2_architecture_inflation(grid: dict[str, dict[str, float]], out: Path) -> None:
    rows = [(m, grid[m]["seen_uniform"], grid[m]["held_matched"]) for m in ORDER]
    rows.sort(key=lambda r: (r[1] - r[2]))

    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, 3.1))
    ys = range(len(rows))
    for y, (m, orig, honest) in zip(ys, rows):
        ax.plot([honest, orig], [y, y], color=GRID, lw=2, zorder=1, solid_capstyle="round")
        ax.plot(orig,   y, "o", ms=9, color=ORANGE, zorder=3, mec=SURF, mew=1.6)
        ax.plot(honest, y, "o", ms=9, color=BLUE,   zorder=3, mec=SURF, mew=1.6)
        delta = orig - honest
        lo_v, hi_v = sorted((orig, honest))
        ax.text(hi_v + 0.012, y, f"{hi_v:.3f}", va="center", ha="left",
                fontsize=8, color=INK)
        ax.text(lo_v - 0.012, y, f"{lo_v:.3f}", va="center", ha="right",
                fontsize=8, color=INK)
        flat = abs(delta) < 0.02
        ax.text(0.985, y + 0.30,
                ("no change" if flat else f"inflation  +{delta:.3f}"),
                va="center", ha="right", fontsize=8, color=INK2 if flat else INK,
                style="italic" if flat else "normal",
                fontweight="normal" if flat else "bold")

    ax.set_yticks(list(ys))
    ax.set_yticklabels([PRETTY[m] for m in (r[0] for r in rows)], fontsize=9)
    ax.set_xlim(0.44, 1.0)
    ax.set_ylim(-0.7, len(rows) - 0.35)
    ax.set_xlabel("AUROC  (held-out test edges)", fontsize=8.5, color=INK2)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    _despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)

    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=8, color=ORANGE,
               label="Conventional protocol  (seen edges + uniform negatives)"),
        Line2D([], [], marker="o", ls="", ms=8, color=BLUE,
               label="Corrected protocol  (held-out edges + degree-matched negatives)"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=1, frameon=False,
        fontsize=8, labelcolor=INK, handletextpad=0.6)

    fig.tight_layout()
    fig.subplots_adjust(left=0.30)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ── Figure 3 — collapse under matched negatives (grouped bars) ────────────────────────
def load_uniform_trained() -> dict[str, dict[str, float]]:
    path = COMP / "comparison_table_checkpoints_v3fixed_baselines_edgesplit_uniform.tsv"
    out: dict[str, dict[str, float]] = {}
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            out[row["model"]] = {
                k: (float("nan") if row[c] == "nan" else float(row[c]))
                for k, c in (("uniform", "auroc_uniform"), ("matched", "auroc_matched"))
            }
    return out


def load_topology() -> dict[str, dict[str, float]]:
    with open(COMP / "topology_baseline_v3fixed_test.json") as fh:
        res = json.load(fh)["results"]
    return {h: {k: res[k][h]["auroc"] for k in ("uniform", "degree_matched")}
            for h in res["uniform"]}


def fig3_negative_sampler_collapse(same_model, topo, out: Path) -> None:
    import numpy as np

    models = [m for m in ORDER if m != "random"] + ["random"]
    unif = [same_model[m]["uniform"] for m in models]
    matc = [same_model[m]["matched"] for m in models]

    x = np.arange(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, 3.0))
    b1 = ax.bar(x - w / 2 - 0.01, unif, w, color=ORANGE, edgecolor=SURF, lw=2,
                label="evaluated with uniform negatives", zorder=3)
    b2 = ax.bar(x + w / 2 + 0.01, matc, w, color=BLUE, edgecolor=SURF, lw=2,
                label="evaluated with degree-matched negatives", zorder=3)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.008,
                    f"{r.get_height():.3f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK)

    gd = topo["gene_degree"]["degree_matched"]
    ax.axhline(0.5, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.axhline(gd, color=ORANGE, lw=1.3, ls=(0, (1, 2)), zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in models], fontsize=7.8)
    ax.set_ylim(0.44, 0.90)
    ax.set_ylabel("AUROC", fontsize=8.5, color=INK2)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _despine(ax, keep=("left",))
    ax.tick_params(axis="x", length=0)
    handles = list(ax.get_legend_handles_labels()[0]) + [
        Line2D([], [], color=INK2, lw=1.1, ls=(0, (4, 3)), label="chance  0.500"),
        Line2D([], [], color=ORANGE, lw=1.3, ls=(0, (1, 2)),
               label=f"gene-degree heuristic  {gd:.3f}"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=7.5,
              labelcolor=INK, ncol=1)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)



# ── Figure 4 — the same protocol collapse, across six graphs (slope chart) ────────────
GRAPH_PRETTY = {
    "own":           "Our graph (primary case)",
    "canonical5430": "Canonical HMDD",
    "cksnp_gnn":     "CKSNP-GNN",
    "digamn":        "DiGAMN",
    "meahne":        "MEAHNE",
    "couplemda":     "CoupleMDA",
}


def load_cross_graph_grid() -> dict[str, dict[str, float]]:
    """The two corners Figure 1 connects, for every graph with a trained protocol grid.

    The five surveyed graphs come from the aggregated trained grid (4 seeds each, every
    cell matched to its own training arm -- see analysis/aggregate_trained_grid.py); our
    own graph's two corners come from the same multiseed artifacts fig1 reads, so both
    arms of this figure are the numbers Table 7 reports.
    """
    out: dict[str, dict[str, float]] = {}

    own = load_headline_grid()
    out["own"] = {"seen_uniform": own["seen_uniform"][0],
                  "held_matched": own["held_matched"][0]}

    with open(COMP / "trained_grid_summary.json") as fh:
        for g in json.load(fh)["graphs"]:
            cells = g["trained_cells"]
            out[g["graph"]] = {"seen_uniform": cells["seen__uniform"]["mean"],
                               "held_matched": cells["held_out__degree_matched"]["mean"]}
    return out


def fig4_cross_graph_collapse(grid: dict[str, dict[str, float]], out: Path) -> None:
    """Job: the same two-condition change, across graphs -> slope chart, as in fig2.

    Deliberately the same visual form as the per-architecture figure: the point is that
    the collapse is a property of the protocol, so it should look identical whether the
    rows are architectures on one graph or one architecture across graphs.
    """
    rows = [(g, grid[g]["seen_uniform"], grid[g]["held_matched"]) for g in GRAPH_PRETTY]
    rows.sort(key=lambda r: (r[1] - r[2]))

    corrected = [r[2] for r in rows]
    lo, hi = min(corrected), max(corrected)

    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, 3.4))
    ys = range(len(rows))

    # The band every graph lands in once the protocol is corrected -- drawn first so the
    # slopes sit on top of it.
    ax.axvspan(lo, hi, color=BLUE, alpha=0.10, lw=0, zorder=0)
    ax.text((lo + hi) / 2, len(rows) - 0.45, f"{lo:.3f}-{hi:.3f}", ha="center",
            va="center", fontsize=7.5, color=BLUE, fontweight="bold")

    for y, (g, orig, honest) in zip(ys, rows):
        own = g == "own"
        ax.plot([honest, orig], [y, y], color=GRID, lw=2, zorder=1, solid_capstyle="round")
        ax.plot(orig,   y, "o", ms=9, color=ORANGE, zorder=3, mec=SURF, mew=1.6)
        ax.plot(honest, y, "o", ms=9, color=BLUE,   zorder=3, mec=SURF, mew=1.6)
        ax.text(orig + 0.012, y, f"{orig:.3f}", va="center", ha="left", fontsize=8, color=INK)
        ax.text(honest - 0.012, y, f"{honest:.3f}", va="center", ha="right", fontsize=8, color=INK)
        ax.text(0.995, y + 0.30, f"inflation  +{orig - honest:.3f}", va="center",
                ha="right", fontsize=8, color=INK,
                fontweight="bold" if own else "normal")

    ax.set_yticks(list(ys))
    ax.set_yticklabels([GRAPH_PRETTY[r[0]] for r in rows], fontsize=9)
    # Our own graph is one row among six; marked, not privileged.
    for tick, (g, _, _) in zip(ax.get_yticklabels(), rows):
        if g == "own":
            tick.set_fontweight("bold")

    ax.set_xlim(0.54, 1.05)
    ax.set_ylim(-0.7, len(rows) - 0.35)
    ax.set_xlabel("AUROC", fontsize=8.5, color=INK2)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    _despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)

    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=8, color=ORANGE,
               label="Conventional protocol  (seen edges + uniform negatives)"),
        Line2D([], [], marker="o", ls="", ms=8, color=BLUE,
               label="Corrected protocol  (held-out edges + degree-matched negatives)"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=1, frameon=False,
        fontsize=8, labelcolor=INK, handletextpad=0.6)

    fig.tight_layout()
    fig.subplots_adjust(left=0.32)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)

    headline = load_headline_grid()
    arch_grid = load_architecture_grid()
    same_model = load_uniform_trained()
    topo = load_topology()
    cross = load_cross_graph_grid()

    targets = [
        ("fig1_protocol_grid.pdf",              lambda p: fig1_protocol_grid(headline, p)),
        ("fig2_architecture_inflation.pdf",     lambda p: fig2_architecture_inflation(arch_grid, p)),
        ("fig3_negative_sampler_collapse.pdf",  lambda p: fig3_negative_sampler_collapse(same_model, topo, p)),
        ("fig4_cross_graph_collapse.pdf",       lambda p: fig4_cross_graph_collapse(cross, p)),
    ]
    for name, fn in targets:
        fn(FIGS / name)
        png = FIGS / name.replace(".pdf", ".png")
        fn(png)
        print(f"wrote {FIGS.relative_to(ROOT)}/{name}  (+ .png for review)")

    print(f"\nsanity — Figure 1 headline: seen+uniform {headline['seen_uniform'][0]:.4f} "
          f"held+matched {headline['held_matched'][0]:.4f} "
          f"(expect 0.9867 / 0.6276, EVALUATION_AUDIT.md)")
    assert abs(headline["seen_uniform"][0] - 0.9867) < 0.001, "seen+uniform drifted from the canonical headline"
    assert abs(headline["held_matched"][0] - 0.6276) < 0.001, "held+matched drifted from the canonical headline"

    m = arch_grid["random"]
    print(f"sanity — random control must sit at chance: "
          f"{m['seen_uniform']:.4f} / {m['held_matched']:.4f}")


if __name__ == "__main__":
    main()
