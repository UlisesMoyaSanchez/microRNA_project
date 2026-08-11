"""
make_slide_figures.py — the three figures the beamer deck needs for the current results.

Reads ONLY committed artifacts under results/comparison/, so every mark on every slide
traces back to a job ID and a JSON/TSV — the same rule EVALUATION_AUDIT.md holds its own
numbers to. Nothing here is typed in by hand except the labels.

Outputs (vector PDF, because beamer scales it without resampling):
  results/figures/slides_01_inflacion_por_arquitectura.pdf
  results/figures/slides_02_rejilla_protocolo.pdf
  results/figures/slides_03_colapso_negativos.pdf

Palette: categorical #1A6E9E (the deck's blue family, snapped to clear the OKLCH lightness
band and chroma floor) + #BE6E23 (the deck's msorange). Verified with the data-viz skill's
six checks — CVD dE 19.2, normal-vision dE 26.0, both above the 8.0 / 15.0 gates. The
sequential ramp for the heatmap is one hue, light->dark, min adjacent dL 0.109 and a pale end
at 2.02:1 against white so the lightest cell still reads as a mark. Do not substitute
"nicer" colours without re-running that validator.

Usage:  python analysis/make_slide_figures.py
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

ROOT = Path(__file__).resolve().parents[1]
COMP = ROOT / "results" / "comparison"
FIGS = ROOT / "results" / "figures"

# ── validated palette ─────────────────────────────────────────────────────────
BLUE   = "#1A6E9E"   # categorical slot 1 — the honest protocol
ORANGE = "#BE6E23"   # categorical slot 2 — the original (inflated) protocol
RAMP   = ["#8FBDD6", "#4E8FB6", "#1A6E9E", "#0E4E73"]   # sequential, light->dark
INK    = "#0B0B0B"   # text primary
INK2   = "#52514E"   # text secondary
GRID   = "#D8D8D6"   # recessive grid
SURF   = "#FFFFFF"   # slide surface

# Text wears ink tokens, never a series colour; a coloured mark beside it carries identity.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
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

CELLS = {           # cell -> (table suffix, column with train/eval negatives consistent)
    "seen_uniform":  ("transductive_uniform", "auroc_uniform"),
    "seen_matched":  ("transductive",         "auroc_matched"),
    "held_uniform":  ("edgesplit_uniform",    "auroc_uniform"),
    "held_matched":  ("edgesplit",            "auroc_matched"),
}

# Spanish labels for the deck. 'random' is the control and is named as such.
PRETTY = {
    "hgt_v2":             "HGT  (el modelo del proyecto)",
    "homo_gcn":           "GCN homogéneo",
    "ablation_no_coexpr": "HGT sin co-expresión",
    "mlp":                "MLP  (¡sin grafo!)",
    "random":             "Aleatorio  (control, sin entrenar)",
}
ORDER = ["hgt_v2", "homo_gcn", "ablation_no_coexpr", "mlp", "random"]


def load_grid() -> dict[str, dict[str, float]]:
    """model -> cell -> AUROC, from the four committed cross-architecture tables."""
    out: dict[str, dict[str, float]] = {}
    for cell, (suffix, col) in CELLS.items():
        path = COMP / f"comparison_table_checkpoints_v3fixed_baselines_{suffix}.tsv"
        with open(path) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                val = row[col]
                out.setdefault(row["model"], {})[cell] = float("nan") if val == "nan" else float(val)
    return out


def load_topology() -> dict[str, dict[str, float]]:
    with open(COMP / "topology_baseline_v3fixed_test.json") as fh:
        res = json.load(fh)["results"]
    return {h: {k: res[k][h]["auroc"] for k in ("uniform", "degree_matched")}
            for h in ("gene_degree", "adamic_adar")}


def _despine(ax, keep=("left", "bottom")):
    for side, sp in ax.spines.items():
        sp.set_visible(side in keep)


# ── Figure 1 — inflation per architecture (slope chart) ───────────────────────
def fig_inflation(grid, out: Path) -> None:
    """Job: change between two conditions, across categories -> slope chart.

    Horizontal so the architecture names are readable without rotation, sorted by
    inflation so the control lands at the bottom as the visual anchor.
    """
    # Ascending, so the largest inflation ends up at the TOP (y grows upward) and the
    # untrained control sits at the bottom as the anchor.
    rows = [(m, grid[m]["seen_uniform"], grid[m]["held_matched"]) for m in ORDER]
    rows.sort(key=lambda r: (r[1] - r[2]))

    fig, ax = plt.subplots(figsize=(10.2, 4.3))
    ys = range(len(rows))
    for y, (m, orig, honest) in zip(ys, rows):
        # 2px line, ends anchored by >=8px markers; a 2px surface ring on overlap.
        ax.plot([honest, orig], [y, y], color=GRID, lw=2, zorder=1, solid_capstyle="round")
        ax.plot(orig,   y, "o", ms=11, color=ORANGE, zorder=3, mec=SURF, mew=2)
        ax.plot(honest, y, "o", ms=11, color=BLUE,   zorder=3, mec=SURF, mew=2)
        delta = orig - honest
        # Direct labels placed by VALUE order, not by which series they belong to: on the
        # control row the honest point is the higher of the two, and assuming otherwise
        # printed the two labels on top of each other.
        lo_v, hi_v = sorted((orig, honest))
        ax.text(hi_v + 0.012, y, f"{hi_v:.3f}", va="center", ha="left",
                fontsize=10, color=INK)
        ax.text(lo_v - 0.012, y, f"{lo_v:.3f}", va="center", ha="right",
                fontsize=10, color=INK)
        flat = abs(delta) < 0.02
        ax.text(0.985, y + 0.30,
                ("sin cambio" if flat else f"inflación  +{delta:.3f}"),
                va="center", ha="right", fontsize=10.5, color=INK2 if flat else INK,
                style="italic" if flat else "normal",
                fontweight="normal" if flat else "bold")

    ax.set_yticks(list(ys))
    ax.set_yticklabels([PRETTY[m] for m in (r[0] for r in rows)], fontsize=11)
    ax.set_xlim(0.44, 1.0)
    ax.set_ylim(-0.7, len(rows) - 0.35)
    ax.set_xlabel("AUROC  (aristas de validación retenidas)", fontsize=10.5, color=INK2)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    _despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)

    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=10, color=ORANGE,
               label="Protocolo original  (aristas vistas + controles al azar)"),
        Line2D([], [], marker="o", ls="", ms=10, color=BLUE,
               label="Protocolo honesto  (aristas retenidas + controles pareados)"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=1, frameon=False,
        fontsize=10.5, labelcolor=INK, handletextpad=0.6)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ── Figure 2 — the protocol 2x2 (heatmap) ────────────────────────────────────
def fig_grid(grid, out: Path, model: str = "hgt_v2") -> None:
    """Job: magnitude across two crossed factors -> heatmap, one hue light->dark."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    cmap = LinearSegmentedColormap.from_list("msblues", RAMP)
    norm = Normalize(vmin=0.45, vmax=1.0)

    M = np.array([[grid[model]["seen_uniform"],  grid[model]["seen_matched"]],
                  [grid[model]["held_uniform"],  grid[model]["held_matched"]]])

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for i in range(2):
        for j in range(2):
            v = M[i, j]
            # 2px surface gap between fills, so adjacent cells never touch.
            ax.add_patch(Rectangle((j + 0.01, i + 0.01), 0.98, 0.98,
                                   facecolor=cmap(norm(v)), edgecolor=SURF, lw=2))
            # Label ink flips to white only where the fill is dark enough to need it.
            ax.text(j + 0.5, i + 0.5, f"{v:.3f}", ha="center", va="center",
                    fontsize=19, fontweight="bold",
                    color=SURF if norm(v) > 0.55 else INK)
    # Extra room on the right: the diagonal's delta label goes OUTSIDE the matrix, because
    # every position inside it collided with either the arrow or a cell value.
    ax.set_xlim(0, 2.62); ax.set_ylim(2, 0)
    ax.set_xticks([0.5, 1.5]); ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["controles al azar\n(uniformes)", "controles pareados\npor popularidad"],
                       fontsize=10.5)
    ax.set_yticklabels(["aristas\nVISTAS", "aristas\nRETENIDAS"], fontsize=10.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.annotate("", xy=(1.5, 1.5), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color=ORANGE,
                                shrinkA=30, shrinkB=30))
    ax.text(2.08, 1.0, f"−{M[0, 0] - M[1, 1]:.3f}", color=ORANGE, fontsize=17,
            fontweight="bold", ha="left", va="center")
    ax.text(2.08, 1.30, "lo que costó\nevaluar bien", color=INK2, fontsize=10,
            ha="left", va="center")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ── Figure 3 — collapse under matched negatives (grouped bars) ───────────────
def load_uniform_trained() -> dict[str, dict[str, float]]:
    """Both eval columns from the SAME uniform-trained, held-out table.

    This figure's whole claim is 'one model, two evaluations'. Taking the matched column
    from the edgesplit (matched-TRAINED) table instead would compare two different models
    and reproduce the mismatch trap the audit documents — a hard-negative-trained model
    cannot exploit uniform negatives, so that pairing measures the mismatch, not difficulty.
    """
    path = COMP / "comparison_table_checkpoints_v3fixed_baselines_edgesplit_uniform.tsv"
    out: dict[str, dict[str, float]] = {}
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            out[row["model"]] = {
                k: (float("nan") if row[c] == "nan" else float(row[c]))
                for k, c in (("uniform", "auroc_uniform"), ("matched", "auroc_matched"))
            }
    return out


def fig_collapse(same_model, topo, out: Path) -> None:
    """Job: same models, two evaluation conditions -> grouped bars + reference lines."""
    import numpy as np

    models = [m for m in ORDER if m != "random"] + ["random"]
    unif = [same_model[m]["uniform"] for m in models]
    matc = [same_model[m]["matched"] for m in models]

    x = np.arange(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10.2, 4.1))
    # Rounded data-ends anchored to the baseline; 2px surface gap between adjacent bars.
    b1 = ax.bar(x - w / 2 - 0.01, unif, w, color=ORANGE, edgecolor=SURF, lw=2,
                label="evaluado con controles al azar", zorder=3)
    b2 = ax.bar(x + w / 2 + 0.01, matc, w, color=BLUE, edgecolor=SURF, lw=2,
                label="evaluado con controles pareados", zorder=3)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.008,
                    f"{r.get_height():.3f}", ha="center", va="bottom",
                    fontsize=9.5, color=INK)

    # Reference lines are labelled in the legend, not in the plot: every matched bar lands
    # in the 0.50-0.56 band, so in-plot annotations there collided with the bars and with
    # each other.
    gd = topo["gene_degree"]["degree_matched"]
    ax.axhline(0.5, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.axhline(gd, color=ORANGE, lw=1.4, ls=(0, (1, 2)), zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[m].replace("  (", "\n(") for m in models], fontsize=10)
    ax.set_ylim(0.44, 0.90)
    ax.set_ylabel("AUROC", fontsize=10.5, color=INK2)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _despine(ax, keep=("left",))
    ax.tick_params(axis="x", length=0)
    handles = list(ax.get_legend_handles_labels()[0]) + [
        Line2D([], [], color=INK2, lw=1.2, ls=(0, (4, 3)), label="azar  0.500"),
        Line2D([], [], color=ORANGE, lw=1.4, ls=(0, (1, 2)),
               label=f"heurística de popularidad  {gd:.3f}"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=10,
              labelcolor=INK, ncol=2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    grid, topo = load_grid(), load_topology()
    same_model = load_uniform_trained()

    targets = [
        ("slides_01_inflacion_por_arquitectura.pdf", lambda p: fig_inflation(grid, p)),
        ("slides_02_rejilla_protocolo.pdf",          lambda p: fig_grid(grid, p)),
        ("slides_03_colapso_negativos.pdf",          lambda p: fig_collapse(same_model, topo, p)),
    ]
    for name, fn in targets:
        fn(FIGS / name)
        # A PNG twin, only so the result can actually be looked at before shipping.
        png = FIGS / name.replace(".pdf", ".png")
        fn(png)
        print(f"wrote {FIGS.name}/{name}  (+ .png for review)")

    m = grid["hgt_v2"]
    print(f"\nsanity — HGT 2x2: seen+unif {m['seen_uniform']:.4f}  held+match {m['held_matched']:.4f}"
          f"  inflation {m['seen_uniform']-m['held_matched']:+.4f}")
    print(f"sanity — random must sit at chance: "
          f"{grid['random']['seen_uniform']:.4f} / {grid['random']['held_matched']:.4f}")


if __name__ == "__main__":
    main()
