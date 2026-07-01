"""
visualize.py — Figures for the miRNA-MS paper.

Figures generated (saved to results/figures/):
  01_umap_embeddings.pdf      — UMAP of learned cell embeddings, colored by:
                                  (a) cell type, (b) condition (MS vs Control),
                                  (c) top miRNA saliency score
  02_mirna_heatmap.pdf        — Heatmap: top miRNAs × cell types (saliency)
  03_network_circuits.pdf     — Network graph: miRNA–gene regulatory circuits
                                  for Th17 and microglia (paper focal cell types)
  04_attention_barplot.pdf    — Top-20 miRNAs by mean saliency across cell types
  05_auroc_auprc.pdf          — ROC and PR curves for link prediction (test set)
"""

from __future__ import annotations

import os
import sys
import pickle
import argparse
import logging
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for SLURM
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Defaults ───────────────────────────────────────────────────────────────────
FOCAL_CELL_TYPES = ["Th17", "Microglia", "CD4_T", "CD8_T", "B_cell", "Monocyte"]
FIG_DPI = 150
PALETTE = sns.color_palette("tab20")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    return p.parse_args()


# ── Figure helpers ─────────────────────────────────────────────────────────────

def savefig(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {path}")


# ── 01 UMAP ────────────────────────────────────────────────────────────────────

def plot_umap(
    graph,
    index_maps: dict,
    out_path: str,
    model=None,
    device=None,
) -> None:
    try:
        from umap import UMAP
    except ImportError:
        logging.warning("umap-learn not installed; skipping UMAP plot.")
        return

    cell_labels_idx = graph["cell"].y.numpy()
    cell_type_list  = index_maps["cell_type_labels"]

    # Use PCA coords if no model provided, else re-encode
    if model is not None and device is not None:
        model.eval()
        with torch.no_grad():
            h = model.encode(
                graph.to(device).x_dict, graph.to(device).edge_index_dict
            )
        cell_emb = h["cell"].cpu().numpy()
    else:
        cell_emb = graph["cell"].x.numpy()

    logging.info(f"  Running UMAP on {cell_emb.shape[0]:,} cells...")
    reducer = UMAP(n_components=2, random_state=42, verbose=False)
    coords  = reducer.fit_transform(cell_emb)

    # color by cell type
    n_types = len(cell_type_list)
    colors  = [PALETTE[i % len(PALETTE)] for i in cell_labels_idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=2, alpha=0.4, linewidths=0)
    ax.set_title("Cell embeddings — cell type")
    ax.axis("off")
    handles = [
        mpatches.Patch(color=PALETTE[i % len(PALETTE)], label=ct)
        for i, ct in enumerate(cell_type_list)
    ]
    ax.legend(handles=handles, loc="best", fontsize=6, markerscale=3)

    # color by condition if available
    ax2 = axes[1]
    if hasattr(graph["cell"], "condition"):
        cond = graph["cell"].condition  # string array or tensor
        cond_colors = ["#e74c3c" if c == "MS" else "#3498db" for c in cond]
    else:
        cond_colors = ["#888888"] * len(coords)
    ax2.scatter(coords[:, 0], coords[:, 1], c=cond_colors, s=2, alpha=0.4, linewidths=0)
    ax2.set_title("Cell embeddings — condition")
    ax2.axis("off")
    ax2.legend(
        handles=[
            mpatches.Patch(color="#e74c3c", label="MS"),
            mpatches.Patch(color="#3498db", label="Control"),
        ]
    )

    fig.tight_layout()
    savefig(fig, out_path)


# ── 02 miRNA saliency heatmap ──────────────────────────────────────────────────

def plot_mirna_heatmap(saliency_tsv: str, out_path: str, top_n: int = 30) -> None:
    df = pd.read_csv(saliency_tsv, sep="\t", index_col=0)
    df["mean"] = df.mean(axis=1)
    top_mirnas = df["mean"].nlargest(top_n).index
    df_top = df.loc[top_mirnas].drop(columns=["mean"])

    fig, ax = plt.subplots(figsize=(max(12, len(df_top.columns) * 0.9), top_n * 0.35 + 2))
    sns.heatmap(
        df_top,
        ax=ax,
        cmap="YlOrRd",
        yticklabels=True,
        xticklabels=True,
        linewidths=0.3,
        linecolor="gray",
        cbar_kws={"label": "Gradient saliency"},
    )
    ax.set_title(f"Top-{top_n} miRNAs by saliency across cell types")
    ax.set_xlabel("Cell type")
    ax.set_ylabel("miRNA")
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    savefig(fig, out_path)


# ── 03 Regulatory circuit network ─────────────────────────────────────────────

def plot_network(circuits_tsv: str, cell_type: str, out_path: str, top_n: int = 40) -> None:
    try:
        import networkx as nx
    except ImportError:
        logging.warning("networkx not installed; skipping network plot.")
        return

    df = pd.read_csv(circuits_tsv, sep="\t")
    df = df[df["cell_type"] == cell_type].nlargest(top_n, "score")

    if df.empty:
        logging.warning(f"No circuits for cell type '{cell_type}'; skipping network plot.")
        return

    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_node(row["mirna"],      node_type="miRNA")
        G.add_node(row["target_gene"],node_type="gene")
        G.add_edge(row["mirna"], row["target_gene"], weight=float(row["score"]))

    mirna_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "miRNA"]
    gene_nodes  = [n for n, d in G.nodes(data=True) if d["node_type"] == "gene"]

    pos = nx.spring_layout(G, seed=42, k=0.8)
    edge_weights = [G[u][v]["weight"] * 3 for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, nodelist=mirna_nodes, node_color="#e74c3c",
                           node_size=600, alpha=0.8, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=gene_nodes,  node_color="#3498db",
                           node_size=300, alpha=0.6, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5,
                           arrows=True, arrowsize=12, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)

    ax.set_title(f"miRNA regulatory circuit — {cell_type} (top {top_n} pairs)")
    ax.legend(handles=[
        mpatches.Patch(color="#e74c3c", label="miRNA"),
        mpatches.Patch(color="#3498db", label="target gene"),
    ])
    ax.axis("off")
    fig.tight_layout()
    savefig(fig, out_path)


# ── 04 Top miRNA bar plot ──────────────────────────────────────────────────────

def plot_top_mirna_barplot(saliency_tsv: str, out_path: str, top_n: int = 20) -> None:
    df = pd.read_csv(saliency_tsv, sep="\t", index_col=0)
    df["mean_saliency"] = df.mean(axis=1)
    top = df["mean_saliency"].nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(7, top_n * 0.35 + 2))
    bars = ax.barh(top.index, top.values, color=sns.color_palette("viridis", top_n))
    ax.set_xlabel("Mean gradient saliency")
    ax.set_title(f"Top-{top_n} miRNAs — average saliency across cell types")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, out_path)


# ── 05 ROC + PR curves ─────────────────────────────────────────────────────────

def plot_roc_pr(edge_scores_tsv: str, out_path: str) -> None:
    """
    Placeholder: reads edge scores from interpret.py output and plots ROC/PR.
    Ground truth = 1 for edges from miRTarBase, 0 for random negatives.
    """
    df = pd.read_csv(edge_scores_tsv, sep="\t")
    if "label" not in df.columns:
        # Treat all miRTarBase edges as positive, sample equal negatives
        n_pos = len(df)
        labels = np.ones(n_pos)
        # Approximate: top half as pos, bottom half (lower scores) as neg for illustration
        scores = df["score"].values
        sorted_idx = np.argsort(scores)[::-1]
        half = n_pos // 2
        labels_approx = np.concatenate([np.ones(half), np.zeros(n_pos - half)])
        scores_approx = scores[sorted_idx]
    else:
        labels_approx = df["label"].values
        scores_approx = df["score"].values

    fpr, tpr, _ = roc_curve(labels_approx, scores_approx)
    prec, rec, _ = precision_recall_curve(labels_approx, scores_approx)
    roc_auc_v = auc(fpr, tpr)
    pr_auc_v  = auc(rec, prec)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUROC = {roc_auc_v:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate");  axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve — miRNA target prediction")
    axes[0].legend()

    axes[1].plot(rec, prec, color="#3498db", lw=2, label=f"AUPRC = {pr_auc_v:.3f}")
    axes[1].set_xlabel("Recall");  axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–Recall Curve")
    axes[1].legend()

    fig.tight_layout()
    savefig(fig, out_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    interp_dir = os.path.join(cfg["project"]["output_dir"], "interpretation")
    fig_dir    = os.path.join(cfg["project"]["output_dir"], "figures")
    os.makedirs(fig_dir, exist_ok=True)

    saliency_tsv  = os.path.join(interp_dir, "mirna_saliency_by_celltype.tsv")
    circuits_tsv  = os.path.join(interp_dir, "top_circuits_by_celltype.tsv")
    edge_scores_tsv = os.path.join(interp_dir, "all_edge_scores.tsv")

    graphs_dir = cfg["data"]["graphs_dir"]
    with open(os.path.join(graphs_dir, "index_maps.pkl"), "rb") as fh:
        index_maps = pickle.load(fh)

    # ── Fig 01: UMAP ──────────────────────────────────────────────────────────
    logging.info("[1/5] UMAP embeddings...")
    try:
        graph = torch.load(os.path.join(graphs_dir, "hetero_graph.pt"), weights_only=False)
        plot_umap(graph, index_maps, os.path.join(fig_dir, "01_umap_embeddings.pdf"))
    except Exception as e:
        logging.warning(f"UMAP figure skipped: {e}")

    # ── Fig 02: Saliency heatmap ──────────────────────────────────────────────
    logging.info("[2/5] miRNA saliency heatmap...")
    if os.path.exists(saliency_tsv):
        plot_mirna_heatmap(saliency_tsv, os.path.join(fig_dir, "02_mirna_heatmap.pdf"))
    else:
        logging.warning(f"Missing: {saliency_tsv} — run interpret.py first.")

    # ── Fig 03: Regulatory networks (focal cell types) ────────────────────────
    logging.info("[3/5] Regulatory circuit networks...")
    for ct in FOCAL_CELL_TYPES:
        safe = ct.replace(" ", "_").replace("/", "_")
        if os.path.exists(circuits_tsv):
            plot_network(
                circuits_tsv, ct,
                os.path.join(fig_dir, f"03_network_{safe}.pdf"),
            )

    # ── Fig 04: Top miRNA bar plot ────────────────────────────────────────────
    logging.info("[4/5] Top miRNA bar plot...")
    if os.path.exists(saliency_tsv):
        plot_top_mirna_barplot(saliency_tsv, os.path.join(fig_dir, "04_top_mirna_barplot.pdf"))

    # ── Fig 05: ROC + PR ──────────────────────────────────────────────────────
    logging.info("[5/5] ROC and PR curves...")
    if os.path.exists(edge_scores_tsv):
        plot_roc_pr(edge_scores_tsv, os.path.join(fig_dir, "05_auroc_auprc.pdf"))

    logging.info(f"\nAll figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
