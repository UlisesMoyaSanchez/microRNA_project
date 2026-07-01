"""
hetero_gnn.py — miRNAGraphTransformer: the central model architecture.

Architecture:
  1. NodeProjector   — maps each node type to a shared hidden_channels space
  2. HGTLayer × L   — L layers of Heterogeneous Graph Transformer message passing
  3. TargetPredictor — link-prediction head: scores miRNA→gene interactions
  4. CellTypeClassifier — node classification head: predicts cell type

The model produces:
  - node embeddings for all types (used for interpretation)
  - edge_logits for (miRNA, gene) pairs in a mini-batch
  - cell_logits for cell nodes in the mini-batch

Reference: Hu et al. "Heterogeneous Graph Transformer" (WWW 2020)
Adapted for multi-modal miRNA + scRNA-seq integration in Multiple Sclerosis.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from models.layers import (
    NodeProjector,
    HGTLayer,
    TargetPredictor,
    CellTypeClassifier,
)


class miRNAGraphTransformer(nn.Module):
    """
    Heterogeneous Graph Transformer for miRNA–gene–cell integration.

    Args:
        hidden_channels:  Embedding dimensionality for all node types.
        num_heads:        Number of attention heads per HGT layer.
        num_layers:       Number of stacked HGT layers.
        dropout:          Dropout probability (applied after each HGT layer).
        num_cell_types:   Number of cell-type classes for the classification head.
        metadata:         PyG graph metadata = (node_types, edge_types).
    """

    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        num_cell_types: int,
        metadata: tuple,
    ):
        super().__init__()

        node_types: list[str] = metadata[0]

        self.projector = NodeProjector(node_types, hidden_channels)

        self.layers = nn.ModuleList([
            HGTLayer(hidden_channels, metadata, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.target_predictor   = TargetPredictor(hidden_channels)
        self.cell_classifier    = CellTypeClassifier(hidden_channels, num_cell_types)

        self.hidden_channels = hidden_channels
        self.metadata        = metadata

    # ── Forward ───────────────────────────────────────────────────────────────

    def encode(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run projection + HGT message passing. Returns final node embeddings."""
        h = self.projector(x_dict)
        for layer in self.layers:
            h = layer(h, edge_index_dict)
        return h

    def forward(
        self,
        x_dict:         dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        mirna_idx: torch.Tensor | None = None,
        gene_idx:  torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x_dict:          Node feature dict {'miRNA': ..., 'gene': ..., 'cell': ...}
            edge_index_dict: Edge index dict per relation type
            mirna_idx:       (E,) miRNA node indices for link-prediction pairs
            gene_idx:        (E,) gene node indices for link-prediction pairs

        Returns dict with keys:
            'embeddings'    — full node embedding dict
            'edge_logits'   — (E,) link-prediction scores (if mirna_idx given)
            'cell_logits'   — (N_cell, num_classes) classification logits
        """
        h = self.encode(x_dict, edge_index_dict)

        out: dict[str, torch.Tensor] = {"embeddings": h}

        # Link prediction: score (miRNA, gene) pairs
        if mirna_idx is not None and gene_idx is not None:
            mirna_emb = h["miRNA"][mirna_idx]
            gene_emb  = h["gene"][gene_idx]
            out["edge_logits"] = self.target_predictor(mirna_emb, gene_emb)

        # Cell type classification
        if "cell" in h:
            out["cell_logits"] = self.cell_classifier(h["cell"])

        return out

    # ── Convenience: predict on a full HeteroData batch ─────────────────────

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        metadata: tuple,
        num_cell_types: int,
    ) -> "miRNAGraphTransformer":
        mcfg = cfg["model"]
        return cls(
            hidden_channels=mcfg["hidden_channels"],
            num_heads=mcfg["num_heads"],
            num_layers=mcfg["num_layers"],
            dropout=mcfg["dropout"],
            num_cell_types=num_cell_types,
            metadata=metadata,
        )
