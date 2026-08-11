"""
layers.py — Reusable building blocks for the miRNA heterogeneous GNN.

HGTLayer       — single Heterogeneous Graph Transformer layer
NodeProjector  — projects each node type to a common hidden dimension
"""

import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear


class NodeProjector(nn.Module):
    """
    Projects each node type from its raw feature dimension to `hidden_channels`.
    Uses PyG's lazy Linear (in_channels=-1) so it adapts to any input at first forward.
    """

    def __init__(self, node_types: list[str], hidden_channels: int):
        super().__init__()
        self.lins = nn.ModuleDict({
            node_type: Linear(-1, hidden_channels)
            for node_type in node_types
        })

    def forward(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            node_type: self.lins[node_type](x).relu()
            for node_type, x in x_dict.items()
            if node_type in self.lins
        }


class HGTLayer(nn.Module):
    """
    Single HGT (Heterogeneous Graph Transformer) layer.
    Wraps torch_geometric.nn.HGTConv with dropout and residual connection.
    """

    def __init__(
        self,
        hidden_channels: int,
        metadata: tuple,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.conv = HGTConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            metadata=metadata,
            heads=num_heads,
        )
        self.norms = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_channels)
            for node_type in metadata[0]
        })
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        h_dict = self.conv(x_dict, edge_index_dict)
        out = {}
        for node_type, h in h_dict.items():
            residual = x_dict.get(node_type)
            h = self.dropout(h)
            if residual is not None and residual.shape == h.shape:
                h = h + residual          # residual connection
            h = self.norms[node_type](h)
            out[node_type] = h
        return out


class TargetPredictor(nn.Module):
    """
    Link-prediction head: scores a (miRNA, gene) pair for target interaction.
    Input: concatenated embeddings [miRNA_emb || gene_emb]
    Output: scalar logit per pair
    """

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, 1),
        )

    def forward(
        self,
        mirna_emb: torch.Tensor,  # (E, hidden_channels)
        gene_emb:  torch.Tensor,  # (E, hidden_channels)
    ) -> torch.Tensor:            # (E,)
        return self.mlp(torch.cat([mirna_emb, gene_emb], dim=-1)).squeeze(-1)


class CellTypeClassifier(nn.Module):
    """
    Node classification head: predicts cell type from cell embedding.
    """

    def __init__(self, hidden_channels: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def forward(self, cell_emb: torch.Tensor) -> torch.Tensor:  # (N, num_classes)
        return self.classifier(cell_emb)
