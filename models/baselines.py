"""
baselines.py — Comparison models for ablation & baseline evaluation.

Three baselines to compare against miRNAGraphTransformer (V2):

  1. MLPBaseline          — ignores all graph structure; processes each node type
                            independently with an MLP; same task heads as the HGT.
  2. HomoGCNBaseline      — projects all nodes to a shared space and runs a
                            standard 2-layer GCN on a *homogeneous* graph (all
                            edge types merged); cannot model edge-type semantics.
  3. RandomBaseline       — outputs random scores; sets the expected performance
                            floor. Useful to confirm the dataset is non-trivial.

All baselines expose the same forward() signature as miRNAGraphTransformer so
they can be dropped into the existing evaluate() pipeline with zero changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv

from models.layers import TargetPredictor, CellTypeClassifier, NodeProjector


# ── 1. MLP Baseline ─────────────────────────────────────────────────────────

class MLPBaseline(nn.Module):
    """
    Pure MLP: no message-passing, no graph structure.
    Each node type is projected to hidden_channels via a 2-layer MLP.
    Link prediction and cell classification use the same heads as the HGT.

    This isolates whether the graph topology adds value beyond raw features.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
        num_cell_types: int,
        metadata: tuple,
    ):
        super().__init__()
        node_types: list[str] = metadata[0]
        self.hidden_channels = hidden_channels
        self.metadata = metadata

        # Per-type MLP encoders
        def _make_mlp(depth: int, hidden: int, drop: float) -> nn.Sequential:
            layers: list[nn.Module] = [nn.LazyLinear(hidden), nn.ReLU(), nn.Dropout(drop)]
            for _ in range(depth - 1):
                layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(drop)]
            return nn.Sequential(*layers)

        self.encoders = nn.ModuleDict({
            nt: _make_mlp(max(2, num_layers), hidden_channels, dropout)
            for nt in node_types
        })

        self.target_predictor = TargetPredictor(hidden_channels)
        self.cell_classifier  = CellTypeClassifier(hidden_channels, num_cell_types)

    def encode(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],  # ignored — MLP has no MP
    ) -> dict[str, torch.Tensor]:
        return {nt: self.encoders[nt](x) for nt, x in x_dict.items() if nt in self.encoders}

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        mirna_idx: torch.Tensor | None = None,
        gene_idx:  torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h = self.encode(x_dict, edge_index_dict)
        out: dict[str, torch.Tensor] = {"embeddings": h}
        if mirna_idx is not None and gene_idx is not None:
            out["edge_logits"] = self.target_predictor(h["miRNA"][mirna_idx], h["gene"][gene_idx])
        if "cell" in h:
            out["cell_logits"] = self.cell_classifier(h["cell"])
        return out

    @classmethod
    def from_config(cls, cfg: dict, metadata: tuple, num_cell_types: int) -> "MLPBaseline":
        m = cfg["model"]
        t = cfg["training"]
        return cls(
            hidden_channels=m["hidden_channels"],
            num_layers=m["num_layers"],
            dropout=m.get("dropout", 0.2),
            num_cell_types=num_cell_types,
            metadata=metadata,
        )


# ── 2. Homogeneous GCN Baseline ──────────────────────────────────────────────

class HomoGCNBaseline(nn.Module):
    """
    Homogeneous GCN: merges all node/edge types into a single node set and a
    single adjacency matrix.  Two GCNConv layers with residual connections.

    This tests whether *type-aware* message passing (HGT) matters, vs. a
    simpler GCN that treats miRNA, gene, and cell as identical node types.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
        num_cell_types: int,
        metadata: tuple,
    ):
        super().__init__()
        node_types: list[str] = metadata[0]
        self.hidden_channels  = hidden_channels
        self.metadata         = metadata
        self.node_types       = node_types
        self.dropout_p        = dropout

        # Project each node type to hidden_channels
        self.projectors = nn.ModuleDict({
            nt: nn.Sequential(nn.LazyLinear(hidden_channels), nn.ReLU())
            for nt in node_types
        })

        # Homogeneous GCN layers (act on merged node+edge tensor)
        self.convs = nn.ModuleList([
            GCNConv(hidden_channels, hidden_channels, add_self_loops=True)
            for _ in range(max(2, num_layers))
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(max(2, num_layers))
        ])

        # Task heads reuse the same implementations as the HGT
        self.target_predictor = TargetPredictor(hidden_channels)
        self.cell_classifier  = CellTypeClassifier(hidden_channels, num_cell_types)

    def _merge_nodes(
        self,
        x_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, tuple[int, int]]]:
        """
        Concatenate all node features into one tensor.
        Returns (x_merged, offsets) where offsets[nt] = (start, end).
        """
        projected = {nt: self.projectors[nt](x) for nt, x in x_dict.items() if nt in self.projectors}
        offsets: dict[str, tuple[int, int]] = {}
        chunks: list[torch.Tensor] = []
        cursor = 0
        for nt in self.node_types:
            if nt not in projected:
                continue
            n = projected[nt].shape[0]
            offsets[nt] = (cursor, cursor + n)
            chunks.append(projected[nt])
            cursor += n
        x_merged = torch.cat(chunks, dim=0)  # (total_nodes, hidden)
        return x_merged, offsets, projected

    def _merge_edges(
        self,
        edge_index_dict: dict[tuple, torch.Tensor],
        offsets: dict[str, tuple[int, int]],
        device: torch.device,
    ) -> torch.Tensor:
        """Merge all edge types into one homogeneous edge_index, remapping node IDs."""
        edges: list[torch.Tensor] = []
        for (src_type, _, dst_type), edge_index in edge_index_dict.items():
            if src_type not in offsets or dst_type not in offsets:
                continue
            src_offset = offsets[src_type][0]
            dst_offset = offsets[dst_type][0]
            shifted = edge_index.clone()
            shifted[0] += src_offset
            shifted[1] += dst_offset
            edges.append(shifted)
        if not edges:
            total = sum(e - s for s, e in offsets.values())
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        return torch.cat(edges, dim=1)

    def encode(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        device = next(iter(x_dict.values())).device
        x_merged, offsets, _ = self._merge_nodes(x_dict)
        edge_index = self._merge_edges(edge_index_dict, offsets, device)

        h = x_merged
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)
            h_new = F.dropout(h_new, p=self.dropout_p, training=self.training)
            if h.shape == h_new.shape:
                h_new = h_new + h  # residual
            h = norm(h_new)

        # Split back to per-type dicts
        return {nt: h[s:e] for nt, (s, e) in offsets.items()}

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        mirna_idx: torch.Tensor | None = None,
        gene_idx:  torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h = self.encode(x_dict, edge_index_dict)
        out: dict[str, torch.Tensor] = {"embeddings": h}
        if mirna_idx is not None and gene_idx is not None:
            out["edge_logits"] = self.target_predictor(h["miRNA"][mirna_idx], h["gene"][gene_idx])
        if "cell" in h:
            out["cell_logits"] = self.cell_classifier(h["cell"])
        return out

    @classmethod
    def from_config(cls, cfg: dict, metadata: tuple, num_cell_types: int) -> "HomoGCNBaseline":
        m = cfg["model"]
        return cls(
            hidden_channels=m["hidden_channels"],
            num_layers=m["num_layers"],
            dropout=m.get("dropout", 0.2),
            num_cell_types=num_cell_types,
            metadata=metadata,
        )


# ── 3. Random Baseline ───────────────────────────────────────────────────────

class RandomBaseline(nn.Module):
    """
    Outputs random scores for both tasks. Establishes the performance floor
    (expected AUROC ≈ 0.50, accuracy ≈ 1/num_cell_types).
    Has no learnable parameters; no training needed.
    """

    def __init__(self, num_cell_types: int, metadata: tuple, **kwargs):
        super().__init__()
        self.num_cell_types  = num_cell_types
        self.metadata        = metadata
        self.hidden_channels = 1  # dummy
        # Dummy parameter so DDP / optimizer don't complain
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def encode(self, x_dict, edge_index_dict) -> dict[str, torch.Tensor]:
        return {nt: torch.randn(x.shape[0], 1, device=x.device) for nt, x in x_dict.items()}

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        mirna_idx: torch.Tensor | None = None,
        gene_idx:  torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        device = next(iter(x_dict.values())).device
        out: dict[str, torch.Tensor] = {"embeddings": self.encode(x_dict, edge_index_dict)}
        if mirna_idx is not None:
            out["edge_logits"] = torch.randn(mirna_idx.shape[0], device=device)
        n_cells = x_dict["cell"].shape[0]
        out["cell_logits"] = torch.randn(n_cells, self.num_cell_types, device=device)
        return out

    @classmethod
    def from_config(cls, cfg: dict, metadata: tuple, num_cell_types: int) -> "RandomBaseline":
        return cls(num_cell_types=num_cell_types, metadata=metadata)
