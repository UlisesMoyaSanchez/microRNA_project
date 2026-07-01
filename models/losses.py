"""
losses.py — Combined training loss for miRNAGraphTransformer.

Three terms:
  1. Link prediction loss (BCE with logits) — miRNA→gene target prediction
  2. Cell-type classification loss (CrossEntropy) — supervised cell annotation
  3. Attention sparsity (L1 on miRNA embeddings) — encourages sparse regulation

The weights are configurable via config.yaml (training section).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """
    Weighted combination of:
      - link_loss    : BCE on (miRNA, gene) edge prediction
      - clf_loss     : CrossEntropy on cell type classification
      - sparsity_loss: L1 norm on miRNA embeddings (encourages sparse circuits)
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        classification_weight: float = 0.5,
        sparsity_weight:       float = 0.01,
        pos_weight:            float | None = None,
    ):
        super().__init__()
        self.w_link     = reconstruction_weight
        self.w_clf      = classification_weight
        self.w_sparsity = sparsity_weight

        # pos_weight for BCE: compensates for class imbalance (many negative edges)
        # computed as (n_negative / n_positive); set externally after data loading
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.register_buffer("pos_weight", pw)

    def link_loss(
        self,
        logits: torch.Tensor,  # (E,) raw scores
        labels: torch.Tensor,  # (E,) float 0/1
    ) -> torch.Tensor:
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pw)

    def classification_loss(
        self,
        logits: torch.Tensor,  # (N, C) raw scores
        labels: torch.Tensor,  # (N,) int
    ) -> torch.Tensor:
        valid = labels >= 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        return F.cross_entropy(logits[valid], labels[valid])

    def sparsity_loss(self, mirna_emb: torch.Tensor) -> torch.Tensor:
        return mirna_emb.abs().mean()

    def forward(
        self,
        edge_logits:  torch.Tensor | None,       # (E,) — None when no link-prediction
        edge_labels:  torch.Tensor | None,       # (E,) — None when no link-prediction
        cell_logits:  torch.Tensor,              # (N_cell, C)
        cell_labels:  torch.Tensor,              # (N_cell,)
        mirna_emb:    torch.Tensor | None = None,  # (N_mirna, D)
    ) -> dict[str, torch.Tensor]:
        ref_device = cell_logits.device

        if edge_logits is not None and edge_labels is not None:
            l_link = self.link_loss(edge_logits, edge_labels)
        else:
            l_link = torch.tensor(0.0, device=ref_device)

        l_clf  = self.classification_loss(cell_logits, cell_labels)

        l_sparse = (
            self.sparsity_loss(mirna_emb)
            if mirna_emb is not None
            else torch.tensor(0.0, device=ref_device)
        )

        total = (
            self.w_link     * l_link
            + self.w_clf    * l_clf
            + self.w_sparsity * l_sparse
        )

        return {
            "loss":         total,
            "link_loss":    l_link,
            "clf_loss":     l_clf,
            "sparse_loss":  l_sparse,
        }
