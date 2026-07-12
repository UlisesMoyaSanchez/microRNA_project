"""
splits.py — Edge-level train/val/test split and negative sampling for link prediction.

Two problems this module exists to fix, both of which made the pre-2026-07 numbers
unpublishable as "prediction":

1. There was no edge-level split. `train.split_graph()` partitions `cell` nodes only, so
   the miRNA→gene edges scored at evaluation were the same edges supervised during
   training. `diagnose_leakage.py` ruled out the *message-passing* half of that leak
   (masking the scored pair costs only ~0.009 AUROC), but not weight-level memorization.
   Only a real held-out edge set can. `build_edge_split()` provides one.

   The subtlety is the reverse relation. `build_heterograph.py` adds both
   (miRNA, regulates, gene) and (gene, regulated_by, miRNA). Holding an edge out of the
   forward relation while leaving its reverse in place leaves it reachable in one hop and
   the split is worthless. `RandomLinkSplit(rev_edge_types=...)` strips both in lockstep.

2. Negatives were uniform and drawn against the wrong exclusion set. Uniform negatives
   overstate specificity: `eval_hard_negatives.py` measured AUROC 0.9758 against uniform
   negatives but 0.8828 against degree-matched ones, while a model-free "score the gene by
   how many miRNAs target it" heuristic went 0.7760 → 0.5150. And the exclusion set passed
   to `negative_sampling` was the ≤512-edge batch *subsample*, so a sampled "negative"
   could be a genuine positive from elsewhere in the graph.

   `LinkSampler` draws degree-matched negatives — for a positive (m, g), a negative (m, g')
   where g' is not a target of m and sits in the same log-spaced in-degree bin as g — and
   excludes against every known positive in the batch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling

REL_FWD = ("miRNA", "regulates", "gene")
REL_REV = ("gene", "regulated_by", "miRNA")

N_DEGREE_BINS = 8

log = logging.getLogger(__name__)


# ── Pair-key helpers ───────────────────────────────────────────────────────────

def pair_keys(edge_index: torch.Tensor, n_gene: int) -> torch.Tensor:
    """Encode (miRNA, gene) pairs as single ints so they can be set-compared."""
    return edge_index[0].long() * n_gene + edge_index[1].long()


# ── The split ──────────────────────────────────────────────────────────────────

@dataclass
class EdgeSplit:
    """
    mp_graph   — the graph the encoder is allowed to see. Its REL_FWD/REL_REV contain
                 ONLY training message-passing edges; val and test edges are absent in
                 both directions. Used for every loader (train, val and test), so all
                 three splits are scored under an identical encoder view.
    train_sup / val_sup / test_sup — supervision targets, global (2, N) miRNA→gene.
                 Disjoint from each other, and val/test are disjoint from mp_graph.
    all_pos    — every known positive edge. The exclusion set for negative sampling:
                 negatives must avoid held-out positives too, or a test edge gets handed
                 to the model labelled 0.
    """
    mp_graph:  HeteroData
    train_sup: torch.Tensor
    val_sup:   torch.Tensor
    test_sup:  torch.Tensor
    all_pos:   torch.Tensor


def build_edge_split(
    graph: HeteroData,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    disjoint_train_ratio: float = 0.3,
) -> EdgeSplit:
    """
    Hold out val/test miRNA→gene edges from message passing AND from supervision.

    disjoint_train_ratio additionally splits the training edges so the ones used as
    supervision targets are not also fed to the encoder — without it the model can still
    read a training edge off the graph while being asked to predict it, which is the
    habit we are trying to break.
    """
    if REL_FWD not in graph.edge_types:
        raise ValueError(
            f"{REL_FWD} absent — cannot build an edge split for a graph with no "
            "miRNA→gene edges (expected only for the no_mirna ablation, which should "
            "not call this)."
        )

    all_pos = graph[REL_FWD].edge_index.clone()

    torch.manual_seed(seed)
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=False,
        edge_types=REL_FWD,
        rev_edge_types=REL_REV,
        disjoint_train_ratio=disjoint_train_ratio,
        neg_sampling_ratio=0.0,          # negatives come from LinkSampler, not here
        add_negative_train_samples=False,
    )
    train_data, val_data, test_data = transform(graph)

    split = EdgeSplit(
        mp_graph=train_data,
        train_sup=train_data[REL_FWD].edge_label_index,
        val_sup=val_data[REL_FWD].edge_label_index,
        test_sup=test_data[REL_FWD].edge_label_index,
        all_pos=all_pos,
    )

    log.info(
        f"Edge split — {all_pos.shape[1]:,} positives → "
        f"mp {split.mp_graph[REL_FWD].edge_index.shape[1]:,} | "
        f"train_sup {split.train_sup.shape[1]:,} | "
        f"val_sup {split.val_sup.shape[1]:,} | "
        f"test_sup {split.test_sup.shape[1]:,}"
    )
    return split


def assert_no_edge_leakage(split: EdgeSplit) -> None:
    """
    Gating check. If this does not pass, no number downstream of it means anything.

    Verifies that held-out edges are absent from the encoder's input in BOTH directions
    (the reverse-relation trap), and that the three supervision sets do not overlap.
    """
    n_gene = split.mp_graph["gene"].num_nodes

    mp_fwd = split.mp_graph[REL_FWD].edge_index
    mp_keys = pair_keys(mp_fwd, n_gene)
    # REL_REV is (gene → miRNA); flip it so it is comparable as (miRNA, gene).
    mp_rev_keys = pair_keys(split.mp_graph[REL_REV].edge_index.flip(0), n_gene)

    train_k = pair_keys(split.train_sup, n_gene)
    val_k   = pair_keys(split.val_sup,   n_gene)
    test_k  = pair_keys(split.test_sup,  n_gene)

    def overlap(a: torch.Tensor, b: torch.Tensor) -> int:
        return int(torch.isin(a, b).sum())

    checks = [
        ("val_sup  ∩ mp forward edges",  overlap(val_k,  mp_keys)),
        ("test_sup ∩ mp forward edges",  overlap(test_k, mp_keys)),
        ("val_sup  ∩ mp reverse edges",  overlap(val_k,  mp_rev_keys)),
        ("test_sup ∩ mp reverse edges",  overlap(test_k, mp_rev_keys)),
        ("val_sup  ∩ train_sup",         overlap(val_k,  train_k)),
        ("test_sup ∩ train_sup",         overlap(test_k, train_k)),
        ("test_sup ∩ val_sup",           overlap(test_k, val_k)),
        ("train_sup ∩ mp forward edges", overlap(train_k, mp_keys)),
    ]

    failed = [(name, n) for name, n in checks if n > 0]
    for name, n in checks:
        log.info(f"  {'FAIL' if n > 0 else 'ok  '}  {name}: {n}")
    if failed:
        raise AssertionError(
            "Edge split leaks: " + "; ".join(f"{name} = {n}" for name, n in failed)
        )

    # The forward and reverse message-passing relations must describe the same edge set,
    # or one of them was filtered and the other was not.
    if mp_keys.numel() != mp_rev_keys.numel():
        raise AssertionError(
            f"mp forward ({mp_keys.numel()}) and reverse ({mp_rev_keys.numel()}) edge "
            "counts differ — the reverse relation was not stripped in lockstep."
        )


# ── Degree-matched negatives ───────────────────────────────────────────────────

def gene_in_degree(edge_index: torch.Tensor, n_gene: int) -> torch.Tensor:
    """How many miRNAs target each gene. Pass TRAINING edges only when building the
    sampler for a training run — binning on the full edge set leaks held-out structure
    into the choice of negatives."""
    deg = torch.zeros(n_gene, dtype=torch.long)
    deg.scatter_add_(
        0, edge_index[1].long().cpu(), torch.ones(edge_index.shape[1], dtype=torch.long)
    )
    return deg


def degree_bins(deg: torch.Tensor, n_bins: int = N_DEGREE_BINS) -> torch.Tensor:
    """Assign each gene to a log-spaced in-degree bin. Bin 0 = never targeted."""
    logd = torch.log1p(deg.float())
    hi = float(logd.max()) if float(logd.max()) > 0 else 1.0
    edges = torch.linspace(0, hi, n_bins + 1)[1:-1]
    return torch.bucketize(logd, edges)


def sample_degree_matched_negatives(
    scored_pos:      torch.Tensor,   # (2, k) local (miRNA, gene) — the positives
    exclusion_pos:   torch.Tensor,   # (2, P) every true edge among batch nodes, local
    gene_bins_local: torch.Tensor,   # (n_gene_local,) degree bin per local gene
    n_gene_local:    int,
    generator:       torch.Generator,
    device:          torch.device,
    n_tries:         int = 16,
) -> tuple[torch.Tensor, int]:
    """
    For each positive (m, g), a negative (m, g') with g' in the same degree bin as g and
    (m, g') not a true edge. Same miRNA and an equally-popular gene, so neither miRNA
    promiscuity nor gene popularity can separate positive from negative — only regulatory
    structure can.

    Falls back to any non-target of m when the bin offers no valid candidate in n_tries
    draws, so the result always has k columns. Returns (negatives, n_fallback); a high
    fallback count means the degree match is not clean and the metric is closer to the
    uniform-negative one.

    Vectorised: the training loop calls this once per batch for every epoch, which a
    per-pair Python loop with a `set` membership test is far too slow for.

    Runs on CPU regardless of `device` (the generator is CPU-seeded, and this is index
    bookkeeping, not arithmetic); the result is moved to `device` on the way out.
    """
    k = scored_pos.shape[1]
    if k == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device), 0

    scored_pos      = scored_pos.cpu()
    gene_bins_local = gene_bins_local.cpu()
    true_keys = pair_keys(exclusion_pos.cpu(), n_gene_local)

    m = scored_pos[0].long()                      # (k,)
    g = scored_pos[1].long()                      # (k,)

    # Group local genes by degree bin, CSR-style, so candidates can be drawn per bin
    # without a Python-level dict.
    order        = torch.argsort(gene_bins_local)               # (n_gene_local,)
    sorted_genes = order
    sorted_bins  = gene_bins_local[order]
    n_bins       = int(gene_bins_local.max()) + 1
    counts       = torch.bincount(sorted_bins, minlength=n_bins)          # (n_bins,)
    starts       = torch.cat([torch.zeros(1, dtype=torch.long),
                              counts.cumsum(0)[:-1]])                     # (n_bins,)

    tgt_bin   = gene_bins_local[g]                # (k,) bin each positive gene sits in
    tgt_count = counts[tgt_bin].clamp(min=1)      # (k,)
    tgt_start = starts[tgt_bin]                   # (k,)

    # n_tries candidates per positive, all drawn from its gene's own degree bin.
    rnd  = torch.rand((k, n_tries), generator=generator)
    offs = (rnd * tgt_count.unsqueeze(1).float()).long()                  # (k, n_tries)
    cand = sorted_genes[tgt_start.unsqueeze(1) + offs]                    # (k, n_tries)

    valid = ~torch.isin(pair_keys(
        torch.stack([m.unsqueeze(1).expand_as(cand).reshape(-1), cand.reshape(-1)]),
        n_gene_local,
    ), true_keys).view(k, n_tries)

    # First valid candidate per row.
    has_valid  = valid.any(dim=1)
    first_valid = valid.float().argmax(dim=1)                             # (k,)
    picked = cand.gather(1, first_valid.unsqueeze(1)).squeeze(1)          # (k,)

    n_fallback = int((~has_valid).sum())
    if n_fallback:
        # Bin exhausted: any gene m does not target. Keeps the column count at k, at the
        # cost of that pair no longer being degree-matched — hence the reported rate.
        fb_idx = (~has_valid).nonzero(as_tuple=True)[0]
        fb_rnd = torch.randint(
            0, n_gene_local, (len(fb_idx), n_tries), generator=generator
        )
        fb_valid = ~torch.isin(pair_keys(
            torch.stack([
                m[fb_idx].unsqueeze(1).expand_as(fb_rnd).reshape(-1),
                fb_rnd.reshape(-1),
            ]),
            n_gene_local,
        ), true_keys).view(len(fb_idx), n_tries)
        fb_first = fb_valid.float().argmax(dim=1)
        picked[fb_idx] = fb_rnd.gather(1, fb_first.unsqueeze(1)).squeeze(1)

    return torch.stack([m, picked]).to(device), n_fallback


# ── Batch-level pair sampling ──────────────────────────────────────────────────

def map_to_local(
    edge_global: torch.Tensor,
    batch,
    device: torch.device,
) -> torch.Tensor | None:
    """
    Remap global miRNA→gene edges to a NeighborLoader batch's local indices via n_id,
    keeping only edges with both endpoints in the batch. Returns None if none survive.
    """
    mirna_g = batch["miRNA"].n_id.to(device)
    gene_g  = batch["gene"].n_id.to(device)

    n_mirna_g = max(int(edge_global[0].max()), int(mirna_g.max())) + 1
    n_gene_g  = max(int(edge_global[1].max()), int(gene_g.max())) + 1

    mirna_g2l = torch.full((n_mirna_g,), -1, dtype=torch.long, device=device)
    mirna_g2l[mirna_g] = torch.arange(len(mirna_g), device=device)
    gene_g2l = torch.full((n_gene_g,), -1, dtype=torch.long, device=device)
    gene_g2l[gene_g] = torch.arange(len(gene_g), device=device)

    src = mirna_g2l[edge_global[0].to(device)]
    dst = gene_g2l[edge_global[1].to(device)]
    keep = (src >= 0) & (dst >= 0)
    if int(keep.sum()) == 0:
        return None
    return torch.stack([src[keep], dst[keep]])


class LinkSampler:
    """
    Turns a batch plus a global supervision edge set into (mirna_idx, gene_idx, labels).

    Holds the two things that must be consistent across train and eval: the exclusion set
    (every known positive, so no held-out edge is ever labelled 0) and the degree bins
    used to match negatives to positives.
    """

    def __init__(
        self,
        all_pos_global: torch.Tensor,
        deg: torch.Tensor,
        seed: int,
        hard: bool = True,
        n_bins: int = N_DEGREE_BINS,
    ) -> None:
        self.all_pos = all_pos_global
        self.gene_bins_global = degree_bins(deg, n_bins)
        self.hard = hard
        self.generator = torch.Generator().manual_seed(seed)
        self.n_fallback = 0
        self.n_negatives = 0

    @property
    def fallback_pct(self) -> float:
        return 100.0 * self.n_fallback / max(self.n_negatives, 1)

    def sample(
        self,
        batch,
        sup_global: torch.Tensor,
        device: torch.device,
        max_pairs: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[None, None, None]:
        """Positives are drawn from sup_global only; negatives avoid every known edge."""
        scored_pos = map_to_local(sup_global, batch, device)
        if scored_pos is None:
            return None, None, None

        if scored_pos.shape[1] > max_pairs:
            perm = torch.randperm(
                scored_pos.shape[1], generator=self.generator, device=device
            )[:max_pairs]
            scored_pos = scored_pos[:, perm]

        n_mirna = batch["miRNA"].num_nodes
        n_gene  = batch["gene"].num_nodes
        k = scored_pos.shape[1]

        # Exclude against every true edge among the batch's nodes — not just the
        # supervision subsample, which would let genuine positives be sampled as
        # negatives.
        exclusion = map_to_local(self.all_pos, batch, device)
        if exclusion is None:
            exclusion = scored_pos

        if self.hard:
            gene_bins_local = self.gene_bins_global[batch["gene"].n_id.cpu()]
            neg, n_fb = sample_degree_matched_negatives(
                scored_pos, exclusion, gene_bins_local, n_gene,
                self.generator, device,
            )
            self.n_fallback += n_fb
        else:
            neg = negative_sampling(
                edge_index=exclusion,
                num_nodes=(n_mirna, n_gene),
                num_neg_samples=k,
                method="sparse",
            ).to(device)

        if neg.shape[1] == 0:
            return None, None, None
        self.n_negatives += neg.shape[1]

        mirna_idx = torch.cat([scored_pos[0], neg[0]]).to(device)
        gene_idx  = torch.cat([scored_pos[1], neg[1]]).to(device)
        labels    = torch.cat([
            torch.ones(k, device=device),
            torch.zeros(neg.shape[1], device=device),
        ])
        return mirna_idx, gene_idx, labels
