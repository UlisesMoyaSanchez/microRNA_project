"""
build_survey_bipartite_graph.py — A surveyed paper's association matrix as a HeteroData
graph this project's own model can train on.

Why this exists: the HMDD audit so far compares each paper's *reported* number against
our model-free floor. That leaves one question open -- does a trained model move across
the protocol grid the same way the floor does? Answering it needs a trained model on
their graph, and the papers publish a bare binary association matrix: no node features,
no cells, nothing but edges.

Two deliberate decisions, both worth stating so nobody later reads them as bugs:

1. **Disease nodes are stored under the node type `gene`.** The type names are hardcoded
   throughout `training/splits.py` (REL_FWD/REL_REV, `gene_in_degree`, the degree-matched
   sampler) and in the model's link head (`models/hetero_gnn.py`). Structurally a
   miRNA-disease matrix *is* the same bipartite graph as a miRNA-gene one, so reusing the
   names lets the entire split, leakage-check and negative-sampling machinery run
   unmodified -- which is the point, since that machinery is what the comparison is
   about. Renaming would mean forking five call sites and re-validating each. The column
   axis here is diseases; only the label says `gene`.

2. **Node features are random-normal**, the same recipe `build_heterograph.py`
   already uses for miRNA nodes (`build_mirna_features`), treated as learned embeddings.
   The paper's data carries no features, so this is not a simplification of their setup;
   it is their setup.

No `cell` nodes are created. `miRNAGraphTransformer.forward` already guards its
classification head with `if "cell" in h`, so the head is simply never invoked, and
`training/train_survey_bipartite.py` trains full-batch rather than seeding batches from
cell nodes the way `train.py` does.

Usage:
  python data/03_build_graph/build_survey_bipartite_graph.py --graph meahne
  python data/03_build_graph/build_survey_bipartite_graph.py --graph couplemda
"""

from __future__ import annotations

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# The graph definitions come from analysis/density_sweep.py rather than a second copy
# here: it already keys by DISTINCT GRAPH (so the canonical HMDD matrix shared by MGCNSS,
# NIMGSA and HLGNN-MDA appears once, not three times) and already handles CoupleMDA, which
# publishes pair lists instead of a matrix. Two sources of truth for "which graphs are
# there" is exactly how the three-papers-one-matrix duplication went unnoticed before.
from analysis.density_sweep import GRAPHS, load_positives

FEATURE_SEED = 42


def build_features(n: int, dim: int) -> torch.Tensor:
    """Same recipe as build_heterograph.build_mirna_features: random normal, treated as
    a learned embedding during training. Seeded so a rebuild is reproducible."""
    g = torch.Generator().manual_seed(FEATURE_SEED)
    return torch.randn(n, dim, generator=g)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", required=True, choices=list(GRAPHS),
                   help="A distinct graph, not a paper: canonical5430 covers MGCNSS, "
                        "NIMGSA and HLGNN-MDA, which share one byte-identical matrix.")
    p.add_argument("--out-root", default="data/graphs_survey")
    p.add_argument("--init-dim", type=int, default=64,
                   help="Width of the random-normal node embeddings, both sides.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    spec = GRAPHS[args.graph]
    edges, n_mirna, n_disease = load_positives(spec)
    edges = edges.contiguous()
    log.info(f"{args.graph}: {n_mirna} miRNA x {n_disease} diseases, "
             f"{edges.shape[1]:,} associations "
             f"(density {edges.shape[1] / (n_mirna * n_disease):.5f}) "
             f"[{', '.join(spec['papers'])}]")

    data = HeteroData()
    data["miRNA"].x = build_features(n_mirna, args.init_dim)
    data["gene"].x = build_features(n_disease, args.init_dim)   # diseases; see docstring
    data["miRNA", "regulates", "gene"].edge_index = edges
    data["gene", "regulated_by", "miRNA"].edge_index = edges.flip(0)

    out_dir = Path(args.out_root) / args.graph
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_dir / "hetero_graph.pt")

    # index_maps is written for parity with data/graphs_v3fixed. cell_type_labels is
    # empty on purpose: there are no cells, and the trainer for these graphs never reads
    # it -- but a downstream tool that does will fail loudly rather than silently.
    with open(out_dir / "index_maps.pkl", "wb") as fh:
        pickle.dump({"cell_type_labels": [], "n_mirna": n_mirna,
                     "n_disease": n_disease}, fh)

    manifest = {
        "graph": args.graph,
        "papers": spec["papers"],
        "source": spec.get("matrix") or spec.get("pair_lists"),
        "n_mirna": int(n_mirna),
        "n_disease": int(n_disease),
        "n_associations": int(edges.shape[1]),
        "density": float(edges.shape[1] / (n_mirna * n_disease)),
        "init_dim": args.init_dim,
        "feature_seed": FEATURE_SEED,
        "disease_nodes_stored_as": "gene",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / "graph_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    log.info(f"Wrote {out_dir}/hetero_graph.pt, index_maps.pkl, graph_manifest.json")
    log.info(f"Graph: {data}")


if __name__ == "__main__":
    main()
