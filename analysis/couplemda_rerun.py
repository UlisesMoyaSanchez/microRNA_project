"""
couplemda_rerun.py -- CoupleMDA's own released code (li2025couplemda, miRNA-disease), as released and with
its test negatives changed one way at a time.

CoupleMDA (github.com/lizhj39/CoupleMDA @ c9128cb, no licence; data: Zou et al.'s heterogeneous graph,
shipped in the repository) reports AUROC 95.36 +- 0.35. What its released code does:

  * a fixed test set (data/Zou/link.dat.test, 1,528 miRNA-disease edges) that never enters the graph;
    ~11% of the remaining miRNA-disease edges are moved to validation and deleted from the graph; the
    metapath instances are enumerated on that training graph; early stopping on validation AUROC;
  * negatives, for training, validation and test alike: for each positive (miRNA, disease) the DISEASE
    is kept and the miRNA replaced by a random one not linked to it (data_loader.get_train_valid_neg,
    get_test_neigh_w_random). The paper says negatives "were sampled from node pairs that did not have
    any existing edges". Keeping the disease matches disease popularity by construction; miRNA
    popularity is left free.

So the edge split is clean by reading; this rerun measures what the test negatives contribute. The model
is trained once per seed exactly as released (run_Zou.run_model_Zou, train mode), and the selected
checkpoint is scored on the same 1,528 test positives against:
  theirs            their own sampler (the released number)
  uniform           random miRNA and random disease, 1:1 (the paper's wording)
  matched_mirna     disease kept, miRNA drawn from the same training-degree bin (exact sampler)
  matched_disease   miRNA kept, disease drawn from the same training-degree bin (our usual convention)
protocheck 0.2.0 audits each set in both orientations (miRNA x disease and disease x miRNA, so that the
column-degree heuristic covers either side); the floor reported is the higher of the two.

Seeds: their data_loader.set_seed(5555) fixes the validation split and every negative; other seeds are
passed through that same function. Runs on CPU (DGL's CUDA wheels are not reachable from the DGX).

Usage: python analysis/couplemda_rerun.py --repo PATH [--seeds 5555 1 2] [--epochs 100]
Writes results/comparison/couplemda_rerun.json (suffix _smoke when --epochs < 100).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics as st
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from protocheck.audit import audit  # noqa: E402
from training.splits import degree_bins, sample_degree_matched_negatives_exact  # noqa: E402

MD = 12   # miRNA-disease relation id in their info.dat


def audit_both(tr, te, trn, neg, auc_v, n_m, n_d):
    """tr/te/trn/neg: (k, 2) local (miRNA, disease) pairs. Audit as miRNA x disease and disease x miRNA."""
    out = {}
    for name, (rows, cols, f) in {"mirna_x_disease": (n_m, n_d, lambda a: a.T),
                                  "disease_x_mirna": (n_d, n_m, lambda a: a[:, ::-1].T)}.items():
        rep = audit(rows, cols, f(tr), f(te), encoder_edges=f(tr), train_neg=f(trn), eval_neg=f(neg),
                    model_auroc=auc_v)
        o = {}
        for fi in rep.findings:
            ev = fi.evidence or {}
            o[fi.check] = {"status": fi.status, "message": fi.message}
            if fi.check == "model_free_baseline":
                o[fi.check].update(floor=ev.get("reference_floor"), heuristic=ev.get("reference_heuristic"),
                                   margin=ev.get("margin"))
        out[name] = o
    best = max(out, key=lambda k: out[k]["model_free_baseline"]["floor"])
    out["floor"] = out[best]["model_free_baseline"]["floor"]
    out["floor_heuristic"] = f"{out[best]['model_free_baseline']['heuristic']} ({best})"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[5555, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--work", type=Path, default=ROOT / "results/couplemda_work")
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    mdir = a.repo / "methods" / "CoupleMDA"
    os.chdir(mdir)                       # their paths are relative ('../../data/Zou')
    sys.path[:0] = [str(mdir), str(a.repo)]
    import utils                         # their package; com_reminder (an e-mail notifier) is not released
    stub = types.ModuleType("utils.com_reminder")
    stub.send_email = lambda *x, **k: None
    sys.modules["utils.com_reminder"] = stub
    utils.com_reminder = stub
    # The released tree does not import as published: model/__init__.py imports MAGNN_nc and MAGNN_nc_mb,
    # which are not in the repository (node-classification variants, unused here), and MAGNN_lp.py imports
    # its helpers from methods.CoupleGNN.utils.tools, the folder's name before it was renamed CoupleMDA.
    # Both are bridged in sys.modules; no file of theirs is edited.
    for name, cls in (("model.MAGNN_nc", "MAGNN_nc"), ("model.MAGNN_nc_mb", "MAGNN_nc_mb")):
        m = types.ModuleType(name)
        setattr(m, cls, None)
        sys.modules[name] = m
    import utils.tools as UT
    for name in ("methods", "methods.CoupleGNN", "methods.CoupleGNN.utils"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["methods.CoupleGNN.utils.tools"] = UT
    import scripts.data_loader as DLM
    from model.MAGNN_lp import MAGNN_lp
    from utils.pytorchtools import datalodaer_gen
    spec = importlib.util.spec_from_file_location("run_Zou", mdir / "run_Zou.py")
    RZ = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(RZ)          # defines run_model_Zou and its parameters; __main__ does not run

    out = {"meta": {"repo_commit": "c9128cb", "seeds": a.seeds, "epochs": a.epochs, "device": "cpu",
                    "reported_auroc": 0.9536}, "runs": []}
    captured = {}

    def test_hook(self, dl, batch_size, datas, param, test_edge_type=0):
        """Replaces MAGNN_lp.test: draws their test negatives first (same RNG state as their test()),
        then scores every negative set with their evaluate()."""
        test_neigh, test_label = dl.get_test_neigh_w_random()
        X = np.array(test_neigh[test_edge_type]).T
        y = np.array(test_label[test_edge_type])
        P, Nt = X[y == 1], X[y == 0]
        m0, m1 = dl.nodes["shift"][0], dl.nodes["shift"][1]
        n_m, n_d = dl.nodes["count"][0], dl.nodes["count"][1]
        loc = lambda E: np.c_[E[:, 0] - m0, E[:, 1] - m1]  # noqa: E731
        tr = loc(np.array(dl.train_pos[MD]).T)
        trn = loc(np.array(dl.train_neg[MD]).T)
        allp = set(map(tuple, loc(np.array(dl.ori_links["data"][MD].nonzero()).T).tolist())) | \
            set(map(tuple, loc(np.array(dl.valid_pos[MD]).T).tolist()))
        allp_arr = np.array(sorted(allp))
        Pl = loc(P)
        rng = np.random.default_rng(captured["seed"])
        uni = []
        seen = set()
        while len(uni) < len(Pl):
            c = (int(rng.integers(n_m)), int(rng.integers(n_d)))
            if c not in allp and c not in seen:
                seen.add(c), uni.append(c)
        uni = np.array(uni)
        g = torch.Generator().manual_seed(captured["seed"])
        deg_m = torch.from_numpy(np.bincount(tr[:, 0], minlength=n_m)).long()
        deg_d = torch.from_numpy(np.bincount(tr[:, 1], minlength=n_d)).long()
        mm = sample_degree_matched_negatives_exact(torch.from_numpy(Pl[:, ::-1].T.copy()).long(),
                                                   torch.from_numpy(allp_arr[:, ::-1].T.copy()).long(),
                                                   degree_bins(deg_m), n_m, g)[0].numpy().T[:, ::-1]
        md = sample_degree_matched_negatives_exact(torch.from_numpy(Pl.T.copy()).long(),
                                                   torch.from_numpy(allp_arr.T.copy()).long(),
                                                   degree_bins(deg_d), n_d, g)[0].numpy().T
        sets = {"theirs": loc(Nt), "uniform": uni, "matched_mirna": mm.copy(), "matched_disease": md}
        res = {"n_test_pos": int(len(P))}
        for name, neg in sets.items():
            negg = np.c_[neg[:, 0] + m0, neg[:, 1] + m1]
            loader = datalodaer_gen(P, negg, batch_size=batch_size, shuffle=False)
            _, r = self.evaluate(loader, datas, param)
            v = float(r["roc_auc"])
            res[name] = {"auc": v, "ap": float(r["AP"]), "audit": audit_both(tr, Pl, trn, neg, v, n_m, n_d)}
            print(f"  {name:16s} AUROC {v:.4f}  floor {res[name]['audit']['floor']:.4f} "
                  f"({res[name]['audit']['floor_heuristic']})", flush=True)
        captured["res"] = res

    MAGNN_lp.test = test_hook
    t0 = time.time()
    for seed in a.seeds:
        DLM.set_seed.__defaults__ = (seed,)     # their data_loader calls set_seed() with no argument
        # load_Zou_data caches the metapath instances of the first training graph it sees; another seed's
        # validation split differs, so the cache is rebuilt for every seed
        for pkl in ("adj_and_idx1.pkl", "adj2.pkl"):
            (a.repo / "data" / "Zou" / pkl).unlink(missing_ok=True)
        captured["seed"] = seed
        RZ.checkpoint_path = str(a.work / f"checkpoint_seed{seed}.pt")
        RZ.logfile_path = str(a.work / f"output_Zou_seed{seed}.txt")
        RZ.p_data_path = str(a.work / f"p_data_Zou_seed{seed}.csv")
        print(f"=== seed {seed}", flush=True)
        RZ.run_model_Zou(0, 64, 8, 128, "linear", a.epochs, 30, 256, 500, 1, "Zou", "train")
        out["runs"].append({"seed": seed, **captured["res"]})
        print(f"seed {seed} done [{time.time() - t0:.0f}s]", flush=True)

    summ = {}
    for name in ("theirs", "uniform", "matched_mirna", "matched_disease"):
        v = [r[name]["auc"] for r in out["runs"]]
        f = [r[name]["audit"]["floor"] for r in out["runs"]]
        summ[name] = {"auc": st.mean(v), "sd": st.stdev(v) if len(v) > 1 else 0.0, "floor": st.mean(f),
                      "floor_heuristic": out["runs"][0][name]["audit"]["floor_heuristic"]}
    out["summary"] = summ
    print("\n=== summary ===\n", json.dumps(summ, indent=1))
    path = ROOT / ("results/comparison/couplemda_rerun" + ("" if a.epochs >= 100 else "_smoke") + ".json")
    path.write_text(json.dumps(out, indent=2, default=float))
    print("Wrote", path)


if __name__ == "__main__":
    main()
