"""
hgdti_rerun.py -- HGDTI (Yu et al. 2022, yu2022hgdti) rerun with its own released code, as released and
under progressively corrected protocols.

HGDTI's materials (Google Drive folder linked from the paper; no licence, kept outside this repo at
external_code/HGDTI) ship src/HGDTI.py, args.py, the DTINet-derived data and the model module only as
compiled files (tools.so, tools.cpython-3x.pyc). tools.py is RECONSTRUCTED from tools.cpython-37.pyc
with decompyle3 (sha256 da4e0532b9494fb4..., unmodified apart from a header); HGDTI.py is imported
unmodified. What the code does (read 2026-11-14, BITACORA):

  * evaluation passes the model a drug-protein adjacency built FROM THE EVALUATED PAIRS' LABELS:
    model(DTItest, *DTI_normalize(DTItest)) puts every test positive in the graph the drug and
    protein embeddings aggregate over (node_het_agg: d_p_agg = drug_protein_normalize @ protein_agg);
    the validation pass does the same with the validation labels;
  * "reliable" negatives: the half of all unknown pairs least similar to the known interactions,
    computed from ALL known DTIs (test positives included), then 10 per positive;
  * 72% of proteins (1,088 / 1,512) have no known DTI;
  * 10-fold StratifiedKFold over pairs, 10% of training as validation, epoch by validation AUPR.

Steps, per fold (their sample_neg_links / sample_links / StratifiedKFold / train_test_split, in their
RNG order; their hyper-parameters):
  1 released      their train_and_evaluate unchanged (validation and test adjacency from their own labels)
  2 train_graph   same splits and negatives; every forward pass uses the TRAINING adjacency; epoch by
                  validation AUPR as theirs
  3 uniform       the step-2 model on uniform test negatives (unknown pairs outside the training and
                  validation sets, 10 per positive, as their ratio)
  4 matched       the step-2 model on degree-matched test negatives (same drug, a protein in the same
                  training-degree bin; exact sampler with nearest-bin fallback; 1 per positive)

Each step is audited with protocheck 0.2.0 (encoder edges = the adjacency the test pass used).

Usage: python analysis/hgdti_rerun.py --repo PATH [--seeds 10 11 12] [--epochs 1000]
Writes results/comparison/hgdti_rerun_s<seeds>.json (suffix _smoke when --epochs < 1000 or --folds < 10).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from protocheck.audit import audit  # noqa: E402
from training.splits import degree_bins, sample_degree_matched_negatives_exact  # noqa: E402

D_N, P_N = 708, 1512


def load_their_module(repo: Path, cuda: bool):
    src = repo / "src"
    # tools.py calls read_args() at import time, so argv must already carry the data path.
    sys.argv = [sys.argv[0], "--data_path", str(repo / "data") + "/", "--result_path", "/tmp/",
                "--model_path", "/tmp/", "--cuda", "1" if cuda else "0"]
    sys.path.insert(0, str(src))
    spec = importlib.util.spec_from_file_location("hgdti_main", src / "HGDTI.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # defines model_class; its __main__ block does not run
    mod.args = mod.read_args()     # their methods read the global `args`
    return mod


def pairs(arr):
    return np.asarray(arr)[:, :2].T.astype(np.int64)


def run_model(mod, mo, model, data, adj):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(data, *adj)).cpu().numpy()


def train_with_train_graph(mod, mo, DTItrain, DTIvalid, epochs):
    """Their train_and_evaluate loop (same model, optimiser, loss, schedule and early stopping), but
    the validation pass uses the TRAINING adjacency; returns the model at the best validation AUPR."""
    a = mod.args
    model = mod.tools.HetAgg(a, mo.drug_drug_normalize, mo.drug_chemical_normalize,
                             mo.drug_disease_normalize, mo.drug_sideeffect_normalize,
                             mo.protein_protein_normalize, mo.protein_sequence_normalize,
                             mo.protein_disease_normalize)
    model.init_weights()
    if mo.gpu:
        model.cuda()
    opt = optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    adj_tr = mo.DTI_normalize(DTItrain)
    best_aupr, best_state, bad = 0.0, None, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(DTItrain, *adj_tr)
        loss = mod.tools.binary_cross_entropy_loss(mo.gpu, out, torch.from_numpy(DTItrain[:, 2]).float())
        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                v_auc, v_aupr = mo.validation(model(DTIvalid, *adj_tr), DTIvalid[:, 2])
            if v_aupr >= best_aupr:
                best_aupr, bad = v_aupr, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    return model, adj_tr, best_aupr


def auc(scores, labels):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def audit_step(train_pos, test_pos, enc, train_neg, eval_neg, model_auc):
    rep = audit(D_N, P_N, train_pos, test_pos, encoder_edges=enc, train_neg=train_neg,
                eval_neg=eval_neg, model_auroc=model_auc)
    out = {}
    for f in rep.findings:
        ev = f.evidence or {}
        out[f.check] = {"status": f.status, "message": f.message}
        if f.check == "model_free_baseline":
            out[f.check].update(floor=ev.get("reference_floor"), heuristic=ev.get("reference_heuristic"),
                                floor_corrected=ev.get("floor_corrected", {}).get("floor"),
                                margin=ev.get("margin"))
        if f.check == "edge_leakage":
            out[f.check]["fraction_heldout_in_encoder"] = ev.get("fraction_heldout_in_encoder")
        if f.check == "negative_matching":
            out[f.check]["tv_eval_neg_vs_heldout_pos"] = ev.get("tv_eval_neg_vs_heldout_pos")
        if f.check == "dead_candidates":
            out[f.check]["eval_negatives_on_dead_columns"] = ev.get("eval_negatives_on_dead_columns")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=10, help="run only the first N folds (smoke tests)")
    a_cli = ap.parse_args()
    argv0 = sys.argv[:]
    cuda = torch.cuda.is_available()
    mod = load_their_module(a_cli.repo, cuda)
    sys.argv = argv0
    mod.args.epochs = a_cli.epochs
    from sklearn.model_selection import StratifiedKFold, train_test_split

    Y = np.loadtxt(a_cli.repo / "data" / "mat_drug_protein.txt")
    all_pos = torch.from_numpy(np.argwhere(Y == 1).T.copy()).long()
    out = {"meta": {"hgdti_tools": "reconstructed from tools.cpython-37.pyc (sha256 da4e0532b9494fb4)",
                    "epochs": a_cli.epochs, "seeds": a_cli.seeds, "cuda": cuda,
                    "n_dti": int(Y.sum()), "proteins_without_dti": int((Y.sum(0) == 0).sum())},
           "runs": []}
    t0 = time.time()
    for seed in a_cli.seeds:
        # their __main__: seeds, then model_class (draws its own seeds), then model_train's order
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        mo = mod.model_class(mod.args)
        wp, wn, ns, wpt, wnt, nst = mo.sample_neg_links(0.5)
        data_set, _ = mo.sample_links(wp, wn, ns, wpt, wnt, nst)
        kf = StratifiedKFold(n_splits=10, shuffle=True)
        folds = []
        for tr_i, te_i in kf.split(data_set[:, :2], data_set[:, 2]):
            DTItrain, DTItest = data_set[tr_i], data_set[te_i]
            DTItrain, DTIvalid = train_test_split(DTItrain, test_size=0.1)
            folds.append((DTItrain, DTIvalid, DTItest))
        for k, (DTItrain, DTIvalid, DTItest) in enumerate(folds[:a_cli.folds]):
            tr_pos, tr_neg = pairs(DTItrain[DTItrain[:, 2] == 1]), pairs(DTItrain[DTItrain[:, 2] == 0])
            te_pos = pairs(DTItest[DTItest[:, 2] == 1])
            te_neg = pairs(DTItest[DTItest[:, 2] == 0])
            rec = {"seed": seed, "fold": k + 1, "n_test_pos": int(te_pos.shape[1]),
                   "n_test_neg": int(te_neg.shape[1])}

            # 1 released: their function, unchanged
            auc1, aupr1 = mo.train_and_evaluate(DTItrain=DTItrain, DTIvalid=DTIvalid, DTItest=DTItest,
                                                num_step=k + 1)
            rec["released"] = {"auc": float(auc1), "aupr": float(aupr1),
                               "audit": audit_step(tr_pos, te_pos, te_pos, tr_neg, te_neg, float(auc1))}

            # 2 train_graph: every pass on the training adjacency
            model, adj_tr, _ = train_with_train_graph(mod, mo, DTItrain, DTIvalid, a_cli.epochs)
            s2 = run_model(mod, mo, model, DTItest, adj_tr)
            auc2 = auc(s2, DTItest[:, 2])
            rec["train_graph"] = {"auc": auc2, "audit": audit_step(tr_pos, te_pos, tr_pos, tr_neg, te_neg, auc2)}

            # 3 uniform test negatives (outside train and validation pairs), 10 per positive
            used = set(map(tuple, np.vstack([DTItrain[:, :2], DTIvalid[:, :2]]).tolist()))
            cand = np.argwhere(Y == 0)
            cand = cand[[tuple(c) not in used for c in cand.tolist()]]
            rng = np.random.default_rng(100 * seed + k)
            un = cand[rng.choice(len(cand), size=10 * te_pos.shape[1], replace=False)]
            test3 = np.vstack([np.c_[te_pos.T, np.ones(te_pos.shape[1], int)],
                               np.c_[un, np.zeros(len(un), int)]]).astype(int)
            s3 = run_model(mod, mo, model, test3, adj_tr)
            auc3 = auc(s3, test3[:, 2])
            rec["uniform"] = {"auc": auc3, "audit": audit_step(tr_pos, te_pos, tr_pos, tr_neg, un.T, auc3)}

            # 4 degree-matched test negatives (protein degree from the training positives)
            col_deg = torch.from_numpy(np.bincount(tr_pos[1], minlength=P_N)).long()
            neg, n_unif, n_nb = sample_degree_matched_negatives_exact(
                torch.from_numpy(te_pos).long(), all_pos, degree_bins(col_deg), P_N,
                torch.Generator().manual_seed(1000 * seed + k))
            neg = neg.numpy()
            test4 = np.vstack([np.c_[te_pos.T, np.ones(te_pos.shape[1], int)],
                               np.c_[neg.T, np.zeros(neg.shape[1], int)]]).astype(int)
            s4 = run_model(mod, mo, model, test4, adj_tr)
            auc4 = auc(s4, test4[:, 2])
            rec["matched"] = {"auc": auc4, "uniform_fallback": n_unif, "neighbour_fallback": n_nb,
                              "audit": audit_step(tr_pos, te_pos, tr_pos, tr_neg, neg, auc4)}
            out["runs"].append(rec)
            fl = {s: rec[s]["audit"]["model_free_baseline"]["floor"] for s in ("released", "train_graph", "uniform", "matched")}
            print(f"seed {seed} fold {k + 1}: released {auc1:.4f} (floor {fl['released']:.4f}) | "
                  f"train_graph {auc2:.4f} (floor {fl['train_graph']:.4f}) | uniform {auc3:.4f} "
                  f"(floor {fl['uniform']:.4f}) | matched {auc4:.4f} (floor {fl['matched']:.4f}) | "
                  f"fallback u{n_unif}/n{n_nb} [{time.time() - t0:.0f}s]", flush=True)

    summ = {}
    for step in ("released", "train_graph", "uniform", "matched"):
        per_seed = [st.mean(r[step]["auc"] for r in out["runs"] if r["seed"] == s) for s in a_cli.seeds]
        fl = [st.mean(r[step]["audit"]["model_free_baseline"]["floor"] for r in out["runs"] if r["seed"] == s)
              for s in a_cli.seeds]
        summ[step] = {"auc_mean": st.mean(per_seed), "auc_sd": st.stdev(per_seed) if len(per_seed) > 1 else 0.0,
                      "floor_mean": st.mean(fl), "per_seed": per_seed}
    out["summary"] = summ
    print("\n=== summary (mean over folds, then seeds) ===")
    for k, v in summ.items():
        print(f"  {k:12} AUC {v['auc_mean']:.4f} +- {v['auc_sd']:.4f}   floor {v['floor_mean']:.4f}")
    tag = "" if (a_cli.epochs >= 1000 and a_cli.folds >= 10) else "_smoke"
    tag = "_s" + "-".join(map(str, a_cli.seeds)) + tag  # one job per seed; merged afterwards
    path = ROOT / f"results/comparison/hgdti_rerun{tag}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print("Wrote", path)


if __name__ == "__main__":
    main()
