"""
hybridgnn_rerun.py -- HybridGNN (Ahmad et al. 2026, ahmad2026hybridgnn) rerun with its own released code,
as released and under progressively corrected protocols, on two matrices.

HybridGNN's repository (github.com/mbasharatahmad/HybridGNN-miRNA-disease, commit f441278; no licence,
so it is cloned outside this repo and its cv_mirna_split.py is imported unmodified) does not implement
the protocol its paper describes (read 2026-11-14, BITACORA):

  * paper: PyG RandomLinkSplit 80/10/10, validation set, Optuna on validation AUC, early stopping;
    code: 5-fold KFold over positives (80/20), no validation set;
  * the message-passing graph is built ONCE from the FULL association matrix (prepare_gcn_data),
    so every test edge is in the graph in every fold;
  * node features: GIP similarity of the FULL matrix -> PCA, and MDMF factorisation trained to
    reconstruct the FULL matrix (test positives are targets of 1);
  * the epoch kept is the one with the best TEST AUC (train_and_evaluate_model: "We'll use
    test_mask/test_labels as the validation here");
  * negatives: uniform 1:1 (paper: 20% / 10% ratios);
  * released data (alldata.xlsx, HMDD v4.0) preprocess to 1,911 x 2,360 / 31,461 pairs, not the
    paper's 728 x 175 / 18,738; the paper's own filter is not in the code.

Hyper-parameters: the optimal values of the paper's Table 1 (no tuned-parameter file is released,
and re-running its Optuna would tune on the test set again).

Steps, per fold (their get_cv_folds 'random' mode, their seed):
  1 released       their train_and_evaluate_model unchanged: full-matrix graph and features, best
                   epoch by test AUC
  2 val_selection  same full-matrix graph and features; 1/8 of the fold's training positives and
                   as many training negatives held out as validation; best epoch by validation AUC
  3 no_leak        graph, GIP similarity, PCA and MDMF built from the fold's training positives only
                   (test and validation positives zeroed); best epoch by validation AUC
  4 matched        the step-3 model scored on degree-matched test negatives (same miRNA, a disease in
                   the same training-degree bin; exact sampler with nearest-bin fallback)

Every step is also audited with protocheck (encoder edges = the graph that step used), which gives
the model-free floor on the same test pairs.

Usage:
  python analysis/hybridgnn_rerun.py --repo PATH --dataset hmdd_v4|canonical [--seeds 42 1 2 3 4]
Writes results/comparison/hybridgnn_rerun_<dataset>.json (suffix _smoke when --epochs < 200).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from protocheck.audit import audit  # noqa: E402
from training.splits import degree_bins, sample_degree_matched_negatives_exact  # noqa: E402

# Paper's Table 1, optimal values.
PARAMS = {"hidden_dim": 117, "learning_rate": 0.0076, "dropout": 0.4493, "num_heads": 12,
          "num_layers": 2, "pca_dim": 32, "latent_dim": 42, "out_channels": 22,
          "lambda_reg": 0.0121, "weight_decay": 1.75e-5}
EPOCHS = 200  # their CLI default; early stopping is commented out in their loop


def load_their_module(repo: Path):
    # Their file imports optuna at top level; it is used only by the tuning path we do not run.
    sys.modules.setdefault("optuna", types.ModuleType("optuna"))
    spec = importlib.util.spec_from_file_location("hybridgnn_cv", repo / "cv_mirna_split.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # defines functions only; its __main__ block does not run
    return mod


def set_globals(mod, **kw):
    for k, v in kw.items():
        setattr(mod, k, v)


def logits_all(model, data, n_m):
    model.eval()
    with torch.no_grad():
        out = model(data)
        return (out[:n_m] @ out[n_m:].t()).view(-1)


def auc_on(logits, idx, labels) -> float:
    return float(roc_auc_score(labels, torch.sigmoid(logits[idx]).cpu().numpy()))


def build_features(mod, mat: pd.DataFrame, n_m, n_d):
    """Their feature pipeline (compute_similarity, prepare_gcn_data, MDMF) on a given matrix."""
    sim_m, sim_d = mod.compute_similarity(mat)
    data, _, _ = mod.prepare_gcn_data(mat, sim_m, sim_d, feature_dim=PARAMS["pca_dim"])
    data.n_mirnas = n_m
    mdmf = mod.MDMF(num_mirnas=n_m, num_diseases=n_d, latent_dim=PARAMS["latent_dim"],
                    lambda_reg=PARAMS["lambda_reg"])
    U, V = mod.train_mdmf(mdmf, torch.tensor(mat.values, dtype=torch.float),
                          torch.tensor(sim_m.values, dtype=torch.float),
                          torch.tensor(sim_d.values, dtype=torch.float))
    data.x = torch.cat([data.x, torch.cat([U, V], dim=0).to(mod.device)], dim=1)
    return data


def new_model(mod, data):
    return mod.HybridGCN_GAT(in_channels=data.x.shape[1], hidden_channels=PARAMS["hidden_dim"],
                             out_channels=PARAMS["out_channels"], dropout=PARAMS["dropout"],
                             num_heads=PARAMS["num_heads"], num_layers=PARAMS["num_layers"])


def train_select_on_val(mod, model, data, tr_idx, tr_lab, va_idx, va_lab, epochs, n_m):
    """Their training step (same forward, loss, optimiser), best epoch by VALIDATION AUC."""
    model.to(mod.device)
    opt = torch.optim.Adam(model.parameters(), lr=PARAMS["learning_rate"],
                           weight_decay=PARAMS["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    tr_lab_t = torch.tensor(tr_lab, dtype=torch.float, device=mod.device)
    best, best_state = -np.inf, None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data)
        logits = (out[:n_m] @ out[n_m:].t()).view(-1)
        loss_fn(logits[tr_idx], tr_lab_t).backward()
        opt.step()
        v = auc_on(logits_all(model, data, n_m), va_idx, va_lab)
        if v > best + 1e-6:
            best, best_state = v, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best


def flat_to_pairs(idx, n_d):
    idx = np.asarray(idx)
    return np.vstack([idx // n_d, idx % n_d]).astype(np.int64)


def audit_step(n_m, n_d, train_pos_flat, test_pos_flat, enc_flat, train_neg_flat, eval_neg_flat, auc):
    rep = audit(n_m, n_d, flat_to_pairs(train_pos_flat, n_d), flat_to_pairs(test_pos_flat, n_d),
                encoder_edges=flat_to_pairs(enc_flat, n_d),
                train_neg=flat_to_pairs(train_neg_flat, n_d),
                eval_neg=flat_to_pairs(eval_neg_flat, n_d), model_auroc=auc)
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
    return out


def load_matrix(mod, dataset: str, repo: Path) -> pd.DataFrame:
    if dataset == "hmdd_v4":
        mat, _, _ = mod.preprocess_data(mod.load_data(str(repo / "alldata.xlsx")))
        return mat
    m = np.loadtxt(ROOT / "data/raw/hmdd_survey/nimgsa/matrix.csv", delimiter=",").astype(int)
    mat = pd.DataFrame(m, index=[f"m{i}" for i in range(m.shape[0])],
                       columns=[f"d{j}" for j in range(m.shape[1])])
    mat.index.name, mat.columns.name = "miRNA", "disease"
    return mat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--dataset", required=True, choices=["hmdd_v4", "canonical"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--folds", type=int, default=5, help="run only the first N folds (smoke tests)")
    args = ap.parse_args()

    mod = load_their_module(args.repo)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mat = load_matrix(mod, args.dataset, args.repo)
    n_m, n_d = mat.shape
    Y = mat.values.astype(np.int64)
    all_pos_flat = np.flatnonzero(Y.ravel() == 1)
    all_pos_t = torch.from_numpy(flat_to_pairs(all_pos_flat, n_d)).long()
    print(f"{args.dataset}: {n_m} x {n_d}, {len(all_pos_flat)} associations, device {dev}", flush=True)

    out = {"meta": {"repo_commit": "f441278", "dataset": args.dataset, "shape": [n_m, n_d],
                    "n_associations": int(len(all_pos_flat)), "params": PARAMS, "epochs": args.epochs,
                    "seeds": args.seeds, "device": str(dev), "torch": torch.__version__}, "runs": []}
    t0 = time.time()
    for seed in args.seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        set_globals(mod, seed=seed, device=dev, matrix=mat, n_mirnas=n_m, n_diseases=n_d,
                    epochs=args.epochs, patience=20, output_folder="unused")
        # Step 1 features: full matrix, built once for all folds, as cross_validate_with_best_params.
        data_full = build_features(mod, mat, n_m, n_d)
        folds = mod.get_cv_folds(mat, split_mode="random", n_folds=5)
        for k, (tr_mask, tr_lab, te_mask, te_lab) in enumerate(folds[:args.folds]):
            tr_lab, te_lab = tr_lab.numpy(), te_lab.numpy()
            tr_pos, tr_neg = tr_mask[tr_lab == 1], tr_mask[tr_lab == 0]
            te_pos, te_neg = te_mask[te_lab == 1], te_mask[te_lab == 0]
            rec = {"seed": seed, "fold": k + 1, "n_test_pos": int(len(te_pos))}

            # 1 released: their function, unchanged (best epoch by test AUC)
            model = new_model(mod, data_full)
            auc1 = mod.train_and_evaluate_model(
                model, data_full, train_mask=tr_mask, train_labels=torch.tensor(tr_lab, dtype=torch.float, device=dev),
                test_mask=te_mask, test_labels=torch.tensor(te_lab, dtype=torch.float, device=dev),
                learning_rate=PARAMS["learning_rate"], weight_decay=PARAMS["weight_decay"],
                max_epochs=args.epochs, patience=20)[0]
            rec["released"] = {"auc": auc1, "audit": audit_step(n_m, n_d, tr_pos, te_pos, all_pos_flat,
                                                                tr_neg, te_neg, auc1)}

            # validation carve-out from the fold's training pairs (same for steps 2 and 3)
            rng = np.random.default_rng(10_000 + 100 * seed + k)
            va_pos = rng.choice(tr_pos, size=len(tr_pos) // 8, replace=False)
            va_neg = rng.choice(tr_neg, size=len(va_pos), replace=False)
            fit_pos, fit_neg = np.setdiff1d(tr_pos, va_pos), np.setdiff1d(tr_neg, va_neg)
            fit_idx = np.concatenate([fit_pos, fit_neg])
            fit_lab = np.r_[np.ones(len(fit_pos)), np.zeros(len(fit_neg))]
            va_idx = np.concatenate([va_pos, va_neg])
            va_lab = np.r_[np.ones(len(va_pos)), np.zeros(len(va_neg))]

            # 2 val_selection: full-matrix graph and features, best epoch by validation AUC
            model = new_model(mod, data_full)
            v2 = train_select_on_val(mod, model, data_full, fit_idx, fit_lab, va_idx, va_lab, args.epochs, n_m)
            auc2 = auc_on(logits_all(model, data_full, n_m), te_mask, te_lab)
            rec["val_selection"] = {"auc": auc2, "best_val_auc": v2,
                                    "audit": audit_step(n_m, n_d, fit_pos, te_pos, all_pos_flat,
                                                        fit_neg, te_neg, auc2)}

            # 3 no_leak: everything built from the fold's fitting positives only
            Ytr = Y.copy().ravel()
            Ytr[np.concatenate([te_pos, va_pos])] = 0
            mat_tr = pd.DataFrame(Ytr.reshape(n_m, n_d), index=mat.index, columns=mat.columns)
            mat_tr.index.name, mat_tr.columns.name = "miRNA", "disease"
            data_tr = build_features(mod, mat_tr, n_m, n_d)
            model = new_model(mod, data_tr)
            v3 = train_select_on_val(mod, model, data_tr, fit_idx, fit_lab, va_idx, va_lab, args.epochs, n_m)
            logits3 = logits_all(model, data_tr, n_m)
            auc3 = auc_on(logits3, te_mask, te_lab)
            rec["no_leak"] = {"auc": auc3, "best_val_auc": v3,
                              "audit": audit_step(n_m, n_d, fit_pos, te_pos, fit_pos, fit_neg, te_neg, auc3)}

            # 4 matched: the step-3 model on degree-matched test negatives
            col_deg = torch.from_numpy(Ytr.reshape(n_m, n_d).sum(0)).long()
            neg, n_unif, n_nb = sample_degree_matched_negatives_exact(
                torch.from_numpy(flat_to_pairs(te_pos, n_d)).long(), all_pos_t, degree_bins(col_deg),
                n_d, torch.Generator().manual_seed(1000 * seed + k))
            neg_flat = (neg[0] * n_d + neg[1]).numpy()
            m_idx = np.concatenate([te_pos, neg_flat])
            m_lab = np.r_[np.ones(len(te_pos)), np.zeros(len(neg_flat))]
            auc4 = auc_on(logits3, m_idx, m_lab)
            rec["matched"] = {"auc": auc4, "uniform_fallback": n_unif, "neighbour_fallback": n_nb,
                              "audit": audit_step(n_m, n_d, fit_pos, te_pos, fit_pos, fit_neg, neg_flat, auc4)}
            out["runs"].append(rec)
            fl = {s: rec[s]["audit"]["model_free_baseline"]["floor"] for s in ("released", "val_selection", "no_leak", "matched")}
            print(f"seed {seed} fold {k + 1}: released {auc1:.4f} (floor {fl['released']:.4f}) | "
                  f"val_sel {auc2:.4f} | no_leak {auc3:.4f} (floor {fl['no_leak']:.4f}) | "
                  f"matched {auc4:.4f} (floor {fl['matched']:.4f}) | fallback u{n_unif}/n{n_nb} "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    summ = {}
    for step in ("released", "val_selection", "no_leak", "matched"):
        per_seed = [st.mean(r[step]["auc"] for r in out["runs"] if r["seed"] == s) for s in args.seeds]
        fl_seed = [st.mean(r[step]["audit"]["model_free_baseline"]["floor"] for r in out["runs"] if r["seed"] == s)
                   for s in args.seeds]
        summ[step] = {"auc_mean": st.mean(per_seed), "auc_sd": st.stdev(per_seed) if len(per_seed) > 1 else 0.0,
                      "floor_mean": st.mean(fl_seed), "per_seed": per_seed}
    out["summary"] = summ
    print("\n=== summary (mean over folds, then seeds) ===")
    for k, v in summ.items():
        print(f"  {k:14} AUC {v['auc_mean']:.4f} +- {v['auc_sd']:.4f}   floor {v['floor_mean']:.4f}")
    tag = "" if args.epochs >= EPOCHS else "_smoke"
    path = ROOT / f"results/comparison/hybridgnn_rerun_{args.dataset}{tag}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print("Wrote", path)


if __name__ == "__main__":
    main()
