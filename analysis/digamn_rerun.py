"""
digamn_rerun.py -- DiGAMN's own released code (liu2024digamn, miRNA-disease), as released and corrected step by
step.

DiGAMN (github.com/yinboliu-git/GAMN @ 6695f4e, no licence; data in the repository) reports AUC 0.9635 on its
Data1 (folder MDA-CF: 917 miRNAs x 792 diseases, 14,550 associations; the matrix of this paper's E3) and says
"We masked 20% of the association information during cross-validation in the test set to prevent information
leakage". What its released code does (main_bio.py, data_utils.py, train_model.py):

  * one graph of every miRNA-disease association in both directions, plus miRNA and disease K-means cluster
    nodes (data5/, sequence and semantic similarity, not association-derived) and self-loops;
  * negatives: as many zero pairs as positives, drawn once with torch.randint before any seed is set;
    5-fold KFold (random_state 100) over positives and negatives together;
  * "edge removal": for each test positive, graph.remove_edges((src, tgt)). DGL's remove_edges takes EDGE IDS,
    so this deletes the two edges whose ids equal the two node indices, not the test edge: about 40% of the
    miRNA->disease edges are deleted from the front of the list, test and training alike, and every test
    positive's disease->miRNA edge stays (verified with DGL 2.1.0; BITACORA 2026-11-16);
  * full-batch training, Adam lr 1e-3, weight decay 2e-4, up to 5,000 epochs, test metrics every 20 epochs;
    the script reports the checkpoint with the best fold-mean test AUPR (main_bio.mul_func), after a grid of
    448 hyper-parameter settings on the same folds.

The repository does not import as published (data_utils/model/train_model import `global_args`; the file is
`globel_args.py`, which also pins CUDA device 0): a module `global_args` with device = cpu is provided in
memory. Their model class (model.DiGAMN_Model) and graph builder (data_utils.read_data) run unmodified; the
training loop is theirs line for line, with the removal step and checkpoint choice made switchable.

Steps (per seed, five folds, the paper's case-study setting: 4 GAMN layers, 4 attention layers, 8 hidden
channels, memory 8):
  1 released     their removal (node ids passed as edge ids); test-selected checkpoint (and the last one)
  2 removed      every test positive's edge removed in both directions; test-selected checkpoint (and last)
  3 validation   step 2, plus one eighth of the training pairs held out for validation (their positive edges
                 also removed); checkpoint by validation AUPR, the paper's own selection metric
  4 matched      the step-3 checkpoint scored on degree-matched test negatives (miRNA kept, disease from the
                 same training-degree bin; exact sampler)
protocheck 0.2.0 audits steps 1-4 in both orientations; the floor reported is the higher of the two.

Usage: python analysis/digamn_rerun.py --repo PATH [--seeds 521 1 2] [--epochs 5000]
Writes results/comparison/digamn_rerun_s<seeds>.json (suffix _smoke when --epochs < 5000).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import statistics as st
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from protocheck.audit import audit  # noqa: E402
from training.splits import degree_bins, sample_degree_matched_negatives_exact  # noqa: E402

EVAL_EVERY = 20


def audit_both(tr, te, trn, neg, enc, auc_v, n_m, n_d):
    """(k, 2) local (miRNA, disease) arrays. Audit as miRNA x disease and disease x miRNA; keep the higher floor."""
    out = {}
    for name, (rows, cols, f) in {"mirna_x_disease": (n_m, n_d, lambda a: a.T),
                                  "disease_x_mirna": (n_d, n_m, lambda a: a[:, ::-1].T)}.items():
        rep = audit(rows, cols, f(tr), f(te), encoder_edges=f(enc), train_neg=f(trn), eval_neg=f(neg),
                    model_auroc=auc_v)
        o = {}
        for fi in rep.findings:
            ev = fi.evidence or {}
            o[fi.check] = {"status": fi.status, "message": fi.message}
            if fi.check == "model_free_baseline":
                o[fi.check].update(floor=ev.get("reference_floor"), heuristic=ev.get("reference_heuristic"),
                                   margin=ev.get("margin"))
        out[name] = o
    best = max(("mirna_x_disease", "disease_x_mirna"), key=lambda k: out[k]["model_free_baseline"]["floor"])
    out["floor"] = out[best]["model_free_baseline"]["floor"]
    out["floor_heuristic"] = f"{out[best]['model_free_baseline']['heuristic']} ({best})"
    out["margin_status"] = out[best]["model_free_baseline"]["status"]
    return out


def md_pairs_in_graph(graph, n_m, n_d):
    """miRNA-disease pairs still connected (either direction) in the graph, as local (miRNA, disease)."""
    s, t = (x.cpu().numpy() for x in graph.edges())
    a = (s < n_m) & (t >= n_m) & (t < n_m + n_d)
    b = (t < n_m) & (s >= n_m) & (s < n_m + n_d)
    P = np.r_[np.c_[s[a], t[a] - n_m], np.c_[t[b], s[b] - n_m]]
    return np.unique(P, axis=0)


def run_fold(M, graph0, y, tr, te, va, param, mode, epochs, their_metrics):
    """Their train_model loop. mode: 'released' (their removal) or 'removed' (true removal of test + val pos)."""
    import dgl  # noqa: F401
    graph = graph0.clone()
    ye, yy = y["y_edge"], y["y"].reshape(-1)
    model = M.DiGAMN_Model(param)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0002)
    te_pos = te[(yy[te] == 1).numpy()]
    if mode == "released":
        for _src, _tgt in zip(ye[0][te_pos], ye[1][te_pos]):
            graph.remove_edges((_src, _tgt))          # exactly as released: node ids taken as edge ids
    else:
        drop = te_pos if va is None else np.r_[te_pos, va[(yy[va] == 1).numpy()]]
        s, t = ye[0][drop], ye[1][drop]
        eids = torch.cat([graph.edge_ids(s, t), graph.edge_ids(t, s)])
        graph.remove_edges(eids)
    s_tr, t_tr = ye[0][tr], ye[1][tr]
    ck, best_state, best_ap = [], None, -1.0
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        rep = model(graph, None)
        preds = model.predict(rep[s_tr], rep[t_tr])
        loss = F.binary_cross_entropy_with_logits(preds, yy[tr])
        loss.backward()
        opt.step()
        if epoch % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                rep = model(graph, None)
                sc = {k: model.predict(rep[ye[0][ix]], rep[ye[1][ix]]).numpy()
                      for k, ix in (("test", te), ("val", va)) if ix is not None}
            model.train()
            row = {"epoch": epoch, "scores": sc}
            yt = yy[te].numpy()
            with contextlib.redirect_stdout(io.StringIO()):
                m, names = their_metrics(yt.reshape(1, -1), sc["test"].reshape(1, -1))
            row["test_their"] = dict(zip(names[2:], [float(v) for v in m[2:]]))
            row["test_auc"] = float(roc_auc_score(yt, sc["test"]))
            if va is not None:
                row["val_ap"] = float(average_precision_score(yy[va].numpy(), sc["val"]))
                if row["val_ap"] > best_ap:
                    best_ap, best_state = row["val_ap"], copy.deepcopy(model.state_dict())
            ck.append(row)
    if best_state is not None:
        model.load_state_dict(best_state)   # the validation-selected checkpoint
    return ck, graph, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--dataset", default="MDA-CF")
    ap.add_argument("--seeds", type=int, nargs="+", default=[521, 1, 2])
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "16")))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ga = types.ModuleType("global_args")
    ga.device = torch.device("cpu")
    sys.modules["global_args"] = ga
    os.chdir(a.repo)
    sys.path.insert(0, str(a.repo))
    import data_utils as DU
    import model as M

    class Config:  # main_bio.Config with the paper's case-study setting
        def __init__(self):
            self.kfold, self.self_encode_len, self.globel_random = 5, 256, 100
            self.other_args = {"arg_name": [], "arg_value": []}
            self.epochs, self.print_epoch, self.lr, self.reg = a.epochs, EVAL_EVERY, 0.01, 0.0005
            self.num_rels, self.dropout, self.xr, self.di_num = 0, 0.1, None, 5
            self.MGAT_layer, self.GAT_layer, self.n_hid, self.Memory_size = 4, 4, 8, 8

    class Paths:  # main_bio.Data_paths
        def __init__(self, df, di):
            b = f"./data{di}/{df}/"
            self.md, self.mc, self.dc = b + "m_d.csv", b + "m_sc.csv", b + "d_sc.csv"
            self.mm, self.dd = [b + "m_gs.csv", b + "m_ss.csv"], [b + "d_gs.csv", b + "d_ss.csv"]

    out = {"meta": {"repo_commit": "6695f4e", "dataset": a.dataset, "seeds": a.seeds, "epochs": a.epochs,
                    "eval_every": EVAL_EVERY, "reported_auroc": 0.9635,
                    "setting": {"MGAT_layer": 4, "GAT_layer": 4, "n_hid": 8, "Memory_size": 8}}, "runs": []}
    t0 = time.time()
    for seed in a.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        param = Config()
        graph, _, y = DU.read_data(param, Paths(a.dataset, param.di_num))
        n_m, n_d = param.m_numb, param.d_numb
        ye, yy = y["y_edge"], y["y"].reshape(-1)
        loc = lambda ix: np.c_[ye[0][ix].numpy(), ye[1][ix].numpy() - n_m]  # noqa: E731
        allpos = np.argwhere(np.loadtxt(f"./data{param.di_num}/{a.dataset}/m_d.csv", delimiter=",") != 0)
        kf = KFold(n_splits=5, shuffle=True, random_state=param.globel_random)
        per_step = {k: [] for k in ("released", "removed", "validation", "matched")}
        for fold, (tr, te) in enumerate(kf.split(np.arange(len(yy)))):
            rng = np.random.default_rng(seed * 100 + fold)
            tr_s = rng.permutation(tr)
            nv = len(tr) // 8
            va, tr_v = np.sort(tr_s[:nv]), np.sort(tr_s[nv:])
            pos_te = (yy[te] == 1).numpy()
            P, Nu = loc(te)[pos_te], loc(te)[~pos_te]
            fold_out = {"seed": seed, "fold": fold + 1, "n_test_pos": int(pos_te.sum())}
            for step, (mode, trx, vax) in {"released": ("released", tr, None), "removed": ("removed", tr, None),
                                           "validation": ("removed", tr_v, va)}.items():
                torch.manual_seed(521 + 1000 * seed + fold)   # their set_seed(521), varied per seed
                ck, g, mdl = run_fold(M, graph, y, trx, te, vax, param, mode, a.epochs, DU.get_metrics)
                enc = md_pairs_in_graph(g, n_m, n_d)
                enc_set = set(map(tuple, enc.tolist()))
                trp = loc(trx)[(yy[trx] == 1).numpy()]
                trn = loc(trx)[(yy[trx] == 0).numpy()]
                last = ck[-1]
                if vax is None:
                    best = ck   # chosen across folds below (their fold-mean test AUPR)
                    sel = None
                else:
                    sel = max(ck, key=lambda r: r["val_ap"])   # first maximum, as the snapshot keeps
                rec = {"test_auc_by_ckpt": [r["test_auc"] for r in ck],
                       "their_prc_by_ckpt": [r["test_their"]["prc"] for r in ck],
                       "their_auc_by_ckpt": [r["test_their"]["auc"] for r in ck],
                       "last_auc": last["test_auc"], "n_enc_md_pairs": int(len(enc)),
                       "test_pos_in_graph": float(np.mean([tuple(p) in enc_set for p in P.tolist()]))}
                use = sel if sel is not None else last
                rec["audit_last_or_selected"] = audit_both(trp, P, trn, Nu, enc, use["test_auc"], n_m, n_d)
                if sel is not None:
                    rec.update(selected_epoch=sel["epoch"], selected_auc=sel["test_auc"])
                    # step 4: same checkpoint, degree-matched negatives
                    g_t = torch.Generator().manual_seed(seed * 100 + fold)
                    deg_d = torch.from_numpy(np.bincount(trp[:, 1], minlength=n_d)).long()
                    mdn = sample_degree_matched_negatives_exact(torch.from_numpy(P.T.copy()).long(),
                                                                torch.from_numpy(allpos.T.copy()).long(),
                                                                degree_bins(deg_d), n_d, g_t)[0].numpy().T
                    sc_pos = sel["scores"]["test"][pos_te]
                    mdl.eval()   # the validation-selected checkpoint (restored in run_fold)
                    with torch.no_grad():
                        rep = mdl(g, None)
                        s_neg = mdl.predict(rep[torch.from_numpy(mdn[:, 0])],
                                            rep[torch.from_numpy(mdn[:, 1] + n_m)]).numpy()
                        s_pos = mdl.predict(rep[torch.from_numpy(P[:, 0])],
                                            rep[torch.from_numpy(P[:, 1] + n_m)]).numpy()
                    auc_m = float(roc_auc_score(np.r_[np.ones(len(s_pos)), np.zeros(len(s_neg))], np.r_[s_pos, s_neg]))
                    fold_out["matched"] = {"auc": auc_m, "rescore_max_abs_diff": float(np.abs(s_pos - sc_pos).max()),
                                           "audit": audit_both(trp, P, trn, mdn, enc, auc_m, n_m, n_d)}
                    per_step["matched"].append(auc_m)
                fold_out[step] = rec
                print(f"seed {seed} fold {fold + 1} {step:10s} last {last['test_auc']:.4f}"
                      + (f" sel@{sel['epoch']} {sel['test_auc']:.4f}" if sel else "")
                      + f" | test pos in graph {rec['test_pos_in_graph']:.1%} | floor "
                      f"{rec['audit_last_or_selected']['floor']:.4f} [{time.time() - t0:.0f}s]", flush=True)
            if "matched" in fold_out:
                print(f"seed {seed} fold {fold + 1} matched    {fold_out['matched']['auc']:.4f} floor "
                      f"{fold_out['matched']['audit']['floor']:.4f} (rescore diff "
                      f"{fold_out['matched']['rescore_max_abs_diff']:.1e})", flush=True)
            out["runs"].append(fold_out)
        # their reported number: checkpoint with the best fold-mean test AUPR (their get_metrics), per step
        for step in ("released", "removed"):
            rs = [r[step] for r in out["runs"] if r["seed"] == seed]
            prc = np.mean([r["their_prc_by_ckpt"] for r in rs], axis=0)
            i = int(prc.argmax())
            for r in rs:
                r["their_selected_ckpt"] = i
            print(f"seed {seed} {step}: their selection ckpt {(i + 1) * EVAL_EVERY}: "
                  f"AUC {np.mean([r['test_auc_by_ckpt'][i] for r in rs]):.4f} (their AUC "
                  f"{np.mean([r['their_auc_by_ckpt'][i] for r in rs]):.4f}); last "
                  f"{np.mean([r['last_auc'] for r in rs]):.4f}", flush=True)

    def fold_means(fn):
        per_seed = [np.mean([fn(r) for r in out["runs"] if r["seed"] == s]) for s in a.seeds]
        return {"mean": float(np.mean(per_seed)), "sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0}
    summ = {}
    for step in ("released", "removed"):
        summ[step] = {"their_selection": fold_means(lambda r: r[step]["test_auc_by_ckpt"][r[step]["their_selected_ckpt"]]),
                      "last": fold_means(lambda r: r[step]["last_auc"]),
                      "floor_last": fold_means(lambda r: r[step]["audit_last_or_selected"]["floor"]),
                      "test_pos_in_graph": fold_means(lambda r: r[step]["test_pos_in_graph"])}
    summ["validation"] = {"selected": fold_means(lambda r: r["validation"]["selected_auc"]),
                          "floor": fold_means(lambda r: r["validation"]["audit_last_or_selected"]["floor"])}
    summ["matched"] = {"auc": fold_means(lambda r: r["matched"]["auc"]),
                       "floor": fold_means(lambda r: r["matched"]["audit"]["floor"])}
    out["summary"] = summ
    print("\n=== summary ===\n", json.dumps(summ, indent=1))
    tag = "_".join(map(str, a.seeds))
    path = ROOT / f"results/comparison/digamn_rerun_s{tag}{'' if a.epochs >= 5000 else '_smoke'}.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print("Wrote", path)


if __name__ == "__main__":
    main()
