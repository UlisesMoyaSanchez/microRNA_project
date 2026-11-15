"""
hgdti_floor_diagnostic.py -- what drives the 0.740 model-free floor on HGDTI degree-matched negatives?

Independent 10-fold split of the positives (KFold, random_state=0; not HGDTI folds), the same exact
degree-matched sampler, protocheck heuristics, plus sequence homology (Similarity_Matrix_Proteins) of the
candidate protein to the drug s training targets. Run: python analysis/hgdti_floor_diagnostic.py DATA_DIR REPO
(BITACORA 2026-11-15).
"""
import sys, numpy as np, torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
sys.path.insert(0, sys.argv[2])
from protocheck.checks import TopologyScorer
from training.splits import degree_bins, sample_degree_matched_negatives_exact
D = sys.argv[1]
Y = np.loadtxt(f"{D}/mat_drug_protein.txt"); nd, npr = Y.shape
Sp = np.loadtxt(f"{D}/Similarity_Matrix_Proteins.txt"); Sd = np.loadtxt(f"{D}/mat_drug_drug.txt")
PP = np.loadtxt(f"{D}/mat_protein_protein.txt")
print("shapes", Y.shape, Sp.shape, Sd.shape, PP.shape)
pos = np.argwhere(Y == 1).T; allp = torch.from_numpy(pos.copy()).long()
res = {}
for k, (tr, te) in enumerate(KFold(10, shuffle=True, random_state=0).split(pos.T)):
    trp, tep = pos[:, tr], pos[:, te]
    A = np.zeros_like(Y); A[trp[0], trp[1]] = 1
    cd = torch.from_numpy(A.sum(0).astype(np.int64))
    neg = sample_degree_matched_negatives_exact(torch.from_numpy(tep).long(), allp, degree_bins(cd), npr,
                                                torch.Generator().manual_seed(k))[0].numpy()
    P = np.c_[tep, neg]; lab = np.r_[np.ones(tep.shape[1]), np.zeros(neg.shape[1])]
    sc = TopologyScorer(trp, nd, npr).score(P)
    Spz = Sp.copy(); np.fill_diagonal(Spz, 0); Sdz = Sd.copy(); np.fill_diagonal(Sdz, 0)
    PPz = PP.copy(); np.fill_diagonal(PPz, 0)
    # guilt by association: drug's training targets similar to candidate protein (max), and drugs similar to d hitting p
    sc["prot_sim_max (seq. homology to d's known targets)"] = np.array([ (Spz[p, A[d] == 1].max() if A[d].sum() else 0) for d, p in P.T])
    sc["drug-drug interactors hit p"] = np.array([ (Sdz[d, A[:, p] == 1].max() if A[:, p].sum() else 0) for d, p in P.T])
    sc["ppi_to_known_targets"] = np.array([ PPz[p, A[d] == 1].sum() for d, p in P.T])
    sc["exact_deg_p"] = A.sum(0)[P[1]]
    # common_neigh>0 shares
    cn = sc["common_neigh"]
    sc["_frac_cn>0_pos"] = None
    for n, v in sc.items():
        if v is None: continue
        res.setdefault(n, []).append(roc_auc_score(lab, v))
    res.setdefault("cn>0 pos", []).append((cn[lab == 1] > 0).mean()); res.setdefault("cn>0 neg", []).append((cn[lab == 0] > 0).mean())
    # adamic-adar after removing protein-similarity-explained: restrict to pairs where max homology < 0.3
    low = sc["prot_sim_max (seq. homology to d's known targets)"] < 0.3
    if len(set(lab[low])) == 2:
        res.setdefault("adamic_adar | low homology", []).append(roc_auc_score(lab[low], sc["adamic_adar"][low]))
        res.setdefault("share low homology pos/neg", []).append(f"{low[lab==1].mean():.2f}/{low[lab==0].mean():.2f}")
for n, v in res.items():
    print(f"{n:55s}", v[0] if isinstance(v[0], str) else f"{np.mean(v):.4f}")
# last fold, deeper look
h = sc["prot_sim_max (seq. homology to d's known targets)"]; aa = sc["adamic_adar"]
print("homology quantiles pos", np.quantile(h[lab==1],[.1,.25,.5,.75]).round(3), "neg", np.quantile(h[lab==0],[.1,.25,.5,.75]).round(3))
for lo, hi in [(0,.05),(.05,.2),(.2,.5),(.5,1.01)]:
    m = (h>=lo)&(h<hi)
    if len(set(lab[m]))==2: print(f"homology [{lo},{hi}) n_pos {int(lab[m].sum())} n_neg {int((1-lab[m]).sum())} AA AUROC {roc_auc_score(lab[m], aa[m]):.3f}")
    else: print(f"homology [{lo},{hi}) n_pos {int(lab[m].sum())} n_neg {int((1-lab[m]).sum())}")
m = aa>0
print("AA>0 pairs: homology AUROC", round(roc_auc_score(lab[m], h[m]),3), " AA=0 pairs: homology AUROC", round(roc_auc_score(lab[~m], h[~m]),3))
print("spearman(AA,homology) among negatives+pos", round(__import__('scipy.stats',fromlist=['x']).spearmanr(aa,h)[0],3))
# drug degree of positives
print("drug degree (train) of test positives: median", np.median(A.sum(1)[tep[0]]), "share drugs with deg>=2", (A.sum(1)[tep[0]]>=2).mean().round(3))
