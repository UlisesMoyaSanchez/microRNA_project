# Evaluation Audit — miRNA-MS Project

**Status:** canonical results document. Supersedes `results/archive_pre_audit/REPORT.md`
and `results/archive_pre_audit/EXEC_SUMMARY.md`, both of which report numbers we now know
to be artifacts.
**Last updated:** 2026-07-13
**Spanish summary for clinical collaborators:** [`RESUMEN_AUDITORIA.md`](RESUMEN_AUDITORIA.md)

---

## The headline

The project's published link-prediction result — **AUROC 0.9836** — does not survive a
correct evaluation. Retrained under a verified leak-free edge split with popularity-matched
negatives, the same architecture scores **AUROC 0.6467**.

For scale: a **no-learning heuristic** (Adamic–Adar, a two-line formula) scores **0.5911**
on the identical held-out edges. The four-layer, 512-channel heterogeneous graph transformer
buys about **5.6 AUROC points over trivial graph structure**.

And the sharpest result of the audit: **under the sloppy protocol, a scorer that ignores the
miRNA entirely and only counts how many miRNAs already target the gene (AUROC 0.8723) beats
the trained graph transformer (0.8132).** The evaluation protocol was not merely flattering
the model — it was rewarding a model that had learned less than a one-line heuristic.

**Cell-type classification is unaffected and remains genuine: accuracy 0.9950.** The two
tasks separated cleanly. Cell typing is real. The regulatory link head was an artifact of
how it was measured.

---

## Data provenance (corrected — this is load-bearing)

The 44,186 miRNA→gene edges are **miRDB v6.0 sequence-based predictions, score ≥ 80**.
They are **not** experimentally validated interactions, and they are **not** from miRTarBase
(miRTarBase's server was unavailable during data collection).

The archived `REPORT.md` described them as *"validadas experimentalmente (miRTarBase,
miRDB)"*. That is wrong, and it is not a cosmetic error: **miRDB edges are determined by
seed-sequence complementarity, and the graph contains no sequence information at all.**
Gene node features are 1-D (mean log-normalized expression); miRNA features are a learnable
embedding with no biological content. So for a miRNA–gene pair the model has never seen,
there is no sequence signal available to generalize from — only topology. This is the most
likely explanation for the ceiling documented below, and it is central to the paper's
argument.

---

## What was asked, and what came back

Four experiments, each narrowing the question. The order matters — it is the method.

### 1. Was the model reading the answer off the graph? — **No.**

`training/diagnose_leakage.py` · job **5593** · `results/comparison/leakage_diagnostic.json`

The miRNA→gene edges were both supervision targets *and* inputs to the encoder. Message
passing could have been handing the link head the very edge it was asked to predict. Same
checkpoint, same pairs, three encoder views:

| Encoder view | AUROC |
|---|:--:|
| (a) Graph intact — as published | 0.9853 |
| (c) **Scored pair masked out of message passing** | **0.9766** (−0.009) |
| (b) miRNA↔gene relation removed entirely | 0.5551 (−0.430) |

**Establishes:** masking the scored edge costs essentially nothing. The model was *not*
reading the answer off the graph. Row (b) collapsing to chance is expected and healthy — it
means the signal lives in the interaction topology, which is what a graph model should
exploit.

**Does not rule out:** weight-level memorization. This checkpoint had those exact pairs as
training targets, so no post-hoc analysis of it can settle the question. Only a retrain can.

### 2. Was it specificity, or popularity bias? — **Partly popularity, but not only.**

`training/eval_hard_negatives.py` · jobs **5595 / 5596** · `results/comparison/hard_negatives.json`

Two negative samplers × two scorers, on the same pairs. A negative is *degree-matched* when
the decoy gene has the same miRNA and a gene of equal in-degree — so neither miRNA
promiscuity nor gene popularity can separate positive from negative.

| Scorer | Uniform negatives | Degree-matched negatives |
|---|:--:|:--:|
| Gene-degree heuristic (model-free) | 0.7760 | **0.5150** |
| HGT V2 | 0.9758 | **0.8828** |

**Establishes:** uniform negatives inflate the metric by ~9 AUROC points. A model-free
"guess the popular gene" rule reaches 0.776 against them — so the published number was weak
evidence on its own. But the heuristic collapses to chance under matched negatives while the
model held 0.8828, which at the time read as genuine specificity.

**Does not rule out:** the same thing. 0.8828 was still measured on a checkpoint trained on
those pairs.

### 3. Does it survive a real held-out split? — **No. This is the finding.**

`training/splits.py`, `configs/config_v2_edgesplit.yaml` · job **5605**

Split (verified leak-free by `training/test_edge_split.py`; all eight checks at zero):

```
44,186 positives = 24,745 message-passing + 10,605 train-supervision + 4,418 val + 4,418 test
```

Held-out edges are absent from supervision **and** from the encoder's input, **in both
directions** — the reverse relation `(gene, regulated_by, miRNA)` is stripped in lockstep via
`RandomLinkSplit(rev_edge_types=...)`. Without that, a held-out edge stays reachable in one
hop and the split is worthless.

| | AUROC |
|---|:--:|
| Published protocol | 0.9836 |
| **Held-out edges + degree-matched negatives** | **0.6467** |
| Cell-type accuracy (same run) | **0.9950** |

Converged at 144 epochs, model selected on `val_auroc`. The link head **overfits from epoch
1**: training loss falls to 0.038 while validation loss climbs to 6.9, and only AUROC creeps
upward. Cell accuracy rises monotonically to 0.9950 throughout.

**Establishes:** the leak was weight-level memorization — precisely the possibility neither
diagnostic above could exclude. On genuinely unseen pairs the model retains 0.6467.

### 4. Is 0.65 the model's achievement, or the task's ceiling? — **Close to the ceiling.**

`training/eval_topology_baseline.py` · job **5604** · `results/comparison/topology_baseline.json`

Model-free heuristics on the **same** 4,418 held-out edges, computed from training edges only
(35,350 = message-passing + train-supervision). No learning, no checkpoint.

| Scorer | Uniform negatives | Degree-matched negatives |
|---|:--:|:--:|
| `gene_degree` — **ignores the miRNA entirely** | **0.8723** | 0.5123 |
| `pref_attach` | 0.8371 | 0.5071 |
| `common_neigh` | 0.8582 | 0.5836 |
| `adamic_adar` — best heuristic | 0.8616 | **0.5911** |
| **HGT V2, retrained (job 5605)** | — | **0.6467** |

Two conclusions, and the second is the paper.

**(a) The HGT beats trivial topology by 5.6 points** (0.6467 vs 0.5911). Real, but small. A
4-layer/512-channel transformer buys ~5 AUROC points over a formula from 1999. It is not
performing meaningful regulatory inference; it is doing slightly-better-than-trivial graph
completion. This also rules out architecture tuning as a rescue: nothing bridges 0.65 → 0.9
when there is no sequence signal to generalize from.

**(b) Under uniform negatives, a scorer that ignores the miRNA entirely reaches 0.8723.**
Gene popularity alone — no learning, no model, no miRNA. This is the strongest single number
in the project, because it is a control no reviewer can argue with:

| Protocol | Popularity heuristic | Deep model |
|---|:--:|:--:|
| **Sloppy** (uniform negatives) | **0.8723** | 0.9836 |
| **Honest** (held-out + matched negatives) | **0.5123** | 0.6467 |

Under the protocol used in the published version, the gap between a one-line heuristic and a
state-of-the-art graph transformer is ~11 points. **The evaluation protocol, not the model,
was doing most of the work.**

---

## 5. Attributing the collapse — which sin cost what?

Jobs **5605** (matched negatives) and **5607** (uniform negatives), both trained *and*
evaluated under the same held-out split, so training and evaluation negatives always agree:

| | Uniform negatives | Degree-matched negatives |
|---|:--:|:--:|
| **Edges seen in training** (published) | **0.9836** | 0.8828 |
| **Edges held out** (honest) | 0.8132 | **0.6467** |

| Effect | Cost |
|---|:--:|
| An honest **split** alone (negatives held uniform) | **−0.170** |
| Honest **negatives** alone (edges held seen) | **−0.101** |
| Both — published → honest | **−0.337** |

The two effects are **super-additive**: −0.170 + −0.101 = −0.271, but the true total is
−0.337. Fixing only one of the two problems substantially understates the damage, which is
itself worth reporting — a paper that holds out edges but keeps uniform negatives still
reports an inflated number.

> **A methodological warning worth stating explicitly.** An earlier attempt (job 5606)
> scored the *matched-negative* checkpoint against uniform negatives and got **0.5533** —
> *lower* than against matched negatives, which is nonsense on its face. That figure is a
> **train/eval mismatch, not a difficulty measurement**: a model trained on degree-matched
> negatives learns to ignore gene degree, and then cannot exploit the easy degree signal
> that uniform negatives hand it. It is not reported as a result. Attribution requires the
> negative distribution to be the same at training and evaluation time — hence job 5607.

### The most damning single fact in the audit

Put the trained model next to the model-free heuristic **on the same held-out edges, under
the same sloppy protocol** (uniform negatives):

| Scorer, held-out edges + uniform negatives | AUROC |
|---|:--:|
| `gene_degree` — **no learning, no model, ignores the miRNA** | **0.8723** |
| HGT V2, trained end-to-end with uniform negatives (job 5607) | **0.8132** |

**The one-line popularity heuristic beats the graph transformer by 6 AUROC points.** Under
the evaluation protocol used in the published version, the deep model is not merely
unnecessary — it is *worse* than counting how many miRNAs already target the gene.

---

## What no longer stands

- **The regulatory circuits are not findings.** `results/interpretation/top_circuits_by_celltype.tsv`,
  `hsa-miR-23a-3p → CCL7` (score 0.986), `hsa-miR-146a-3p` in Th17, `hsa-miR-140-5p` in
  oligodendrocytes — every one of these is ranked by a link head whose genuine generalization
  is ~5 points above a no-learning heuristic. They must not be presented as validated, and
  **external validation against the literature has been suspended**: checking them now would
  risk validating an artifact and would spend collaborators' credibility, not just GPU hours.
- **"V2 is the only model that does link prediction."** That was the `HeteroData.get()` bug
  returning `None` for the tuple edge-type key and silently disabling the link head, which
  produced `nan` for every baseline. Fixed, the baselines have real numbers
  (`homo_gcn` 0.9170, `ablation_no_coexpr` 0.9374 transductive), and `random` lands at
  0.5126 — the smoke test confirming the metric is finally computed correctly.
- **`val_loss` as a cross-model column.** A model without a link head optimizes a strictly
  smaller objective, which is why `ablation_no_mirna` showed the "best" loss while being the
  worst model. Reported per-task (`link_loss`, `clf_loss`) instead.

## What does stand

- **Cell-type classification: 0.9950**, on a real cell-level split. Genuine throughout.
- **The pipeline and the audit instrumentation**, which is now the project's main asset:
  `training/splits.py` (leak-free edge split, reverse relation stripped in lockstep),
  `training/test_edge_split.py` (gating leakage test),
  `training/diagnose_leakage.py`, `training/eval_hard_negatives.py`,
  `training/eval_topology_baseline.py`, `training/eval_heldout_grid.py`.

---

## Reproduction

All runs are SLURM jobs on the DGX (`ssh dgxum`,
`/raid/home/umoya/scripts/microRNA_project`). PyG is not installed on the laptop.

```bash
# Gate first: if the split leaks, nothing below means anything. CPU-only.
python training/test_edge_split.py --config configs/config_v2_edgesplit.yaml

# 1. Message-passing leakage (job 5593)
sbatch --export=ALL,CONFIG=configs/config_v2.yaml,CHECKPOINT=checkpoints_v2/best_model.pt \
    training/slurm_diagnose_leakage.sh

# 2. Popularity bias (jobs 5595/5596)
sbatch --export=ALL,CONFIG=configs/config_v2.yaml,CKPT=checkpoints_v2/best_model.pt \
    training/slurm_hard_negatives.sh

# 3. The honest retrain (job 5605)
sbatch --export=ALL,CONFIG=configs/config_v2_edgesplit.yaml training/slurm_train.sh

# 4. Topology-only controls (job 5604)
sbatch --export=ALL,CONFIG=configs/config_v2_edgesplit.yaml training/slurm_topology_baseline.sh

# 5. Attribution — uniform-negative retrain (job 5607)
sbatch --export=ALL,CONFIG=configs/config_v2_edgesplit_uniform.yaml training/slurm_train.sh
```

Every number above traces to a job ID and a JSON artifact under `results/comparison/`.

## Still open

- **Multi-seed (3–5 seeds).** A methods paper must show the effect is not seed noise. The
  design is final, so this is now worth the GPU time. It is the last experimental gap.
- Re-pick the venue: this is an evaluation/methods paper, not a biology-discovery one.
- Draft the manuscript from this document.
