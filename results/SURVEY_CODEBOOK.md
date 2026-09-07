# Literature-survey codebook

Decision rules for classifying a paper along the four dimensions of the evaluation-practice
survey (`results/literature_survey.tsv`, Table 1 of the JBI manuscript).

Written 2026-09-01, ahead of the second-rater pass. Until this file existed the rules lived
only in the first rater's head and in the `evidence_quote` column; a second rater cannot be
said to measure the same construct without them.

## Scope of the unit being classified

One row = one **paper**, classified from **its own text**, primary source only (publisher page
or PMC full text). Not from its abstract, not from a citing paper's description of it, not from
its repository README, and **not from its code**.

The last exclusion is deliberate and load-bearing. The survey's claim is about what a *reader
or reviewer* can establish from the methods section as written. A fact that is recoverable only
by cloning the repository and reading `train.py` is, for this survey's purpose, **not reported**.
Where a paper's prose is silent but its code would settle the question, the call is `unclear`,
not the answer the code gives.

## Cardinal rule for `unclear`

`unclear` is a statement about **the paper**, not about the rater's effort. Use it when the
paper's prose does not settle the question, however obvious the answer may seem from context,
convention, or the method's design. Do **not** infer:

- from what the architecture would "have to" do to work;
- from what is standard practice in that subfield;
- from a library name alone, **unless** the library's cited API fixes the behaviour (see D2);
- from the absence of a complaint about leakage.

Conversely, `unclear` is **not** a hedge for a claim the paper does make weakly. If the paper
states the fact, the call is `yes`/`no` even where the phrasing is one clause in one sentence.

Every `yes` and every `no` must carry a **verbatim quote** from the paper that, read alone,
supports the call. If no such quote can be produced, the call is `unclear` by construction.

---

## D1. `cv_over_edges` — is the evaluation unit an edge?

Does the paper split/cross-validate over **individual associations (edges)**, as opposed to over
**entities (nodes)** or over whole **datasets**?

- **`yes`** — the paper describes splitting associations, interactions, pairs, or samples into
  folds/train/test. Includes k-fold CV over positive pairs and single random train/val/test
  splits of the edge set.
- **`no`** — the split is over nodes (all edges of a held-out miRNA/drug/disease move together),
  or over independent datasets, or the paper evaluates only on an external benchmark set.
- **`unclear`** — the paper says it cross-validates but never says over what.

> Boundary note: a paper that splits over edges *and additionally* reports a cold-start /
> node-wise experiment is `yes` — the headline number is the edge-wise one. Record the
> node-wise arm in the quote.

## D2. `testedges_removed_from_encoder_graph` — is the test edge invisible during message passing?

Are held-out (validation/test) positive edges **deleted from the adjacency structure the encoder
propagates over**, or do they remain in the graph while being scored?

- **`yes`** — the paper states the held-out edges are removed, masked, deleted, or excluded from
  the training graph / adjacency matrix / similarity matrices computed from associations. Also
  `yes` when the paper names an API whose documented behaviour is precisely this and cites it as
  used for the split (e.g. PyG `RandomLinkSplit` in its default edge-splitting mode).
- **`no`** — the paper states, or its described procedure entails on its own terms, that the
  full association matrix is the encoder input and only the *labels* are held out. The clearest
  signal: similarity/adjacency matrices are described as built once from all known associations,
  before or independently of the split.
- **`unclear`** — the paper describes the split at the level of "samples", "positive and negative
  pairs", or "folders" and never states what happens to the graph the encoder sees. **This is
  the modal case and it is the survey's actual finding — do not resolve it by inference.**

> Boundary note, the hardest call in the codebook: "we divided the *associations* into training
> and test sets" is **`unclear`**, not `yes`. Dividing the associations into sets says nothing
> about which of those sets is wired into the encoder. Only an explicit statement about the
> graph / adjacency / matrix / message passing earns `yes`.

## D3. `negative_sampling` — where do the negatives come from?

Record both a free-text description (with ratio if stated) and one category:

- **`uniform_unlabeled`** — negatives drawn uniformly at random from pairs with no recorded
  association. The dominant convention. Any stated ratio (1:1, 1:5, 1:10, 1:50) stays in this
  category; the ratio goes in the free text.
- **`all_unlabeled`** — every non-positive pair is treated as negative, no sampling.
- **`non_uniform`** — negatives are *selected* rather than drawn uniformly: reliability-score
  filtering, distance/similarity-based selection, clustering, hard-negative mining, or any
  procedure whose stated purpose is to make the negatives more credible.
- **`not_described`** — the paper reports negatives exist but never says how they were obtained.
- **`no_negatives`** — the task is formulated as ranking/regression without an explicit negative
  class.

> A paper whose negative-selection scheme is one of its own contributions is `non_uniform` even
> when a uniform variant appears in an ablation.

## D4. `model_free_baseline` — is an untrained comparator in the results table?

**Revised 2026-09-01 after the second-rater pass.** The original wording gave a definition
("no trained parameters") and then an enumerated list of qualifying baselines, and the two
disagreed: six papers compared against untrained network-propagation methods that met the
definition but were absent from the list. The first rater applied the list and recorded `no`;
that produced a false "0 of 22" headline. **The definition governs. The list is illustrative,
never exhaustive.**

The question: does the paper report, **in a results or comparison table/figure**, at least one
comparator whose predictions require **no parameters fitted to the association data**?

### The operational test

Ask of the comparator, not of the surveying paper: *does anything get fitted?* If the method
runs directly on the graph and/or the similarity matrices and emits a score, it is model-free.
Free hyperparameters that are set rather than learned (a restart probability, a path-length
decay, a number of hops) do **not** make a method trained.

**Qualifies (untrained).** Common neighbours, Adamic--Adar, resource allocation, L3,
preferential attachment, Katz, personalised PageRank, raw similarity-matrix scores, degree /
popularity predictors, and the field's classical propagation and scoring methods --
random walk with restart (RWR, RWRH), structural perturbation (SPM), path-based scoring
(PBMDA), bipartite network projection (BNPMDA), within-and-between scoring (WBSMDA),
heterogeneous network propagation (HNM), forward similarity integration (FSI), and
heterogeneous graph inference (HGIMDA).

**Does not qualify (trained).** Anything with parameters fitted to data: SVM, random forest,
XGBoost, gradient boosting, logistic regression; matrix factorisation and matrix completion
(IMCMDA, MCLPMDA, NIMCGCN, ELMDA); autoencoders; node embeddings (Dgn2vec, node2vec,
HerGePred); and every GNN.

### Two rules that decide the hard cases

1. **Whether a comparator is trained is a fact about that comparator, not about the surveying
   paper's description of it.** Where the surveying paper misdescribes a method -- and this
   happens -- classify from the comparator's own original publication and **record the
   discrepancy** in the quote field. (Encountered: one paper describes WBSMDA, a 2016
   score-based method, as "an attention-aware multi-view graph convolutional network".) This is
   a deliberate exception to the codebook's general "classify from the paper's prose" rule: D4
   asks whether an untrained number was put on the page, which the surveying paper's prose
   cannot make false by mislabelling it.
2. **In the table, not in the pipeline.** A topological score used for data augmentation,
   feature construction, candidate generation, or graph densification is not a baseline. It
   must be evaluated on the same task and reported alongside the trained model. (ModulePred
   uses L3 for augmentation -- not qualifying -- *and* reports RWR/RWRH as compared methods --
   qualifying. Check the whole paper, not the first topological term you find.)

- **`yes`** -- at least one untrained comparator appears in a results/comparison table or figure.
- **`no`** -- every comparator is trained, or there are no comparators.
- **`unclear`** -- comparators exist but cannot be identified from text, tables or figures.

### Companion fields, recorded whenever D4 = yes

The count alone understates what these rows show, so also record:

- `mf_comparator` -- the untrained method(s) by name.
- `mf_score` / `mf_metric` -- its reported number and metric.
- `paper_score` -- the paper's own headline number **on the same metric and dataset**, so the
  two are comparable.
- `margin` -- `paper_score - mf_score`. This is the quantity of interest: a small margin means
  the paper's own table already shows a trained model barely clearing an untrained floor,
  unremarked.

## D5. `replicable` — can the paper's graph be re-run by us?

Whether the association data behind the paper can be obtained and re-scored under both
protocols, so its inflation can be **measured rather than taken from its own reported number**.

- **`have_locally`** -- already downloaded and instrumented under `data/graphs_survey`.
- **`obtainable`** -- a bundled repository, supplementary file, or named public release
  (an HMDD version, a public benchmark) that could be downloaded. Give the location.
- **`not_obtainable`** -- no data release, or the graph cannot be reconstructed from what is
  published. A dead repository link counts as `not_obtainable`; say so.

---

## Output contract

One row per paper, tab-separated:

```
paper	cv_over_edges	testedges_removed_from_encoder_graph	negative_sampling_category	negative_sampling_text	model_free_baseline	quote_D1	quote_D2	quote_D3	quote_D4	source_retrieved
```

`source_retrieved` records whether the full text was actually reached (`full_text`,
`abstract_only`, `paywalled`, `failed`). A row whose source could not be reached is **not**
classified as `unclear` — it is dropped from the agreement computation, because a rater who
could not read the paper disagrees with nothing.
