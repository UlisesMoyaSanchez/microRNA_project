* On 7 surveyed papers’ own data, the same heuristic beats one trained
model outright.| Comment: Do We review only 7 papers ?

* Rigorous data handling and transparent evaluation reporting are foundational
to reliable network biology, yet Graph Neural Network (GNN) papers predict-
ing microRNA–target interactions routinely report AUROC in the 0.91–0.99
range without establishing model free baselines.| Comment: could be great to explain waht is model-free baseline. 

* We built a heterogeneous graph from a public single-cell
RNA-seq cohort and miRDB v6.0 miRNA–target interactions, training a het-
erogeneous graph transformer to score miRNA–gene pairs. | Comment: Maybe here we need to update this description,, in addition that we compute the model free baselines of maybe 12 cases.

* The conventional protocol (edges seen during message
passing, uniform negatives) reports AUROC 0.9867 ± 0.0011. Correcting both
flaws simultaneously drops this to 0.6276 ± 0.0070, within 3.6 points of a one-
line, no-learning topological heuristic (Adamic/Adar, 0.5912); the inflation is
super-additive, so correcting either flaw alone understates the total. | Comment: These results are only about our model  maybe we need toupdate taking into account all the model free baselines. 




