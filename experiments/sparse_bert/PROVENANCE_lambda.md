# Provenance: Sparse BERT checkpoint and its rule weight (2026-10-24)

This note documents the configuration of the single "Sparse BERT" production
checkpoint whose metrics appear as the **Sparse BERT** row of Table 1 in the
ICMLC 2027 submission (P@1 0.649, R@3 0.754, email AUPRC 0.9962, F1 0.9693).
The checkpoint itself (`sparse_bert_best_model.pt`, ~255 MB) is **not
redistributed** (checkpoints are excluded from this repository); `results.json`
in this directory is the frozen output it produced.

## What the record shows

- The checkpoint was produced by `experiments/scripts/run_sparse_bert.py`
  (training date: August 2026; best epoch 6; temperature 0.735).
- In that script, the rule-loss weight `lambda_rule` defaults to **3.0** in
  all three places it is defined (function signature, `config.get` default,
  and the `--lambda-rule` argparse default).
- There is **no record anywhere in the repository of an override**
  (`--lambda-rule 1.0` or otherwise) for the August run, and the checkpoint
  stores no config field. `experiments/archive/training_log.json` records the
  contemporaneous main recipe as `lambda=3.0` ("Final model uses
  lambda_rule=3.0").
- Conclusion: the August Sparse BERT endpoint is a **prior-heavy operating
  point at λ_rule ≈ 3.0**, the same nature as the CNN four-candidate λ grid.
  It is reported for completeness; it is **not** the validation-selected
  λ_rule = 1.0 recipe (the λ = 1.0 weight was selected on 2026-10-09 by a
  validation-only protocol, m10, before any ground-truth evaluation).

## Controlled three-seed estimate

The validation-selected recipe's BERT effect is estimated under a protocol
dated and frozen before the ground truth was opened
(`ABLATION_PROTOCOL_2026-10-22.md`), on a lightweight frozen-encoder
replication pipeline (cached 768-d DistilBERT sentence vectors; attention
head only, ~80 s/run; seeds 123/456/789):

- plain BERT-ABMIL: P@1 0.417, R@3 0.650
- Sparse BERT (λ_rule = 1.0): P@1 0.607, R@3 0.735
- paired per-email bootstrap, sparse-minus-plain P@1: **+0.19,
  95% CI [+0.14, +0.24]** (n = 202 gold-positive emails)
- the margin survives every adjudication bracket of the 56 unresolved bags
  (0.15–0.19); frozen JSONs: `experiments/icicml_runs/ablation_gt.json`,
  `gt_final.json`, `sparse_brackets_gt.json`.

A fidelity check mounted the production checkpoint's attention head on the
same cached sentence vectors and reproduced its row's localization metrics
(0.6485 / 0.7544 / k@90 5.34); the production head is also behaviorally
closest to a λ = 3 replication head (per-email Spearman 0.942, top-1
agreement 0.837) rather than a λ = 1 head (0.876–0.894 / 0.757–0.767).

Per the project's one-time terminal-evaluation rule, no model was retrained
to replace the 0.649 point; the August number is disclosed, not optimized.
