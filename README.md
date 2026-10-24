# RgSA — Rule-Guided Sparse Attention for Weakly Supervised Phishing Sentence Localization

Accompanying code and frozen experiment outputs for the manuscript **"Weakly
Supervised Evidence Localization in Phishing Emails via Rule-Guided Sparse
Attention"** (Zhong Chen, Liwen Jing, Yi Xie).

Given only email-level labels, an Attention-Based Multiple Instance Learning
(ABMIL) model is shaped by two priors — a rule-alignment loss built from
cognitive-trigger labeling functions, and an L1 sparsity penalty with
deterministic temperature annealing — to localize the sentences that are
plausible evidence of the phishing act.

## Layout

```
main.py                         entry point (training/evaluation)
src/                            models, dataset, baselines, metrics, rules
  models/abmil.py               ABMIL attention (plain and sparse)
  baselines/                    ABMIL, Snorkel, BERT-ABMIL, AI2TALE faithful reimplementation
  rules.py / rules_baseline.py  cognitive-trigger labeling functions
  evaluate_sentence_localization.py  GT ranking metrics and paired bootstrap CIs
experiments/
  configs/                      all YAML configurations (defaults.yaml is the production run)
  scripts/                      training chains, GT evaluations, figure generation
  icicml_runs/                  frozen outputs of the 2026-10-22 pre-registered round
    ABLATION_PROTOCOL_2026-10-22.md   dated, frozen protocol (read first)
    ablation_gt.json / random_anchor.json / sparse_brackets_gt.json /
    fixed_lambda_gt.json        terminal-GT aggregates and per-email weights
  agent_annotation/             annotation protocol, prompts, adjudication, gold benchmark
    exports/gold_final.csv      258-email / 1,955-sentence adjudicated benchmark
    outputs/ensemble_human_kappa_round3_reconstruction.json
    scripts/reconstruct_ensemble_human_kappa.py   recomputes the published κ snapshot
  human_annotation/             30-email two-annotator pilot + protocol + agreement report
  *_results.json, main_table_data.json, ...       frozen scalar results behind every paper number
phishingdata/kwords_stem.json   cognitive-trigger keyword stems (shipped)
```

Every number in the paper traces to a frozen JSON here; the manuscript's
response letter maps each reported value to `file → field → value`.

## Requirements

- Python 3.10+; `pip install -r requirements.txt`
- macOS MPS is the default device (`device: "mps"` in the YAML); change to
  `cuda` or `cpu` to run elsewhere.

## Data

The corpus is the public AI2TALE dataset (40,000 pre-tokenized emails).
Obtain it from the AI2TALE authors/repository and place these files in
`phishingdata/`:

- `x_y_tr_m.mat`, `x_y_te_m.mat` — train/test tensors (~1.7 GB; not redistributed)
- `ind_word.pickle`, `ind_word_vocab_size_m.mat` — vocabulary

`phishingdata/kwords_stem.json` (the rule vocabulary) is included. The
adjudicated benchmark `experiments/agent_annotation/exports/gold_final.csv`
is redistributed under the original dataset's terms.

## Reproducing the paper

```bash
# Production 64-d CNN pipeline, single seed (≈154 min on MPS is the frozen
# transformer reference; the lightweight CNN trains in ≈20 min):
python main.py --config experiments/configs/defaults.yaml

# Three-seed headline runs use experiments/scripts/m_seed789_run.sh etc.
```

Terminal-GT analyses were run once, after GT was opened, under the dated
frozen protocol `experiments/icicml_runs/ABLATION_PROTOCOL_2026-10-22.md`
(all pre-specified variants reported; no selection):

| Paper element | Script | Frozen output |
|---|---|---|
| Random-ranking analytic anchor + MC check | `experiments/scripts/icicml_random_anchor.py` | `icicml_runs/random_anchor.json` |
| Loss-term decomposition, paired CIs | `experiments/scripts/icicml_ablation_gt.py` | `icicml_runs/ablation_gt.json` |
| Unresolved-bag adjudication brackets | `experiments/scripts/icicml_sparse_brackets.py` | `icicml_runs/sparse_brackets_gt.json` |
| Fixed-weight label-fraction control | `experiments/scripts/icicml_fixed_lambda_gt.py` | `icicml_runs/fixed_lambda_gt.json` |
| Ensemble–human κ snapshot | `experiments/agent_annotation/scripts/reconstruct_ensemble_human_kappa.py` | `agent_annotation/outputs/ensemble_human_kappa_round3_reconstruction.json` |
| Main table aggregates | `experiments/scripts/m_table_data.py`, `m_sparse_gt_eval.py` | `experiments/main_table_data.json`, `final_gt_eval.json` |
| Figures | `experiments/scripts/gen_figures.py`, `src/plot_*.py` | `experiments/*.png/.pdf` |

The frozen-encoder (replication) pipeline — attention head trained on cached
DistilBERT <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> vectors, ≈80 s/run — supports the bracket and loss-term
conclusions only; its magnitudes are not interchangeable with the production
pipeline (see the manuscript footnote). Regenerate its embedding cache with
`experiments/scripts/icicml_cache_bert.py` (~680 MB; not shipped).

## Not included

- **Model checkpoints** (`.pt`, ≈15 GB): deleted after backup; frozen score
  JSONs are sufficient to reproduce every table. Scripts expecting a
  checkpoint document it (e.g. set `SPARSE_CKPT` for
  `experiments/scripts/m4_email_probs.py`).
- **`ai2tale_src/`**: the third-party AI2TALE reference implementation is a
  runtime dependency of the AI2TALE baseline
  (`src/baselines/run_ai2tale.py`); obtain it from the AI2TALE release. Our
  faithful PyTorch reimplementation used for the reported numbers is
  `src/baselines/ai2tale_faithful.py`.
- Raw corpus tensors and embedding caches (see Data above).

## License

Code released under the MIT License ([LICENSE](LICENSE)). The AI2TALE
dataset, name, and the original AI2TALE method remain property of their
authors; please cite Nguyen et al., ICLR 2025, when using the corpus.
