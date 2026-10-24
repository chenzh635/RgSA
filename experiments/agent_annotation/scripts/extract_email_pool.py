"""Extract per-email metrics + raw sentences for the whole phishing test set.

Adapted from experiments/scripts/spearman_stratified.py (same model, same
conventions). Produces the sampling pool for the spec's failure-mode strata
(groundtruth_annotation_spec §2):
  mismatch  = per-email Spearman(attn, rule) < 0
  short     = exactly 2 sentences (1-sentence emails are degenerate for P@1
              and were excluded from the original 150 as well)
  diffuse   = attention entropy > 1.5 and >= 6 sentences

Output (SIDECAR — contains model-derived fields + text, NEVER give to an
annotation agent):
  experiments/agent_annotation/tasks/_email_pool_metrics.json
    {email_idx: {n_sentences, spearman, entropy, max_attn, pred_prob,
                 sentences: [str, ...]}}

Run: python3 experiments/agent_annotation/scripts/extract_email_pool.py
     (MPS; sandbox off)
"""

import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'experiments/scripts'))

from eval_faithfulness import build_model, ckpt, load_config, load_state
from src.dataset import create_dataloaders
from src.evaluate import load_keywords
from src.rules import batch_score

CKPT = ckpt('multi_seed_sparse_seed123/sparse_best_model.pt')
OUT = os.path.join(project_root, 'experiments/agent_annotation/tasks/_email_pool_metrics.json')


def main():
    config = load_config()
    # optional argv[1] = device override ('cpu' avoids MPS contention with training)
    if len(sys.argv) > 1:
        device = torch.device(sys.argv[1])
    else:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    _, test_loader = create_dataloaders(config)
    kw_path = os.path.join(project_root,
                           config.get('keywords_file', 'phishingdata/kwords_stem.json'))
    keywords = load_keywords(kw_path)

    model = build_model('cnn_sparse')
    model.load_state_dict(load_state(CKPT), strict=False)
    model.to(device)
    model.eval()

    rows = {}
    with torch.no_grad():
        for batch in test_loader:
            tokens = batch['tokens'].to(device)
            labels = batch['label'].to(device)
            sent_lengths = batch['sent_lengths'].to(device)
            raw_sentences = batch['raw_sentences']
            ids = batch['idx'] if 'idx' in batch else batch.get('email_idx')
            output, attn_weights, _, _ = model(tokens, sent_lengths, temperature=0.1)
            probs = torch.sigmoid(output).cpu().numpy()
            rule_scores = batch_score(raw_sentences, keywords)
            B = attn_weights.shape[0]
            for b in range(B):
                if labels[b].item() != 1:
                    continue
                v = ids[b]
                eid = int(v.item() if hasattr(v, 'item') else v)
                n_raw = len(raw_sentences[b])
                n_valid = int((sent_lengths[b] > 0).sum().item())
                n_use = min(n_valid, rule_scores.shape[1], n_raw)
                attn_b = attn_weights[b, :n_use].cpu().numpy()
                p = attn_b / max(attn_b.sum(), 1e-12)
                ent = float(-(p * np.log(p + 1e-10)).sum())
                sp = None
                if n_use >= 2 and len(set(attn_b.tolist())) >= 2:
                    rule_b = rule_scores[b, :n_use]
                    if len(set(rule_b.tolist())) >= 2:
                        corr, _ = spearmanr(attn_b, rule_b)
                        if not np.isnan(corr):
                            sp = float(corr)
                rows[eid] = {
                    'n_sentences': n_raw,
                    'spearman': sp,
                    'entropy': ent,
                    'max_attn': float(attn_b.max()) if n_use else None,
                    'pred_prob': float(probs[b]),
                    'sentences': [raw_sentences[b][i] for i in range(n_raw)],
                }
            if len(rows) % 1000 < 32:
                print(f'  ... {len(rows)} phishing emails', flush=True)

    with open(OUT, 'w') as f:
        json.dump(rows, f)
    n_sp = sum(1 for r in rows.values() if r['spearman'] is not None)
    n_neg = sum(1 for r in rows.values()
                if r['spearman'] is not None and r['spearman'] < 0)
    n_short = sum(1 for r in rows.values() if r['n_sentences'] == 2)
    n_diff = sum(1 for r in rows.values()
                 if r['entropy'] > 1.5 and r['n_sentences'] >= 6)
    print(f'\npool: {len(rows)} phishing emails | spearman defined {n_sp} '
          f'| mismatch(<0) {n_neg} | short(=2) {n_short} | diffuse {n_diff}')
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main()
