"""Per-email Spearman stratified by sentence count (review S4).

Recomputes the headline localization metric on the lambda_rule=3.0 multi-seed
checkpoints (seed 123), reporting:
  - mean Spearman on all defined emails vs >=3 / >=5-sentence subsets
  - stratification by sentence count (2 / 3 / 4-9 / 10+)
  - top-1 attention mass stats (>0.97 fraction, mean max_attn, e^H)
  - paired Wilcoxon sparse vs ABMIL over commonly defined emails

Output: experiments/spearman_stratified.json
Run: python3 experiments/scripts/spearman_stratified.py   (MPS; sandbox off)
"""

import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'experiments/scripts'))

from eval_faithfulness import build_model, ckpt, load_config, load_state
from src.dataset import create_dataloaders
from src.evaluate import load_keywords
from src.rules import batch_score

MODELS = {
    'sparse_seed123': ('cnn_sparse', ckpt('multi_seed_sparse_seed123/sparse_best_model.pt')),
    'abmil_seed123': ('cnn_abmil', ckpt('multi_seed_abmil_seed123/abmil_best_model.pt')),
}


def per_email_stats(model, kind, test_loader, device, keywords):
    is_sparse = kind == 'cnn_sparse'
    rows = {}
    with torch.no_grad():
        for batch in test_loader:
            tokens = batch['tokens'].to(device)
            labels = batch['label'].to(device)
            sent_lengths = batch['sent_lengths'].to(device)
            raw_sentences = batch['raw_sentences']
            ids = batch['email_idx'] if 'email_idx' in batch else None

            if is_sparse:
                _, attn_weights, _, _ = model(tokens, sent_lengths, temperature=0.1)
            else:
                _, attn_weights, _ = model(tokens, sent_lengths)

            rule_scores = batch_score(raw_sentences, keywords)
            B, S = attn_weights.shape
            for b in range(B):
                if labels[b].item() != 1:
                    continue
                n_valid = int((sent_lengths[b] > 0).sum().item())
                n_rule = rule_scores.shape[1]
                n_use = min(n_valid, n_rule)
                attn_b = attn_weights[b, :n_use].cpu().numpy()
                p = attn_b / max(attn_b.sum(), 1e-12)
                ent = float(-(p * np.log(p + 1e-10)).sum())
                entry = {
                    'n_sentences': n_use,
                    'max_attn': float(attn_b.max()) if n_use else None,
                    'entropy': ent,
                    'spearman': None,
                }
                if n_use >= 2 and len(set(attn_b.tolist())) >= 2:
                    rule_b = rule_scores[b, :n_use]
                    if len(set(rule_b.tolist())) >= 2:
                        corr, _ = spearmanr(attn_b, rule_b)
                        if not np.isnan(corr):
                            entry['spearman'] = float(corr)
                key = int(ids[b].item()) if ids is not None else len(rows)
                rows[key] = entry
    return rows


def summarize(rows):
    sp = [(r['n_sentences'], r['spearman']) for r in rows.values() if r['spearman'] is not None]
    vals = [s for _, s in sp]
    out = {
        'n_phishing': len(rows),
        'n_spearman_defined': len(sp),
        'n_excluded_1sent': sum(1 for r in rows.values() if r['n_sentences'] < 2),
        'mean_all': float(np.mean(vals)),
        'mean_ge3': float(np.mean([s for n, s in sp if n >= 3])),
        'n_ge3': sum(1 for n, _ in sp if n >= 3),
        'mean_ge5': float(np.mean([s for n, s in sp if n >= 5])),
        'n_ge5': sum(1 for n, _ in sp if n >= 5),
        'strata': {},
    }
    for lo, hi, tag in ((2, 2, '2'), (3, 3, '3'), (4, 9, '4-9'), (10, 10**9, '10+')):
        v = [s for n, s in sp if lo <= n <= hi]
        out['strata'][tag] = {'n': len(v), 'mean': float(np.mean(v)) if v else None}
    two = [s for n, s in sp if n == 2]
    out['two_sent_frac_plus1'] = float(np.mean([s > 0.99 for s in two])) if two else None
    ma = [(r['n_sentences'], r['max_attn']) for r in rows.values() if r['max_attn'] is not None]
    out['max_attn_mean'] = float(np.mean([m for _, m in ma]))
    out['max_attn_frac_gt97'] = float(np.mean([m > 0.97 for _, m in ma]))
    for lo, hi, tag in ((1, 2, 'le2'), (3, 5, '3-5'), (6, 10**9, '6+')):
        v = [m for n, m in ma if lo <= n <= hi]
        out[f'max_attn_{tag}'] = {'n': len(v), 'mean': float(np.mean(v)) if v else None,
                                   'frac_gt97': float(np.mean([m > 0.97 for m in v])) if v else None}
    ents = [r['entropy'] for r in rows.values()]
    out['entropy_mean'] = float(np.mean(ents))
    out['effective_support_eH_mean'] = float(np.mean(np.exp(ents)))
    return out


def main():
    config = load_config()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    _, test_loader = create_dataloaders(config)
    kw_path = os.path.join(project_root, config.get('keywords_file', 'phishingdata/kwords_stem.json'))
    keywords = load_keywords(kw_path)

    results = {}
    per_model_rows = {}
    for name, (kind, path) in MODELS.items():
        model = build_model(kind)
        model.load_state_dict(load_state(path), strict=False)
        model.to(device)
        model.eval()
        rows = per_email_stats(model, kind, test_loader, device, keywords)
        per_model_rows[name] = rows
        results[name] = summarize(rows)
        print(f"\n== {name} ==")
        print(json.dumps(results[name], indent=1))
        del model

    common = [k for k in per_model_rows['sparse_seed123']
              if k in per_model_rows['abmil_seed123']
              and per_model_rows['sparse_seed123'][k]['spearman'] is not None
              and per_model_rows['abmil_seed123'][k]['spearman'] is not None]
    a = [per_model_rows['sparse_seed123'][k]['spearman'] for k in common]
    b = [per_model_rows['abmil_seed123'][k]['spearman'] for k in common]
    stat, p = wilcoxon(a, b)
    results['wilcoxon_sparse_vs_abmil'] = {
        'n_paired': len(common), 'mean_sparse': float(np.mean(a)),
        'mean_abmil': float(np.mean(b)), 'statistic': float(stat), 'p_value': float(p),
    }
    print("\n== Wilcoxon paired (sparse vs abmil) ==")
    print(json.dumps(results['wilcoxon_sparse_vs_abmil'], indent=1))

    out_path = os.path.join(project_root, 'experiments/spearman_stratified.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=1)
    print(f"\nsaved -> {out_path}")


if __name__ == '__main__':
    main()
