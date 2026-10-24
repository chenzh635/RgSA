"""Failure case analysis for Sparse ABMIL phishing localization.

Identifies and categorizes failure modes:
1. Low Spearman emails (attention disagrees with rules)
2. Misclassified emails (wrong prediction)
3. High-entropy emails (diffuse attention)
"""

import os
import sys
import json
import yaml
import numpy as np
import torch
from scipy.stats import spearmanr

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models.abmil import SparseABMIL
from src.dataset import create_dataloaders
from src.rules import batch_score


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(config, checkpoint_path, device):
    model = SparseABMIL(
        vocab_size=config.get('vocab_size', 311774),
        embed_dim=config.get('embed_dim', 150),
        hidden_dim=config.get('hidden_dim', 150),
        attention_hidden=config.get('attention_hidden', 64),
        use_sparse=True,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def analyze_failures(model, test_loader, keywords, device, output_path):
    """Run detailed failure analysis on the test set."""

    results = {
        'phishing_emails': [],
        'summary': {},
    }

    all_spearmans = []
    all_entropies = []
    all_pred_probs = []
    all_labels_list = []

    with torch.no_grad():
        for batch in test_loader:
            tokens = batch['tokens'].to(device)
            labels = batch['label'].to(device)
            sent_lengths = batch['sent_lengths'].to(device)
            raw_sentences = batch['raw_sentences']

            output, attn_weights, attn_scores, _ = model(tokens, sent_lengths, temperature=0.1)
            pred_probs = torch.sigmoid(output).cpu().numpy()
            rule_scores = batch_score(raw_sentences, keywords)

            B, S = attn_weights.shape

            for b in range(B):
                label = labels[b].item()
                pred_prob = float(pred_probs[b])
                pred = 1 if pred_prob > 0.5 else 0

                entry = {
                    'label': label,
                    'pred': pred,
                    'pred_prob': pred_prob,
                    'correct': pred == label,
                    'n_sentences': int((sent_lengths[b] > 0).sum().item()),
                }

                valid_mask = sent_lengths[b] > 0
                n_valid = int(valid_mask.sum().item())

                if n_valid > 0:
                    attn_b = attn_weights[b, :n_valid].cpu().numpy()
                    p = attn_b / (attn_b.sum() + 1e-10)
                    entropy = float(-np.sum(p * np.log(p + 1e-10)))
                    entry['entropy'] = entropy
                    entry['max_attn'] = float(np.max(attn_b))
                    entry['attn_weights'] = attn_b.tolist()

                    # Top-3 attention sentences
                    top_indices = np.argsort(-attn_b)[:3]
                    entry['top_sentences'] = []
                    for idx in top_indices:
                        if idx < len(raw_sentences[b]):
                            entry['top_sentences'].append({
                                'idx': int(idx),
                                'text': raw_sentences[b][idx][:120],
                                'attn': float(attn_b[idx]),
                            })

                    # Rule scores
                    n_rule = rule_scores.shape[1]
                    n_use = min(n_valid, n_rule)
                    rule_b = rule_scores[b, :n_use]
                    entry['rule_scores'] = rule_b.tolist()

                    # Spearman for phishing emails
                    if label == 1 and n_use >= 2:
                        attn_use = attn_b[:n_use]
                        if len(set(attn_use)) >= 2 and len(set(rule_b)) >= 2:
                            corr, pval = spearmanr(attn_use, rule_b)
                            if not np.isnan(corr):
                                entry['spearman'] = float(corr)
                                entry['spearman_pval'] = float(pval)
                                all_spearmans.append(corr)

                    all_entropies.append(entropy)

                all_pred_probs.append(pred_prob)
                all_labels_list.append(label)

                if label == 1:
                    results['phishing_emails'].append(entry)
                else:
                    # Track benign emails for misclassification analysis
                    results.setdefault('benign_emails', []).append(entry)

    # Categorize failures
    phishing = results['phishing_emails']

    # 1. Low Spearman (bottom 5%)
    spearman_values = sorted([e['spearman'] for e in phishing if 'spearman' in e])
    if spearman_values:
        threshold_5pct = spearman_values[max(0, len(spearman_values) // 20)]
        low_spearman = [e for e in phishing if e.get('spearman', 1.0) <= threshold_5pct]
    else:
        low_spearman = []

    # 2. Misclassified phishing emails
    misclassified = [e for e in phishing if not e['correct']]

    # 3. High entropy (top 5%)
    entropy_values = sorted([e['entropy'] for e in phishing if 'entropy' in e])
    if entropy_values:
        threshold_95pct = entropy_values[min(len(entropy_values) - 1, int(0.95 * len(entropy_values)))]
        high_entropy = [e for e in phishing if e.get('entropy', 0) >= threshold_95pct]
    else:
        high_entropy = []

    # 4. Benign emails misclassified as phishing
    benign_misclassified = [e for e in results.get('benign_emails', []) if not e['correct']]

    # Summary statistics
    results['summary'] = {
        'total_phishing': len(phishing),
        'total_benign': len(results.get('benign_emails', [])),
        'n_low_spearman': len(low_spearman),
        'spearman_threshold': float(threshold_5pct) if spearman_values else None,
        'n_misclassified_phishing': len(misclassified),
        'n_misclassified_benign': len(benign_misclassified),
        'n_high_entropy': len(high_entropy),
        'entropy_threshold': float(threshold_95pct) if entropy_values else None,
        'spearman_stats': {
            'mean': float(np.mean(spearman_values)) if spearman_values else None,
            'median': float(np.median(spearman_values)) if spearman_values else None,
            'std': float(np.std(spearman_values)) if spearman_values else None,
            'min': float(np.min(spearman_values)) if spearman_values else None,
            'max': float(np.max(spearman_values)) if spearman_values else None,
        },
        'entropy_stats': {
            'mean': float(np.mean(entropy_values)) if entropy_values else None,
            'median': float(np.median(entropy_values)) if entropy_values else None,
            'std': float(np.std(entropy_values)) if entropy_values else None,
        },
    }

    # Detailed failure cases (for paper)
    results['failure_cases'] = {
        'low_spearman': sorted(low_spearman, key=lambda x: x.get('spearman', 1.0))[:10],
        'misclassified_phishing': misclassified[:10],
        'high_entropy': sorted(high_entropy, key=lambda x: x.get('entropy', 0), reverse=True)[:10],
        'benign_misclassified': benign_misclassified[:10],
    }

    # Categorize failure patterns
    patterns = categorize_failure_patterns(phishing)
    results['patterns'] = patterns

    # Remove raw attn_weights from saved data to keep file small
    for entry_list in [results['phishing_emails'], results.get('benign_emails', [])]:
        for e in entry_list:
            e.pop('attn_weights', None)
            e.pop('rule_scores', None)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


def categorize_failure_patterns(phishing_emails):
    """Categorize common failure patterns."""
    patterns = {
        'short_emails': [],         # 1-2 sentences, hard to localize
        'diffuse_attention': [],     # attention spread across many sentences
        'rule_attention_mismatch': [],  # model attends to different sentences than rules
        'confident_wrong': [],       # high confidence but wrong prediction
    }

    for e in phishing_emails:
        n_sent = e['n_sentences']
        spearman = e.get('spearman', None)
        entropy = e.get('entropy', None)
        correct = e['correct']

        # Short emails (1-2 sentences)
        if n_sent <= 2:
            patterns['short_emails'].append(e)

        # Diffuse attention (entropy > 1.5)
        if entropy is not None and entropy > 1.5:
            patterns['diffuse_attention'].append(e)

        # Rule-attention mismatch
        if spearman is not None and spearman < 0:
            patterns['rule_attention_mismatch'].append(e)

        # Confident but wrong
        if not correct and e['pred_prob'] > 0.8:
            patterns['confident_wrong'].append(e)

    # Convert to counts and percentages
    total = len(phishing_emails)
    pattern_summary = {}
    for name, cases in patterns.items():
        pattern_summary[name] = {
            'count': len(cases),
            'pct': round(100 * len(cases) / total, 1) if total > 0 else 0,
        }

    return pattern_summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(project_root, 'experiments', 'configs', 'defaults.yaml'))
    parser.add_argument('--checkpoint', default=os.path.join(project_root, 'experiments', 'archive', 'sparse_best_model.pt'))
    parser.add_argument('--output', default=os.path.join(project_root, 'experiments', 'archive', 'failure_analysis.json'))
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    config = load_config(args.config)
    model = load_model(config, args.checkpoint, device)
    print("Model loaded.")

    # Load test dataset using the standard data loading pipeline
    _, test_loader = create_dataloaders(config)
    print(f"Test set: {len(test_loader.dataset)} emails")

    # Load keywords
    keywords_path = os.path.join(project_root, 'phishingdata', 'kwords_stem.json')
    with open(keywords_path) as f:
        keywords = json.load(f)

    # Run analysis
    print("Running failure analysis...")
    results = analyze_failures(model, test_loader, keywords, device, args.output)

    # Print summary
    s = results['summary']
    print("\n=== Failure Analysis Summary ===")
    print(f"Total phishing emails: {s['total_phishing']}")
    print(f"Low Spearman (bottom 5%): {s['n_low_spearman']} (threshold: {s['spearman_threshold']:.3f})")
    print(f"Misclassified phishing: {s['n_misclassified_phishing']}")
    print(f"Misclassified benign: {s['n_misclassified_benign']}")
    print(f"High entropy (top 5%): {s['n_high_entropy']} (threshold: {s['entropy_threshold']:.3f})")

    print(f"\nSpearman: mean={s['spearman_stats']['mean']:.3f}, median={s['spearman_stats']['median']:.3f}, std={s['spearman_stats']['std']:.3f}")
    print(f"Spearman range: [{s['spearman_stats']['min']:.3f}, {s['spearman_stats']['max']:.3f}]")
    print(f"Entropy: mean={s['entropy_stats']['mean']:.3f}, median={s['entropy_stats']['median']:.3f}, std={s['entropy_stats']['std']:.3f}")

    print("\n=== Failure Patterns ===")
    for name, info in results['patterns'].items():
        print(f"  {name}: {info['count']} ({info['pct']}%)")

    print(f"\nDetailed results saved to: {args.output}")

    # Print top-5 worst Spearman cases
    print("\n=== Top-5 Worst Spearman Cases ===")
    for i, case in enumerate(results['failure_cases']['low_spearman'][:5]):
        print(f"\n--- Case {i+1}: Spearman={case.get('spearman', 'N/A'):.3f}, Entropy={case.get('entropy', 'N/A'):.3f}, "
              f"N_sentences={case['n_sentences']}, Correct={case['correct']} ---")
        for sent in case.get('top_sentences', []):
            print(f"  [{sent['attn']:.4f}] {sent['text']}")


if __name__ == '__main__':
    main()
