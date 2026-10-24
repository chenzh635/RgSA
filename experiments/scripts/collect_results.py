"""Run all evaluations and collect results for the paper."""

import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_curve, auc, f1_score

from src.dataset import create_dataloaders
from src.models.abmil import SparseABMIL, ABMILBaseline
from src.evaluate import evaluate
from src.rules import load_keywords, batch_score, score_sentences


def load_config(path='experiments/configs/defaults.yaml'):
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_model(model_type, config, device):
    if model_type == 'sparse':
        model = SparseABMIL(
            hidden_dim=config.get('hidden_dim', 64),
            attention_hidden=config.get('attention_hidden', 64),
            classifier_hidden=config.get('classifier_hidden', [128, 64]),
        )
    else:
        model = ABMILBaseline(
            hidden_dim=config.get('hidden_dim', 64),
            attention_hidden=config.get('attention_hidden', 64),
            classifier_hidden=config.get('classifier_hidden', [128, 64]),
        )
    model = model.to(device)
    return model


def load_checkpoint(model, model_type, device):
    ckpt_path = os.path.join('experiments', f'{model_type}_best_model.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join('experiments', 'best_model.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  Loaded checkpoint from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
        return True
    return False


def rules_baseline_full(test_loader, config):
    """Rules-only baseline with the new 4 metrics."""
    keywords_path = config.get('keywords_file', 'phishingdata/kwords_stem.json')
    if not os.path.isabs(keywords_path):
        project_root = os.path.dirname(os.path.abspath(__file__))
        keywords_path = os.path.join(project_root, keywords_path)
    keywords = load_keywords(keywords_path)

    all_labels = []
    all_scores = []  # max rule score per email (continuous prediction)
    all_entropies = []
    all_spearmans = []

    for batch in test_loader:
        labels = batch['label']
        raw_sentences = batch['raw_sentences']
        sent_lengths = batch['sent_lengths']

        rule_scores = batch_score(raw_sentences, keywords)

        B = len(labels)
        for b in range(B):
            valid_mask = sent_lengths[b] > 0
            n_valid = int(valid_mask.sum().item())
            scores = rule_scores[b, :n_valid]

            all_labels.append(labels[b].item())
            all_scores.append(float(scores.max()))

            # Entropy from normalized rule scores
            if scores.sum() > 0 and n_valid > 0:
                p = scores / scores.sum()
                entropy = -sum(float(pi) * np.log(float(pi) + 1e-10) for pi in p if float(pi) > 1e-10)
                all_entropies.append(entropy)

            # Spearman for phishing emails: rule scores vs themselves = perfect correlation
            if labels[b].item() == 1 and n_valid >= 2 and len(set(scores)) >= 2:
                all_spearmans.append(1.0)

    # Email-Level AUPRC
    if len(set(all_labels)) < 2:
        email_auprc = 0.0
    else:
        precision_arr, recall_arr, _ = precision_recall_curve(all_labels, all_scores)
        email_auprc = auc(recall_arr, precision_arr)

    # Email-Level F1
    preds = [1 if s > 0.3 else 0 for s in all_scores]
    email_f1 = f1_score(all_labels, preds, zero_division=0)

    # Spearman: rules-only = self-correlation = 1.0
    spearman_corr = np.mean(all_spearmans) if all_spearmans else 0.0

    # Entropy
    entropy = np.mean(all_entropies) if all_entropies else 0.0

    return {
        'email_auprc': email_auprc,
        'email_f1': email_f1,
        'spearman_corr': spearman_corr,
        'attention_entropy': entropy,
    }


def plot_attention_distribution(sparse_attns, abmil_attns, output_path='experiments/attention_dist.png'):
    """Compare attention weight distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Sparse ABMIL
    axes[0].hist(sparse_attns, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    axes[0].set_title('Sparse ABMIL Attention Weights')
    axes[0].set_xlabel('Attention Weight')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)

    # ABMIL
    axes[1].hist(abmil_attns, bins=50, color='coral', alpha=0.7, edgecolor='white')
    axes[1].set_title('ABMIL Baseline Attention Weights')
    axes[1].set_xlabel('Attention Weight')
    axes[1].set_ylabel('Count')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved attention distribution plot to {output_path}")


def plot_email_example(sentences, sparse_attn, abmil_attn, rule_scores, idx, output_dir='experiments/analysis'):
    """Plot attention comparison for a single email."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    n = len(sentences)
    x = np.arange(n)

    ax1.bar(x, sparse_attn, color='steelblue', alpha=0.8)
    ax1.set_ylabel('Sparse ABMIL\nAttention')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(sparse_attn.max(), color='red', linestyle='--', alpha=0.5, label=f'top-1: idx={sparse_attn.argmax()}')
    ax1.legend(fontsize=8)

    ax2.bar(x, abmil_attn, color='coral', alpha=0.8)
    ax2.set_ylabel('ABMIL Baseline\nAttention')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(abmil_attn.max(), color='red', linestyle='--', alpha=0.5, label=f'top-1: idx={abmil_attn.argmax()}')
    ax2.legend(fontsize=8)

    ax3.bar(x, rule_scores, color='forestgreen', alpha=0.8)
    ax3.set_ylabel('Rule Score')
    ax3.set_xlabel('Sentence Index')
    ax3.grid(True, alpha=0.3)

    short_sents = [s[:60] + '...' if len(s) > 60 else s for s in sentences]
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_sents, rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    fig.savefig(f'{output_dir}/compare_email_{idx}.png', dpi=150, bbox_inches='tight')
    plt.close()


def collect_all_results():
    config = load_config()
    device = torch.device('cpu')
    _, test_loader = create_dataloaders(config)

    results = {}

    # Rules-only baseline (no model needed)
    print("\n1. Rules-only baseline...")
    results['Rules-only'] = rules_baseline_full(test_loader, config)
    print(f"   Email-AUPRC: {results['Rules-only']['email_auprc']:.4f}")
    print(f"   Email-F1: {results['Rules-only']['email_f1']:.4f}")
    print(f"   Spearman: {results['Rules-only']['spearman_corr']:.4f}")
    print(f"   Entropy: {results['Rules-only']['attention_entropy']:.4f}")

    # ABMIL Baseline
    print("\n2. ABMIL Baseline...")
    model = load_model('abmil', config, device)
    if load_checkpoint(model, 'abmil', device):
        results['ABMIL'] = evaluate(model, test_loader, config, device)
    else:
        print("   WARNING: No checkpoint found. Running with untrained model.")
        results['ABMIL'] = evaluate(model, test_loader, config, device)
    print(f"   Email-AUPRC: {results['ABMIL']['email_auprc']:.4f}")
    print(f"   Email-F1: {results['ABMIL']['email_f1']:.4f}")
    print(f"   Spearman: {results['ABMIL']['spearman_corr']:.4f}")
    print(f"   Entropy: {results['ABMIL']['attention_entropy']:.4f}")

    # Sparse ABMIL
    print("\n3. Sparse ABMIL (Ours)...")
    model = load_model('sparse', config, device)
    if load_checkpoint(model, 'sparse', device):
        results['Sparse ABMIL'] = evaluate(model, test_loader, config, device)
    else:
        print("   WARNING: No checkpoint found. Running with untrained model.")
        results['Sparse ABMIL'] = evaluate(model, test_loader, config, device)
    print(f"   Email-AUPRC: {results['Sparse ABMIL']['email_auprc']:.4f}")
    print(f"   Email-F1: {results['Sparse ABMIL']['email_f1']:.4f}")
    print(f"   Spearman: {results['Sparse ABMIL']['spearman_corr']:.4f}")
    print(f"   Entropy: {results['Sparse ABMIL']['attention_entropy']:.4f}")

    # Convert numpy types to Python floats for JSON serialization
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    results = convert_types(results)

    # Save results
    output_dir = 'experiments'
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Qualitative analysis: compare attention on sample phishing emails
    print("\n4. Qualitative analysis...")
    os.makedirs('experiments/analysis', exist_ok=True)

    sparse_model = load_model('sparse', config, device)
    abmil_model = load_model('abmil', config, device)
    load_checkpoint(sparse_model, 'sparse', device)
    load_checkpoint(abmil_model, 'abmil', device)
    sparse_model.eval()
    abmil_model.eval()

    keywords_path = config.get('keywords_file', 'phishingdata/kwords_stem.json')
    if not os.path.isabs(keywords_path):
        project_root = os.path.dirname(os.path.abspath(__file__))
        keywords_path = os.path.join(project_root, keywords_path)
    keywords = load_keywords(keywords_path)

    all_sparse_attns = []
    all_abmil_attns = []
    shown = 0

    for batch in test_loader:
        tokens = batch['tokens']
        sent_lengths = batch['sent_lengths']
        raw_sentences = batch['raw_sentences']
        labels = batch['label']

        with torch.no_grad():
            _, sparse_attn, _, _ = sparse_model(tokens, sent_lengths, temperature=0.1)
            _, abmil_attn, _ = abmil_model(tokens, sent_lengths)

        B = len(labels)
        for b in range(B):
            valid_mask = sent_lengths[b] > 0
            n_valid = valid_mask.sum().item()

            sa = sparse_attn[b][valid_mask].numpy()
            aa = abmil_attn[b][valid_mask].numpy()

            all_sparse_attns.extend(sa.tolist())
            all_abmil_attns.extend(aa.tolist())

            if labels[b].item() == 1 and shown < 3 and n_valid > 2:
                sentences = raw_sentences[b][:n_valid]
                rule_scores = score_sentences(sentences, keywords)
                plot_email_example(sentences, sa, aa, rule_scores, shown + 1)
                shown += 1

    # Attention distribution comparison
    plot_attention_distribution(all_sparse_attns, all_abmil_attns)

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"{'Metric':<20} {'Rules-only':>12} {'ABMIL':>12} {'Sparse ABMIL':>12}")
    print(f"{'='*60}")
    for metric in ['email_auprc', 'email_f1', 'spearman_corr', 'attention_entropy']:
        rule_v = f"{results['Rules-only'][metric]:.4f}"
        abmil_v = f"{results['ABMIL'][metric]:.4f}"
        sparse_v = f"{results['Sparse ABMIL'][metric]:.4f}"
        print(f"{metric:<20} {rule_v:>12} {abmil_v:>12} {sparse_v:>12}")
    print(f"{'='*60}")


if __name__ == '__main__':
    collect_all_results()
