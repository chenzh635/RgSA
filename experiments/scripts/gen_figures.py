"""Generate qualitative analysis figures for the paper."""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.dataset import create_dataloaders
from src.models.abmil import SparseABMIL, ABMILBaseline
from src.rules import load_keywords, score_sentences


def load_config(path='experiments/configs/defaults.yaml'):
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def plot_attention_distribution(sparse_attns, abmil_attns, output_path='experiments/attention_dist.png'):
    """Compare attention weight distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(sparse_attns, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    axes[0].set_title('Sparse ABMIL Attention Weights')
    axes[0].set_xlabel('Attention Weight')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(abmil_attns, bins=50, color='coral', alpha=0.7, edgecolor='white')
    axes[1].set_title('ABMIL Baseline Attention Weights')
    axes[1].set_xlabel('Attention Weight')
    axes[1].set_ylabel('Count')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    # Also save as vector PDF
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved attention distribution to {output_path} and {pdf_path}")


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


def main():
    config = load_config()
    device = torch.device('cpu')
    _, test_loader = create_dataloaders(config)
    os.makedirs('experiments/analysis', exist_ok=True)

    # Load models
    sparse_model = SparseABMIL(
        hidden_dim=config.get('hidden_dim', 64),
        attention_hidden=config.get('attention_hidden', 64),
        classifier_hidden=config.get('classifier_hidden', [128, 64]),
    ).to(device)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sparse_ckpt = os.path.join(project_root, 'experiments', 'archive', 'sparse_best_model.pt')
    if not os.path.exists(sparse_ckpt):
        sparse_ckpt = os.path.join(project_root, 'experiments', 'sparse_best_model.pt')
    ckpt = torch.load(sparse_ckpt, map_location=device)
    sparse_model.load_state_dict(ckpt['model_state_dict'], strict=False)
    sparse_model.eval()
    print(f"Loaded sparse model from {sparse_ckpt}")

    abmil_model = ABMILBaseline(
        hidden_dim=config.get('hidden_dim', 64),
        attention_hidden=config.get('attention_hidden', 64),
        classifier_hidden=config.get('classifier_hidden', [128, 64]),
    ).to(device)
    abmil_ckpt = os.path.join(project_root, 'experiments', 'archive', 'abmil_best_model.pt')
    if not os.path.exists(abmil_ckpt):
        abmil_ckpt = os.path.join(project_root, 'experiments', 'abmil_best_model.pt')
    ckpt = torch.load(abmil_ckpt, map_location=device)
    abmil_model.load_state_dict(ckpt['model_state_dict'], strict=False)
    abmil_model.eval()
    print(f"Loaded ABMIL model from epoch {ckpt.get('epoch', '?')}")

    # Load keywords
    keywords_path = config.get('keywords_file', 'phishingdata/kwords_stem.json')
    if not os.path.isabs(keywords_path):
        keywords_path = os.path.join(project_root, keywords_path)
    keywords = load_keywords(keywords_path)

    all_sparse_attns = []
    all_abmil_attns = []
    shown = 0

    print("\nGenerating qualitative examples...")
    with torch.no_grad():
        for batch in test_loader:
            tokens = batch['tokens']
            sent_lengths = batch['sent_lengths']
            raw_sentences = batch['raw_sentences']
            labels = batch['label']

            _, sparse_attn, _, _ = sparse_model(tokens, sent_lengths, temperature=0.1)
            _, abmil_attn, _ = abmil_model(tokens, sent_lengths)

            B = len(labels)
            for b in range(B):
                valid_mask = sent_lengths[b] > 0
                n_valid = int(valid_mask.sum().item())

                sa = sparse_attn[b][valid_mask].numpy()
                aa = abmil_attn[b][valid_mask].numpy()

                all_sparse_attns.extend(sa.tolist())
                all_abmil_attns.extend(aa.tolist())

                if labels[b].item() == 1 and shown < 3 and n_valid > 2:
                    sentences = raw_sentences[b][:n_valid]
                    rule_scores = score_sentences(sentences, keywords)
                    plot_email_example(sentences, sa, aa, rule_scores, shown + 1)
                    print(f"  Saved compare_email_{shown + 1}.png ({n_valid} sentences)")
                    shown += 1

    # Attention distribution comparison
    plot_attention_distribution(all_sparse_attns, all_abmil_attns)

    print(f"\nDone. Generated {shown} email comparisons + attention distribution plot.")


if __name__ == '__main__':
    main()
