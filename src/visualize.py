"""Qualitative analysis: visualize attention weights on sample emails."""

import os
import json
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


def visualize_attention(model_name='sparse', n_samples=5):
    config = load_config()
    device = torch.device('cpu')

    _, test_loader = create_dataloaders(config)

    # Load model
    if model_name == 'sparse':
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

    ckpt_path = os.path.join(config['save_dir'], 'best_model.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded {model_name} from {ckpt_path}")
    else:
        print(f"No checkpoint found at {ckpt_path}")
        return

    model.eval()

    # Load keywords for rule scoring
    keywords_path = config.get('keywords_file', 'phishingdata/kwords_stem.json')
    if not os.path.isabs(keywords_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        keywords_path = os.path.join(project_root, keywords_path)
    keywords = load_keywords(keywords_path)

    # Find phishing emails and visualize
    phishing_count = 0
    output_dir = 'experiments/analysis'
    os.makedirs(output_dir, exist_ok=True)

    for batch_idx, batch in enumerate(test_loader):
        if phishing_count >= n_samples:
            break

        labels = batch['label']
        tokens = batch['tokens']
        sent_lengths = batch['sent_lengths']
        raw_sentences = batch['raw_sentences']

        with torch.no_grad():
            is_sparse = hasattr(model, 'use_sparse') and model.use_sparse
            if is_sparse:
                output, attn_weights, _, _ = model(tokens, sent_lengths, temperature=0.1)
            else:
                output, attn_weights, _ = model(tokens, sent_lengths)

        B = len(labels)
        for b in range(B):
            if labels[b].item() != 1:
                continue
            if phishing_count >= n_samples:
                break

            valid_mask = sent_lengths[b] > 0
            n_valid = valid_mask.sum().item()
            attn_b = attn_weights[b][valid_mask].numpy()
            sentences = raw_sentences[b][:n_valid]

            # Rule scores
            rule_scores = score_sentences(sentences, keywords)

            # Plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

            x_pos = np.arange(n_valid)
            bar_width = 0.35

            ax1.bar(x_pos, attn_b, bar_width, color='steelblue', alpha=0.8)
            ax1.set_ylabel('Attention Weight')
            ax1.set_title(f'Phishing Email #{phishing_count+1} ({n_valid} sentences)')
            ax1.grid(True, alpha=0.3)

            ax2.bar(x_pos, rule_scores, bar_width, color='coral', alpha=0.8)
            ax2.set_ylabel('Rule Score')
            ax2.set_xlabel('Sentence Index')
            ax2.grid(True, alpha=0.3)

            # Add sentence text as x-tick labels on bottom plot
            short_sents = [s[:50] + '...' if len(s) > 50 else s for s in sentences]
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(short_sents, rotation=45, ha='right', fontsize=7)

            plt.tight_layout()
            fig.savefig(f'{output_dir}/{model_name}_email_{phishing_count+1}.png', dpi=150, bbox_inches='tight')
            plt.close()

            # Print details
            top1_idx = attn_b.argmax()
            print(f"\n--- Phishing Email #{phishing_count+1} ---")
            print(f"  Sentences: {n_valid}")
            print(f"  Top-1 sentence (idx={top1_idx}, attn={attn_b[top1_idx]:.4f}):")
            print(f"    {sentences[top1_idx]}")
            print(f"  Rule score for top-1: {rule_scores[top1_idx]:.4f}")

            phishing_count += 1


if __name__ == '__main__':
    visualize_attention('sparse')
