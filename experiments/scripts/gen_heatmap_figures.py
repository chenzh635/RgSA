"""Generate block-matrix heatmap figures for the paper.

For each sample phishing email, renders sentences as colored blocks where
color intensity encodes the attention/risk score. Only top-3 sentences are
labeled below the figure.
"""

import os
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from src.dataset import create_dataloaders
from src.models.abmil import SparseABMIL, ABMILBaseline
from src.rules import load_keywords, score_sentences


def load_config(path='experiments/configs/defaults.yaml'):
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def find_good_examples(sparse_model, abmil_model, test_loader, keywords, n_examples=4):
    """Find phishing emails where sparse attention is highly concentrated on a high-rule-score sentence."""
    examples = []

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
                if labels[b].item() != 1:
                    continue

                sentences = raw_sentences[b]
                n_valid = len(sentences)
                if n_valid < 4 or n_valid > 15:
                    continue
                rich_count = sum(1 for s in sentences if len(s.split()) > 5)
                if rich_count < 3:
                    continue

                sa = sparse_attn[b][:n_valid].numpy()
                aa = abmil_attn[b][:n_valid].numpy()
                rule_scores = score_sentences(sentences, keywords)

                p = sa / sa.sum()
                entropy = -(p * np.log(p + 1e-10)).sum()
                max_entropy = np.log(n_valid)
                norm_entropy = entropy / max_entropy

                top1_idx = sa.argmax()
                top1_rule = rule_scores[top1_idx]

                score = (1 - norm_entropy) + top1_rule

                examples.append({
                    'sentences': sentences,
                    'sparse_attn': sa,
                    'abmil_attn': aa,
                    'rule_scores': rule_scores,
                    'entropy': entropy,
                    'norm_entropy': norm_entropy,
                    'top1_rule': top1_rule,
                    'score': score,
                    'label': labels[b].item(),
                })

    examples.sort(key=lambda x: -x['score'])
    return examples[:n_examples]


def get_top3_sentences(sentences, sparse_attn, abmil_attn, rule_scores):
    """Get the top-3 sentences by combined importance across all models."""
    n = len(sentences)
    # Combined score: normalize each and average
    sa_norm = sparse_attn / (sparse_attn.max() + 1e-10)
    aa_norm = abmil_attn / (abmil_attn.max() + 1e-10)
    combined = (sa_norm + aa_norm + rule_scores) / 3.0
    top3_idx = np.argsort(combined)[::-1][:3]
    return sorted(top3_idx.tolist())


def score_to_color(score, cmap):
    """Map a score [0, 1] to an RGB color."""
    return cmap(score)


def plot_block_heatmap(sentences, sparse_attn, abmil_attn, rule_scores, idx, output_dir):
    """Render block-matrix heatmap for one email example."""
    n = len(sentences)
    top3_idx = get_top3_sentences(sentences, sparse_attn, abmil_attn, rule_scores)

    # Three-row layout: Sparse ABMIL, ABMIL Baseline, Rule Score
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 4.0),
                             gridspec_kw={'height_ratios': [1, 1, 1], 'hspace': 0.15})

    # Color map: light gray (0) -> orange -> dark red (1)
    cmap = LinearSegmentedColormap.from_list('risk_heat', [
        (0.92, 0.92, 0.92),   # light gray for near-zero
        (0.95, 0.75, 0.55),   # light orange
        (1.0, 0.55, 0.30),    # orange
        (0.85, 0.12, 0.08),   # dark red
    ])

    # Normalize scores for display
    max_sparse = sparse_attn.max() + 1e-10
    max_abmil = abmil_attn.max() + 1e-10
    max_rule = rule_scores.max() + 1e-10

    row_labels = ['Sparse ABMIL', 'ABMIL Baseline', 'Rule Score']
    score_arrays = [sparse_attn / max_sparse, abmil_attn / max_abmil, rule_scores / max_rule]

    fig.subplots_adjust(left=0.15, bottom=0.28)

    for row in range(3):
        ax = axes[row]
        scores = score_arrays[row]
        ax.set_xlim(0, n)
        ax.set_ylim(0, 1)
        ax.axis('off')

        for i in range(n):
            color = score_to_color(scores[i], cmap)
            rect = Rectangle((i, 0.1), 0.92, 0.8, facecolor=color,
                             edgecolor='white', linewidth=1.5, alpha=0.95)
            ax.add_patch(rect)

        for ti in top3_idx:
            rect = Rectangle((ti, 0.1), 0.92, 0.8, facecolor='none',
                             edgecolor='black', linewidth=2.5, alpha=0.8)
            ax.add_patch(rect)

    fig.canvas.draw()

    for row, ax in enumerate(axes):
        bbox = ax.get_position()
        y_center = (bbox.y0 + bbox.y1) / 2.0
        fig.text(0.01, y_center, row_labels[row], fontsize=7.5,
                 fontweight='bold', rotation=0, ha='left', va='center',
                 transform=fig.transFigure)

    y_start = 0.24
    for rank, si in enumerate(top3_idx):
        sent_text = sentences[si]
        if len(sent_text) > 100:
            sent_text = sent_text[:97] + '...'
        label = f'S{rank+1}: {sent_text}'
        fig.text(0.05, y_start - rank * 0.055, label, fontsize=7,
                 fontweight='bold' if rank == 0 else 'normal',
                 bbox=dict(boxstyle='round,pad=0.04', facecolor='lightyellow',
                           edgecolor='gray' if rank > 0 else 'darkred',
                           alpha=0.6))

    output_path = os.path.join(output_dir, f'block_heatmap_email_{idx}.png')
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path} ({n} sentences, top-3: {top3_idx})")


def plot_combined_block_heatmap(examples, output_dir):
    """Create a single figure with all examples as block matrices."""
    n_examples = len(examples)
    row_labels = ['Sparse ABMIL', 'ABMIL Baseline', 'Rule Score']
    n_rows = n_examples * 3

    fig, axes = plt.subplots(n_rows, 1, figsize=(9.5, 2.2 * n_examples),
                             gridspec_kw={'height_ratios': [1] * n_rows,
                                          'hspace': 0.06,
                                          'left': 0.15},
                             constrained_layout=False)
    if n_rows == 1:
        axes = [axes]

    # Adjust bottom margin for text labels
    fig.subplots_adjust(bottom=0.18 + 0.04 * n_examples)

    cmap = LinearSegmentedColormap.from_list('risk_heat', [
        (0.92, 0.92, 0.92),
        (0.95, 0.75, 0.55),
        (1.0, 0.55, 0.30),
        (0.85, 0.12, 0.08),
    ])

    # Draw blocks first
    ax_idx = 0
    for ex_idx, ex in enumerate(examples):
        sentences = ex['sentences']
        sa = ex['sparse_attn']
        aa = ex['abmil_attn']
        rs = ex['rule_scores']
        n = len(sentences)
        top3_idx = get_top3_sentences(sentences, sa, aa, rs)

        max_sparse = sa.max() + 1e-10
        max_abmil = aa.max() + 1e-10
        max_rule = rs.max() + 1e-10
        score_arrays = [sa / max_sparse, aa / max_abmil, rs / max_rule]

        for row in range(3):
            ax = axes[ax_idx]
            scores = score_arrays[row]
            ax.set_xlim(0, n)
            ax.set_ylim(0, 1)
            ax.axis('off')

            for i in range(n):
                color = score_to_color(scores[i], cmap)
                rect = Rectangle((i, 0.1), 0.92, 0.8, facecolor=color,
                                 edgecolor='white', linewidth=1.2, alpha=0.95)
                ax.add_patch(rect)

            for ti in top3_idx:
                rect = Rectangle((ti, 0.1), 0.92, 0.8, facecolor='none',
                                 edgecolor='black', linewidth=2.0, alpha=0.7)
                ax.add_patch(rect)

            if row == 2 and ex_idx < n_examples - 1:
                ax.axhline(1.05, color='gray', linewidth=0.5, linestyle='--')

            ax_idx += 1

    # Now draw to compute actual positions
    fig.canvas.draw()

    # Add row labels using actual axis positions
    for ax_idx, ax in enumerate(axes):
        bbox = ax.get_position()
        y_center = (bbox.y0 + bbox.y1) / 2.0
        label = row_labels[ax_idx % 3]
        fig.text(0.01, y_center, label, fontsize=7, fontweight='bold',
                 rotation=0, ha='left', va='center', transform=fig.transFigure)

    # Add example labels at bottom
    y_pos = 0.15
    for ex_idx, ex in enumerate(examples):
        sentences = ex['sentences']
        sa = ex['sparse_attn']
        aa = ex['abmil_attn']
        rs = ex['rule_scores']
        top3_idx = get_top3_sentences(sentences, sa, aa, rs)

        fig.text(0.03, y_pos, f'Example {ex_idx + 1} (entropy={ex["norm_entropy"]:.3f}):',
                 fontsize=7, fontweight='bold')
        y_pos -= 0.03
        for rank, si in enumerate(top3_idx):
            sent = sentences[si]
            if len(sent) > 90:
                sent = sent[:87] + '...'
            fig.text(0.05, y_pos, f'  S{rank+1}: {sent}', fontsize=6.5,
                     style='italic')
            y_pos -= 0.025
        y_pos -= 0.02

    output_path = os.path.join(output_dir, 'block_heatmap_combined.png')
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved combined figure {output_path}")


def main():
    config = load_config()
    device = torch.device('cpu')
    _, test_loader = create_dataloaders(config)
    output_dir = 'experiments/analysis'
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    sparse_model = SparseABMIL(
        hidden_dim=config.get('hidden_dim', 64),
        attention_hidden=config.get('attention_hidden', 64),
        classifier_hidden=config.get('classifier_hidden', [128, 64]),
        use_sparse=True,
    ).to(device)
    ckpt = torch.load('experiments/sparse_best_model.pt', map_location=device)
    sparse_model.load_state_dict(ckpt['model_state_dict'], strict=False)
    sparse_model.eval()

    abmil_model = ABMILBaseline(
        hidden_dim=config.get('hidden_dim', 64),
        attention_hidden=config.get('attention_hidden', 64),
        classifier_hidden=config.get('classifier_hidden', [128, 64]),
    ).to(device)
    ckpt = torch.load('experiments/abmil_best_model.pt', map_location=device)
    abmil_model.load_state_dict(ckpt['model_state_dict'], strict=False)
    abmil_model.eval()

    # Load keywords
    keywords_path = config.get('keywords_file', 'phishingdata/kwords_stem.json')
    if not os.path.isabs(keywords_path):
        keywords_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), keywords_path)
    keywords = load_keywords(keywords_path)

    print("Searching for best qualitative examples...")
    examples = find_good_examples(sparse_model, abmil_model, test_loader, keywords, n_examples=4)

    for i, ex in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"  Sentences: {len(ex['sentences'])}")
        print(f"  Sparse entropy (normalized): {ex['norm_entropy']:.3f}")
        print(f"  Top-1 rule score: {ex['top1_rule']:.2f}")
        print(f"  Top-1 sentence: {ex['sentences'][ex['sparse_attn'].argmax()][:80]}...")

    # Plot individual block heatmaps
    print("\nGenerating individual block heatmap figures...")
    for i, ex in enumerate(examples):
        plot_block_heatmap(
            ex['sentences'], ex['sparse_attn'], ex['abmil_attn'], ex['rule_scores'],
            i + 1, output_dir
        )

    # Plot combined figure
    print("\nGenerating combined block heatmap figure...")
    plot_combined_block_heatmap(examples, output_dir)

    # Save metadata
    metadata = []
    for ex in examples:
        metadata.append({
            'sentences': ex['sentences'],
            'sparse_attn': [float(x) for x in ex['sparse_attn']],
            'abmil_attn': [float(x) for x in ex['abmil_attn']],
            'rule_scores': [float(x) for x in ex['rule_scores']],
            'entropy': float(ex['entropy']),
        })
    with open(os.path.join(output_dir, 'block_heatmap_examples.json'), 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Generated {len(examples)} block heatmap examples.")


if __name__ == '__main__':
    main()
