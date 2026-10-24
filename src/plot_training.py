"""Extract training curves from log output and generate plots."""

import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def extract_epochs(output_file):
    """Parse epoch summary lines from training output."""
    epochs = []
    pattern = r'^Epoch (\d+): avg_loss=([\d.]+), acc=([\d.]+), bce=([\d.]+), sparse=([\d.]+), rule=([\d.]+)'
    with open(output_file, 'r') as f:
        for line in f:
            m = re.match(pattern, line.strip())
            if m:
                epochs.append({
                    'epoch': int(m.group(1)),
                    'loss': float(m.group(2)),
                    'acc': float(m.group(3)),
                    'bce': float(m.group(4)),
                    'sparse': float(m.group(5)),
                    'rule': float(m.group(6)),
                })
    return epochs


def plot_training_curves(sparse_file, abmil_file, output_path='experiments/training_curves.png'):
    sparse_epochs = extract_epochs(sparse_file)
    abmil_epochs = extract_epochs(abmil_file)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Accuracy
    s_epochs = [e['epoch'] for e in sparse_epochs]
    s_acc = [e['acc'] for e in sparse_epochs]
    a_epochs = [e['epoch'] for e in abmil_epochs]
    a_acc = [e['acc'] for e in abmil_epochs]

    axes[0].plot(s_epochs, s_acc, 'o-', label='Sparse ABMIL', color='steelblue', linewidth=2)
    axes[0].plot(a_epochs, a_acc, 's-', label='ABMIL Baseline', color='coral', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Loss decomposition (Sparse ABMIL)
    s_bce = [e['bce'] for e in sparse_epochs]
    s_sparse = [e['sparse'] for e in sparse_epochs]
    s_rule = [e['rule'] for e in sparse_epochs]
    s_total = [e['loss'] for e in sparse_epochs]

    axes[1].plot(s_epochs, s_total, 'ko-', label='Total', linewidth=2)
    axes[1].plot(s_epochs, s_bce, 'b--', label='BCE', alpha=0.7)
    axes[1].plot(s_epochs, s_sparse, 'g--', label='L1 Sparse', alpha=0.7)
    axes[1].plot(s_epochs, s_rule, 'r--', label='Rule Alignment', alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Sparse ABMIL Loss Decomposition')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # BCE comparison
    axes[2].plot(s_epochs, s_bce, 'o-', label='Sparse ABMIL (BCE)', color='steelblue', linewidth=2)
    axes[2].plot(a_epochs, [e['bce'] for e in abmil_epochs], 's-', label='ABMIL (BCE)', color='coral', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('BCE Loss')
    axes[2].set_title('Classification Loss Comparison')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot training curves from log files')
    parser.add_argument('sparse_log', help='Path to Sparse ABMIL training log')
    parser.add_argument('abmil_log', help='Path to ABMIL baseline training log')
    parser.add_argument('--output', default='experiments/training_curves.png',
                        help='Output path (default: experiments/training_curves.png)')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plot_training_curves(args.sparse_log, args.abmil_log, output_path=args.output)
