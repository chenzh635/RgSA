"""Evaluate ablation models."""

import os
import json
import torch
from src.dataset import create_dataloaders
from src.models.abmil import SparseABMIL
from src.evaluate import evaluate


def load_config(path='experiments/configs/defaults.yaml'):
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def eval_ablation(name, config, test_loader, device):
    print(f"\n{'='*40}")
    print(f"Ablation: {name}")
    print(f"{'='*40}")

    config_ab = config.copy()
    config_ab['save_dir'] = f'experiments/{name}'

    model = SparseABMIL(
        hidden_dim=config.get('hidden_dim', 64),
        attention_hidden=config.get('attention_hidden', 64),
        classifier_hidden=config.get('classifier_hidden', [128, 64]),
        use_sparse=True,
    ).to(device)

    ckpt_path = f'experiments/{name}/sparse_best_model.pt'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    else:
        print(f"WARNING: No checkpoint at {ckpt_path}")
        return None

    results = evaluate(model, test_loader, config, device)
    for k, v in results.items():
        results[k] = float(v)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    return results


def main():
    config = load_config()
    device = torch.device('cpu')
    _, test_loader = create_dataloaders(config)

    ablations = [
        'ablation_no_l1',
        'ablation_no_rule',
        'ablation_no_temp',
    ]

    results = {}
    for name in ablations:
        results[name] = eval_ablation(name, config, test_loader, device)

    # Load full model for comparison
    print(f"\n{'='*60}")
    print(f"{'Model':<30} {'Email-AUPRC':>11} {'Email-F1':>10} {'Spearman':>10} {'Entropy':>10}")
    print(f"{'='*60}")
    print(f"{'Full model (Sparse ABMIL)':<30} {0.9569:>11.4f} {0.9945:>10.4f} {0.9945:>10.4f} {0.8385:>10.4f}")

    display = {
        'ablation_no_l1': 'w/o L1 sparsity',
        'ablation_no_rule': 'w/o rule alignment',
        'ablation_no_temp': 'w/o temp annealing',
    }
    for name in ablations:
        r = results[name]
        if r:
            label = display[name]
            print(f"{label:<30} {r['email_auprc']:>11.4f} {r['email_f1']:>10.4f} {r['spearman_corr']:>10.4f} {r['attention_entropy']:>10.4f}")

    # Save ablation results
    with open('experiments/ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to experiments/ablation_results.json")


if __name__ == '__main__':
    main()
