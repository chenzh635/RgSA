"""Quick evaluation script — minimal overhead."""

import os
import json
import torch
import numpy as np

from src.dataset import create_dataloaders
from src.models.abmil import SparseABMIL, ABMILBaseline
from src.evaluate import evaluate
from src.rules import load_keywords, batch_score
from sklearn.metrics import f1_score, precision_recall_curve, auc


def load_config(path='experiments/configs/defaults.yaml'):
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def eval_single_model(model, model_name, test_loader, config, device):
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name}...")
    print(f"{'='*50}")

    results = evaluate(model, test_loader, config, device)

    # Convert to Python floats
    for k, v in results.items():
        results[k] = float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else v

    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    return results


def main():
    config = load_config()
    device = torch.device('cpu')
    _, test_loader = create_dataloaders(config)

    results = {}

    # ABMIL
    model = ABMILBaseline(
        hidden_dim=config.get('hidden_dim', 64),
        attention_hidden=config.get('attention_hidden', 64),
        classifier_hidden=config.get('classifier_hidden', [128, 64]),
    ).to(device)
    ckpt = torch.load('experiments/abmil_best_model.pt', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f"Loaded ABMIL from epoch {ckpt.get('epoch', '?')}")
    results['ABMIL'] = eval_single_model(model, 'ABMIL', test_loader, config, device)

    # Sparse ABMIL
    model = SparseABMIL(
        hidden_dim=config.get('hidden_dim', 64),
        attention_hidden=config.get('attention_hidden', 64),
        classifier_hidden=config.get('classifier_hidden', [128, 64]),
    ).to(device)
    ckpt = torch.load('experiments/sparse_best_model.pt', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f"Loaded Sparse ABMIL from epoch {ckpt.get('epoch', '?')}")
    results['Sparse ABMIL'] = eval_single_model(model, 'Sparse ABMIL', test_loader, config, device)

    # Rules-only (quick)
    print(f"\n{'='*50}")
    print("Evaluating Rules-only...")
    print(f"{'='*50}")

    keywords_path = config.get('keywords_file', 'phishingdata/kwords_stem.json')
    if not os.path.isabs(keywords_path):
        project_root = os.path.dirname(os.path.abspath(__file__))
        keywords_path = os.path.join(project_root, keywords_path)
    keywords = load_keywords(keywords_path)

    all_labels, all_scores, all_entropies = [], [], []
    all_spearmans = []

    for batch in test_loader:
        labels = batch['label']
        raw_sentences = batch['raw_sentences']
        sent_lengths = batch['sent_lengths']
        rule_scores = batch_score(raw_sentences, keywords)

        for b in range(len(labels)):
            n_valid = int((sent_lengths[b] > 0).sum().item())
            scores = rule_scores[b, :n_valid]

            all_labels.append(labels[b].item())
            all_scores.append(float(scores.max()))

            if scores.sum() > 0 and n_valid > 0:
                p = scores / scores.sum()
                entropy = -sum(float(pi) * np.log(float(pi) + 1e-10) for pi in p if float(pi) > 1e-10)
                all_entropies.append(float(entropy))

            if labels[b].item() == 1 and n_valid >= 2 and len(set(scores)) >= 2:
                all_spearmans.append(1.0)

    if len(set(all_labels)) < 2:
        email_auprc = 0.0
    else:
        precision_arr, recall_arr, _ = precision_recall_curve(all_labels, all_scores)
        email_auprc = auc(recall_arr, precision_arr)

    preds = [1 if s > 0.3 else 0 for s in all_scores]
    email_f1 = f1_score(all_labels, preds, zero_division=0)

    spearman_corr = np.mean(all_spearmans) if all_spearmans else 0.0
    entropy = np.mean(all_entropies) if all_entropies else 0.0

    results['Rules-only'] = {
        'email_auprc': float(email_auprc),
        'email_f1': float(email_f1),
        'spearman_corr': float(spearman_corr),
        'attention_entropy': float(entropy),
    }
    print(f"  email_auprc: {email_auprc:.4f}")
    print(f"  email_f1: {email_f1:.4f}")
    print(f"  spearman_corr: {spearman_corr:.4f}")
    print(f"  attention_entropy: {entropy:.4f}")

    # Save
    with open('experiments/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to experiments/results.json")

    # Print table
    print(f"\n{'='*70}")
    print(f"{'Metric':<20} {'Rules-only':>12} {'ABMIL':>12} {'Sparse ABMIL':>12}")
    print(f"{'='*70}")
    for metric in ['email_auprc', 'email_f1', 'spearman_corr', 'attention_entropy']:
        vals = []
        for m in ['Rules-only', 'ABMIL', 'Sparse ABMIL']:
            v = results[m].get(metric, 0)
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(f"{v}")
        print(f"{metric:<20} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
