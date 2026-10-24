"""Run full evaluation with new 4 metrics for all existing models."""
import os, json, torch, yaml, numpy as np
from src.dataset import create_dataloaders
from src.models.abmil import SparseABMIL, ABMILBaseline
from src.evaluate import (
    compute_email_auprc, compute_email_f1,
    compute_spearman_correlation, compute_attention_entropy
)
from src.rules import load_keywords, batch_score
from sklearn.metrics import precision_recall_curve, auc, f1_score


def load_config(path='experiments/configs/defaults.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def rules_baseline(test_loader, config):
    """Rules-only with new metrics."""
    keywords = load_keywords('phishingdata/kwords_stem.json')
    all_labels, all_scores, all_entropies = [], [], []

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

    precision_arr, recall_arr, _ = precision_recall_curve(all_labels, all_scores)
    email_auprc = auc(recall_arr, precision_arr)
    preds = [1 if s > 0.3 else 0 for s in all_scores]
    email_f1 = f1_score(all_labels, preds, zero_division=0)
    entropy = np.mean(all_entropies) if all_entropies else 0.0

    return {
        'email_auprc': float(email_auprc),
        'email_f1': float(email_f1),
        'spearman_corr': 1.0,  # Rules self-correlation
        'attention_entropy': float(entropy),
    }


def eval_model(model, test_loader, config):
    """Evaluate a model with new metrics."""
    return {
        'email_auprc': float(compute_email_auprc(model, test_loader, config, torch.device('cpu'))),
        'email_f1': float(compute_email_f1(model, test_loader, config, torch.device('cpu'))),
        'spearman_corr': float(compute_spearman_correlation(model, test_loader, config, torch.device('cpu'))),
        'attention_entropy': float(compute_attention_entropy(model, test_loader, config, torch.device('cpu'))),
    }


def load_and_eval(name, ckpt_path, model_class, test_loader, config):
    print(f"\nEvaluating {name}...")
    model = model_class(
        hidden_dim=config.get('hidden_dim', 64),
        attention_hidden=config.get('attention_hidden', 64),
        classifier_hidden=config.get('classifier_hidden', [128, 64]),
    )
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    results = eval_model(model, test_loader, config)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    return results


def main():
    config = load_config()
    print("Loading test data...")
    _, test_loader = create_dataloaders(config)
    print(f"  Test batches: {len(test_loader)}")

    results = {}

    # Rules-only
    print("\n1. Rules-only baseline...")
    results['Rules-only'] = rules_baseline(test_loader, config)
    for k, v in results['Rules-only'].items():
        print(f"  {k}: {v:.4f}")

    # ABMIL
    results['ABMIL'] = load_and_eval(
        'ABMIL', 'experiments/abmil_best_model.pt',
        ABMILBaseline, test_loader, config
    )

    # Sparse ABMIL
    results['Sparse ABMIL'] = load_and_eval(
        'Sparse ABMIL', 'experiments/sparse_best_model.pt',
        SparseABMIL, test_loader, config
    )

    # Ablations
    ablation_names = {
        'ablation_no_l1': 'w/o L1 sparsity',
        'ablation_no_rule': 'w/o rule alignment',
        'ablation_no_temp': 'w/o temp annealing',
    }
    results_ablation = {}
    for dir_name, display_name in ablation_names.items():
        ckpt_path = f'experiments/{dir_name}/sparse_best_model.pt'
        if os.path.exists(ckpt_path):
            results_ablation[display_name] = load_and_eval(
                display_name, ckpt_path, SparseABMIL, test_loader, config
            )

    # Save
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    with open('experiments/ablation_results.json', 'w') as f:
        json.dump(results_ablation, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Model':<25} {'AUPRC':>8} {'F1':>8} {'Spearman':>10} {'Entropy':>10}")
    print(f"{'='*70}")
    for name in ['Rules-only', 'ABMIL', 'Sparse ABMIL']:
        r = results[name]
        print(f"{name:<25} {r['email_auprc']:>8.4f} {r['email_f1']:>8.4f} {r['spearman_corr']:>10.4f} {r['attention_entropy']:>10.4f}")
    print(f"{'='*70}")

    print(f"\nAblation results:")
    for name, r in results_ablation.items():
        print(f"  {name:<25} {r['email_auprc']:>8.4f} {r['email_f1']:>8.4f} {r['spearman_corr']:>10.4f} {r['attention_entropy']:>10.4f}")

    print("\nDone. Saved to experiments/results.json and experiments/ablation_results.json")


if __name__ == '__main__':
    main()
