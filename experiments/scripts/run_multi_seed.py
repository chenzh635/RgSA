"""Multi-seed statistical significance experiments.

Trains 4 core models (ABMIL, Sparse ABMIL, BERT-ABMIL, Snorkel) with
multiple random seeds and reports mean ± std for all metrics.

Usage:
    python experiments/scripts/run_multi_seed.py [--seeds 123,456,789] [--models abmil,sparse,bert,snorkel]

Estimated MPS time: ~12h total (CNN models ~2h, BERT-ABMIL ~8h, Snorkel ~1.5h)
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config():
    import yaml
    with open(os.path.join(project_root, 'experiments/configs/defaults.yaml'), 'r') as f:
        return yaml.safe_load(f)


def train_and_eval_abmil(seed, config, device, use_sparse=True):
    """Train and evaluate ABMIL or Sparse ABMIL with a given seed."""
    from src.dataset import create_dataloaders
    from src.models.abmil import SparseABMIL, ABMILBaseline
    from src.train import train
    from src.evaluate import evaluate

    set_seed(seed)
    model_name = 'sparse' if use_sparse else 'abmil'
    save_dir = os.path.join(project_root, 'experiments', f'multi_seed_{model_name}_seed{seed}')
    os.makedirs(save_dir, exist_ok=True)

    config_run = config.copy()
    config_run['save_dir'] = save_dir

    train_loader, test_loader = create_dataloaders(config_run)

    if use_sparse:
        model = SparseABMIL(
            hidden_dim=config_run.get('hidden_dim', 64),
            attention_hidden=config_run.get('attention_hidden', 64),
            classifier_hidden=config_run.get('classifier_hidden', [128, 64]),
            use_sparse=True,
        )
    else:
        model = ABMILBaseline(
            hidden_dim=config_run.get('hidden_dim', 64),
            attention_hidden=config_run.get('attention_hidden', 64),
            classifier_hidden=config_run.get('classifier_hidden', [128, 64]),
        )
    model = model.to(device)

    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} with seed={seed}")
    print(f"{'='*60}")

    start_time = time.time()
    train(model, train_loader, test_loader, config_run, device)
    train_time = time.time() - start_time

    # Load best checkpoint
    ckpt_path = os.path.join(save_dir, f'{model_name}_best_model.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

    results = evaluate(model, test_loader, config_run, device)
    results['train_time_min'] = train_time / 60.0
    results['seed'] = seed

    # Save individual results
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  {model_name} seed={seed}: AUPRC={results['email_auprc']:.4f}, "
          f"F1={results['email_f1']:.4f}, Spearman={results['spearman_corr']:.4f}, "
          f"Entropy={results['attention_entropy']:.4f}, Time={train_time/60:.1f}min")

    return results


def train_and_eval_bert(seed, config, device):
    """Train and evaluate BERT-ABMIL with a given seed."""
    from src.baselines.bert_abmil import BertABMIL, create_bert_dataloaders, train_bert_abmil

    set_seed(seed)
    save_dir = os.path.join(project_root, 'experiments', f'multi_seed_bert_seed{seed}')
    os.makedirs(save_dir, exist_ok=True)

    config_run = config.copy()
    config_run['save_dir'] = save_dir

    train_loader, test_loader = create_bert_dataloaders(config_run)

    model = BertABMIL(
        attention_hidden=config_run.get('attention_hidden', 64),
        classifier_hidden=config_run.get('classifier_hidden', [128, 64]),
    ).to(device)

    print(f"\n{'='*60}")
    print(f"Training BERT-ABMIL with seed={seed}")
    print(f"{'='*60}")

    start_time = time.time()
    train_bert_abmil(model, train_loader, test_loader, config_run, device)
    train_time = time.time() - start_time

    # Load best and evaluate using the unified eval from run_all_baselines
    ckpt_path = os.path.join(save_dir, 'bert_abmil_best_model.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

    # Evaluate BERT-ABMIL with standard metrics
    results = evaluate_bert_abmil_unified(model, test_loader, config_run, device)
    results['train_time_min'] = train_time / 60.0
    results['seed'] = seed

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  BERT seed={seed}: AUPRC={results['email_auprc']:.4f}, "
          f"F1={results['email_f1']:.4f}, Time={train_time/60:.1f}min")

    return results


def train_and_eval_snorkel(seed, config, device):
    """Train and evaluate Snorkel with a given seed."""
    from src.dataset import create_dataloaders
    from src.baselines.snorkel_baseline import train_snorkel, evaluate_snorkel

    set_seed(seed)
    save_dir = os.path.join(project_root, 'experiments', f'multi_seed_snorkel_seed{seed}')
    os.makedirs(save_dir, exist_ok=True)

    config_run = config.copy()
    config_run['save_dir'] = save_dir

    train_loader, test_loader = create_dataloaders(config_run)

    print(f"\n{'='*60}")
    print(f"Training Snorkel with seed={seed}")
    print(f"{'='*60}")

    start_time = time.time()
    model = train_snorkel(train_loader, config_run, device)
    train_time = time.time() - start_time

    results = evaluate_snorkel(model, test_loader, config_run, device)
    results['train_time_min'] = train_time / 60.0
    results['seed'] = seed

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Snorkel seed={seed}: AUPRC={results['email_auprc']:.4f}, "
          f"F1={results['email_f1']:.4f}, Spearman={results['spearman_corr']:.4f}, "
          f"Entropy={results['attention_entropy']:.4f}, Time={train_time/60:.1f}min")

    return results


def evaluate_bert_abmil_unified(model, test_loader, config, device):
    """Evaluate BERT-ABMIL using the standard 4 metrics."""
    from src.evaluate import load_keywords
    from src.rules import batch_score
    from sklearn.metrics import precision_recall_curve, auc, f1_score
    from scipy.stats import spearmanr

    model.eval()
    keywords_path = config.get('keywords_file', 'phishingdata/kwords_stem.json')
    if not os.path.isabs(keywords_path):
        keywords_path = os.path.join(project_root, keywords_path)
    keywords = load_keywords(keywords_path) if os.path.exists(keywords_path) else None

    all_scores, all_labels = [], []
    all_attn_weights, all_raw_sentences, all_sent_lengths = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            labels = batch['label'].to(device)
            sent_lengths = batch['sent_lengths'].to(device)
            raw_sentences = batch['raw_sentences']

            output, attn_weights, h = model(raw_sentences, sent_lengths)
            scores = torch.sigmoid(output).cpu().numpy()

            all_scores.extend(scores.flatten().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            B = attn_weights.shape[0]
            for b in range(B):
                all_attn_weights.append(attn_weights[b].cpu().numpy())
                all_raw_sentences.append(raw_sentences[b])
                all_sent_lengths.append(sent_lengths[b])

    # Email AUPRC
    if len(set(all_labels)) >= 2:
        precision, recall, _ = precision_recall_curve(all_labels, all_scores)
        email_auprc = auc(recall, precision)
    else:
        email_auprc = 0.0

    # Email F1
    preds = [1 if s > 0.5 else 0 for s in all_scores]
    email_f1 = f1_score(all_labels, preds, zero_division=0)

    # Spearman correlation (attention vs rule scores for phishing emails)
    correlations = []
    entropies = []
    for i in range(len(all_labels)):
        if all_labels[i] != 1:
            continue
        sl = all_sent_lengths[i]
        n_valid = int((sl > 0).sum().item()) if hasattr(sl, 'sum') else int((sl > 0).sum())
        if n_valid < 2:
            continue

        attn_b = all_attn_weights[i][:n_valid]

        # Entropy
        p = attn_b / (attn_b.sum() + 1e-10)
        ent = -(p * np.log(p + 1e-10)).sum()
        entropies.append(float(ent))

        # Spearman
        if keywords and len(set(attn_b)) >= 2:
            rule_b = batch_score([all_raw_sentences[i]], keywords)[0, :n_valid]
            if len(set(rule_b)) >= 2:
                corr, _ = spearmanr(attn_b, rule_b)
                if not np.isnan(corr):
                    correlations.append(corr)

    spearman_corr = np.mean(correlations) if correlations else 0.0
    attention_entropy = np.mean(entropies) if entropies else 0.0

    return {
        'email_auprc': float(email_auprc),
        'email_f1': float(email_f1),
        'spearman_corr': float(spearman_corr),
        'attention_entropy': float(attention_entropy),
    }


def aggregate_results(all_results):
    """Compute mean ± std across seeds for each model."""
    metrics = ['email_auprc', 'email_f1', 'spearman_corr', 'attention_entropy']
    summary = {}

    for model_name, results_list in all_results.items():
        summary[model_name] = {}
        for metric in metrics:
            values = [r[metric] for r in results_list if metric in r]
            if values:
                summary[model_name][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'values': [float(v) for v in values],
                }
    return summary


def print_summary(summary):
    """Print a formatted summary table."""
    print("\n" + "=" * 100)
    print(f"{'Model':<25} {'AUPRC':>18} {'F1':>18} {'Spearman':>18} {'Entropy':>18}")
    print("=" * 100)

    for model_name, metrics in summary.items():
        auprc = metrics.get('email_auprc', {})
        f1 = metrics.get('email_f1', {})
        spearman = metrics.get('spearman_corr', {})
        entropy = metrics.get('attention_entropy', {})

        print(f"{model_name:<25} "
              f"{auprc.get('mean', 0):.4f}±{auprc.get('std', 0):.4f}  "
              f"{f1.get('mean', 0):.4f}±{f1.get('std', 0):.4f}  "
              f"{spearman.get('mean', 0):.4f}±{spearman.get('std', 0):.4f}  "
              f"{entropy.get('mean', 0):.4f}±{entropy.get('std', 0):.4f}")

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Multi-seed statistical significance experiments')
    parser.add_argument('--seeds', type=str, default='123,456,789',
                        help='Comma-separated list of seeds (default: 123,456,789)')
    parser.add_argument('--models', type=str, default='abmil,sparse,bert,snorkel',
                        help='Comma-separated list of models to run (default: all)')
    parser.add_argument('--lambda-rule', type=float, default=None,
                        help='Override lambda_rule (default: use config value, typically 0.5)')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training, only aggregate existing results')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    models = [m.strip() for m in args.models.split(',')]

    config = load_config()
    if args.lambda_rule is not None:
        config['lambda_rule'] = args.lambda_rule
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print(f"Models: {models}")
    print(f"lambda_rule: {config['lambda_rule']}")

    all_results = {}

    for model_name in models:
        all_results[model_name] = []

        for seed in seeds:
            if args.skip_train:
                # Try to load existing results
                if model_name in ('abmil', 'sparse'):
                    result_dir = os.path.join(project_root, 'experiments',
                                              f'multi_seed_{model_name}_seed{seed}')
                elif model_name == 'bert':
                    result_dir = os.path.join(project_root, 'experiments',
                                              f'multi_seed_bert_seed{seed}')
                elif model_name == 'snorkel':
                    result_dir = os.path.join(project_root, 'experiments',
                                              f'multi_seed_snorkel_seed{seed}')
                else:
                    continue

                result_path = os.path.join(result_dir, 'results.json')
                if os.path.exists(result_path):
                    with open(result_path) as f:
                        all_results[model_name].append(json.load(f))
                    print(f"Loaded existing results for {model_name} seed={seed}")
                else:
                    print(f"No existing results for {model_name} seed={seed}, skipping")
                continue

            # Train and evaluate
            if model_name == 'abmil':
                result = train_and_eval_abmil(seed, config, device, use_sparse=False)
            elif model_name == 'sparse':
                result = train_and_eval_abmil(seed, config, device, use_sparse=True)
            elif model_name == 'bert':
                result = train_and_eval_bert(seed, config, device)
            elif model_name == 'snorkel':
                result = train_and_eval_snorkel(seed, config, device)
            else:
                print(f"Unknown model: {model_name}")
                continue

            all_results[model_name].append(result)

    # Aggregate and print
    if any(len(v) > 0 for v in all_results.values()):
        summary = aggregate_results(all_results)
        print_summary(summary)

        # Save summary
        output_path = os.path.join(project_root, 'experiments', 'multi_seed_summary.json')
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to {output_path}")
    else:
        print("No results to aggregate.")


if __name__ == '__main__':
    main()
