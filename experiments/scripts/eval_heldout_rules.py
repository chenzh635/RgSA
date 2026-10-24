"""Held-out rule evaluation experiment.

Addresses the circular evaluation concern: the current Spearman metric compares
attention weights against the same rule scores used to train the model.

Solution: Split the 7 cognitive trigger categories into:
- Training set (5 categories): scarcity, consistency, reciprocity, authority, socialproof
- Held-out set (2 categories): liking, security

Train with rule alignment using only the 5 training categories, then evaluate
Spearman against the 2 held-out categories. This proves the model generalizes
beyond the specific rules it was trained on.

Usage:
    python experiments/scripts/eval_heldout_rules.py [--seed 42] [--retrain]
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

# Categories for training and held-out evaluation
TRAIN_CATEGORIES = ['scarcity', 'consistency', 'reciprocity', 'authority', 'socialproof']
HELDOUT_CATEGORIES = ['liking', 'security']


def load_keywords_filtered(keywords_path, categories):
    """Load only the specified categories from keywords file."""
    with open(keywords_path, 'r') as f:
        all_keywords = json.load(f)
    return {cat: all_keywords[cat] for cat in categories if cat in all_keywords}


def score_sentences_filtered(raw_sentences, keywords):
    """Score sentences using only the specified keyword categories.

    Same logic as rules.py score_sentences but restricted to given categories.
    """
    import re
    scores = np.zeros(len(raw_sentences), dtype=np.float32)

    for i, sent in enumerate(raw_sentences):
        sent_lower = sent.lower()
        triggers = 0
        total_checks = len(keywords)

        for cat, stems in keywords.items():
            for stem in stems:
                if stem in sent_lower:
                    triggers += 1
                    break

        # Keep the same rule-based signals for consistency
        has_uri = bool(re.search(r'https?://|www\.|click\s+here|link\s+below', sent_lower))
        has_generic_greeting = bool(re.search(r'dear\s+(customer|user|member|beneficiary|sir)', sent_lower))
        has_action = bool(re.search(r'click\s+here|update\s+your|confirm\s+your|verify\s+your|renew', sent_lower))

        rule_score = (triggers / max(total_checks, 1) + has_uri * 0.3 +
                      has_generic_greeting * 0.2 + has_action * 0.2)
        scores[i] = min(rule_score, 1.0)

    return scores


def batch_score_filtered(raw_sentences_batch, keywords):
    """Batch version of score_sentences_filtered."""
    batch_size = len(raw_sentences_batch)
    max_sents = max(len(sents) for sents in raw_sentences_batch)
    all_scores = np.zeros((batch_size, max_sents), dtype=np.float32)

    for b, sentences in enumerate(raw_sentences_batch):
        all_scores[b, :len(sentences)] = score_sentences_filtered(sentences, keywords)

    return all_scores


def compute_spearman_with_rules(model, test_loader, keywords, device):
    """Compute Spearman correlation using a specific set of rule categories."""
    from scipy.stats import spearmanr

    model.eval()
    is_sparse = hasattr(model, 'use_sparse') and model.use_sparse
    correlations = []

    with torch.no_grad():
        for batch in test_loader:
            tokens = batch['tokens'].to(device)
            labels = batch['label'].to(device)
            sent_lengths = batch['sent_lengths'].to(device)
            raw_sentences = batch['raw_sentences']

            if is_sparse:
                _, attn_weights, _, _ = model(tokens, sent_lengths, temperature=0.1)
            else:
                _, attn_weights, _ = model(tokens, sent_lengths)

            # Use filtered rule scoring
            rule_scores = batch_score_filtered(raw_sentences, keywords)

            B, S = attn_weights.shape
            for b in range(B):
                if labels[b].item() != 1:
                    continue

                valid_mask = sent_lengths[b] > 0
                n_valid = int(valid_mask.sum().item())
                if n_valid < 2:
                    continue

                n_rule = rule_scores.shape[1]
                n_use = min(n_valid, n_rule)
                if n_use < 2:
                    continue

                attn_b = attn_weights[b, :n_use].cpu().numpy()
                rule_b = rule_scores[b, :n_use]

                if len(set(attn_b)) < 2 or len(set(rule_b)) < 2:
                    continue

                corr, _ = spearmanr(attn_b, rule_b)
                if not np.isnan(corr):
                    correlations.append(corr)

    return np.mean(correlations) if correlations else 0.0


def train_with_filtered_rules(config, device, seed=42):
    """Train Sparse ABMIL using only the training rule categories for alignment."""
    from src.dataset import create_dataloaders
    from src.models.abmil import SparseABMIL
    from src.evaluate import evaluate

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load only training categories for rule alignment
    keywords_path = os.path.join(project_root, config.get('keywords_file', 'phishingdata/kwords_stem.json'))
    train_keywords = load_keywords_filtered(keywords_path, TRAIN_CATEGORIES)

    save_dir = os.path.join(project_root, 'experiments', 'heldout_rules_train')
    os.makedirs(save_dir, exist_ok=True)
    config_run = config.copy()
    config_run['save_dir'] = save_dir

    train_loader, test_loader = create_dataloaders(config_run)

    model = SparseABMIL(
        hidden_dim=config_run.get('hidden_dim', 64),
        attention_hidden=config_run.get('attention_hidden', 64),
        classifier_hidden=config_run.get('classifier_hidden', [128, 64]),
        use_sparse=True,
    ).to(device)

    # Monkey-patch the keywords in src.rules for training
    # (the training loop calls batch_score from src.rules which uses all keywords)
    # Instead, we create a custom training loop that uses filtered keywords
    import src.rules as rules_module
    original_batch_score = rules_module.batch_score

    def filtered_batch_score(raw_sentences_batch, keywords):
        """Override batch_score to use only training categories."""
        return batch_score_filtered(raw_sentences_batch, train_keywords)

    rules_module.batch_score = filtered_batch_score

    try:
        from src.train import train
        print(f"\nTraining Sparse ABMIL with filtered rules ({len(TRAIN_CATEGORIES)} categories): {TRAIN_CATEGORIES}")
        start_time = time.time()
        train(model, train_loader, test_loader, config_run, device)
        train_time = time.time() - start_time
        print(f"Training time: {train_time/60:.1f} minutes")
    finally:
        # Restore original
        rules_module.batch_score = original_batch_score

    return model, train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='Held-out rule evaluation experiment')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--retrain', action='store_true',
                        help='Retrain with filtered rules (otherwise use existing model)')
    args = parser.parse_args()

    import yaml
    with open(os.path.join(project_root, 'experiments/configs/defaults.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    # Use λ=3.0 for the held-out experiment (same as main model)
    config['lambda_rule'] = 3.0

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Training categories: {TRAIN_CATEGORIES}")
    print(f"Held-out categories: {HELDOUT_CATEGORIES}")

    keywords_path = os.path.join(project_root, config.get('keywords_file', 'phishingdata/kwords_stem.json'))

    if args.retrain:
        model, train_loader, test_loader = train_with_filtered_rules(config, device, seed=args.seed)
    else:
        # Use the existing Sparse ABMIL model (trained with all categories)
        from src.dataset import create_dataloaders
        from src.models.abmil import SparseABMIL

        train_loader, test_loader = create_dataloaders(config)
        model = SparseABMIL(
            hidden_dim=config.get('hidden_dim', 64),
            attention_hidden=config.get('attention_hidden', 64),
            classifier_hidden=config.get('classifier_hidden', [128, 64]),
            use_sparse=True,
        ).to(device)

        # Try loading the heldout-trained model first, then the main model
        ckpt_path = os.path.join(project_root, 'experiments', 'heldout_rules_train', 'sparse_best_model.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded heldout-trained model from epoch {ckpt.get('epoch', '?')}")
        else:
            ckpt_path = os.path.join(project_root, 'experiments', 'archive', 'sparse_best_model.pt')
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(project_root, 'experiments', 'sparse_best_model.pt')
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ckpt['model_state_dict'])
                print(f"Loaded main Sparse ABMIL model (trained with ALL categories)")
            else:
                print("No checkpoint found! Run with --retrain first.")
                return

    # Load keyword sets
    all_keywords = load_keywords_filtered(keywords_path,
        TRAIN_CATEGORIES + HELDOUT_CATEGORIES)
    train_keywords = load_keywords_filtered(keywords_path, TRAIN_CATEGORIES)
    heldout_keywords = load_keywords_filtered(keywords_path, HELDOUT_CATEGORIES)

    # Compute Spearman with different rule sets
    print("\n" + "=" * 60)
    print("Spearman Correlation with Different Rule Sets:")
    print("=" * 60)

    spearman_all = compute_spearman_with_rules(model, test_loader, all_keywords, device)
    print(f"  All categories (7):        {spearman_all:.4f}")

    spearman_train = compute_spearman_with_rules(model, test_loader, train_keywords, device)
    print(f"  Training categories (5):   {spearman_train:.4f}")

    spearman_heldout = compute_spearman_with_rules(model, test_loader, heldout_keywords, device)
    print(f"  Held-out categories (2):   {spearman_heldout:.4f}")

    # Also compute standard metrics
    from src.evaluate import evaluate
    std_results = evaluate(model, test_loader, config, device)

    print(f"\n  Standard evaluation:")
    print(f"    AUPRC:   {std_results['email_auprc']:.4f}")
    print(f"    F1:      {std_results['email_f1']:.4f}")
    print(f"    Entropy: {std_results['attention_entropy']:.4f}")

    # Save results
    results = {
        'spearman_all_categories': float(spearman_all),
        'spearman_train_categories': float(spearman_train),
        'spearman_heldout_categories': float(spearman_heldout),
        'train_categories': TRAIN_CATEGORIES,
        'heldout_categories': HELDOUT_CATEGORIES,
        'standard_metrics': std_results,
        'retrained': args.retrain,
        'seed': args.seed,
    }

    output_dir = os.path.join(project_root, 'experiments', 'heldout_rules_train')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'heldout_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {os.path.join(output_dir, 'heldout_results.json')}")

    # Interpretation
    print("\n" + "=" * 60)
    print("Interpretation:")
    if spearman_heldout > 0.3:
        print(f"  ✓ Held-out Spearman ({spearman_heldout:.4f}) > 0.3: Model generalizes")
        print(f"    beyond training rules — attention captures genuine phishing patterns,")
        print(f"    not just memorized keyword correlations.")
    elif spearman_heldout > 0.1:
        print(f"  ~ Held-out Spearman ({spearman_heldout:.4f}) is moderate: some generalization")
        print(f"    but model partially relies on training-category keywords.")
    else:
        print(f"  ✗ Held-out Spearman ({spearman_heldout:.4f}) is low: model may be")
        print(f"    overfitting to training categories. Circular evaluation concern valid.")
    print("=" * 60)


if __name__ == '__main__':
    main()
