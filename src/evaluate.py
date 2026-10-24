"""Evaluation metrics for phishing sentence localization."""

import json
import os

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, auc, f1_score
from scipy.stats import spearmanr


def load_keywords(keywords_path):
    """Load cognitive trigger keywords."""
    with open(keywords_path, 'r') as f:
        return json.load(f)


def compute_email_auprc(model, test_loader, config, device):
    """Email-Level AUPRC: model output vs real email labels.

    Uses the sigmoid of the email-level logit as the prediction score.
    """
    model.eval()
    is_sparse = hasattr(model, 'use_sparse') and model.use_sparse
    all_scores, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            tokens = batch['tokens'].to(device)
            labels = batch['label'].to(device)
            sent_lengths = batch['sent_lengths'].to(device)

            if is_sparse:
                output, _, _, _ = model(tokens, sent_lengths, temperature=0.1)
            else:
                output, _, _ = model(tokens, sent_lengths)

            scores = torch.sigmoid(output).cpu().numpy()
            all_scores.extend(scores.flatten().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    if len(set(all_labels)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    return auc(recall, precision)


def compute_email_f1(model, test_loader, config, device):
    """Email-Level F1: binary F1 at threshold 0.5.

    Uses the sigmoid of the email-level logit as the prediction score.
    """
    model.eval()
    is_sparse = hasattr(model, 'use_sparse') and model.use_sparse
    all_scores, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            tokens = batch['tokens'].to(device)
            labels = batch['label'].to(device)
            sent_lengths = batch['sent_lengths'].to(device)

            if is_sparse:
                output, _, _, _ = model(tokens, sent_lengths, temperature=0.1)
            else:
                output, _, _ = model(tokens, sent_lengths)

            scores = torch.sigmoid(output).cpu().numpy()
            all_scores.extend(scores.flatten().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    preds = [1 if s > 0.5 else 0 for s in all_scores]
    return f1_score(all_labels, preds, zero_division=0)


def compute_spearman_correlation(model, test_loader, config, device):
    """Spearman Rank Correlation: attention weights vs rule scores.

    For each phishing email, compute the Spearman correlation between
    the attention weights and the rule scores over sentences.
    Returns the mean correlation across all phishing emails.
    """
    keywords_path = config.get('keywords_file', 'phishingdata/kwords_stem.json')
    if not os.path.isabs(keywords_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        keywords_path = os.path.join(project_root, keywords_path)
    keywords = load_keywords(keywords_path) if os.path.exists(keywords_path) else None
    if keywords is None:
        return 0.0

    from src.rules import batch_score

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

            rule_scores = batch_score(raw_sentences, keywords)

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

                # Skip if all values are identical (no rank information)
                if len(set(attn_b)) < 2 or len(set(rule_b)) < 2:
                    continue

                corr, _ = spearmanr(attn_b, rule_b)
                if not np.isnan(corr):
                    correlations.append(corr)

    return np.mean(correlations) if correlations else 0.0


def compute_attention_entropy(model, test_loader, config, device):
    """Attention Entropy: Shannon entropy of the attention distribution.

    Lower entropy = more concentrated attention (higher model confidence).
    """
    model.eval()
    is_sparse = hasattr(model, 'use_sparse') and model.use_sparse
    total_entropy = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            tokens = batch['tokens'].to(device)
            sent_lengths = batch['sent_lengths'].to(device)

            if is_sparse:
                _, attn_weights, _, _ = model(tokens, sent_lengths, temperature=0.1)
            else:
                _, attn_weights, _ = model(tokens, sent_lengths)

            B, S = attn_weights.shape
            for b in range(B):
                valid_mask = sent_lengths[b] > 0
                p = attn_weights[b][valid_mask]
                if p.sum() > 0:
                    p = p / p.sum()
                    entropy = -(p * (p + 1e-10).log()).sum().item()
                    total_entropy += entropy
                total_samples += 1

    return total_entropy / total_samples if total_samples > 0 else 0


def check_cognitive_triggers(sentence, keywords):
    """Check if a sentence contains any cognitive trigger keywords.

    Args:
        sentence: string to check
        keywords: dict of {category: [word_stems]} from kwords_stem.json

    Returns:
        True if any cognitive trigger category matches the sentence
    """
    sent_lower = sentence.lower()
    for cat, stems in keywords.items():
        if cat == 'english_words':
            continue
        for stem in stems:
            if stem in sent_lower:
                return True
    return False


def evaluate(model, test_loader, config, device):
    """Run all evaluation metrics."""
    was_training = model.training
    model.eval()

    print("Computing Email-Level AUPRC...")
    email_auprc = compute_email_auprc(model, test_loader, config, device)

    print("Computing Email-Level F1...")
    email_f1 = compute_email_f1(model, test_loader, config, device)

    print("Computing Spearman Rank Correlation...")
    spearman = compute_spearman_correlation(model, test_loader, config, device)

    print("Computing Attention Entropy...")
    entropy = compute_attention_entropy(model, test_loader, config, device)

    # Restore original training mode
    if was_training:
        model.train()
    else:
        model.eval()

    return {
        'email_auprc': email_auprc,
        'email_f1': email_f1,
        'spearman_corr': spearman,
        'attention_entropy': entropy,
    }
