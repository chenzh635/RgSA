"""Rules-only baseline evaluation."""

import json
import os
import numpy as np
import torch

from src.rules import load_keywords, batch_score
from src.dataset import create_dataloaders


def load_config(path='experiments/configs/defaults.yaml'):
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def rules_baseline_eval():
    config = load_config()
    train_loader, test_loader = create_dataloaders(config)

    keywords_path = config.get('keywords_file', 'phishingdata/kwords_stem.json')
    if not os.path.isabs(keywords_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        keywords_path = os.path.join(project_root, keywords_path)
    keywords = load_keywords(keywords_path)

    # Evaluate rules-only baseline
    all_labels = []
    all_preds_email = []  # email-level predictions
    all_attn_scores = []  # rule scores as pseudo-attention
    all_raw_sentences = []
    all_email_labels = []

    for batch in test_loader:
        labels = batch['label']
        raw_sentences = batch['raw_sentences']
        sent_lengths = batch['sent_lengths']

        rule_scores = batch_score(raw_sentences, keywords)

        B = len(labels)
        for b in range(B):
            valid_mask = sent_lengths[b] > 0
            n_valid = valid_mask.sum().item()
            scores_b = rule_scores[b, :n_valid]

            # Email-level prediction: max rule score
            all_labels.append(labels[b].item())
            all_preds_email.append(1 if scores_b.max() > 0.3 else 0)

            # Store for localization metrics
            all_attn_scores.append(scores_b)
            all_raw_sentences.append(raw_sentences[b])
            all_email_labels.append(labels[b].item())

    # Email classification accuracy
    correct = sum(1 for l, p in zip(all_labels, all_preds_email) if l == p)
    print(f"Rules-only email accuracy: {correct/len(all_labels):.4f}")

    # Label-Accuracy: top-1 sentence by rule score predicts email label
    # For rules-only, the "classifier" is whether top-1 sentence has rule score > 0.3
    label_acc_correct = 0
    total = 0
    for scores, email_label in zip(all_attn_scores, all_email_labels):
        if len(scores) == 0:
            continue
        top1_score = scores.max()
        pred = 1 if top1_score > 0.3 else 0
        if pred == email_label:
            label_acc_correct += 1
        total += 1
    print(f"Label-Accuracy (rules): {label_acc_correct/total:.4f}")

    # Cognitive-TP: for phishing emails, does top-1 sentence contain triggers?
    from src.evaluate import check_cognitive_triggers
    tp_count = 0
    total_phishing = 0
    for scores, sentences, email_label in zip(all_attn_scores, all_raw_sentences, all_email_labels):
        if email_label != 1:
            continue
        total_phishing += 1
        if len(scores) == 0:
            continue
        top1_idx = scores.argmax()
        if top1_idx < len(sentences):
            if check_cognitive_triggers(sentences[top1_idx], keywords):
                tp_count += 1
    print(f"Cognitive-TP (rules): {tp_count/total_phishing if total_phishing > 0 else 0:.4f}")

    # Attention Entropy (use normalized rule scores as distribution)
    total_entropy = 0
    total_samples = 0
    for scores in all_attn_scores:
        if len(scores) == 0 or scores.sum() == 0:
            continue
        p = scores / scores.sum()
        entropy = 0.0
        for pi in p:
            if pi > 1e-10:
                entropy -= pi * np.log(pi + 1e-10)
        total_entropy += entropy
        total_samples += 1
    print(f"Entropy (rules): {total_entropy/total_samples if total_samples > 0 else 0:.4f}")


if __name__ == '__main__':
    rules_baseline_eval()
