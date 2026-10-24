"""Unified evaluation: run all baselines and produce a single results table.

Usage:
    python run_all_baselines.py [--skip-train] [--skip-ai2tale]

Options:
    --skip-train    Skip training (load existing checkpoints)
    --skip-ai2tale  Skip AI2TALE (requires TensorFlow)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def load_config():
    import yaml
    with open(os.path.join(project_root, 'experiments/configs/defaults.yaml'), 'r') as f:
        return yaml.safe_load(f)


def evaluate_existing_models(config, device, test_loader, results):
    """Evaluate models that already have checkpoints saved."""
    from src.models.abmil import ABMILBaseline, SparseABMIL
    from src.evaluate import evaluate

    # Sparse ABMIL (already trained)
    print("\n" + "=" * 60)
    print("Evaluating Sparse ABMIL (Ours)...")
    model_sparse = SparseABMIL(
        hidden_dim=config.get('hidden_dim', 64),
        attention_hidden=config.get('attention_hidden', 64),
        classifier_hidden=config.get('classifier_hidden', [128, 64]),
        use_sparse=True,
    ).to(device)
    ckpt_path = os.path.join(project_root, 'experiments', 'sparse_best_model.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model_sparse.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  Loaded from epoch {ckpt.get('epoch', '?')}")
        results['Sparse ABMIL (Ours)'] = evaluate(model_sparse, test_loader, config, device)
    else:
        print("  WARNING: No checkpoint found, skipping")

    # ABMIL baseline
    print("\n" + "=" * 60)
    print("Evaluating ABMIL baseline...")
    model_abmil = ABMILBaseline(
        hidden_dim=config.get('hidden_dim', 64),
        attention_hidden=config.get('attention_hidden', 64),
        classifier_hidden=config.get('classifier_hidden', [128, 64]),
    ).to(device)
    ckpt_path = os.path.join(project_root, 'experiments', 'abmil_best_model.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model_abmil.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  Loaded from epoch {ckpt.get('epoch', '?')}")
        results['ABMIL'] = evaluate(model_abmil, test_loader, config, device)
    else:
        print("  WARNING: No checkpoint found, skipping")

    # Rules-only baseline
    print("\n" + "=" * 60)
    print("Evaluating Rules-only baseline...")
    results['Rules-only'] = evaluate_rules_only(config, test_loader, device)

    # Ablation models
    print("\n" + "=" * 60)
    print("Evaluating ablation models...")
    ablation_names = ['ablation_no_l1', 'ablation_no_rule', 'ablation_no_temp']
    for name in ablation_names:
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
            print(f"  Loaded {name} from epoch {ckpt.get('epoch', '?')}")
            display_name = f"w/o {name.split('ablation_no_')[1]}"
            results[display_name] = evaluate(model, test_loader, config, device)
        else:
            print(f"  WARNING: No checkpoint for {name}, skipping")


def evaluate_rules_only(config, test_loader, device):
    """Evaluate rules-only baseline using the same metrics."""
    from src.evaluate import check_cognitive_triggers, load_keywords
    from src.rules import batch_score

    keywords_path = config.get('keywords_file', 'phishingdata/kwords_stem.json')
    if not os.path.isabs(keywords_path):
        keywords_path = os.path.join(project_root, keywords_path)
    keywords = load_keywords(keywords_path) if os.path.exists(keywords_path) else None
    if keywords is None:
        return {k: 0 for k in ['label_accuracy', 'cognitive_tp', 'attention_entropy',
                                'rule_agreement', 'f1', 'auprc']}

    email_labels_all = []
    email_rule_scores = []
    email_raw_sentences = []
    email_sent_lengths = []

    for batch in test_loader:
        labels = batch['label'].numpy()
        sent_lengths = batch['sent_lengths']
        raw_sentences = batch['raw_sentences']

        for b in range(len(labels)):
            sentences = raw_sentences[b]
            if not sentences:
                email_labels_all.append(int(labels[b]))
                email_rule_scores.append(np.array([]))
                email_raw_sentences.append([])
                email_sent_lengths.append(sent_lengths[b])
                continue

            rule_s = batch_score([sentences], keywords)[0]
            # Normalize to attention-like weights
            scores_sum = rule_s.sum()
            if scores_sum > 0:
                scores_norm = rule_s / scores_sum
            else:
                scores_norm = np.ones(len(sentences)) / len(sentences)

            email_labels_all.append(int(labels[b]))
            email_rule_scores.append(scores_norm)
            email_raw_sentences.append(sentences)
            email_sent_lengths.append(sent_lengths[b])

    # Compute metrics
    label_acc_correct = 0
    label_acc_total = 0
    cog_tp_count = 0
    cog_tp_total = 0
    entropy_vals = []
    rule_agree_sum = 0
    rule_agree_count = 0
    all_scores_cont = []
    all_rule_binary = []

    for i, label in enumerate(email_labels_all):
        scores = email_rule_scores[i]
        sentences = email_raw_sentences[i]
        sl = email_sent_lengths[i]
        if len(scores) == 0:
            continue

        n_valid = int((sl > 0).sum())
        scores_valid = scores[:n_valid]
        top1_idx = scores_valid.argmax()

        # Label-Accuracy: for rules-only, top-1 score > 0 means phishing
        top1_score = scores_valid[top1_idx]
        if label == 1:
            if np.any(scores_valid > 0):
                label_acc_correct += 1
        else:
            if not np.any(scores_valid > 0.01):
                label_acc_correct += 1
        label_acc_total += 1

        # Cognitive-TP
        if label == 1 and keywords:
            cog_tp_total += 1
            if top1_idx < len(sentences):
                if check_cognitive_triggers(sentences[top1_idx], keywords):
                    cog_tp_count += 1

        # Entropy
        p = scores_valid / (scores_valid.sum() + 1e-10)
        ent = -(p * np.log(p + 1e-10)).sum()
        entropy_vals.append(ent)

        # Rule Agreement
        if label == 1 and n_valid >= 2 and keywords:
            rule_s_raw = batch_score([sentences], keywords)[0, :n_valid]
            k = min(3, n_valid)
            top_k_attn = set(scores_valid.argsort()[::-1][:k])
            top_k_rule = set(rule_s_raw.argsort()[::-1][:k])
            overlap = len(top_k_attn & top_k_rule)
            rule_agree_sum += overlap / k
            rule_agree_count += 1

            all_scores_cont.extend(scores_valid.tolist())
            all_rule_binary.extend((rule_s_raw > 0.3).astype(int).tolist())

    from sklearn.metrics import precision_recall_curve, auc, f1_score

    f1 = 0
    auprc = 0
    if len(all_rule_binary) > 0 and sum(all_rule_binary) > 0:
        threshold = np.mean(all_scores_cont) if len(all_scores_cont) > 3 else 0.5
        f1 = f1_score(all_rule_binary, [1 if s > threshold else 0 for s in all_scores_cont], zero_division=0)
        precision, recall, _ = precision_recall_curve(all_rule_binary, all_scores_cont)
        auprc = auc(recall, precision)

    return {
        'label_accuracy': label_acc_correct / label_acc_total if label_acc_total > 0 else 0,
        'cognitive_tp': cog_tp_count / cog_tp_total if cog_tp_total > 0 else 0,
        'attention_entropy': np.mean(entropy_vals) if entropy_vals else 0,
        'rule_agreement': rule_agree_sum / rule_agree_count if rule_agree_count > 0 else 0,
        'f1': f1,
        'auprc': auprc,
    }


def run_new_baselines(config, device, test_loader, results, skip_train=False, skip_ai2tale=False):
    """Train and evaluate new baselines: BERT-ABMIL, Snorkel, AI2TALE."""

    # ── BERT-ABMIL ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BERT-ABMIL Baseline")
    print("=" * 60)
    ckpt_path = os.path.join(project_root, 'experiments/bert_abmil/bert_abmil_best_model.pt')
    if os.path.exists(ckpt_path) and skip_train:
        print("Loading existing BERT-ABMIL checkpoint...")
    elif skip_train:
        print("Skipping BERT-ABMIL (no checkpoint found)")
    else:
        from src.baselines.bert_abmil import BertABMIL, create_bert_dataloaders, train_bert_abmil

        train_loader_bert, test_loader_bert = create_bert_dataloaders(config)
        model_bert = BertABMIL(
            attention_hidden=config.get('attention_hidden', 64),
            classifier_hidden=config.get('classifier_hidden', [128, 64]),
        ).to(device)
        train_bert_abmil(model_bert, train_loader_bert, test_loader_bert, config, device)

    # Evaluate BERT-ABMIL
    if os.path.exists(ckpt_path):
        from src.baselines.bert_abmil import BertABMIL, create_bert_dataloaders

        train_loader_bert, test_loader_bert = create_bert_dataloaders(config)
        model_bert = BertABMIL(
            attention_hidden=config.get('attention_hidden', 64),
            classifier_hidden=config.get('classifier_hidden', [128, 64]),
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model_bert.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"Loaded BERT-ABMIL from epoch {ckpt.get('epoch', '?')}")

        # Evaluate using our metrics with BERT-ABMIL specific evaluation
        results['BERT-ABMIL'] = evaluate_bert_abmil(model_bert, test_loader_bert, config, device)

    # ── Snorkel ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Snorkel Baseline")
    print("=" * 60)
    ckpt_path_snorkel = os.path.join(project_root, 'experiments/snorkel/snorkel_best_model.pt')
    if os.path.exists(ckpt_path_snorkel) and skip_train:
        print("Loading existing Snorkel checkpoint...")
    elif skip_train:
        print("Skipping Snorkel (no checkpoint found)")
    else:
        from src.dataset import create_dataloaders
        from src.baselines.snorkel_baseline import train_snorkel

        train_loader, test_loader = create_dataloaders(config)
        train_snorkel(train_loader, config, device)

    # Evaluate Snorkel
    if os.path.exists(ckpt_path_snorkel):
        from src.dataset import create_dataloaders
        from src.baselines.snorkel_baseline import SentenceClassifier, evaluate_snorkel

        train_loader, test_loader = create_dataloaders(config)
        vocab_size = 311774
        model_snorkel = SentenceClassifier(vocab_size=vocab_size).to(device)
        ckpt = torch.load(ckpt_path_snorkel, map_location=device)
        model_snorkel.load_state_dict(ckpt['model_state_dict'])

        results['Snorkel'] = evaluate_snorkel(model_snorkel, test_loader, config, device)

    # ── AI2TALE ──────────────────────────────────────────────────────
    if skip_ai2tale:
        print("\nSkipping AI2TALE (--skip-ai2tale)")
    else:
        print("\n" + "=" * 60)
        print("AI2TALE Baseline")
        print("=" * 60)
        try:
            from src.baselines.run_ai2tale import run_ai2tale_baseline
            ai2tale_results = run_ai2tale_baseline(config, evaluate_metrics=True)
            if ai2tale_results:
                results['AI2TALE'] = ai2tale_results
        except Exception as e:
            print(f"AI2TALE evaluation failed: {e}")
            print("Try: pip install tensorflow tensorflow-probability keras")


def evaluate_bert_abmil(model, test_loader, config, device):
    """Evaluate BERT-ABMIL using our standard metrics."""
    from src.evaluate import check_cognitive_triggers, load_keywords
    from src.rules import batch_score

    model.eval()
    keywords_path = config.get('keywords_file', 'phishingdata/kwords_stem.json')
    if not os.path.isabs(keywords_path):
        keywords_path = os.path.join(project_root, keywords_path)
    keywords = load_keywords(keywords_path) if os.path.exists(keywords_path) else None

    email_labels_all = []
    email_attn_weights = []
    email_raw_sentences = []
    email_sent_lengths = []
    email_embeddings = []

    with torch.no_grad():
        for batch in test_loader:
            labels = batch['label'].to(device)
            sent_lengths = batch['sent_lengths'].to(device)
            raw_sentences = batch['raw_sentences']

            output, attn_weights, h = model(raw_sentences, sent_lengths)

            B = attn_weights.shape[0]
            for b in range(B):
                email_labels_all.append(labels[b].item())
                email_attn_weights.append(attn_weights[b].cpu().numpy())
                email_raw_sentences.append(raw_sentences[b])
                email_sent_lengths.append(sent_lengths[b])
                email_embeddings.append(h[b].cpu().numpy())

    # Compute metrics
    label_acc_correct = 0
    label_acc_total = 0
    cog_tp_count = 0
    cog_tp_total = 0
    entropy_vals = []
    rule_agree_sum = 0
    rule_agree_count = 0
    all_attn_cont = []
    all_rule_binary = []

    for i, label in enumerate(email_labels_all):
        attn_w = email_attn_weights[i]
        sentences = email_raw_sentences[i]
        sl = email_sent_lengths[i]
        h_b = email_embeddings[i]

        n_valid = int((sl > 0).sum())
        if n_valid == 0:
            continue

        attn_valid = attn_w[:n_valid]
        top1_idx = attn_valid.argmax()

        # Label-Accuracy
        top1_embedding = torch.tensor(h_b[top1_idx:top1_idx+1], device=device)
        top1_logit = model.classifier(top1_embedding).squeeze()
        top1_pred = (torch.sigmoid(top1_logit) > 0.5).long()
        if top1_pred.item() == label:
            label_acc_correct += 1
        label_acc_total += 1

        # Cognitive-TP
        if label == 1 and keywords:
            cog_tp_total += 1
            if top1_idx < len(sentences):
                if check_cognitive_triggers(sentences[top1_idx], keywords):
                    cog_tp_count += 1

        # Entropy
        p = attn_valid / (attn_valid.sum() + 1e-10)
        ent = -(p * np.log(p + 1e-10)).sum()
        entropy_vals.append(ent)

        # Rule Agreement
        if label == 1 and n_valid >= 2 and keywords:
            rule_s = batch_score([sentences], keywords)[0, :n_valid]
            k = min(3, n_valid)
            top_k_attn = set(attn_valid.argsort()[::-1][:k])
            top_k_rule = set(rule_s.argsort()[::-1][:k])
            overlap = len(top_k_attn & top_k_rule)
            rule_agree_sum += overlap / k
            rule_agree_count += 1

            all_attn_cont.extend(attn_valid.tolist())
            all_rule_binary.extend((rule_s > 0.3).astype(int).tolist())

    from sklearn.metrics import precision_recall_curve, auc, f1_score

    f1 = 0
    auprc = 0
    if len(all_rule_binary) > 0 and sum(all_rule_binary) > 0:
        f1 = f1_score(all_rule_binary, [1 if a > np.mean(all_attn_cont) else 0 for a in all_attn_cont], zero_division=0)
        precision, recall, _ = precision_recall_curve(all_rule_binary, all_attn_cont)
        auprc = auc(recall, precision)

    return {
        'label_accuracy': label_acc_correct / label_acc_total if label_acc_total > 0 else 0,
        'cognitive_tp': cog_tp_count / cog_tp_total if cog_tp_total > 0 else 0,
        'attention_entropy': np.mean(entropy_vals) if entropy_vals else 0,
        'rule_agreement': rule_agree_sum / rule_agree_count if rule_agree_count > 0 else 0,
        'f1': f1,
        'auprc': auprc,
    }


def print_results_table(results):
    """Print a formatted results table."""
    print("\n" + "=" * 90)
    print(f"{'Model':<30} {'LabelAcc':>10} {'CogTP':>10} {'Entropy':>10} {'RuleAgree':>10} {'F1':>10} {'AUPRC':>10}")
    print("=" * 90)

    # Order: existing first, then new baselines
    order = [
        'Rules-only',
        'ABMIL',
        'BERT-ABMIL',
        'Snorkel',
        'AI2TALE',
        'Sparse ABMIL (Ours)',
        'w/o l1 sparsity',
        'w/o rule alignment',
        'w/o temp annealing',
    ]

    for name in order:
        if name not in results:
            continue
        r = results[name]
        print(f"{name:<30} {r['label_accuracy']:>10.4f} {r['cognitive_tp']:>10.4f} "
              f"{r['attention_entropy']:>10.4f} {r['rule_agreement']:>10.4f} "
              f"{r['f1']:>10.4f} {r['auprc']:>10.4f}")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-train', action='store_true', help='Skip training, use existing checkpoints')
    parser.add_argument('--skip-ai2tale', action='store_true', help='Skip AI2TALE (requires TensorFlow)')
    args = parser.parse_args()

    config = load_config()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    from src.dataset import create_dataloaders
    train_loader, test_loader = create_dataloaders(config)

    results = {}

    # Evaluate existing models
    evaluate_existing_models(config, device, test_loader, results)

    # Train and evaluate new baselines
    run_new_baselines(config, device, test_loader, results,
                      skip_train=args.skip_train, skip_ai2tale=args.skip_ai2tale)

    # Print and save results
    print_results_table(results)

    # Save
    output_path = os.path.join(project_root, 'experiments', 'all_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == '__main__':
    main()
