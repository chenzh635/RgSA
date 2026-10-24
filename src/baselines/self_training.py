"""Self-Training experiment for phishing sentence localization.

Uses iterative pseudo-labeling to improve sentence localization:
1. Start with pre-trained Sparse ABMIL
2. Use attention weights as soft pseudo-labels for sentences
3. Train a sentence-level classifier on pseudo-labels
4. Combine sentence classifier output with ABMIL attention
5. Iterate: use combined scores to generate refined pseudo-labels

This is an OPTIMIZATION experiment, not a baseline.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_curve, auc, f1_score
from tqdm import tqdm


class SentenceClassifier(torch.nn.Module):
    """Sentence-level classifier for self-training refinement."""

    def __init__(self, vocab_size=311774, embed_dim=150, hidden_dim=150):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = torch.nn.Conv1d(embed_dim, hidden_dim, 3, padding=1)
        self.pool = torch.nn.AdaptiveMaxPool1d(1)
        self.dropout = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, tokens):
        x = self.embedding(tokens)       # (N, T, E)
        x = x.transpose(1, 2)            # (N, E, T)
        x = self.conv(x)                 # (N, H, T)
        x = F.relu(x)
        x = self.pool(x).squeeze(-1)     # (N, H)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)


def evaluate_combined(abmil, sent_cls, test_loader, config, device, alpha=0.5):
    """Evaluate combined model with new 4 metrics.

    Args:
        abmil: SparseABMIL model
        sent_cls: SentenceClassifier model
        test_loader: test DataLoader
        config: config dict
        device: torch device
        alpha: weight for ABMIL attention (1-alpha for sentence classifier)

    Returns:
        dict with email_auprc, email_f1, spearman_corr, attention_entropy
    """
    from src.rules import batch_score, load_keywords

    keywords_path = config.get('keywords_file', 'phishingdata/kwords_stem.json')
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not os.path.isabs(keywords_path):
        keywords_path = os.path.join(project_root, keywords_path)
    keywords = load_keywords(keywords_path) if os.path.exists(keywords_path) else None

    abmil.eval()
    sent_cls.eval()

    all_labels, all_scores = [], []
    all_entropies, all_spearmans = [], []

    with torch.no_grad():
        for batch in test_loader:
            tokens = batch['tokens'].to(device)
            labels = batch['label'].to(device)
            sent_lengths = batch['sent_lengths'].to(device)
            raw_sentences = batch['raw_sentences']

            _, attn_weights, _, h = abmil(tokens, sent_lengths, temperature=0.1)

            B, S = attn_weights.shape
            rule_scores = batch_score(raw_sentences, keywords)

            for b in range(B):
                all_labels.append(int(labels[b].item()))
                n_valid = min(int(sent_lengths[b].sum().item()), S, len(raw_sentences[b]))
                n_rule = min(n_valid, rule_scores.shape[1])

                if n_valid == 0:
                    all_scores.append(0.0)
                    continue

                # Get sentence classifier scores
                sent_probs = []
                for s in range(n_valid):
                    logits = sent_cls(tokens=batch['tokens'][b, s].unsqueeze(0).to(device))
                    sent_probs.append(torch.sigmoid(logits).item())

                # Combine attention and sentence classifier
                if sent_probs and n_valid > 0:
                    sent_arr = np.array(sent_probs[:n_valid])
                    sent_arr = sent_arr / (sent_arr.max() + 1e-10)
                    attn_b = attn_weights[b, :n_valid].cpu().numpy()
                    # Normalize attention to [0, 1] range
                    attn_b = attn_b / (attn_b.max() + 1e-10)
                    combined = alpha * attn_b + (1 - alpha) * sent_arr
                    combined_sum = combined.sum()
                    if combined_sum > 0:
                        combined = combined / combined_sum
                else:
                    combined = attn_weights[b, :n_valid].cpu().numpy()

                # Email-level score: max of combined distribution (proxy for sigmoid output)
                email_score = combined.max() if len(combined) > 0 else 0.0
                all_scores.append(float(email_score))

                # Entropy
                p = combined / (combined.sum() + 1e-10)
                entropy = -(p * np.log(p + 1e-10)).sum()
                all_entropies.append(float(entropy))

                # Spearman for phishing emails
                if labels[b].item() == 1 and n_rule >= 2:
                    rule_b = rule_scores[b, :n_rule]
                    if len(set(combined[:n_rule])) >= 2 and len(set(rule_b)) >= 2:
                        corr, _ = spearmanr(combined[:n_rule], rule_b)
                        if not np.isnan(corr):
                            all_spearmans.append(float(corr))

    # AUPRC
    if len(set(all_labels)) < 2:
        email_auprc = 0.0
    else:
        precision_arr, recall_arr, _ = precision_recall_curve(all_labels, all_scores)
        email_auprc = auc(recall_arr, precision_arr)

    # F1
    preds = [1 if s > 0.5 else 0 for s in all_scores]
    email_f1 = f1_score(all_labels, preds, zero_division=0)

    spearman_corr = np.mean(all_spearmans) if all_spearmans else 0.0
    attention_entropy = np.mean(all_entropies) if all_entropies else 0.0

    return {
        'email_auprc': float(email_auprc),
        'email_f1': float(email_f1),
        'spearman_corr': float(spearman_corr),
        'attention_entropy': float(attention_entropy),
    }


def self_training_pipeline(train_loader, test_loader, config, device, n_rounds=5, alpha=0.5):
    """Run self-training for sentence localization.

    Args:
        train_loader: training DataLoader
        test_loader: test DataLoader
        config: training config
        device: torch device
        n_rounds: number of self-training rounds
        alpha: weight for ABMIL attention vs sentence classifier (alpha * attn + (1-alpha) * sent)

    Returns:
        round_results: list of dicts with per-round metrics
    """
    from src.models.abmil import SparseABMIL

    save_dir = config.get('save_dir', 'experiments/self_training')
    os.makedirs(save_dir, exist_ok=True)

    vocab_size = 311774

    # Load pre-trained Sparse ABMIL
    sparse_ckpt = os.path.join('experiments', 'sparse_best_model.pt')
    if not os.path.exists(sparse_ckpt):
        print("ERROR: No pre-trained Sparse ABMIL found at experiments/sparse_best_model.pt")
        return []

    print("Loading pre-trained Sparse ABMIL...")
    abmil = SparseABMIL(
        hidden_dim=config.get('hidden_dim', 64),
        attention_hidden=config.get('attention_hidden', 64),
        classifier_hidden=config.get('classifier_hidden', [128, 64]),
        use_sparse=True,
    ).to(device)
    ckpt = torch.load(sparse_ckpt, map_location=device, weights_only=False)
    abmil.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f"  Loaded from epoch {ckpt.get('epoch', '?')}, test_acc={ckpt.get('test_acc', '?'):.4f}")

    # Initialize sentence classifier
    sent_classifier = SentenceClassifier(vocab_size=vocab_size).to(device)

    round_results = []

    # Round 0: Baseline ABMIL only
    print("\nRound 0 (ABMIL only):")
    abmil_results = evaluate_combined(abmil, sent_classifier, test_loader, config, device, alpha=1.0)
    round_results.append({'round': 0, 'method': 'ABMIL only', **abmil_results})
    for k, v in abmil_results.items():
        print(f"  {k}: {v:.4f}")

    # Self-training rounds
    best_spearman = abmil_results['spearman_corr']

    for rnd in range(1, n_rounds + 1):
        print(f"\n{'='*50}")
        print(f"Self-Training Round {rnd}")
        print(f"{'='*50}")

        # Step 1: Generate pseudo-labels
        print("Generating pseudo-labels...")
        all_sent_tokens = []
        all_sent_labels = []

        abmil.eval()
        sent_classifier.eval()

        with torch.no_grad():
            for batch in train_loader:
                tokens = batch['tokens'].to(device)
                labels = batch['label'].to(device)
                sent_lengths = batch['sent_lengths'].to(device)
                raw_sentences = batch['raw_sentences']

                _, attn_weights, _, h = abmil(tokens, sent_lengths, temperature=0.1)

                B, S = attn_weights.shape
                for b in range(B):
                    n_valid_raw = min(int((sent_lengths[b] > 0).sum()), S, len(raw_sentences[b]))
                    if labels[b].item() != 1:
                        for s_idx in range(n_valid_raw):
                            all_sent_tokens.append(tokens[b, s_idx])
                            all_sent_labels.append(0.0)
                        continue

                    n_valid = min(int((sent_lengths[b] > 0).sum()), S, len(raw_sentences[b]))
                    for s_idx in range(n_valid):
                        sent_tokens = tokens[b, s_idx]
                        all_sent_tokens.append(sent_tokens)

                        abmil_score = attn_weights[b, s_idx].item()

                        if rnd > 1:
                            sent_logits = sent_classifier(sent_tokens.unsqueeze(0))
                            sent_prob = torch.sigmoid(sent_logits).item()
                            combined = alpha * abmil_score + (1 - alpha) * sent_prob
                        else:
                            combined = abmil_score

                        all_sent_labels.append(combined)

        tokens_tensor = torch.stack(all_sent_tokens).to(device)
        labels_tensor = torch.tensor(all_sent_labels, dtype=torch.float32).to(device)
        n_sent = len(all_sent_labels)
        n_positive = sum(1 for l in all_sent_labels if l > 0.5)
        print(f"  Total sentences: {n_sent}, Positive (score>0.5): {n_positive}")

        # Step 2: Train sentence classifier on pseudo-labels
        print("Training sentence classifier...")
        optimizer = torch.optim.Adam(sent_classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        batch_size = 256
        epochs = 5

        for epoch in range(epochs):
            sent_classifier.train()
            perm = torch.randperm(n_sent)
            total_loss = 0

            for start_idx in range(0, n_sent, batch_size):
                end_idx = min(start_idx + batch_size, n_sent)
                idx = perm[start_idx:end_idx]
                optimizer.zero_grad()
                output = sent_classifier(tokens_tensor[idx])
                loss = F.binary_cross_entropy_with_logits(output, labels_tensor[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sent_classifier.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * (end_idx - start_idx)

            print(f"  Epoch {epoch+1}: loss={total_loss/n_sent:.4f}")

        # Step 3: Evaluate combined model
        print("Evaluating combined model...")
        results = evaluate_combined(abmil, sent_classifier, test_loader, config, device, alpha=alpha)
        round_results.append({'round': rnd, 'method': f'ABMIL + Self-Training (round {rnd})', **results})
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")

        if results['spearman_corr'] > best_spearman:
            best_spearman = results['spearman_corr']
            torch.save({
                'round': rnd,
                'sent_classifier_state_dict': sent_classifier.state_dict(),
                'results': results,
            }, os.path.join(save_dir, 'self_training_best.pt'))
            print(f"  ** New best model (Spearman={best_spearman:.4f}) **")

    # Save results
    with open(os.path.join(save_dir, 'self_training_results.json'), 'w') as f:
        json.dump(round_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("Self-Training Summary")
    print(f"{'='*60}")
    print(f"{'Round':>6} {'Method':<40} {'AUPRC':>8} {'F1':>8} {'Spearman':>10} {'Entropy':>10}")
    print(f"{'-'*82}")
    for r in round_results:
        print(f"{r['round']:>6} {r['method']:<40} {r['email_auprc']:>8.4f} {r['email_f1']:>8.4f} {r['spearman_corr']:>10.4f} {r['attention_entropy']:>10.4f}")

    return round_results


def run_self_training(config=None, n_rounds=5, alpha=0.5):
    """Entry point."""
    if config is None:
        import yaml
        with open('experiments/configs/defaults.yaml', 'r') as f:
            config = yaml.safe_load(f)
        config['save_dir'] = 'experiments/self_training'

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    from src.dataset import create_dataloaders
    train_loader, test_loader = create_dataloaders(config)

    results = self_training_pipeline(train_loader, test_loader, config, device, n_rounds=n_rounds, alpha=alpha)
    return results


if __name__ == '__main__':
    run_self_training()
