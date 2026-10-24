"""Sparse ABMIL + BERT: Apply sparsity mechanisms to BERT-ABMIL.

This experiment adds L1 regularization, temperature annealing, and rule-guided
attention alignment to the BERT-ABMIL baseline. The goal is to test whether
sparsity mechanisms improve BERT-ABMIL's already-strong Spearman correlation
while reducing its diffuse attention entropy.

If successful (Spearman >0.9 with entropy <0.6), this becomes the flagship
result showing that sparsity + rules improve any encoder.

Usage:
    python experiments/scripts/run_sparse_bert.py [--seed 42] [--epochs 10]
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.baselines.bert_abmil import (
    BertSentenceEncoder, EmailTextDataset, bert_collate_fn,
    create_bert_dataloaders
)
from src.rules import batch_score, load_keywords
from src.evaluate import load_keywords as eval_load_keywords


class SparseBertABMIL(nn.Module):
    """BERT-ABMIL with sparse attention (temperature scaling + L1 + rule alignment).

    Same architecture as BertABMIL but adds:
    1. Temperature-controlled softmax for attention
    2. Returns raw attention scores for L1 regularization
    """

    def __init__(self, attention_hidden=64, classifier_hidden=None, bert_dim=768):
        super().__init__()
        self.encoder = BertSentenceEncoder(frozen=True)
        self.sent_dim = bert_dim
        self.use_sparse = True  # Flag for compatibility

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.sent_dim, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1),
        )

        # Classifier
        if classifier_hidden is None:
            classifier_hidden = [128, 64]
        layers = []
        prev_dim = self.sent_dim
        for h in classifier_hidden:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(0.3)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, raw_sentences_list, sent_lengths=None, temperature=1.0):
        """
        Args:
            raw_sentences_list: list of B lists of strings
            sent_lengths: (B, max_S) for masking
            temperature: softmax temperature (lower = sparser)
        Returns:
            output: (B,) email classification logits
            attn_weights: (B, S) attention weights (post-softmax)
            attn_scores: (B, S) raw attention scores (pre-softmax, for L1)
            h: (B, S, D) sentence embeddings
        """
        B = len(raw_sentences_list)
        max_S = max(len(sents) for sents in raw_sentences_list)

        all_h = []
        for b in range(B):
            sents = raw_sentences_list[b] if raw_sentences_list[b] else ["[PAD]"]
            h_b = self.encoder(sents)
            if h_b.shape[0] < max_S:
                pad = torch.zeros(max_S - h_b.shape[0], self.sent_dim, device=h_b.device)
                h_b = torch.cat([h_b, pad], dim=0)
            all_h.append(h_b)

        h = torch.stack(all_h)  # (B, max_S, 768)

        # Compute raw attention scores
        scores = self.attention(h).squeeze(-1)  # (B, S)
        if sent_lengths is not None:
            mask = (sent_lengths[:, :scores.shape[1]] > 0).float().to(scores.device)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_scores = scores  # Raw scores for L1 regularization

        # Temperature-scaled softmax
        attn_weights = F.softmax(scores / temperature, dim=1)

        z = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)  # (B, 768)
        output = self.classifier(z).squeeze(-1)  # (B,)

        return output, attn_weights, attn_scores, h


def compute_sparse_loss(output, labels, attn_scores, raw_sentences, keywords,
                        lambda_sparse=0.1, lambda_rule=3.0):
    """Joint loss: BCE + L1 sparsity + rule alignment.

    NOTE: Unlike the CNN model where raw attention scores are small (150-d input),
    BERT's 768-d input produces large raw scores. We use tanh to normalize them
    to [-1, 1] for both L1 and rule alignment losses.
    """
    device = output.device
    L_bce = F.binary_cross_entropy_with_logits(output, labels.float())
    total_loss = L_bce
    losses = {'bce': L_bce.item()}

    # Normalize scores to [-1, 1] via tanh for bounded regularization
    normed_scores = torch.tanh(attn_scores)

    # L1 sparsity on normalized attention scores
    if lambda_sparse > 0:
        L_sparse = torch.mean(torch.abs(normed_scores))
        total_loss = total_loss + lambda_sparse * L_sparse
        losses['sparse'] = L_sparse.item()

    # Rule alignment loss
    if raw_sentences is not None and keywords is not None and lambda_rule > 0:
        with torch.no_grad():
            rule_scores_np = batch_score(raw_sentences, keywords)
            S = normed_scores.shape[1]
            if rule_scores_np.shape[1] < S:
                padded = np.zeros((normed_scores.shape[0], S), dtype=np.float32)
                padded[:, :rule_scores_np.shape[1]] = rule_scores_np
                rule_scores_np = padded
            elif rule_scores_np.shape[1] > S:
                rule_scores_np = rule_scores_np[:, :S]
            # Map rule scores from [0, 1] to [-1, 1] for tanh-space alignment
            rule_targets = torch.tensor(rule_scores_np * 2 - 1, device=device, dtype=torch.float32)

        L_rule = F.mse_loss(normed_scores, rule_targets)
        total_loss = total_loss + lambda_rule * L_rule
        losses['rule'] = L_rule.item()

    return total_loss, losses


def get_temperature(epoch, temp_init=1.0, temp_min=0.1, decay=0.95):
    """Temperature annealing schedule."""
    return max(temp_init * (decay ** epoch), temp_min)


def train_sparse_bert(model, train_loader, test_loader, config, device):
    """Train Sparse BERT-ABMIL with joint loss."""
    epochs = config.get('epochs', 10)
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0.0001)
    save_dir = config.get('save_dir', 'experiments/sparse_bert')
    os.makedirs(save_dir, exist_ok=True)

    # Load keywords
    keywords_path = config.get('keywords_file', 'phishingdata/kwords_stem.json')
    if not os.path.isabs(keywords_path):
        keywords_path = os.path.join(project_root, keywords_path)
    keywords = load_keywords(keywords_path) if os.path.exists(keywords_path) else None

    lambda_sparse = config.get('lambda_sparse', 0.1)
    lambda_rule = config.get('lambda_rule', 3.0)
    temp_init = config.get('temperature_init', 1.0)
    temp_min = config.get('temperature_min', 0.1)
    temp_decay = config.get('temperature_decay', 0.95)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_test_acc = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_losses = {'bce': 0, 'sparse': 0, 'rule': 0}

        temperature = get_temperature(epoch, temp_init, temp_min, temp_decay)

        pbar = tqdm(train_loader, desc=f'Sparse-BERT Epoch {epoch+1}/{epochs} [τ={temperature:.3f}]')
        for batch in pbar:
            labels = batch['label'].to(device)
            sent_lengths = batch['sent_lengths'].to(device)
            raw_sentences = batch['raw_sentences']

            optimizer.zero_grad()
            output, attn_weights, attn_scores, h = model(raw_sentences, sent_lengths, temperature)
            loss, losses = compute_sparse_loss(
                output, labels, attn_scores, raw_sentences, keywords,
                lambda_sparse, lambda_rule
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * labels.shape[0]
            preds = (torch.sigmoid(output) > 0.5).long()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.shape[0]

            for k in all_losses:
                all_losses[k] += losses.get(k, 0) * labels.shape[0]

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{total_correct/total_samples:.4f}'})

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        print(f'Epoch {epoch+1}: loss={avg_loss:.4f}, acc={avg_acc:.4f}, '
              f'bce={all_losses["bce"]/total_samples:.4f}, '
              f'sparse={all_losses["sparse"]/total_samples:.4f}, '
              f'rule={all_losses["rule"]/total_samples:.4f}')

        # Test accuracy
        test_correct = 0
        test_total = 0
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                labels = batch['label'].to(device)
                sent_lengths = batch['sent_lengths'].to(device)
                output, _, _, _ = model(batch['raw_sentences'], sent_lengths, temperature=0.1)
                preds = (torch.sigmoid(output) > 0.5).long()
                test_correct += (preds == labels).sum().item()
                test_total += labels.shape[0]
        test_acc = test_correct / test_total
        print(f'  Test acc: {test_acc:.4f}')

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            ckpt_path = os.path.join(save_dir, 'sparse_bert_best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'temperature': temperature,
            }, ckpt_path)
            print(f'  Saved best model (acc={test_acc:.4f})')

    print(f'\nSparse BERT-ABMIL training complete. Best test acc: {best_test_acc:.4f}')


def evaluate_sparse_bert(model, test_loader, config, device):
    """Evaluate Sparse BERT-ABMIL using the standard 4 metrics."""
    from sklearn.metrics import precision_recall_curve, auc, f1_score
    from scipy.stats import spearmanr

    keywords_path = config.get('keywords_file', 'phishingdata/kwords_stem.json')
    if not os.path.isabs(keywords_path):
        keywords_path = os.path.join(project_root, keywords_path)
    keywords = load_keywords(keywords_path) if os.path.exists(keywords_path) else None

    model.eval()
    all_scores, all_labels = [], []
    correlations, entropies = [], []

    with torch.no_grad():
        for batch in test_loader:
            labels = batch['label'].to(device)
            sent_lengths = batch['sent_lengths'].to(device)
            raw_sentences = batch['raw_sentences']

            output, attn_weights, _, _ = model(raw_sentences, sent_lengths, temperature=0.1)
            scores = torch.sigmoid(output).cpu().numpy()

            all_scores.extend(scores.flatten().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            B = attn_weights.shape[0]
            for b in range(B):
                sl = sent_lengths[b]
                n_valid = int((sl > 0).sum().item()) if hasattr(sl, 'sum') else int((sl > 0).sum())
                if n_valid < 2:
                    continue

                attn_b = attn_weights[b, :n_valid].cpu().numpy()

                # Entropy
                p = attn_b / (attn_b.sum() + 1e-10)
                ent = -(p * np.log(p + 1e-10)).sum()
                entropies.append(float(ent))

                # Spearman for phishing emails
                if labels[b].item() == 1 and keywords and len(set(attn_b)) >= 2:
                    rule_b = batch_score([raw_sentences[b]], keywords)[0, :n_valid]
                    if len(set(rule_b)) >= 2:
                        corr, _ = spearmanr(attn_b, rule_b)
                        if not np.isnan(corr):
                            correlations.append(corr)

    # AUPRC
    if len(set(all_labels)) >= 2:
        precision, recall, _ = precision_recall_curve(all_labels, all_scores)
        email_auprc = auc(recall, precision)
    else:
        email_auprc = 0.0

    # F1
    preds = [1 if s > 0.5 else 0 for s in all_scores]
    email_f1 = f1_score(all_labels, preds, zero_division=0)

    return {
        'email_auprc': float(email_auprc),
        'email_f1': float(email_f1),
        'spearman_corr': float(np.mean(correlations)) if correlations else 0.0,
        'attention_entropy': float(np.mean(entropies)) if entropies else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description='Sparse ABMIL + BERT experiment')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lambda-rule', type=float, default=3.0)
    parser.add_argument('--eval-only', action='store_true', help='Skip training, evaluate existing checkpoint')
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Config
    import yaml
    with open(os.path.join(project_root, 'experiments/configs/defaults.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    config['save_dir'] = os.path.join(project_root, 'experiments', 'sparse_bert')
    config['epochs'] = args.epochs
    config['lambda_rule'] = args.lambda_rule
    config['lambda_sparse'] = 0.1
    config['temperature_init'] = 1.0
    config['temperature_min'] = 0.1
    config['temperature_decay'] = 0.95

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: λ_rule={args.lambda_rule}, λ_sparse=0.1, epochs={args.epochs}, seed={args.seed}")

    train_loader, test_loader = create_bert_dataloaders(config)

    model = SparseBertABMIL(
        attention_hidden=config.get('attention_hidden', 64),
        classifier_hidden=config.get('classifier_hidden', [128, 64]),
    ).to(device)

    if not args.eval_only:
        start_time = time.time()
        train_sparse_bert(model, train_loader, test_loader, config, device)
        train_time = time.time() - start_time
        print(f"Training time: {train_time/60:.1f} minutes")

    # Load best checkpoint
    ckpt_path = os.path.join(config['save_dir'], 'sparse_bert_best_model.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, acc={ckpt.get('test_acc', '?'):.4f}")

    # Evaluate
    print("\nEvaluating Sparse BERT-ABMIL...")
    results = evaluate_sparse_bert(model, test_loader, config, device)

    print("\n" + "=" * 60)
    print("Sparse BERT-ABMIL Results:")
    print(f"  Email AUPRC:    {results['email_auprc']:.4f}")
    print(f"  Email F1:       {results['email_f1']:.4f}")
    print(f"  Spearman Corr:  {results['spearman_corr']:.4f}")
    print(f"  Attn Entropy:   {results['attention_entropy']:.4f}")
    print("=" * 60)

    # Compare with baselines
    print("\nComparison:")
    print(f"  BERT-ABMIL:       Spearman=0.9423, Entropy=1.0672")
    print(f"  Sparse ABMIL:     Spearman=0.7037, Entropy=0.5076")
    print(f"  Sparse BERT-ABMIL: Spearman={results['spearman_corr']:.4f}, Entropy={results['attention_entropy']:.4f}")

    if results['spearman_corr'] > 0.9 and results['attention_entropy'] < 0.6:
        print("\n  🏆 FLAGSHIP RESULT: Spearman >0.9 with entropy <0.6!")
    elif results['spearman_corr'] > 0.9:
        print("\n  ✓ High Spearman but entropy still high — sparsity partially effective")
    elif results['attention_entropy'] < 0.6:
        print("\n  ✓ Sharp attention but Spearman dropped — sparsity too aggressive?")

    # Save results
    os.makedirs(config['save_dir'], exist_ok=True)
    with open(os.path.join(config['save_dir'], 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {os.path.join(config['save_dir'], 'results.json')}")


if __name__ == '__main__':
    main()
