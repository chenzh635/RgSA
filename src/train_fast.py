"""Fast training loop with precomputed rule scores."""

import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.rules import load_keywords, batch_score


def compute_loss_fast(output, labels, attn_scores=None, rule_scores=None,
                      lambda_sparse=0.1, lambda_rule=0.5):
    """Compute joint loss using precomputed rule scores.

    L = L_bce(Y, Ŷ) + λ_sparse * Σ|scores| + λ_rule * BCE(rule_scores, attn_weights)
    """
    B = output.shape[0]
    device = output.device

    L_bce = F.binary_cross_entropy_with_logits(output, labels.float())
    total_loss = L_bce
    losses = {'bce': L_bce.item()}

    if attn_scores is not None and lambda_sparse > 0:
        L_sparse = torch.mean(torch.abs(attn_scores))
        total_loss = total_loss + lambda_sparse * L_sparse
        losses['sparse'] = L_sparse.item()

    if rule_scores is not None and lambda_rule > 0:
        S = attn_scores.shape[1]
        # rule_scores shape: (B, S) already padded
        rs = torch.tensor(rule_scores, device=device, dtype=torch.float32)
        rs = torch.clamp(rs, 0.01, 0.99)
        L_rule = F.binary_cross_entropy_with_logits(attn_scores, rs)
        total_loss = total_loss + lambda_rule * L_rule
        losses['rule'] = L_rule.item()

    return total_loss, losses


def get_temperature(epoch, temperature_init=1.0, temperature_min=0.1, decay=0.95):
    temp = temperature_init * (decay ** epoch)
    return max(temp, temperature_min)


def train_fast(model, train_loader, test_loader, config, device):
    """Training loop with precomputed rule scores for speed."""
    epochs = config.get('epochs', 10)
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0.0001)
    log_interval = config.get('log_interval', 50)
    save_dir = config.get('save_dir', 'experiments')
    os.makedirs(save_dir, exist_ok=True)

    # Load keywords
    keywords_path = config.get('keywords_file', 'phishingdata/kwords_stem.json')
    if not os.path.isabs(keywords_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        keywords_path = os.path.join(project_root, keywords_path)
    keywords = load_keywords(keywords_path) if os.path.exists(keywords_path) else None

    lambda_sparse = config.get('lambda_sparse', 0.1)
    lambda_rule = config.get('lambda_rule', 0.5)
    temp_init = config.get('temperature_init', 1.0)
    temp_min = config.get('temperature_min', 0.1)
    temp_decay = config.get('temperature_decay', 0.95)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_test_acc = 0
    is_sparse = hasattr(model, 'use_sparse') and model.use_sparse

    # Precompute rule scores for all training samples
    print("Precomputing rule scores for training set...")
    all_rule_scores = {}
    max_S = 0
    for idx, sample in enumerate(train_loader.dataset):
        raw_sentences = sample['raw_sentences']
        scores = batch_score([raw_sentences], keywords)  # (1, n_sents)
        all_rule_scores[idx] = scores[0]  # (n_sents,)
        max_S = max(max_S, scores.shape[1])

    # Pad all rule scores to max_S
    train_rule_scores = np.zeros((len(train_loader.dataset), max_S), dtype=np.float32)
    for idx in all_rule_scores:
        scores = all_rule_scores[idx]
        train_rule_scores[idx, :len(scores)] = scores
    print(f"  Done. {len(all_rule_scores)} samples, max {max_S} sentences.")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_losses = {'bce': 0, 'sparse': 0, 'rule': 0}

        temperature = get_temperature(epoch, temp_init, temp_min, temp_decay) if is_sparse else 1.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [τ={temperature:.3f}]')
        for batch_idx, batch in enumerate(pbar):
            tokens = batch['tokens'].to(device)
            labels = batch['label'].to(device)
            sent_lengths = batch['sent_lengths'].to(device)
            idxs = batch['idx']

            # Get precomputed rule scores for this batch
            batch_rule_scores = train_rule_scores[idxs] if keywords is not None else None

            optimizer.zero_grad()

            if is_sparse:
                output, attn_weights, attn_scores, h = model(tokens, sent_lengths, temperature)
                loss, losses = compute_loss_fast(
                    output, labels, attn_scores, batch_rule_scores,
                    lambda_sparse, lambda_rule
                )
            else:
                output, attn_weights, h = model(tokens, sent_lengths)
                loss = F.binary_cross_entropy_with_logits(output, labels.float())
                losses = {'bce': loss.item()}

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * labels.shape[0]
            preds = (torch.sigmoid(output) > 0.5).long()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.shape[0]

            for k in all_losses:
                all_losses[k] += losses.get(k, 0) * labels.shape[0]

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{total_correct / total_samples:.4f}',
            })

            if (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / total_samples
                avg_acc = total_correct / total_samples
                pbar.set_description(f'Epoch {epoch+1}/{epochs} [τ={temperature:.3f}] loss={avg_loss:.4f} acc={avg_acc:.4f}')

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        print(f'Epoch {epoch+1}: avg_loss={avg_loss:.4f}, acc={avg_acc:.4f}, '
              f'bce={all_losses["bce"]/total_samples:.4f}, '
              f'sparse={all_losses["sparse"]/total_samples:.4f}, '
              f'rule={all_losses["rule"]/total_samples:.4f}')

        # Evaluate on test set
        test_acc = evaluate_simple(model, test_loader, config, device)
        print(f'  Test accuracy: {test_acc:.4f}')

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            model_suffix = 'sparse' if hasattr(model, 'use_sparse') and model.use_sparse else 'abmil'
            ckpt_path = os.path.join(save_dir, f'{model_suffix}_best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'temperature': temperature,
            }, ckpt_path)
            print(f'  Saved best model (acc={test_acc:.4f})')

    print(f'\nTraining complete. Best test accuracy: {best_test_acc:.4f}')


def evaluate_simple(model, test_loader, config, device):
    """Simple evaluation for training loop progress tracking."""
    model.eval()
    total_correct = 0
    total_samples = 0
    is_sparse = hasattr(model, 'use_sparse') and model.use_sparse

    with torch.no_grad():
        for batch in test_loader:
            tokens = batch['tokens'].to(device)
            labels = batch['label'].to(device)
            sent_lengths = batch['sent_lengths'].to(device)

            if is_sparse:
                output, attn_weights, _, _ = model(tokens, sent_lengths, temperature=0.1)
            else:
                output, attn_weights, _ = model(tokens, sent_lengths)

            preds = (torch.sigmoid(output) > 0.5).long()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.shape[0]

    model.train()
    return total_correct / total_samples if total_samples > 0 else 0
