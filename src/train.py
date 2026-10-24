"""Training loop with joint loss: BCE + L1 sparsity + temperature annealing."""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.rules import batch_score, load_keywords


def compute_loss(output, labels, attn_scores=None, raw_sentences=None, keywords=None,
                 lambda_sparse=0.1, lambda_rule=0.5, rule_flip_frac=0.0,
                 fixed_flip=None):
    """Compute joint loss for Sparse ABMIL.

    L = L_bce(Y, Ŷ) + λ_sparse * Σ|scores| + λ_rule * BCE(rule_scores, attn_weights)

    Args:
        output: (B,) email classification logits
        labels: (B,) email labels
        attn_scores: (B, S) raw attention scores (for L1)
        raw_sentences: list of list of strings
        keywords: keyword dict (for rule scoring)
        lambda_sparse: L1 weight
        lambda_rule: rule alignment weight
        rule_flip_frac: if > 0, flip r -> 1 - r for this fraction of
            sentence-level rule scores during training (prior-corruption stress
            test, review B7/M9). Training-only; evaluation always uses clean
            rules.
        fixed_flip: optional (B, S) boolean array — pre-drawn fixed corruption
            mask (M9); when given, rule_flip_frac must be > 0 and the mask is
            used instead of a fresh per-batch draw (stochastic corruption).
    """
    B = output.shape[0]
    device = output.device

    # Main loss: email classification
    L_bce = F.binary_cross_entropy_with_logits(output, labels.float())

    total_loss = L_bce
    losses = {'bce': L_bce.item()}

    # L1 sparsity on attention scores
    if attn_scores is not None and lambda_sparse > 0:
        L_sparse = torch.mean(torch.abs(attn_scores))
        total_loss = total_loss + lambda_sparse * L_sparse
        losses['sparse'] = L_sparse.item()

    # Rule alignment loss (optional)
    if raw_sentences is not None and keywords is not None and lambda_rule > 0:
        with torch.no_grad():
            rule_scores_np = batch_score(raw_sentences, keywords)
            if rule_flip_frac > 0:
                if fixed_flip is not None:
                    # M9 fixed corruption: same (email, sentence) scores are
                    # flipped every epoch, rather than re-drawn per batch.
                    B, S = rule_scores_np.shape
                    flip_mask = np.asarray(fixed_flip[:B, :S], dtype=bool)
                else:
                    # stochastic corruption: fresh 20% draw each batch
                    flip_mask = np.random.random(rule_scores_np.shape) < rule_flip_frac
                rule_scores_np = np.where(flip_mask, 1.0 - rule_scores_np, rule_scores_np)
            # Pad or truncate to match attn_scores shape
            S = attn_scores.shape[1] if attn_scores is not None else rule_scores_np.shape[1]
            if rule_scores_np.shape[1] < S:
                padded = np.zeros((B, S), dtype=np.float32)
                padded[:, :rule_scores_np.shape[1]] = rule_scores_np
                rule_scores_np = padded
            elif rule_scores_np.shape[1] > S:
                rule_scores_np = rule_scores_np[:, :S]
            rule_scores = torch.tensor(rule_scores_np, device=device, dtype=torch.float32)

        # Normalize rule scores to [0, 1] (they already are, but clamp for safety)
        rule_scores = torch.clamp(rule_scores, 0, 1)

        # We want attention weights (softmax) to align with rule scores
        # Use BCE between rule_scores and attention logits
        if attn_scores is not None:
            # Convert rule scores to logits for BCE with raw scores
            rule_scores_safe = torch.clamp(rule_scores, 0.01, 0.99)
            rule_logits = torch.log(rule_scores_safe / (1 - rule_scores_safe))
            L_rule = F.binary_cross_entropy_with_logits(attn_scores, rule_scores_safe)
            total_loss = total_loss + lambda_rule * L_rule
            losses['rule'] = L_rule.item()

    return total_loss, losses


def get_temperature(epoch, temperature_init=1.0, temperature_min=0.1, decay=0.95):
    """Compute temperature for sparse attention with annealing."""
    temp = temperature_init * (decay ** epoch)
    return max(temp, temperature_min)


def train(model, train_loader, val_loader, config, device):
    """Main training loop.

    M10: checkpoint selection uses val_loader (fixed split from the training
    set); the test set is never seen during training. Selection criterion is
    validation AUPRC, with validation accuracy as tie-breaker.
    """
    epochs = config.get('epochs', 10)
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0.0001)
    log_interval = config.get('log_interval', 50)
    save_dir = config.get('save_dir', 'experiments')
    os.makedirs(save_dir, exist_ok=True)

    # Load keywords for rule scoring
    keywords_path = config.get('keywords_file', 'phishingdata/kwords_stem.json')
    if not os.path.isabs(keywords_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        keywords_path = os.path.join(project_root, keywords_path)
    keywords = load_keywords(keywords_path) if os.path.exists(keywords_path) else None

    # Hyperparameters
    lambda_sparse = config.get('lambda_sparse', 0.1)
    lambda_rule = config.get('lambda_rule', 0.5)
    rule_flip_frac = config.get('rule_flip_frac', 0.0)
    rule_flip_mode = config.get('rule_flip_mode', 'stochastic')
    temp_init = config.get('temperature_init', 1.0)
    temp_min = config.get('temperature_min', 0.1)
    temp_decay = config.get('temperature_decay', 0.95)

    # M9: fixed corruption mask drawn once over the training corpus
    # (max_sentences x dataset size), reused every epoch. Keyed by dataset
    # index so it attaches to the same emails regardless of loader order.
    fixed_matrix = None
    if rule_flip_frac > 0 and rule_flip_mode == 'fixed':
        # Keyed by GLOBAL dataset index: train_loader may wrap a Subset, so
        # size to the underlying dataset and draw corruption only at the
        # train subset's global indices.
        ds = train_loader.dataset
        if hasattr(ds, 'indices'):
            base = ds.dataset
            train_global = np.asarray(ds.indices)
        else:
            base = ds
            train_global = np.arange(len(ds))
        max_s = config.get('max_sentences', 100)
        rng = np.random.default_rng(config.get('seed', 42))
        fixed_matrix = np.zeros((len(base), max_s), dtype=bool)
        fixed_matrix[train_global] = \
            rng.random((len(train_global), max_s)) < rule_flip_frac

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_score = -1.0
    is_sparse = hasattr(model, 'use_sparse') and model.use_sparse

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_losses = {'bce': 0, 'sparse': 0, 'rule': 0}

        # Temperature for this epoch
        temperature = get_temperature(epoch, temp_init, temp_min, temp_decay) if is_sparse else 1.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [τ={temperature:.3f}]')
        for batch_idx, batch in enumerate(pbar):
            tokens = batch['tokens'].to(device)
            labels = batch['label'].to(device)
            sent_lengths = batch['sent_lengths'].to(device)
            raw_sentences = batch['raw_sentences']

            optimizer.zero_grad()

            fixed_flip = None
            if fixed_matrix is not None:
                fixed_flip = fixed_matrix[np.asarray(batch['idx'])]

            if is_sparse:
                output, attn_weights, attn_scores, h = model(tokens, sent_lengths, temperature)
                loss, losses = compute_loss(
                    output, labels, attn_scores, raw_sentences, keywords,
                    lambda_sparse, lambda_rule, rule_flip_frac, fixed_flip
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

        # End of epoch logging
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        print(f'Epoch {epoch+1}: avg_loss={avg_loss:.4f}, acc={avg_acc:.4f}, '
              f'bce={all_losses["bce"]/total_samples:.4f}, '
              f'sparse={all_losses["sparse"]/total_samples:.4f}, '
              f'rule={all_losses["rule"]/total_samples:.4f}')

        # Evaluate on validation split (M10; test set stays invisible)
        val_auprc, val_acc = evaluate_val(model, val_loader, config, device)
        print(f'  Val AUPRC: {val_auprc:.4f}, val accuracy: {val_acc:.4f}')

        # Save best model (primary: val AUPRC, tie-break: accuracy)
        val_score = val_auprc * 100 + val_acc
        if val_score > best_val_score:
            best_val_score = val_score
            model_suffix = 'sparse' if hasattr(model, 'use_sparse') and model.use_sparse else 'abmil'
            ckpt_path = os.path.join(save_dir, f'{model_suffix}_best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auprc': val_auprc,
                'val_acc': val_acc,
                'temperature': temperature,
            }, ckpt_path)
            print(f'  Saved best model (val_auprc={val_auprc:.4f})')

    print(f'\nTraining complete. Best val AUPRC-based score: {best_val_score:.4f}')


def evaluate_val(model, val_loader, config, device):
    """Validation metrics for checkpoint selection (M10): (AUPRC, accuracy)."""
    from sklearn.metrics import average_precision_score

    model.eval()
    is_sparse = hasattr(model, 'use_sparse') and model.use_sparse
    all_scores, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            tokens = batch['tokens'].to(device)
            labels = batch['label'].to(device)
            sent_lengths = batch['sent_lengths'].to(device)

            if is_sparse:
                output, _, _, _ = model(tokens, sent_lengths, temperature=0.1)
            else:
                output, _, _ = model(tokens, sent_lengths)

            all_scores.extend(torch.sigmoid(output).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    model.train()
    scores = np.asarray(all_scores)
    labels = np.asarray(all_labels)
    acc = float(((scores > 0.5).astype(int) == labels).mean())
    auprc = float(average_precision_score(labels, scores)) if len(set(labels)) > 1 else 0.0
    return auprc, acc


def evaluate_simple(model, test_loader, config, device):
    """Legacy test-set progress metric (pre-M10 scripts). Prefer evaluate_val."""
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
