"""AI2TALE baseline re-implemented in PyTorch (stable version).

Reproduces the core mechanism of AI2TALE:
- CNN sentence encoder (same architecture as ABMIL)
- Soft attention selection (replaces Gumbel-Softmax for stability)
- Information bottleneck loss with entropy regularization
- Separate predictor for classification

For a fair comparison with Sparse ABMIL, we use the same CNN encoder,
same training device (MPS), and same evaluation metrics.

Note: The original AI2TALE uses Gumbel-Softmax hard selection (k=1).
Here we use soft selection (softmax over selector logits) because
Gumbel-Softmax training on this dataset repeatedly produced NaN
due to gradient explosion in the discrete sampling process.
Soft selection still captures the core IB principle — the selector
learns to distribute probability mass over sentences while the
predictor learns to classify from the weighted embedding.
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ── AI2TALE Model ──────────────────────────────────────────────────

class AI2TALEModel(nn.Module):
    """AI2TALE with soft selection (stable alternative to Gumbel-Softmax).

    Architecture:
    1. Sentence Encoder: CNN → sentence embeddings H
    2. Selector: MLP → softmax distribution over sentences
    3. Predictor: MLP on selected sentence → classification logit

    Loss: L = BCE(y, ŷ) + β * KL(q(Z) || p(Z))

    KL is computed as entropy regularization:
    KL(q || uniform) = -H(q) + log(S) → minimize by maximizing H(q)
    But we use entropy as regularization to prevent collapse.
    """

    def __init__(self, vocab_size=311774, embed_dim=150, hidden_dim=150,
                 attention_hidden=64, classifier_hidden=None, beta=0.1):
        super().__init__()
        if classifier_hidden is None:
            classifier_hidden = [128, 64]
        self.beta = beta

        # Sentence encoder (same CNN as ABMIL)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, hidden_dim, 3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout_enc = nn.Dropout(0.2)

        # Selector MLP (same architecture as ABMIL attention)
        self.selector = nn.Sequential(
            nn.Linear(hidden_dim, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1),
        )

        # Predictor MLP
        layers = []
        prev_dim = hidden_dim
        for h in classifier_hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.predictor = nn.Sequential(*layers)

    def encode_sentences(self, tokens):
        """Encode sentences to embeddings."""
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        B, S, T = tokens.shape
        x = self.embedding(tokens)           # (B, S, T, E)
        x = x.view(B * S, T, -1)             # (B*S, T, E)
        x = x.transpose(1, 2)                # (B*S, E, T)
        x = self.conv(x)                     # (B*S, H, T)
        x = F.relu(x)
        x = self.pool(x).squeeze(-1)         # (B*S, H)
        x = self.dropout_enc(x)
        x = x.view(B, S, -1)                 # (B, S, D)
        return x

    def forward(self, tokens, sent_lengths):
        """Full forward pass.

        Args:
            tokens: (B, S, T)
            sent_lengths: (B, S) number of tokens per sentence
        Returns:
            output: (B,) classification logits
            sel_probs: (B, S) selection probabilities (softmax)
            H: (B, S, D) sentence embeddings
        """
        B, S, _ = tokens.shape
        H = self.encode_sentences(tokens)    # (B, S, D)

        # Selector scores
        scores = self.selector(H).squeeze(-1)  # (B, S)

        # Mask invalid sentences
        mask = (sent_lengths > 0).float()  # (B, S)
        scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax selection
        sel_probs = F.softmax(scores, dim=1)  # (B, S)

        # Weighted sentence embedding
        z = (sel_probs.unsqueeze(-1) * H).sum(dim=1)  # (B, D)

        # Classification
        output = self.predictor(z).squeeze(-1)  # (B,)

        return output, sel_probs, H

    def compute_ib_loss(self, output, sel_probs, labels, H, sent_lengths):
        """Information Bottleneck loss.

        L = BCE(y, ŷ) + β * KL(q(Z) || p(Z))

        We use entropy as a proxy: KL(q || uniform) = -H(q) + log(S)
        Since log(S) is constant, we minimize -H(q), i.e. maximize entropy.
        """
        bce = F.binary_cross_entropy_with_logits(output, labels.float())

        # Entropy of selection distribution
        B, S = sel_probs.shape
        entropy = 0.0
        for b in range(B):
            n_valid = min(int(sent_lengths[b].sum().item()), S)
            if n_valid > 0:
                q = sel_probs[b, :n_valid]
                # Clamp for numerical stability
                q = torch.clamp(q, min=1e-10)
                q = q / q.sum()
                entropy += -(q * torch.log(q)).sum()

        entropy = entropy / B
        # Entropy is bounded
        entropy = torch.clamp(entropy, 0.0, 10.0)

        # IB loss: classification + regularization
        # We want to minimize KL = -H(q) + log(S), so we add -entropy
        return bce - self.beta * entropy


# ── Training ───────────────────────────────────────────────────────

def train_ai2tale(model, train_loader, test_loader, config, device):
    """Train AI2TALE model."""
    epochs = config.get('epochs', 10)
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0.0001)
    save_dir = config.get('save_dir', 'experiments/ai2tale')
    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_test_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f'AI2TALE Epoch {epoch+1}/{epochs}')
        for batch in pbar:
            tokens = batch['tokens'].to(device)
            labels = batch['label'].to(device)
            sent_lengths = batch['sent_lengths'].to(device)

            optimizer.zero_grad()
            output, sel_probs, H = model(tokens, sent_lengths)

            loss = model.compute_ib_loss(output, sel_probs, labels, H, sent_lengths)

            # Check for NaN before backward
            if torch.isnan(loss):
                print(f"\n  WARNING: NaN loss at epoch {epoch+1}, skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val * labels.shape[0]
            preds = (torch.sigmoid(output) > 0.5).long()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.shape[0]

            pbar.set_postfix({'loss': f'{loss_val:.4f}', 'acc': f'{total_correct/total_samples:.4f}'})

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_correct / max(total_samples, 1)
        print(f'  Epoch {epoch+1}: loss={avg_loss:.4f}, acc={avg_acc:.4f}')

        # Test accuracy
        test_correct = 0
        test_total = 0
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                tokens = batch['tokens'].to(device)
                labels = batch['label'].to(device)
                sent_lengths = batch['sent_lengths'].to(device)
                output, _, _ = model(tokens, sent_lengths)
                preds = (torch.sigmoid(output) > 0.5).long()
                test_correct += (preds == labels).sum().item()
                test_total += labels.shape[0]
        test_acc = test_correct / max(test_total, 1)
        print(f'  Test acc: {test_acc:.4f}')
        model.train()

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            ckpt_path = os.path.join(save_dir, 'ai2tale_best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, ckpt_path)
            print(f'  Saved best model (acc={test_acc:.4f})')

    print(f'\nAI2TALE training complete. Best test acc: {best_test_acc:.4f}')
    return best_test_acc


# ── Evaluation ─────────────────────────────────────────────────────

def evaluate_ai2tale(model, test_loader, config, device):
    """Evaluate AI2TALE with our 4 metrics."""
    from src.rules import load_keywords, batch_score
    from scipy.stats import spearmanr
    from sklearn.metrics import precision_recall_curve, auc, f1_score

    model.eval()
    keywords_path = config.get('keywords_file', 'phishingdata/kwords_stem.json')
    if not os.path.isabs(keywords_path):
        # Go up 2 levels from src/baselines/ to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        keywords_path = os.path.join(project_root, keywords_path)
    keywords = load_keywords(keywords_path) if os.path.exists(keywords_path) else None

    all_labels = []
    all_auprc_scores = []
    all_f1_preds = []
    all_entropies = []
    all_spearmans = []

    with torch.no_grad():
        for batch in test_loader:
            tokens = batch['tokens'].to(device)
            labels = batch['label']
            sent_lengths = batch['sent_lengths'].to(device)
            raw_sentences = batch['raw_sentences']

            output, sel_probs, H = model(tokens, sent_lengths)

            B = len(labels)
            for b in range(B):
                n_valid = min(int(sent_lengths[b].sum().item()), S if 'S' in dir() else sel_probs.shape[1], len(raw_sentences[b]))
                if n_valid == 0:
                    all_labels.append(int(labels[b].item()))
                    all_auprc_scores.append(0.0)
                    all_f1_preds.append(0.0)
                    continue

                scores = sel_probs[b, :n_valid].cpu().numpy()

                # Normalize
                if scores.sum() > 0:
                    scores = scores / scores.sum()

                all_labels.append(int(labels[b].item()))
                all_auprc_scores.append(float(scores.max()))
                all_f1_preds.append(float(scores.max()))

                # Entropy
                p = scores / (scores.sum() + 1e-10)
                entropy = -(p * np.log(p + 1e-10)).sum()
                all_entropies.append(float(entropy))

                # Spearman for phishing emails
                if labels[b].item() == 1 and n_valid >= 2 and keywords:
                    rule_s = batch_score([raw_sentences[b][:n_valid]], keywords)[0, :n_valid]
                    if len(set(scores)) >= 2 and len(set(rule_s)) >= 2:
                        corr, _ = spearmanr(scores, rule_s)
                        if not np.isnan(corr):
                            all_spearmans.append(float(corr))

    # AUPRC
    if len(set(all_labels)) < 2:
        email_auprc = 0.0
    else:
        precision_arr, recall_arr, _ = precision_recall_curve(all_labels, all_auprc_scores)
        email_auprc = auc(recall_arr, precision_arr)

    # F1
    preds = [1 if s > 0.5 else 0 for s in all_f1_preds]
    email_f1 = f1_score(all_labels, preds, zero_division=0)

    spearman_corr = np.mean(all_spearmans) if all_spearmans else 0.0
    entropy = np.mean(all_entropies) if all_entropies else 0.0

    return {
        'email_auprc': float(email_auprc),
        'email_f1': float(email_f1),
        'spearman_corr': float(spearman_corr),
        'attention_entropy': float(entropy),
    }


# ── Entry Point ────────────────────────────────────────────────────

def run_ai2tale(config=None):
    """Train and evaluate AI2TALE."""
    if config is None:
        import yaml
        with open('experiments/configs/defaults.yaml', 'r') as f:
            config = yaml.safe_load(f)
        config['save_dir'] = 'experiments/ai2tale'

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    from src.dataset import create_dataloaders
    train_loader, test_loader = create_dataloaders(config)

    model = AI2TALEModel(
        vocab_size=len(train_loader.dataset.vocab),
        embed_dim=150,
        hidden_dim=150,
        attention_hidden=64,
        classifier_hidden=[128, 64],
        beta=0.1,
    ).to(device)

    print(f'Training AI2TALE (vocab={len(train_loader.dataset.vocab)})...')
    train_ai2tale(model, train_loader, test_loader, config, device)

    # Load best model and evaluate
    ckpt_path = os.path.join(config['save_dir'], 'ai2tale_best_model.pt')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)

    print('\nEvaluating AI2TALE...')
    results = evaluate_ai2tale(model, test_loader, config, device)

    print('AI2TALE Results:')
    for k, v in results.items():
        print(f'  {k}: {v:.4f}')

    os.makedirs(config['save_dir'], exist_ok=True)
    with open(os.path.join(config['save_dir'], 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == '__main__':
    run_ai2tale()
