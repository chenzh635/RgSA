"""Snorkel weak supervision baseline.

Uses a Snorkel-style LabelModel to combine multiple labeling functions (LFs)
into probabilistic sentence-level pseudo-labels, then trains a lightweight
sentence classifier. This represents the "proper" weak supervision approach
compared to our simpler rules-only baseline (which uses manually weighted rules).

The key difference from Rules-only:
- Rules-only: manually assigned weights (0.3, 0.2, etc.), direct thresholding
- Snorkel: learns LF accuracies from their agreement patterns, produces calibrated
  probabilities, then trains a downstream model on those probabilities
"""

import json
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ── Labeling Functions ─────────────────────────────────────────────

def lf_scarcity(sentence):
    """LF: Scarcity/limited availability language."""
    patterns = ['limited', 'hurry', 'expire', 'deadline', 'last chance',
                'act now', 'offer ends', 'remaining', 'few left']
    s = sentence.lower()
    return 1 if any(p in s for p in patterns) else 0


def lf_authority(sentence):
    """LF: Claims of authority or official source."""
    patterns = ['official', 'verified', 'authorized', 'management', 'admin',
                'security team', 'bank', 'fraud', 'legal', 'compliance',
                'regulation', 'authority']
    s = sentence.lower()
    return 1 if any(p in s for p in patterns) else 0


def lf_urgency(sentence):
    """LF: Urgency/time pressure language."""
    patterns = ['immediately', 'right away', 'at once', 'without delay',
                'asap', '24 hour', '48 hour', 'verify now', 'confirm now',
                'take action']
    s = sentence.lower()
    return 1 if any(p in s for p in patterns) else 0


def lf_social_proof(sentence):
    """LF: Social proof / popularity claims."""
    patterns = ['millions of', 'users worldwide', 'everyone', 'popular',
                'trending', 'join', 'community', 'trusted by']
    s = sentence.lower()
    return 1 if any(p in s for p in patterns) else 0


def lf_reciprocity(sentence):
    """LF: Reciprocity / gift / reward language."""
    patterns = ['free', 'gift', 'reward', 'bonus', 'complimentary',
                'you deserve', 'exclusive', 'special offer', 'prize',
                'winner', 'congratulation']
    s = sentence.lower()
    return 1 if any(p in s for p in patterns) else 0


def lf_uri(sentence):
    """LF: Contains a URL or link reference."""
    return 1 if re.search(r'https?://|www\.|click\s+here|link\s+below', sentence.lower()) else 0


def lf_generic_greeting(sentence):
    """LF: Generic/impersonal greeting."""
    return 1 if re.search(r'dear\s+(customer|user|member|beneficiary|sir|madam)', sentence.lower()) else 0


def lf_action_required(sentence):
    """LF: Call-to-action patterns."""
    patterns = ['click here', 'update your', 'confirm your', 'verify your',
                'renew', 'sign in', 'log in', 'login', 'reset your password',
                'suspended', 'locked', 'compromised']
    s = sentence.lower()
    return 1 if any(p in s for p in patterns) else 0


def lf_security_concern(sentence):
    """LF: Security-related language."""
    patterns = ['security', 'unusual activity', 'suspicious', 'unauthorized',
                'fraud', 'phishing', 'compromised', 'breach', 'alert',
                'protect', 'safe']
    s = sentence.lower()
    return 1 if any(p in s for p in patterns) else 0


LABELING_FUNCTIONS = [
    lf_scarcity, lf_authority, lf_urgency, lf_social_proof,
    lf_reciprocity, lf_uri, lf_generic_greeting, lf_action_required,
    lf_security_concern,
]


# ── Label Model (Snorkel-style) ───────────────────────────────────

class SnorkelLabelModel:
    """Simplified Snorkel-style generative label model.

    Estimates LF accuracies from agreement/disagreement patterns,
    then combines LFs into calibrated probabilistic labels.
    This captures the core Snorkel insight: LFs that agree with each
    other more often are likely more accurate and should get higher weight.
    """

    def __init__(self):
        self.lf_weights = None

    def fit(self, lf_matrix):
        """Estimate LF weights from pairwise agreement patterns.

        Args:
            lf_matrix: (N, n_lfs) array of LF votes (0 or 1, -1 = abstain)
        """
        n_samples, n_lfs = lf_matrix.shape

        # For binary LFs (0/1), estimate accuracy via agreement with majority
        lf_accuracies = np.zeros(n_lfs)
        for i in range(n_lfs):
            # Fraction of times this LF agrees with the majority of other LFs
            other_mean = (lf_matrix.sum(axis=1) - lf_matrix[:, i]) / (n_lfs - 1)
            agreement = np.mean((lf_matrix[:, i] > 0.5) == (other_mean > 0.5))
            lf_accuracies[i] = max(agreement, 0.5)  # At least random

        # Convert to log-odds weights (Snorkel-style)
        self.lf_weights = np.log(lf_accuracies / (1 - lf_accuracies + 1e-10) + 1e-10)
        # Normalize to positive
        self.lf_weights = self.lf_weights - self.lf_weights.min() + 0.01
        self.lf_weights = self.lf_weights / self.lf_weights.sum()

    def predict_proba(self, lf_matrix):
        """Weighted average of LF votes.

        Returns:
            proba: (N,) probability of being phishing-related
        """
        return np.average(lf_matrix, axis=1, weights=self.lf_weights)


# ── Sentence Classifier ───────────────────────────────────────────

class SentenceClassifier(nn.Module):
    """CNN sentence classifier trained on Snorkel pseudo-labels.

    Same encoder architecture as the main ABMIL model for fair comparison.
    """

    def __init__(self, vocab_size=311774, embed_dim=150, hidden_dim=150):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, hidden_dim, 3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, tokens):
        """
        Args:
            tokens: (N, T) token IDs
        Returns:
            logits: (N,)
        """
        x = self.embedding(tokens)       # (N, T, E)
        x = x.transpose(1, 2)            # (N, E, T)
        x = self.conv(x)                 # (N, H, T)
        x = F.relu(x)
        x = self.pool(x).squeeze(-1)     # (N, H)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)


# ── Training ──────────────────────────────────────────────────────

def train_snorkel(train_loader, config, device):
    """Train the Snorkel baseline.

    Returns:
        model: trained SentenceClassifier
    """
    save_dir = config.get('save_dir', 'experiments/snorkel')
    os.makedirs(save_dir, exist_ok=True)

    dataset = train_loader.dataset
    vocab = dataset.vocab

    # Step 1: Extract all sentences with their token IDs and email labels
    print("Extracting training sentences...")
    all_tokens = []  # list of (sent_idx, token_tensor)
    all_texts = []
    all_email_labels = []

    for idx in range(len(dataset)):
        item = dataset[idx]
        email_label = item['label'].item()
        raw_sentences = item['raw_sentences']
        tokens = item['tokens']

        for sent_idx, sent_text in enumerate(raw_sentences):
            all_tokens.append(tokens[sent_idx].clone())
            all_texts.append(sent_text)
            all_email_labels.append(email_label)

    # Step 2: Apply LFs
    print(f"Applying {len(LABELING_FUNCTIONS)} LFs to {len(all_texts)} sentences...")
    n_samples = len(all_texts)
    n_lfs = len(LABELING_FUNCTIONS)
    lf_matrix = np.zeros((n_samples, n_lfs), dtype=np.int32)
    for i, text in enumerate(all_texts):
        for j, lf in enumerate(LABELING_FUNCTIONS):
            lf_matrix[i, j] = lf(text)

    lf_coverage = (lf_matrix.sum(axis=0) / n_samples * 100)
    print(f"  LF coverage: {', '.join(f'{c:.1f}%' for c in lf_coverage)}")

    # Step 3: Train LabelModel
    print("Training LabelModel...")
    label_model = SnorkelLabelModel()
    label_model.fit(lf_matrix)
    pseudo_probs = label_model.predict_proba(lf_matrix)

    # Zero out sentences from benign emails (we know those emails have no phishing sentences)
    email_label_arr = np.array(all_email_labels)
    pseudo_probs[email_label_arr == 0] = 0.0

    # Keep continuous pseudo-probs for sentence-level regression
    # This preserves the granularity of LF agreement patterns — sentences
    # with more LF hits and higher weighted agreement get higher scores.
    n_positive = (pseudo_probs > 0.3).sum()
    print(f"  Sentences with pseudo_prob > 0.3: {n_positive}/{n_samples} ({n_positive/n_samples*100:.1f}%)")
    print(f"  Pseudo-prob stats: min={pseudo_probs.min():.4f}, mean={pseudo_probs.mean():.4f}, max={pseudo_probs.max():.4f}")
    print(f"  LF weights: {label_model.lf_weights.round(3)}")

    # Step 4: Train sentence classifier with regression on continuous pseudo-probs
    print("Training sentence classifier (regression on pseudo-probs)...")
    model = SentenceClassifier(vocab_size=len(vocab)).to(device)

    tokens_all = torch.stack(all_tokens).to(device)
    # Clamp to [0, 1] and use as soft targets for BCE (sigmoid output vs pseudo-prob)
    targets_all = torch.tensor(np.clip(pseudo_probs, 0.01, 0.99), dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batch_size = min(config.get('batch_size', 256), 512)
    epochs = config.get('epochs', 10)

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples)
        total_loss = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = perm[start:end]
            batch_tokens = tokens_all[batch_idx]
            batch_targets = targets_all[batch_idx]

            optimizer.zero_grad()
            output = model(batch_tokens)
            # MSE regression: model output (raw logit) should match pseudo-prob target
            # Use sigmoid to map logit to [0, 1] then MSE
            output_prob = torch.sigmoid(output)
            loss = F.mse_loss(output_prob, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * (end - start)

        avg_loss = total_loss / n_samples
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'lf_weights': label_model.lf_weights,
                'epoch': epoch,
            }, os.path.join(save_dir, 'snorkel_best_model.pt'))

    print(f"  Best loss: {best_loss:.4f}")
    return model


# ── Evaluation ────────────────────────────────────────────────────

def evaluate_snorkel(model, test_loader, config, device):
    """Evaluate Snorkel baseline using the same metrics as the main models.

    The sentence classifier's sigmoid output is treated as a soft "attention" score,
    normalized per-email to sum to 1, so it can be compared with ABMIL attention weights.
    """
    from src.rules import batch_score, load_keywords

    model.eval()
    keywords_path = config.get('keywords_file', 'phishingdata/kwords_stem.json')
    if not os.path.isabs(keywords_path):
        # Go up 2 levels from src/baselines/ to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        keywords_path = os.path.join(project_root, keywords_path)
    keywords = load_keywords(keywords_path) if os.path.exists(keywords_path) else None

    # Collect per-email results — use pre-tokenized tokens from the batch,
    # consistent with training.
    email_labels_all = []
    email_sentence_scores = []  # normalized scores per email (like attention weights)
    email_raw_sentences = []
    email_sent_lengths = []
    email_raw_logits = []  # un-normalized logits for F1/AUPRC

    with torch.no_grad():
        for batch in test_loader:
            labels = batch['label'].numpy()
            sent_lengths = batch['sent_lengths']
            raw_sentences = batch['raw_sentences']
            tokens = batch['tokens'].to(device)  # (B, S_max, T_max)

            for b in range(len(labels)):
                sentences = raw_sentences[b]
                if not sentences:
                    email_labels_all.append(int(labels[b]))
                    email_sentence_scores.append(np.array([]))
                    email_raw_sentences.append([])
                    email_sent_lengths.append(sent_lengths[b])
                    email_raw_logits.append(np.array([]))
                    continue

                # Use pre-tokenized token IDs directly (same as training)
                n_sent = len(sentences)
                sent_tokens = tokens[b, :n_sent, :]  # (n_sent, T_max)
                logits = model(sent_tokens).cpu().numpy()  # (n_sent,)

                # Normalize to attention-like weights (softmax)
                logits_stable = logits - logits.max()
                scores = np.exp(logits_stable)
                scores = scores / (scores.sum() + 1e-10)

                email_labels_all.append(int(labels[b]))
                email_sentence_scores.append(scores)
                email_raw_sentences.append(sentences)
                email_sent_lengths.append(sent_lengths[b])
                email_raw_logits.append(logits)

    # Now compute the new 4 metrics using these collected results
    all_labels = email_labels_all
    all_auprc_scores = []  # email-level prediction scores (max sentence score)
    all_f1_preds = []      # for F1 at threshold
    all_entropies = []
    all_spearmans = []

    for i, label in enumerate(all_labels):
        scores = email_sentence_scores[i]
        sentences = email_raw_sentences[i]
        sl = email_sent_lengths[i]

        if len(scores) == 0:
            all_auprc_scores.append(0.0)
            all_f1_preds.append(0.0)
            continue

        n_valid = int((sl > 0).sum().item())
        scores_valid = scores[:n_valid]

        # Email-level prediction: max sentence score
        all_auprc_scores.append(float(scores_valid.max()))
        all_f1_preds.append(float(scores_valid.max()))

        # Entropy
        p = scores_valid / (scores_valid.sum() + 1e-10)
        entropy = -(p * np.log(p + 1e-10)).sum()
        all_entropies.append(float(entropy))

        # Spearman for phishing emails
        if label == 1 and n_valid >= 2 and len(set(scores_valid)) >= 2 and keywords:
            from scipy.stats import spearmanr
            rule_s = batch_score([sentences], keywords)[0, :n_valid]
            if len(set(rule_s)) >= 2:
                corr, _ = spearmanr(scores_valid, rule_s)
                if not np.isnan(corr):
                    all_spearmans.append(float(corr))

    # Email-Level AUPRC
    if len(set(all_labels)) < 2:
        email_auprc = 0.0
    else:
        from sklearn.metrics import precision_recall_curve, auc
        precision_arr, recall_arr, _ = precision_recall_curve(all_labels, all_auprc_scores)
        email_auprc = auc(recall_arr, precision_arr)

    # Email-Level F1
    from sklearn.metrics import f1_score
    preds = [1 if s > 0.5 else 0 for s in all_f1_preds]
    email_f1 = f1_score(all_labels, preds, zero_division=0)

    # Spearman
    spearman_corr = np.mean(all_spearmans) if all_spearmans else 0.0

    # Entropy
    entropy = np.mean(all_entropies) if all_entropies else 0.0

    return {
        'email_auprc': float(email_auprc),
        'email_f1': float(email_f1),
        'spearman_corr': float(spearman_corr),
        'attention_entropy': float(entropy),
    }


def run_snorkel(config=None):
    """Entry point: train and evaluate Snorkel baseline."""
    if config is None:
        import yaml
        with open('experiments/configs/defaults.yaml', 'r') as f:
            config = yaml.safe_load(f)
        config['save_dir'] = 'experiments/snorkel'

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    from src.dataset import create_dataloaders
    train_loader, test_loader = create_dataloaders(config)

    # Load vocab reverse mapping for evaluation
    dataset = train_loader.dataset
    vocab_rev = {tid: word for tid, word in dataset.vocab.items()}
    config['_vocab_rev'] = vocab_rev

    model = train_snorkel(train_loader, config, device)
    results = evaluate_snorkel(model, test_loader, config, device)

    print("\nSnorkel Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    # Save results
    os.makedirs(config['save_dir'], exist_ok=True)
    import json
    with open(os.path.join(config['save_dir'], 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == '__main__':
    run_snorkel()
