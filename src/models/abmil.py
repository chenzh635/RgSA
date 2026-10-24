"""Attention-Based MIL (ABMIL) with optional sparsity constraint."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SentenceEncoder(nn.Module):
    """Simple sentence encoder using token embeddings + CNN.

    Uses the same approach as AI2TALE: token embedding -> Conv1d -> max pooling.
    This is lightweight and doesn't require loading a large pretrained model.
    """

    def __init__(self, vocab_size, embed_dim=150, hidden_dim=150, kernel_size=3, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        """
        Args:
            tokens: (max_sentences, max_tokens) token IDs
        Returns:
            embeddings: (max_sentences, hidden_dim) sentence embeddings
        """
        x = self.embedding(tokens)  # (S, T, E)
        x = x.transpose(1, 2)       # (S, E, T)
        x = self.conv(x)            # (S, H, T)
        x = F.relu(x)
        x = self.pool(x).squeeze(-1)  # (S, H)
        x = self.dropout(x)
        return x


class ABMILBaseline(nn.Module):
    """Standard Attention-Based MIL for email classification.

    Architecture:
        Sentence Encoder -> Attention -> Weighted Pooling -> Classifier
    Attention weights serve as the localization signal.
    """

    def __init__(self, vocab_size=311774, embed_dim=150, hidden_dim=150,
                 attention_hidden=64, classifier_hidden=None):
        super().__init__()
        if classifier_hidden is None:
            classifier_hidden = [128, 64]

        self.encoder = SentenceEncoder(vocab_size, embed_dim, hidden_dim)
        self.sent_dim = hidden_dim

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.sent_dim, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1),
        )

        # Classifier
        layers = []
        prev_dim = self.sent_dim
        for h in classifier_hidden:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(0.3)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def compute_attention(self, h, sent_lengths=None):
        """Compute attention weights.

        Args:
            h: (B, S, D) sentence embeddings
            sent_lengths: (B, S) valid token counts per sentence (for masking)
        Returns:
            attn_weights: (B, S) normalized attention weights
        """
        scores = self.attention(h).squeeze(-1)  # (B, S)

        # Mask padding sentences
        if sent_lengths is not None:
            mask = (sent_lengths > 0).float()
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=1)  # (B, S)
        return attn_weights

    def forward(self, tokens, sent_lengths=None):
        """
        Args:
            tokens: (B, S, T) token IDs
            sent_lengths: (B, S) valid token counts
        Returns:
            output: email classification logit
            attn_weights: (B, S) attention weights (localization signal)
            h: (B, S, D) sentence embeddings
        """
        B, S, T = tokens.shape

        # Encode sentences: apply encoder per sentence
        h = self.encoder(tokens.view(B * S, T))  # (B*S, D)
        h = h.view(B, S, -1)  # (B, S, D)

        # Attention pooling
        attn_weights = self.compute_attention(h, sent_lengths)  # (B, S)
        z = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)  # (B, D)

        # Classification
        output = self.classifier(z).squeeze(-1)  # (B,)

        return output, attn_weights, h


class SparseABMIL(nn.Module):
    """Sparse Attention MIL with temperature annealing and optional L1 regularization.

    Same architecture as ABMILBaseline but:
    1. Uses temperature-controlled softmax for sparse attention
    2. Returns attention scores (pre-softmax) for L1 regularization
    """

    def __init__(self, vocab_size=311774, embed_dim=150, hidden_dim=150,
                 attention_hidden=64, classifier_hidden=None, use_sparse=True):
        super().__init__()
        if classifier_hidden is None:
            classifier_hidden = [128, 64]

        self.encoder = SentenceEncoder(vocab_size, embed_dim, hidden_dim)
        self.sent_dim = hidden_dim
        self.use_sparse = use_sparse

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.sent_dim, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1),
        )

        # Classifier
        layers = []
        prev_dim = self.sent_dim
        for h in classifier_hidden:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(0.3)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def compute_sparse_attention(self, scores, sent_lengths=None, temperature=1.0):
        """Compute sparse attention with temperature control.

        Args:
            scores: (B, S) raw attention scores (pre-softmax)
            sent_lengths: (B, S) valid token counts (for masking)
            temperature: float, controls sparsity (lower = sparser)
        Returns:
            attn_weights: (B, S) normalized attention weights
        """
        # Apply temperature scaling
        if self.use_sparse and temperature > 0:
            scores = scores / temperature

        # Mask padding sentences
        if sent_lengths is not None:
            mask = (sent_lengths > 0).float()
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=1)  # (B, S)
        return attn_weights

    def forward(self, tokens, sent_lengths=None, temperature=1.0):
        """
        Args:
            tokens: (B, S, T) token IDs
            sent_lengths: (B, S) valid token counts
            temperature: float for sparse softmax
        Returns:
            output: email classification logit
            attn_weights: (B, S) normalized attention weights
            attn_scores: (B, S) raw attention scores (for L1 regularization)
            h: (B, S, D) sentence embeddings
        """
        B, S, T = tokens.shape

        # Encode sentences
        h = self.encoder(tokens.view(B * S, T))  # (B*S, D)
        h = h.view(B, S, -1)  # (B, S, D)

        # Raw attention scores
        attn_scores = self.attention(h).squeeze(-1)  # (B, S)

        # Sparse attention
        attn_weights = self.compute_sparse_attention(attn_scores, sent_lengths, temperature)

        # Attention-weighted pooling
        z = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)  # (B, D)

        # Classification
        output = self.classifier(z).squeeze(-1)  # (B,)

        return output, attn_weights, attn_scores, h
