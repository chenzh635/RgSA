"""Faithful PyTorch port of the AI2TALE reference implementation.

Review M1. The previous port (ai2tale_pytorch.AI2TALEModel) diverged from the
authors' released code in three load-bearing ways and collapsed (F1 0.33,
GT P@1 below Random):
  1. objective was BCE - 0.1*H(selection) — maximizing selection entropy;
  2. a softmax over a 2-layer tanh attention MLP replaced the Concrete/Gumbel
     selector with hard top-k test selection;
  3. valid-sentence counting summed token counts.

This module mirrors ai2tale_src/ai2tale.py line by line:
  - sentence encoder: Embedding(vocab, 150) -> Dropout .2 -> Conv1D(k=3,
    valid, 250 filters) -> ReLU -> global max pool, per sentence;
  - selector: Dropout .2 -> Dense 250->100 ReLU -> Dropout .2 ->
    Dense 100->100 ReLU -> Dense 100->1 sigmoid; BinaryConcrete mask in
    training, discrete top-k mask at test;
  - predictor: sum of masked encodings -> Dense 250 ReLU -> Dense 2 softmax;
  - two optimizer steps per batch: predictor+encoder only (selector frozen,
    fair-Bernoulli random mask), then the full model with the encoding-energy
    compression regularizer (paper Eqs. 8-9).

Harness interface matches the old port: forward(tokens, sent_lengths) returns
(binary_logit, sel_probs, H); sel_probs are the selector probabilities masked
to valid sentences and normalized (the ranking the hard top-k selection is
drawn from).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Keras initializer defaults (the reference relies on them; PyTorch defaults
# differ materially — Embedding N(0,1) vs Keras U(-.05,.05)).
def keras_embedding_init(t):
    nn.init.uniform_(t, -0.05, 0.05)


def keras_glorot(t, fan_in, fan_out):
    r = math.sqrt(6.0 / (fan_in + fan_out))
    nn.init.uniform_(t, -r, r)


def _gumbel(shape, device, eps=1e-9):
    u = torch.rand(shape, device=device).clamp_min(eps)
    return -torch.log(-torch.log(u))


class SentenceEncoder(nn.Module):
    """Embeddings class in the reference: per-sentence CNN, 250-d output."""

    def __init__(self, vocab_size, embed_dim=150, out_dim=250):
        super().__init__()
        # Reference Keras Embedding has no mask_zero / fixed padding weight;
        # index 0 is a learned vector. Keep that for fidelity.
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        keras_embedding_init(self.embedding.weight)
        self.dropout = nn.Dropout(0.2)
        self.conv = nn.Conv1d(embed_dim, out_dim, kernel_size=3, padding=0)
        # Keras Conv1D GlorotUniform; fans include the receptive field
        keras_glorot(self.conv.weight, 3 * embed_dim, 3 * out_dim)
        nn.init.zeros_(self.conv.bias)

    def forward(self, tokens):
        # tokens: (B, S, T) -> (B, S, 250)
        B, S, T = tokens.shape
        x = self.embedding(tokens).view(B * S, T, -1)
        x = self.dropout(x).transpose(1, 2)      # (B*S, E, T)
        x = F.relu(self.conv(x))
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        return x.view(B, S, -1)


def _keras_dense(linear):
    keras_glorot(linear.weight, linear.in_features, linear.out_features)
    nn.init.zeros_(linear.bias)


class Selector(nn.Module):
    def __init__(self, in_dim=250, hidden=100):
        super().__init__()
        self.drop_1 = nn.Dropout(0.2)
        self.dense_1 = nn.Linear(in_dim, hidden)
        self.drop_2 = nn.Dropout(0.2)
        self.dense_2 = nn.Linear(hidden, hidden)
        self.dense_logits = nn.Linear(hidden, 1)
        for d in (self.dense_1, self.dense_2, self.dense_logits):
            _keras_dense(d)

    def forward(self, h):
        # p (called logits_T in the reference) in (0,1).
        x = self.drop_1(h)
        x = F.relu(self.dense_1(x))
        x = self.drop_2(x)
        x = F.relu(self.dense_2(x))
        return torch.sigmoid(self.dense_logits(x)).squeeze(-1)


class Predictor(nn.Module):
    def __init__(self, in_dim=250, hidden=250):
        super().__init__()
        self.dense = nn.Linear(in_dim, hidden)
        self.pred = nn.Linear(hidden, 2)
        _keras_dense(self.dense)
        _keras_dense(self.pred)

    def forward(self, h, mask):
        z = (mask.unsqueeze(-1) * h).sum(dim=1)
        return self.pred(F.relu(self.dense(z)))   # logits (B, 2)


class AI2TALEFaithful(nn.Module):
    def __init__(self, vocab_size=311774, k=1, tau=0.5):
        super().__init__()
        self.k = k
        self.tau = tau
        self.encoder = SentenceEncoder(vocab_size)
        self.selector = Selector()
        self.predictor = Predictor()

    # ----------------------------------------------------------- masks
    def concrete_mask(self, p):
        """Reference SampleConcrete training branch: their BinaryConcrete
        parameterization with p (sigmoid output) and 1-p over all slen
        positions."""
        g_a = _gumbel(p.shape, p.device)
        g_b = _gumbel(p.shape, p.device)
        a = torch.exp((g_a + p) / self.tau)
        b = torch.exp((g_b + (1.0 - p)) / self.tau)
        return a / (a + b)

    def hard_mask(self, p):
        """Discrete top-k over all slen=100 positions (reference test branch:
        no padding mask — the selector is trained to assign low p to padded
        positions)."""
        idx = p.topk(self.k, dim=1).indices
        m = torch.zeros_like(p)
        m.scatter_(1, idx, 1.0)
        return m

    def random_soft_mask(self, shape, device):
        """Phase-A mask: fair Bernoulli draw (selector ignored) over all slen
        positions, relaxed at temperature 0.5, matching tfp
        RelaxedBernoulli in the reference."""
        r = (torch.rand(shape, device=device) > 0.5).float().detach()
        g1, g2 = _gumbel(r.shape, device), _gumbel(r.shape, device)
        return torch.sigmoid((r + g1 - g2) / 0.5)

    # ----------------------------------------------------------- forward
    def forward(self, tokens, sent_lengths):
        """Eval-time forward: hard top-k selection, same 3-tuple interface as
        the previous port: (binary_logit, sel_probs, H)."""
        valid = sent_lengths > 0
        h = self.encoder(tokens)
        p = self.selector(h)
        hard = self.hard_mask(p)
        logits2 = self.predictor(h, hard)

        # normalized selector probabilities over valid sentences (ranking)
        p_valid = p * valid.float()
        sel_probs = p_valid / p_valid.sum(dim=1, keepdim=True).clamp_min(1e-9)

        binary_logit = logits2[:, 1] - logits2[:, 0]
        return binary_logit, sel_probs, h


# ---------------------------------------------------------------- training
def compression_regularizer(h, p, sigma=0.3, slen=100):
    """Reference loss_re (paper Eqs. 8-9): selector probability weighted by
    squared encoding energy, plus the sigma log-determinant term."""
    x_embed = h.mean(dim=-1).square()               # (B, S), reference order
    sigma_b = 1.0 / (2.0 * sigma ** 2)
    loss_re = (sigma_b * p * x_embed).sum(dim=1)
    loss_re = loss_re + slen * (np.log(sigma) + 0.5 * sigma ** 2)
    return loss_re


def train_faithful(model, train_loader, val_loader, config, device):
    """Two-phase training. Checkpoint selected on validation accuracy from the
    training split (M10); test set stays invisible during training."""
    import os
    epochs = config.get("epochs", 10)
    lr = config.get("lr", 0.001)
    sigma = config.get("sig", 0.3)
    lam = config.get("lam", 0.1)
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    selector_params = set(model.selector.parameters())

    for epoch in range(epochs):
        model.train()
        n_seen, n_correct, loss_sum = 0, 0, 0.0

        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            labels = batch["label"].to(device)

            # ---- phase 1: encoder + predictor only, selector frozen ----
            for par in selector_params:
                par.requires_grad_(False)
            h = model.encoder(tokens)
            p = model.selector(h)   # run but ignore, as the reference does
            m_rand = model.random_soft_mask(p.shape, device)
            logits2 = model.predictor(h, m_rand)
            loss_a = ce(logits2, labels)
            optimizer.zero_grad()
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # ---- phase 2: full model, Concrete mask + loss_re ----
            for par in selector_params:
                par.requires_grad_(True)
            h = model.encoder(tokens)
            p = model.selector(h)
            m = model.concrete_mask(p)
            logits2 = model.predictor(h, m)
            loss = ce(logits2, labels)
            loss_re = compression_regularizer(h, p, sigma=sigma)
            total = loss + lam * loss_re.mean()
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n_seen += labels.shape[0]
            loss_sum += loss.item() * labels.shape[0]
            n_correct += (logits2.argmax(dim=1) == labels).sum().item()

        train_acc = n_correct / n_seen

        # ---- validation (split from training set; M10) ----
        model.eval()
        v_correct, v_seen = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch["tokens"].to(device)
                labels = batch["label"].to(device)
                sent_lengths = batch["sent_lengths"].to(device)
                binary_logit, _, _ = model(tokens, sent_lengths)
                preds = (binary_logit > 0).long()
                v_correct += (preds == labels).sum().item()
                v_seen += labels.shape[0]
        val_acc = v_correct / v_seen

        print(f"epoch {epoch}: train_acc={train_acc:.4f} "
              f"loss={loss_sum/n_seen:.4f} val_acc={val_acc:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_acc": val_acc},
                       os.path.join(save_dir, "ai2tale_best_model.pt"))

    print(f"done. best val acc: {best_val_acc:.4f}", flush=True)
    return best_val_acc
