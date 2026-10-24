"""BERT-ABMIL: ABMIL with frozen DistilBERT sentence encoder.

Uses a pre-trained DistilBERT model to encode sentences, then applies
the same ABMIL attention pooling and classifier as the standard ABMIL.
The DistilBERT encoder is frozen — only the attention and classifier
layers are trained. This provides a strong baseline to show whether
our improvements come from the ABMIL architecture or just from better
sentence representations.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel


class BertSentenceEncoder(nn.Module):
    """Frozen DistilBERT sentence encoder.

    Takes tokenized sentences, runs through DistilBERT, and extracts
    the [CLS] token representation as the sentence embedding.
    """

    def __init__(self, model_name='distilbert-base-uncased', frozen=True):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.frozen = frozen
        if frozen:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, sentences):
        """
        Args:
            sentences: list of S strings
        Returns:
            embeddings: (S, 768) sentence embeddings from [CLS] token
        """
        max_len = min(max(len(s.split()) for s in sentences) + 4, 512) if sentences else 50
        encoded = self.tokenizer(
            sentences,
            padding='max_length',
            max_length=max_len,
            truncation=True,
            return_tensors='pt',
        )
        encoded = {k: v.to(self.bert.device) for k, v in encoded.items()}
        with torch.set_grad_enabled(not self.frozen):
            outputs = self.bert(**encoded)
        return outputs.last_hidden_state[:, 0, :]  # (S, 768)


class BertABMIL(nn.Module):
    """ABMIL with frozen DistilBERT encoder.

    Same attention + classifier architecture as ABMILBaseline, but
    uses DistilBERT for sentence encoding (768d embeddings).
    """

    def __init__(self, attention_hidden=64, classifier_hidden=None, bert_dim=768):
        super().__init__()
        self.encoder = BertSentenceEncoder(frozen=True)
        self.sent_dim = bert_dim

        # Attention mechanism (same as ABMILBaseline)
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

    def compute_attention(self, h, sent_lengths=None):
        """Compute attention weights (same as ABMILBaseline)."""
        scores = self.attention(h).squeeze(-1)  # (B, S)
        if sent_lengths is not None:
            # sent_lengths may be longer than actual S (e.g., 100 vs actual sentences)
            mask = (sent_lengths[:, :scores.shape[1]] > 0).float().to(scores.device)
            scores = scores.masked_fill(mask == 0, -1e9)
        return F.softmax(scores, dim=1)  # (B, S)

    def forward(self, raw_sentences_list, sent_lengths=None):
        """
        Args:
            raw_sentences_list: list of B lists of strings (variable sentences per email)
            sent_lengths: (B, max_S) valid token counts for masking
        Returns:
            output: (B,) email classification logits
            attn_weights: (B, S) attention weights
            h: (B, S, D) sentence embeddings
        """
        B = len(raw_sentences_list)
        # Determine max sentences in this batch
        max_S = max(len(sents) for sents in raw_sentences_list)

        # Encode each batch of sentences through DistilBERT
        all_h = []
        for b in range(B):
            sents = raw_sentences_list[b] if raw_sentences_list[b] else ["[PAD]"]
            h_b = self.encoder(sents)  # (S_b, 768)
            # Pad to max_S
            if h_b.shape[0] < max_S:
                pad = torch.zeros(max_S - h_b.shape[0], self.sent_dim, device=h_b.device)
                h_b = torch.cat([h_b, pad], dim=0)
            all_h.append(h_b)

        h = torch.stack(all_h)  # (B, max_S, 768)

        attn_weights = self.compute_attention(h, sent_lengths)  # (B, S)
        z = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)  # (B, 768)
        output = self.classifier(z).squeeze(-1)  # (B,)

        return output, attn_weights, h


class EmailTextDataset(Dataset):
    """Dataset that provides raw sentence text for BERT encoding."""

    def __init__(self, mat_path, vocab_path, max_sentences=100, max_tokens=50):
        # Reuse the same loading logic as PhishingEmailDataset
        import pickle
        import numpy as np
        import scipy.io

        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        data = scipy.io.loadmat(mat_path)

        if 'X_tr_non' in data:
            X_non = data['X_tr_non']
            X_non_len = data['X_tr_non_len']
            y_non = data['y_tr_non'].flatten()
            X_vul = data['X_tr_vul']
            X_vul_len = data['X_tr_vul_len']
            y_vul = data['y_tr_vul'].flatten()
            self.X = np.concatenate([X_non, X_vul], axis=0).astype(np.int32)
            self.X_len = np.concatenate([X_non_len, X_vul_len], axis=0)
            self.y = np.concatenate([y_non, y_vul], axis=0).astype(np.int64)
        else:
            self.X = data['X_te'].astype(np.int32)
            self.X_len = data['X_te_len']
            self.y = data['y_te'].flatten().astype(np.int64)

        self.max_sentences = max_sentences
        self.max_tokens = max_tokens

    def _decode_email(self, idx):
        sentences = []
        for sent_idx in range(self.max_sentences):
            tokens = self.X[idx, sent_idx, :]
            lengths = self.X_len[idx, sent_idx, :]
            valid_len = 0
            for j in range(self.max_tokens):
                if lengths[j] > 0 and tokens[j] != 0:
                    valid_len = j + 1
            if valid_len == 0:
                break
            words = []
            for j in range(valid_len):
                tok_id = int(tokens[j])
                if tok_id in self.vocab:
                    words.append(self.vocab[tok_id])
            if words:
                sentences.append(' '.join(words))
        return sentences

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        raw_sentences = self._decode_email(idx)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        # Compute sent_lengths for masking
        n_sents = len(raw_sentences)
        sent_lengths = torch.zeros(self.max_sentences, dtype=torch.long)
        sent_lengths[:n_sents] = 1  # Mark valid sentences
        return {
            'raw_sentences': raw_sentences,
            'label': label,
            'sent_lengths': sent_lengths,
            'idx': idx,
        }


def bert_collate_fn(batch):
    """Collate for BERT dataset — keeps raw sentences as lists."""
    labels = torch.stack([item['label'] for item in batch])
    raw_sentences = [item['raw_sentences'] for item in batch]
    sent_lengths = torch.stack([item['sent_lengths'] for item in batch])
    idxs = [item['idx'] for item in batch]
    return {
        'raw_sentences': raw_sentences,
        'label': labels,
        'sent_lengths': sent_lengths,
        'idx': idxs,
    }


def create_bert_dataloaders(config):
    """Create train/test DataLoaders for BERT-ABMIL."""
    data_dir = config.get('data_dir', 'phishingdata')
    vocab_path = config.get('vocab_file', os.path.join(data_dir, 'ind_word.pickle'))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    vocab_path = os.path.join(project_root, vocab_path) if not os.path.isabs(vocab_path) else vocab_path

    train_mat = os.path.join(project_root, data_dir, 'x_y_tr_m.mat')
    test_mat = os.path.join(project_root, data_dir, 'x_y_te_m.mat')

    max_sentences = config.get('max_sentences', 100)
    max_tokens = config.get('max_tokens_per_sentence', 50)

    train_ds = EmailTextDataset(train_mat, vocab_path, max_sentences, max_tokens)
    test_ds = EmailTextDataset(test_mat, vocab_path, max_sentences, max_tokens)

    batch_size = config.get('batch_size', 8)  # Smaller batch for BERT
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=bert_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, collate_fn=bert_collate_fn)
    return train_loader, test_loader


def train_bert_abmil(model, train_loader, test_loader, config, device):
    """Train BERT-ABMIL (only attention + classifier layers, BERT is frozen)."""
    epochs = config.get('epochs', 10)
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0.0001)
    save_dir = config.get('save_dir', 'experiments/bert_abmil')
    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_test_acc = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f'BERT-ABMIL Epoch {epoch+1}/{epochs}')
        for batch in pbar:
            labels = batch['label'].to(device)
            sent_lengths = batch['sent_lengths'].to(device)
            raw_sentences = batch['raw_sentences']

            optimizer.zero_grad()
            output, attn_weights, h = model(raw_sentences, sent_lengths)
            loss = F.binary_cross_entropy_with_logits(output, labels.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * labels.shape[0]
            preds = (torch.sigmoid(output) > 0.5).long()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.shape[0]

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{total_correct/total_samples:.4f}'})

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        print(f'Epoch {epoch+1}: loss={avg_loss:.4f}, acc={avg_acc:.4f}')

        # Test accuracy
        test_correct = 0
        test_total = 0
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                labels = batch['label'].to(device)
                sent_lengths = batch['sent_lengths'].to(device)
                output, _, _ = model(batch['raw_sentences'], sent_lengths)
                preds = (torch.sigmoid(output) > 0.5).long()
                test_correct += (preds == labels).sum().item()
                test_total += labels.shape[0]
        test_acc = test_correct / test_total
        print(f'  Test acc: {test_acc:.4f}')
        model.train()

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            ckpt_path = os.path.join(save_dir, 'bert_abmil_best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, ckpt_path)
            print(f'  Saved best model (acc={test_acc:.4f})')

    print(f'\nBERT-ABMIL training complete. Best test acc: {best_test_acc:.4f}')


def run_bert_abmil(config=None):
    """Entry point: train and evaluate BERT-ABMIL."""
    if config is None:
        import yaml
        with open('experiments/configs/defaults.yaml', 'r') as f:
            config = yaml.safe_load(f)
        config['save_dir'] = 'experiments/bert_abmil'

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_loader, test_loader = create_bert_dataloaders(config)

    model = BertABMIL(
        attention_hidden=config.get('attention_hidden', 64),
        classifier_hidden=config.get('classifier_hidden', [128, 64]),
    ).to(device)

    train_bert_abmil(model, train_loader, test_loader, config, device)


if __name__ == '__main__':
    run_bert_abmil()
