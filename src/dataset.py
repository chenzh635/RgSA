"""Dataset loading and preprocessing for phishing email data."""

import os
import pickle

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader


class PhishingEmailDataset(Dataset):
    """Loads AI2TALE-format phishing email data.

    Each email is stored as token IDs (max_sentences x max_tokens_per_sentence).
    We decode tokens to text using the vocab pickle and return both.
    """

    def __init__(self, mat_path, vocab_path, max_sentences=100, max_tokens=50):
        # Load vocab: {token_id: word_text}
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        # Load .mat data
        data = scipy.io.loadmat(mat_path)

        # AI2TALE format: separate non-vul and vul arrays
        if 'X_tr_non' in data:
            X_non = data['X_tr_non']  # (N_non, max_sentences, max_tokens)
            X_non_len = data['X_tr_non_len']
            y_non = data['y_tr_non'].flatten()  # all 0

            X_vul = data['X_tr_vul']
            X_vul_len = data['X_tr_vul_len']
            y_vul = data['y_tr_vul'].flatten()  # all 1

            # Combine: benign first, then phishing
            self.X = np.concatenate([X_non, X_vul], axis=0).astype(np.int32)
            self.X_len = np.concatenate([X_non_len, X_vul_len], axis=0)
            self.y = np.concatenate([y_non, y_vul], axis=0).astype(np.int64)
        else:
            # Single array format (test set)
            self.X = data['X_te'].astype(np.int32)
            self.X_len = data['X_te_len']
            self.y = data['y_te'].flatten().astype(np.int64)

        self.max_sentences = max_sentences
        self.max_tokens = max_tokens
        self.vocab = self.vocab  # dict mapping token_id -> word text

    def __len__(self):
        return len(self.X)

    def _decode_email(self, idx):
        """Decode an email from token IDs to list of sentences."""
        sentences = []
        for sent_idx in range(self.max_sentences):
            tokens = self.X[idx, sent_idx, :]
            lengths = self.X_len[idx, sent_idx, :]
            # Determine valid token count
            valid_len = 0
            for j in range(self.max_tokens):
                if lengths[j] > 0 and tokens[j] != 0:
                    valid_len = j + 1
            if valid_len == 0:
                break  # End of email
            words = []
            for j in range(valid_len):
                tok_id = int(tokens[j])
                if tok_id in self.vocab:
                    words.append(self.vocab[tok_id])
            if words:
                sentences.append(' '.join(words))
        return sentences

    def __getitem__(self, idx):
        # Raw sentences (decoded text)
        raw_sentences = self._decode_email(idx)

        # Token tensor: (max_sentences, max_tokens)
        tokens = torch.tensor(self.X[idx], dtype=torch.long)

        # Label
        label = torch.tensor(self.y[idx], dtype=torch.long)

        # Length mask: (max_sentences,)
        sent_lengths = []
        for s in range(self.max_sentences):
            lengths = self.X_len[idx, s, :]
            count = sum(1 for j in range(self.max_tokens) if lengths[j] > 0)
            sent_lengths.append(count)
        sent_lengths = torch.tensor(sent_lengths, dtype=torch.long)

        return {
            'tokens': tokens,          # (max_sentences, max_tokens) - for embedding
            'label': label,            # scalar
            'raw_sentences': raw_sentences,  # list of strings
            'sent_lengths': sent_lengths,    # (max_sentences,) - for masking
            'idx': idx,
        }


def create_dataloaders(config):
    """Create train and test DataLoaders."""
    data_dir = config.get('data_dir', 'phishingdata')
    vocab_path = config.get('vocab_file', os.path.join(data_dir, 'ind_word.pickle'))

    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vocab_path = os.path.join(project_root, vocab_path) if not os.path.isabs(vocab_path) else vocab_path

    train_mat = os.path.join(project_root, data_dir, 'x_y_tr_m.mat')
    test_mat = os.path.join(project_root, data_dir, 'x_y_te_m.mat')

    train_dataset = PhishingEmailDataset(
        train_mat, vocab_path,
        max_sentences=config.get('max_sentences', 100),
        max_tokens=config.get('max_tokens_per_sentence', 50),
    )

    test_dataset = PhishingEmailDataset(
        test_mat, vocab_path,
        max_sentences=config.get('max_sentences', 100),
        max_tokens=config.get('max_tokens_per_sentence', 50),
    )

    def collate_fn(batch):
        """Custom collate to handle variable-length raw_sentences."""
        tokens = torch.stack([item['tokens'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        raw_sentences = [item['raw_sentences'] for item in batch]
        sent_lengths = torch.stack([item['sent_lengths'] for item in batch])
        idxs = [item['idx'] for item in batch]
        return {
            'tokens': tokens,
            'label': labels,
            'raw_sentences': raw_sentences,
            'sent_lengths': sent_lengths,
            'idx': idxs,
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        collate_fn=collate_fn,
    )

    return train_loader, test_loader
