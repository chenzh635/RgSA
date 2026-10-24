"""M1: train the faithful AI2TALE port over seeds 123/456/789.

Uses the fixed validation split (experiments/configs/val_split_indices.json,
M10); checkpoint is selected on validation accuracy. Author default
hyperparameters: batch 128, lr 1e-3, epochs 10, sigma 0.3, lam 0.1,
tau 0.5, k 1.

Usage:
    python experiments/scripts/m1_run_faithful.py --seeds 123
    python experiments/scripts/m1_run_faithful.py --seeds 123,456,789
MPS -> run with sandbox disabled.
"""
import argparse
import json
import os
import sys
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.baselines.ai2tale_faithful import AI2TALEFaithful, train_faithful
from src.dataset import PhishingEmailDataset

SPLIT = os.path.join(project_root, "experiments/configs/val_split_indices.json")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="123")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    train_mat = os.path.join(project_root, "phishingdata/x_y_tr_m.mat")
    vocab = os.path.join(project_root, "phishingdata/ind_word.pickle")
    full = PhishingEmailDataset(train_mat, vocab)

    with open(SPLIT) as f:
        split = json.load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device, flush=True)

    for s in seeds:
        set_seed(s)
        train_subset = Subset(full, split["train_indices"])
        val_subset = Subset(full, split["val_indices"])

        g = torch.Generator(); g.manual_seed(s)
        train_loader = DataLoader(train_subset, batch_size=128, shuffle=True,
                                  generator=g, collate_fn=_collate)
        val_loader = DataLoader(val_subset, batch_size=128, shuffle=False,
                                collate_fn=_collate)

        save_dir = os.path.join(project_root, f"experiments/m1_faithful_seed{s}")
        model = AI2TALEFaithful(vocab_size=311774, k=1, tau=0.5).to(device)

        print(f"\n=== seed {s} ===", flush=True)
        train_faithful(model, train_loader, val_loader,
                       {"epochs": 10, "lr": 0.001, "sig": 0.3, "lam": 0.1,
                        "save_dir": save_dir}, device)


def _collate(batch):
    tokens = torch.stack([b["tokens"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    sent_lengths = torch.stack([b["sent_lengths"] for b in batch])
    return {"tokens": tokens, "label": labels,
            "raw_sentences": [b["raw_sentences"] for b in batch],
            "sent_lengths": sent_lengths}


if __name__ == "__main__":
    main()
