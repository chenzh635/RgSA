"""Rerun Sparse ABMIL under the M10 protocol: fixed validation split, val-set
checkpoint selection, test set invisible.

Used for the condensed lambda sweep {1.0, 2.0, 3.0, 3.5}, the final 3-seed
models, and the M9 fixed-corruption variant.

Examples:
  # lambda sweep, single seed 42
  python experiments/scripts/rerun_sparse_val.py --lambdas 1.0,2.0,3.0,3.5 --tag lambda
  # final 3-seed model at lambda=3.0
  python experiments/scripts/rerun_sparse_val.py --seeds 123,456,789 --tag final
  # M9 fixed corruption
  python experiments/scripts/rerun_sparse_val.py --seeds 42 --tag m9_fixed \
      --flip-frac 0.2 --flip-mode fixed

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

from src.dataset import PhishingEmailDataset
from src.models.abmil import SparseABMIL
from src.train import train

SPLIT = os.path.join(project_root, "experiments/configs/val_split_indices.json")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(s)


def collate(batch):
    return {
        "tokens": torch.stack([b["tokens"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "sent_lengths": torch.stack([b["sent_lengths"] for b in batch]),
        "raw_sentences": [b["raw_sentences"] for b in batch],
        "idx": [b["idx"] for b in batch],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--lambdas", default="3.0")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--flip-frac", type=float, default=0.0)
    ap.add_argument("--flip-mode", default="stochastic",
                    choices=["stochastic", "fixed"])
    ap.add_argument("--force-seed-suffix", action="store_true",
                    help="save under {tag}_seed{seed} even for a single seed")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    lambdas = [float(x) for x in args.lambdas.split(",")]

    train_mat = os.path.join(project_root, "phishingdata/x_y_tr_m.mat")
    vocab = os.path.join(project_root, "phishingdata/ind_word.pickle")
    full = PhishingEmailDataset(train_mat, vocab)
    with open(SPLIT) as f:
        split = json.load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device, flush=True)

    for seed in seeds:
        for lam in lambdas:
            set_seed(seed)
            train_loader = DataLoader(Subset(full, split["train_indices"]),
                                      batch_size=32, shuffle=True,
                                      generator=torch.Generator().manual_seed(seed),
                                      collate_fn=collate)
            val_loader = DataLoader(Subset(full, split["val_indices"]),
                                    batch_size=128, shuffle=False,
                                    collate_fn=collate)

            if len(seeds) > 1 or args.force_seed_suffix:
                save_dir = os.path.join(
                    project_root, f"experiments/val_runs/{args.tag}_seed{seed}")
            else:
                ltag = ("%g" % lam).replace(".", "_")
                save_dir = os.path.join(
                    project_root, f"experiments/val_runs/{args.tag}_lam{ltag}")

            model = SparseABMIL(hidden_dim=64, attention_hidden=64,
                                classifier_hidden=[128, 64],
                                use_sparse=True).to(device)
            config = {
                "epochs": 10, "lr": 0.001, "weight_decay": 0.0001,
                "lambda_sparse": 0.1, "lambda_rule": lam,
                "temperature_init": 1.0, "temperature_min": 0.1,
                "temperature_decay": 0.95,
                "rule_flip_frac": args.flip_frac,
                "rule_flip_mode": args.flip_mode,
                "save_dir": save_dir, "seed": seed,
                "keywords_file": "phishingdata/kwords_stem.json",
            }
            print(f"\n=== seed {seed} lambda {lam} tag {args.tag} "
                  f"flip {args.flip_frac}/{args.flip_mode} ===", flush=True)
            train(model, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()
