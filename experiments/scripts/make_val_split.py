"""M10: create and archive the fixed stratified validation split.

2,000 emails (1,000 benign / 1,000 phishing) held out from the 16,000-email
training set; remaining 14,000 are used for training. Indices are archived so
every rerun (AI2TALE faithful port, M9 fixed corruption, our model, lambda
sweep) uses the identical split. Test set stays untouched.

Usage: python experiments/scripts/make_val_split.py
"""
import json
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.dataset import PhishingEmailDataset

OUT = os.path.join(project_root, "experiments/configs/val_split_indices.json")
SEED = 42


def main():
    train_mat = os.path.join(project_root, "phishingdata/x_y_tr_m.mat")
    vocab = os.path.join(project_root, "phishingdata/ind_word.pickle")
    ds = PhishingEmailDataset(train_mat, vocab)
    labels = ds.y  # 0:8000 benign, 8000:16000 phishing

    rng = np.random.default_rng(SEED)
    val_idx, train_idx = [], []
    for lab in (0, 1):
        pool = np.where(labels == lab)[0]
        rng.shuffle(pool)
        val_idx.extend(pool[:1000].tolist())
        train_idx.extend(pool[1000:].tolist())
    rng.shuffle(val_idx)
    rng.shuffle(train_idx)

    payload = {"seed": SEED, "n_val": len(val_idx), "n_train": len(train_idx),
               "val_indices": sorted(val_idx), "train_indices": sorted(train_idx)}
    with open(OUT, "w") as f:
        json.dump(payload, f)

    # sanity
    vlab = labels[sorted(val_idx)]
    print(f"val={len(val_idx)} (benign={(vlab==0).sum()}, phishing={(vlab==1).sum()})")
    print(f"train={len(train_idx)}; overlap={len(set(val_idx)&set(train_idx))}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
