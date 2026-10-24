"""M10 #62: paired bootstrap of val email AUPRC/accuracy across lambda runs.

Loads the four m10_lambda checkpoints, caches each one's probability on
the fixed 2,000-email validation split (aligned by dataset index), and
bootstraps paired differences in AUPRC and accuracy vs the other three.
If no difference is detectable, val metrics are declared insensitive and
lambda is fixed a priori per the M10 audit.

Output: experiments/m10_val_bootstrap.json
MPS -> run with sandbox disabled.
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.dataset import PhishingEmailDataset
from src.models.abmil import SparseABMIL

SPLIT = os.path.join(project_root,
                     "experiments/configs/val_split_indices.json")
OUT = os.path.join(project_root, "experiments/m10_val_bootstrap.json")
LAMS = [1.0, 2.0, 3.0, 3.5]
DIRS = {1.0: "m10_lambda_lam1", 2.0: "m10_lambda_lam2",
        3.0: "m10_lambda_lam3", 3.5: "m10_lambda_lam3_5"}


def collate(batch):
    return {
        "tokens": torch.stack([b["tokens"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "sent_lengths": torch.stack([b["sent_lengths"] for b in batch]),
        "idx": [b["idx"] for b in batch],
    }


@torch.no_grad()
def val_probs(model, loader, device, n_val, val_pos):
    probs = np.full(n_val, np.nan)
    for batch in loader:
        out, _, _, _ = model(batch["tokens"].to(device),
                             batch["sent_lengths"].to(device),
                             temperature=0.1)
        p = torch.sigmoid(out).cpu().numpy()
        for b, gi in enumerate(batch["idx"]):
            probs[val_pos[gi]] = p[b]
    assert not np.isnan(probs).any()
    return probs


def main():
    from torch.utils.data import DataLoader, Subset
    split = json.load(open(SPLIT))
    val_idx = split["val_indices"]
    val_pos = {gi: j for j, gi in enumerate(val_idx)}
    full = PhishingEmailDataset(
        os.path.join(project_root, "phishingdata/x_y_tr_m.mat"),
        os.path.join(project_root, "phishingdata/ind_word.pickle"))
    loader = DataLoader(Subset(full, val_idx), batch_size=128,
                        shuffle=False, collate_fn=collate)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    labels = np.array([int(full[i]["label"]) for i in val_idx])

    # cache probs aligned to val position
    cache = {}
    for lam in LAMS:
        model = SparseABMIL(hidden_dim=64, attention_hidden=64,
                            classifier_hidden=[128, 64], use_sparse=True)
        ck = os.path.join(project_root, "experiments/val_runs",
                          DIRS[lam], "sparse_best_model.pt")
        state = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model.to(device).eval()
        cache[lam] = val_probs(model, loader, device, len(val_idx), val_pos)
        print(f"lam {lam}: AUPRC {average_precision_score(labels, cache[lam]):.5f}"
              f" acc {((cache[lam] > .5) == labels).mean():.4f}"
              f" (ckpt epoch {state['epoch']})", flush=True)
        del model

    # paired bootstrap
    rng = np.random.default_rng(42)
    n = len(labels)
    B = 5000
    metrics = {}
    point = {}
    for lam in LAMS:
        pred = cache[lam] > 0.5
        point[lam] = {"auprc": average_precision_score(labels, cache[lam]),
                      "acc": float((pred == labels).mean())}

    diffs = {(a, b): {"auprc": [], "acc": []}
             for i, a in enumerate(LAMS) for b in LAMS[i + 1:]}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        y = labels[idx]
        if y.sum() == 0 or y.sum() == n:
            continue
        aup = {lam: average_precision_score(y, cache[lam][idx])
               for lam in LAMS}
        acc = {lam: float(((cache[lam][idx] > .5) == y).mean())
               for lam in LAMS}
        for (a, b), d in diffs.items():
            d["auprc"].append(aup[a] - aup[b])
            d["acc"].append(acc[a] - acc[b])

    out_diffs = {}
    for (a, b), d in diffs.items():
        key = f"lam{a}-lam{b}"
        out_diffs[key] = {}
        for mkey in ("auprc", "acc"):
            arr = np.array(d[mkey])
            lo, hi = np.percentile(arr, [2.5, 97.5])
            out_diffs[key][mkey] = {
                "mean": float(arr.mean()),
                "ci": [float(lo), float(hi)],
                "detectable": bool(lo > 0 or hi < 0),
            }

    detectable = [k for k, v in out_diffs.items()
                  if v["auprc"]["detectable"]]
    payload = {"point": point, "pairwise": out_diffs,
              "any_auprc_detectable": detectable,
              "n_val": n, "n_bootstrap": B}
    json.dump(payload, open(OUT, "w"), indent=1)

    print("\n=== pairwise AUPRC differences (row-lam minus col-lam) ===")
    for k, v in out_diffs.items():
        a = v["auprc"]
        print(f"  {k}: {a['mean']:+.5f} CI [{a['ci'][0]:+.5f}, "
              f"{a['ci'][1]:+.5f}] detectable={a['detectable']}")
    print("detectable AUPRC gaps:", detectable)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
