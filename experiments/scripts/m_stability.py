"""Seed-stability recomputation for the final checkpoints over the full 8k
test set.

Groups (3 seeds each):
  - final sparse: experiments/val_runs/final_seed{s}/sparse_best_model.pt
  - abmil:        experiments/multi_seed_abmil_seed{s}/abmil_best_model.pt

Reuses wave0_analysis.collect (one pass, per-email top-k) and
stability_from (pairwise top-1 agreement / top-3 Jaccard).
Output: experiments/m_stability.json
MPS -> run with sandbox disabled.
"""
import json
import os
import sys

import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "experiments/scripts"))

from wave0_analysis import collect, stability_from, load_config, ckpt
from src.dataset import create_dataloaders
from src.rules import load_keywords

OUT = os.path.join(project_root, "experiments/m_stability.json")
SEEDS = (123, 456, 789)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    print("device:", device, flush=True)
    config = load_config()
    _, test_loader = create_dataloaders(config)
    kpath = config.get("keywords_file", "phishingdata/kwords_stem.json")
    if not os.path.isabs(kpath):
        kpath = os.path.join(project_root, kpath)
    keywords = load_keywords(kpath)

    groups = {
        "final_sparse": ("cnn_sparse",
                         [(f"Sparse seed{s}",
                           os.path.join(project_root,
                                        f"experiments/val_runs/final_seed{s}/"
                                        "sparse_best_model.pt"))
                          for s in SEEDS]),
        "abmil": ("cnn_abmil",
                  [(f"ABMIL seed{s}",
                    ckpt(f"multi_seed_abmil_seed{s}/abmil_best_model.pt"))
                   for s in SEEDS]),
    }

    out = {}
    for gname, (kind, entries) in groups.items():
        collections = {}
        for tag, path in entries:
            # wave0's loader helper signature varies; load directly
            from src.models.abmil import SparseABMIL, ABMILBaseline
            if kind == "cnn_sparse":
                m = SparseABMIL(hidden_dim=64, attention_hidden=64,
                                classifier_hidden=[128, 64], use_sparse=True)
            else:
                m = ABMILBaseline(hidden_dim=64, attention_hidden=64,
                                  classifier_hidden=[128, 64])
            st = torch.load(path, map_location="cpu", weights_only=False)
            m.load_state_dict(st["model_state_dict"])
            m.to(device).eval()
            collections[tag] = collect(m, kind, test_loader, device, keywords)
            print(f"{gname}/{tag}: {len(collections[tag])} emails", flush=True)
            del m
        out[gname] = stability_from(collections)
        stb = out[gname]
        print(f"[{gname}] top1 all-agree {stb['top1_all_agree']:.3f}; "
              "phishing "
              f"{stb['by_label']['phishing']['top1_all_agree']:.3f}/"
              f"J3 {stb['by_label']['phishing']['jaccard_top3_mean']:.3f}; "
              "benign "
              f"{stb['by_label']['benign']['top1_all_agree']:.3f}/"
              f"J3 {stb['by_label']['benign']['jaccard_top3_mean']:.3f}",
              flush=True)

    json.dump(out, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
