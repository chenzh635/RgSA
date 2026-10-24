"""Proxy values (Spearman vs CLEAN rules, entropy) for M9 corruption arms and
their clean lambda=3.0 anchor.

Mirrors m_proxy_refresh.py conventions. Runs (all seed 42):
  - m10_lambda_lam3       clean-prior anchor
  - m9_fixed_lam3         fixed 20% rule flips
  - m9_stoch_lam3         per-batch stochastic 20% flips

Output: experiments/m9_proxy_refresh.json
MPS -> run with sandbox disabled.
"""
import json
import os
import sys

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

sys.path.insert(0, os.path.join(project_root, "experiments/scripts"))
from m_proxy_refresh import run_model
from src.dataset import create_dataloaders
from src.models.abmil import SparseABMIL
from src.rules import load_keywords

OUT = os.path.join(project_root, "experiments/m9_proxy_refresh.json")
RUNS = [("clean_lam3", "m10_lambda_lam3"),
        ("m9_fixed", "m9_fixed_lam3"),
        ("m9_stoch", "m9_stoch_lam3")]


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    print("device:", device)
    import yaml
    with open(os.path.join(project_root,
                           "experiments/configs/defaults.yaml")) as f:
        config = yaml.safe_load(f)
    kpath = os.path.join(project_root,
                         config.get("keywords_file",
                                    "phishingdata/kwords_stem.json"))
    keywords = load_keywords(kpath)
    _, test_loader = create_dataloaders(config)

    out = {}
    for name, dirname in RUNS:
        m = SparseABMIL(hidden_dim=64, attention_hidden=64,
                        classifier_hidden=[128, 64], use_sparse=True)
        ck = os.path.join(project_root, "experiments/val_runs", dirname,
                          "sparse_best_model.pt")
        st = torch.load(ck, map_location="cpu", weights_only=False)
        m.load_state_dict(st["model_state_dict"])
        m.to(device).eval()
        out[name] = run_model(m, "sparse", test_loader, device, keywords)
        print(f"{name}: {out[name]}", flush=True)
        del m

    anchor_s = out["clean_lam3"]["spearman"]
    anchor_e = out["clean_lam3"]["entropy"]
    out["__retention__"] = {}
    for name in ("m9_fixed", "m9_stoch"):
        out["__retention__"][name] = {
            "spearman_retention": out[name]["spearman"] / anchor_s,
            "entropy_ratio": out[name]["entropy"] / anchor_e,
        }
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out["__retention__"], indent=1))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
