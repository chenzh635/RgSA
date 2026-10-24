"""Refresh circular proxy values (Spearman, entropy) for new checkpoints.

Conventions identical to src/evaluate.py:
  Spearman: per phishing email, correlation over valid sentences between
  attention/selector weights and rule scores; emails with <2 usable
  positions or zero rank variation skipped; mean across emails.
  Entropy: raw Shannon entropy over valid sentences (per email, averaged
  over ALL emails).

Runs:
  - faithful AI2TALE seeds 123/456/789
  - final SparseABMIL seeds 123/456/789

Output: experiments/m_proxy_refresh.json
MPS -> run with sandbox disabled.
"""
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.baselines.ai2tale_faithful import AI2TALEFaithful
from src.dataset import create_dataloaders
from src.models.abmil import SparseABMIL
from src.rules import batch_score, load_keywords

OUT = os.path.join(project_root, "experiments/m_proxy_refresh.json")


@torch.no_grad()
def run_model(model, kind, test_loader, device, keywords):
    corrs, entropies = [], []
    for batch in test_loader:
        tokens = batch["tokens"].to(device)
        labels = batch["label"].numpy()
        sent_lengths = batch["sent_lengths"].to(device)
        raw = batch["raw_sentences"]

        if kind == "faith":
            _, weights, _ = model(tokens, sent_lengths)
        else:
            _, weights, _, _ = model(tokens, sent_lengths, temperature=0.1)
        rule = batch_score(raw, keywords)
        valid = (sent_lengths > 0).cpu().numpy()
        w = weights.cpu().numpy()

        for b in range(len(labels)):
            nv = int(valid[b].sum())
            wb = w[b, :nv]
            if wb.sum() > 0:
                p = wb / wb.sum()
                entropies.append(float(-(p * np.log(p + 1e-10)).sum()))
            if labels[b] != 1 or nv < 2:
                continue
            nr = rule.shape[1]
            nu = min(nv, nr)
            if nu < 2:
                continue
            rb = rule[b, :nu]
            ab = w[b, :nu]
            if len(set(ab)) < 2 or len(set(rb)) < 2:
                continue
            c, _ = spearmanr(ab, rb)
            if not np.isnan(c):
                corrs.append(float(c))
    return {"spearman": float(np.mean(corrs)) if corrs else None,
            "entropy": float(np.mean(entropies)) if entropies else None,
            "n_spearman": len(corrs), "n_entropy": len(entropies)}


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
    for seed in (123, 456, 789):
        m = AI2TALEFaithful(vocab_size=311774, k=1, tau=0.5)
        ck = os.path.join(project_root,
                          f"experiments/m1_faithful_seed{seed}/"
                          "ai2tale_best_model.pt")
        st = torch.load(ck, map_location="cpu", weights_only=False)
        m.load_state_dict(st["model_state_dict"])
        m.to(device).eval()
        out[f"AI2TALE seed{seed}"] = run_model(m, "faith", test_loader,
                                               device, keywords)
        print(f"AI2TALE seed{seed}: {out[f'AI2TALE seed{seed}']}", flush=True)
        del m

    for seed in (123, 456, 789):
        m = SparseABMIL(hidden_dim=64, attention_hidden=64,
                        classifier_hidden=[128, 64], use_sparse=True)
        ck = os.path.join(project_root,
                          f"experiments/val_runs/final_seed{seed}/"
                          "sparse_best_model.pt")
        st = torch.load(ck, map_location="cpu", weights_only=False)
        m.load_state_dict(st["model_state_dict"])
        m.to(device).eval()
        out[f"Ours seed{seed}"] = run_model(m, "sparse", test_loader,
                                            device, keywords)
        print(f"Ours seed{seed}: {out[f'Ours seed{seed}']}", flush=True)
        del m

    groups = {"AI2TALE": [f"AI2TALE seed{s}" for s in (123, 456, 789)],
              "Ours": [f"Ours seed{s}" for s in (123, 456, 789)]}
    summary = {}
    for g, keys in groups.items():
        summary[g] = {}
        for mk in ("spearman", "entropy"):
            vals = [out[k][mk] for k in keys]
            summary[g][mk] = {"mean": float(np.mean(vals)),
                              "std": float(np.std(vals))}
    out["__summary__"] = summary
    json.dump(out, open(OUT, "w"), indent=1)
    print("\n=== summary ===")
    for g, s in summary.items():
        print(g, {k: f"{v['mean']:.4f}±{v['std']:.4f}"
                   for k, v in s.items()})
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
