"""Length-stratified per-email Spearman for the appendix robustness
paragraph, production checkpoints: final Sparse ABMIL seed123 (lambda=1.0)
and ABMIL seed123. Output: experiments/m_spearman_robust_appendix.json
MPS -> sandbox disabled.
"""
import json
import os
import sys

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "experiments/scripts"))

from scipy.stats import spearmanr
from wave0_analysis import ckpt
from src.dataset import create_dataloaders
from src.models.abmil import SparseABMIL, ABMILBaseline
from src.rules import batch_score, load_keywords

OUT = os.path.join(project_root, "experiments/m_spearman_robust_appendix.json")


def collect(model, loader, device, keywords, temperature=None):
    by_len = {}
    pairs = []   # (spearman sparse, spearman abmil) filled by caller join
    rows = []
    n2_positive = 0
    n2_total = 0
    for batch in loader:
        tokens = batch["tokens"].to(device)
        lengths = batch["sent_lengths"].to(device)
        labels = batch["label"].numpy()
        rule = batch_score(batch["raw_sentences"], keywords)
        with torch.no_grad():
            if temperature is None:
                _, w, _ = model(tokens, lengths)
            else:
                _, w, _, _ = model(tokens, lengths, temperature=temperature)
        w = w.cpu().numpy()
        for b in range(len(labels)):
            if labels[b] == 0:
                continue
            nu = int((lengths[b].cpu().numpy() > 0).sum())
            nu = min(nu, rule.shape[1])
            if nu < 1:
                continue
            a = w[b, :nu]
            r = rule[b, :nu]
            if nu == 2:
                n2_total += 1
            if nu >= 2 and len(set(a)) > 1 and len(set(r)) > 1:
                rho = float(spearmanr(a, r).statistic)
                if nu == 2 and rho > 0.999:
                    n2_positive += 1
            else:
                rho = None
            rows.append((nu, rho))
    return rows


def summarize(rows):
    defined = [(n, r) for n, r in rows if r is not None]
    out = {
        "n_all": len(rows),
        "n_defined": len(defined),
        "mean_all_defined": float(np.mean([r for _, r in defined])),
        "ge3_mean": float(np.mean([r for n, r in defined if n >= 3])),
        "ge3_n": sum(1 for n, r in defined if n >= 3),
        "ge5_mean": float(np.mean([r for n, r in defined if n >= 5])),
        "ge5_n": sum(1 for n, r in defined if n >= 5),
        "strata": {
            "2": float(np.mean([r for n, r in defined if n == 2])),
            "3": float(np.mean([r for n, r in defined if n == 3])),
            "4-9": float(np.mean([r for n, r in defined if 4 <= n <= 9])),
            "10+": float(np.mean([r for n, r in defined if n >= 10])),
        },
    }
    return out


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)
    import yaml
    config = yaml.safe_load(open(os.path.join(project_root,
                                             "experiments/configs/defaults.yaml")))
    _, loader = create_dataloaders(config)
    keywords = load_keywords(os.path.join(project_root,
                                         config.get("keywords_file",
                                                    "phishingdata/kwords_stem.json")))

    sparse = SparseABMIL(hidden_dim=64, attention_hidden=64,
                         classifier_hidden=[128, 64], use_sparse=True)
    st = torch.load(os.path.join(project_root,
                                 "experiments/val_runs/final_seed123/"
                                 "sparse_best_model.pt"),
                    map_location="cpu", weights_only=False)
    sparse.load_state_dict(st["model_state_dict"])
    sparse.to(device).eval()

    abmil = ABMILBaseline(hidden_dim=64, attention_hidden=64,
                          classifier_hidden=[128, 64])
    st = torch.load(ckpt("multi_seed_abmil_seed123/abmil_best_model.pt"),
                    map_location="cpu", weights_only=False)
    abmil.load_state_dict(st["model_state_dict"])
    abmil.to(device).eval()

    rows_s = collect(sparse, loader, device, keywords, temperature=0.1)
    rows_a = collect(abmil, loader, device, keywords)

    # paired wilcoxon over the same defined emails
    from scipy.stats import wilcoxon
    rs = np.array([r for n, r in rows_s if r is not None])
    ra = np.array([r for n, r in rows_a if r is not None])
    wstat = wilcoxon(rs, ra)

    payload = {"sparse_seed123": summarize(rows_s),
               "abmil_seed123": summarize(rows_a),
               "wilcoxon_p": float(wstat.pvalue),
               "n_paired": int(len(rs))}
    json.dump(payload, open(OUT, "w"), indent=1)
    print(json.dumps(payload, indent=1))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
