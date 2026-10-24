"""Regenerate case-study attention/rule numbers from the production
checkpoints for the four canonical test emails (indices 5858, 4169,
7597, 7899).

Models: final Sparse ABMIL seed123 (lambda_rule=1.0, test temperature 0.1)
and ABMIL seed123. Output: experiments/m_case_study_refresh.json
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

import yaml
from wave0_analysis import ckpt
from src.dataset import PhishingEmailDataset
from src.models.abmil import SparseABMIL, ABMILBaseline
from src.rules import batch_score, load_keywords

EMAIL_IDS = [5852, 5362, 7597, 7899]
TEST_TEMP = 0.1
OUT = os.path.join(project_root, "experiments/m_case_study_refresh.json")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)
    with open(os.path.join(project_root, "experiments/configs/defaults.yaml")) as f:
        config = yaml.safe_load(f)
    ds = PhishingEmailDataset(
        os.path.join(project_root, "phishingdata/x_y_te_m.mat"),
        os.path.join(project_root, "phishingdata/ind_word.pickle"))
    kpath = config.get("keywords_file", "phishingdata/kwords_stem.json")
    if not os.path.isabs(kpath):
        kpath = os.path.join(project_root, kpath)
    keywords = load_keywords(kpath)

    tokens = torch.stack([ds[i]["tokens"] for i in EMAIL_IDS]).to(device)
    lengths = torch.stack([ds[i]["sent_lengths"] for i in EMAIL_IDS]).to(device)
    raws = [ds[i]["raw_sentences"] for i in EMAIL_IDS]
    rules = batch_score(raws, keywords)

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

    with torch.no_grad():
        out_s, w_s, _, _ = sparse(tokens, lengths, temperature=TEST_TEMP)
        out_a, w_a, _ = abmil(tokens, lengths)

    prob_s = torch.sigmoid(out_s).cpu().numpy()
    prob_a = torch.sigmoid(out_a).cpu().numpy()
    w_s = w_s.cpu().numpy()
    w_a = w_a.cpu().numpy()

    payload = {"email_ids": EMAIL_IDS, "test_temperature": TEST_TEMP,
               "emails": []}
    for b, (ei, raw) in enumerate(zip(EMAIL_IDS, raws)):
        n = len(raw)
        ws = w_s[b, :n]
        wa = w_a[b, :n]

        def ent(w):
            w = np.asarray(w, dtype=float)
            w = w / w.sum()
            return float(-(w * np.log(w + 1e-12)).sum())

        entry = {
            "test_idx": ei,
            "n_sentences": n,
            "sentences": raw,
            "sparse_alpha": [float(x) for x in ws],
            "abmil_alpha": [float(x) for x in wa],
            "rule": [float(rules[b, j]) for j in range(n)],
            "sparse_entropy": ent(ws),
            "abmil_entropy": ent(wa),
            "sparse_prob": float(prob_s[b]),
            "abmil_prob": float(prob_a[b]),
            "sparse_top1": int(np.argmax(ws)),
            "abmil_top1": int(np.argmax(wa)),
        }
        payload["emails"].append(entry)
        print(f"email {ei} n={n} sparse_H={ent(ws):.3f} abmil_H={ent(wa):.3f} "
              f"prob_s={prob_s[b]:.4f} prob_a={prob_a[b]:.4f}")

    json.dump(payload, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
