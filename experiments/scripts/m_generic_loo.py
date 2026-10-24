"""Generic leave-one-out deletion analysis (SparseABMIL / ABMILBaseline).

Same protocol as m1_loo.py but for attention-MIL models: attention weights
(normalized over valid sentences) are the mixture weights; for each sentence
j we zero it, renormalize, and record
    delta_j = logit_full - logit_-j.

Reported per seed: fraction of emails whose top-1 deletion lowers the logit,
top-1 attention/LOO coincidence, mean Spearman(w, delta) (overall + robust:
n_valid >= 5, delta std > .1). 4,000 phishing test emails; <3 valid excluded.

Usage:
  python experiments/scripts/m_generic_loo.py --model sparse \
      --tag final --seeds 123,456,789
  python experiments/scripts/m_generic_loo.py --model abmil \
      --dir experiments/multi_seed_abmil_seed{seed} --seeds 123,456,789

MPS -> run with sandbox disabled.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.dataset import create_dataloaders
from src.models.abmil import ABMILBaseline, SparseABMIL

EPS = 1e-9


def _spearman(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


@torch.no_grad()
def loo_for_seed(model, sparse, test_loader, device):
    per_email = {}
    for batch in test_loader:
        labels = batch["label"].numpy()
        if (labels == 1).sum() == 0:
            continue
        tokens = batch["tokens"].to(device)
        sent_lengths = batch["sent_lengths"].to(device)
        if sparse:
            _, attn_w, _, h = model(tokens, sent_lengths, temperature=0.1)
        else:
            _, attn_w, h = model(tokens, sent_lengths)
        valid = (sent_lengths > 0).cpu().numpy()
        h = h.cpu().numpy()
        attn_w = attn_w.cpu().numpy()

        for b in np.where(labels == 1)[0]:
            eid = int(batch["idx"][b])
            nv = int(valid[b].sum())
            if nv < 3:
                per_email[eid] = {"excluded": "too_short", "n_valid": nv}
                continue
            w = attn_w[b, :nv]
            hb = h[b, :nv]
            if w.sum() <= EPS:
                per_email[eid] = {"excluded": "no_selection", "n_valid": nv}
                continue

            def logit_of(weights):
                z = torch.tensor((weights[:, None] * hb).sum(axis=0),
                                 device=device).unsqueeze(0)
                return float(model.classifier(z)[0, 0])

            full = logit_of(w)
            deltas = np.full(nv, np.nan)
            for j in range(nv):
                rest = 1.0 - w[j]
                if rest <= EPS:
                    continue
                wn = w.copy()
                wn[j] = 0.0
                wn /= rest
                deltas[j] = full - logit_of(wn)
            defined = ~np.isnan(deltas)
            entry = {
                "n_valid": nv,
                "top1_attn": int(w.argmax()),
                "spearman_attn_delta": _spearman(w[defined], deltas[defined]),
            }
            if defined.any():
                entry["top1_loo"] = int(np.nanargmax(deltas))
                entry["top1_match"] = entry["top1_loo"] == entry["top1_attn"]
                entry["top1_delta_positive"] = (
                    bool(deltas[entry["top1_attn"]] > 0)
                    if defined[entry["top1_attn"]] else None)
                entry["delta_std"] = float(np.nanstd(deltas))
            per_email[eid] = entry
    return per_email


def summarize(per_email):
    ents = [e for e in per_email.values() if "excluded" not in e]
    robust = [e for e in ents
              if e.get("n_valid", 0) >= 5 and e.get("delta_std", 0) > 0.1]

    def mean(key, rows):
        vals = [e[key] for e in rows if e.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "n_emails": len(per_email),
        "n_excluded": len(per_email) - len(ents),
        "n_robust": len(robust),
        "frac_top1_delta_positive": mean("top1_delta_positive", ents),
        "top1_match_rate": mean("top1_match", ents),
        "mean_spearman_attn_delta": mean("spearman_attn_delta", ents),
        "mean_spearman_attn_delta_robust": mean("spearman_attn_delta", robust),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["sparse", "abmil"], default="sparse")
    ap.add_argument("--tag", default=None,
                    help="val_runs subdir prefix (sparse multi-seed runs)")
    ap.add_argument("--dir", default=None,
                    help="absolute checkpoint dir template, use {seed}")
    ap.add_argument("--seeds", default="123,456,789")
    ap.add_argument("--suffix", default="seed{seed}",
                    help="val_runs suffix template for single-seed runs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    sparse = args.model == "sparse"

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    print("device:", device)
    import yaml
    with open(os.path.join(project_root,
                           "experiments/configs/defaults.yaml")) as f:
        config = yaml.safe_load(f)
    _, test_loader = create_dataloaders(config)

    out = {}
    for seed in seeds:
        if sparse:
            model = SparseABMIL(hidden_dim=64, attention_hidden=64,
                                classifier_hidden=[128, 64], use_sparse=True)
            if args.tag:
                d = (f"experiments/val_runs/{args.tag}_"
                     f"{args.suffix.format(seed=seed)}")
            else:
                d = args.dir.format(seed=seed)
            ck = os.path.join(project_root, d, "sparse_best_model.pt")
        else:
            model = ABMILBaseline(hidden_dim=64, attention_hidden=64,
                                  classifier_hidden=[128, 64])
            d = args.dir.format(seed=seed)
            ck = os.path.join(project_root, d, "abmil_best_model.pt")
        state = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model.to(device).eval()
        per_email = loo_for_seed(model, sparse, test_loader, device)
        s = summarize(per_email)
        out[f"seed{seed}"] = {"summary": s}
        print(f"[seed{seed}] n={s['n_emails']} excluded={s['n_excluded']} "
              f"pos={s['frac_top1_delta_positive']:.4f} "
              f"match={s['top1_match_rate']:.4f} "
              f"rho={s['mean_spearman_attn_delta']:.4f} "
              f"rho_robust={s['mean_spearman_attn_delta_robust']}",
              flush=True)
        del model

    vals = {k: [out[f"seed{s}"]["summary"][k] for s in seeds]
            for k in out["seed123"]["summary"]}
    agg = {}
    for k, v in vals.items():
        if all(x is not None and not isinstance(x, str) for x in v):
            agg[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
    out["__agg__"] = agg
    json.dump(out, open(args.out, "w"), indent=1)
    print("\n=== mean +/- std ===")
    for k, st in agg.items():
        print(f"  {k}: {st['mean']:.4f} +/- {st['std']:.4f}")
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
