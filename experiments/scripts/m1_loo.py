"""M1: leave-one-out deletion analysis for the faithful AI2TALE port.

Same protocol as the paper's LOO table (wave0_analysis.summarize_loo):
the selector probabilities (sel_probs: normalized over valid sentences)
serve as mixture weights into the predictor head; for each sentence j we
zero its weight, renormalize the rest, and record
    delta_j = logit_full - logit_-j.
Reported: fraction of emails whose top-1 sentence deletion lowers the
logit, top-1 attention/LOO argmax coincidence, and mean Spearman(w,
delta) (overall + robust subset: >=5 defined sentences, delta std >.1).

Computed on the 4,000 phishing test emails; emails with <3 valid
sentences excluded, matching the paper's convention.

Usage: python experiments/scripts/m1_loo.py --seeds 123,456,789
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

from src.baselines.ai2tale_faithful import AI2TALEFaithful
from src.dataset import create_dataloaders

OUT = os.path.join(project_root, "experiments/m1_faithful_loo.json")
EPS = 1e-9


def _spearman(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def head_logit(model, z):
    """Predictor head on latent z (..., 250) -> binary logit."""
    x = torch.relu(model.predictor.dense(z))
    lg = model.predictor.pred(x)
    return lg[..., 1] - lg[..., 0]


@torch.no_grad()
def loo_for_seed(model, test_loader, device):
    per_email = {}
    for batch in test_loader:
        labels = batch["label"].numpy()
        if (labels == 1).sum() == 0:
            continue
        tokens = batch["tokens"].to(device)
        sent_lengths = batch["sent_lengths"].to(device)
        h = model.encoder(tokens)
        p = model.selector(h)
        valid = sent_lengths > 0
        pv = p * valid.float()
        sel = pv / pv.sum(dim=1, keepdim=True).clamp_min(1e-9)
        h = h.cpu().numpy()
        sel = sel.cpu().numpy()
        valid = valid.cpu().numpy()

        for b in np.where(labels == 1)[0]:
            eid = int(batch["idx"][b])
            nv = int(valid[b].sum())
            if nv < 3:
                per_email[eid] = {"excluded": "too_short", "n_valid": nv}
                continue
            w = sel[b, :nv]
            hb = h[b, :nv]
            if w.sum() <= EPS:
                per_email[eid] = {"excluded": "no_selection", "n_valid": nv}
                continue
            z_full = torch.tensor((w[:, None] * hb).sum(axis=0),
                                  device=device).unsqueeze(0)
            full = float(head_logit(model, z_full)[0])
            deltas = np.full(nv, np.nan)
            for j in range(nv):
                rest = 1.0 - w[j]
                if rest <= EPS:
                    continue
                wn = w.copy()
                wn[j] = 0.0
                wn /= rest
                z = torch.tensor((wn[:, None] * hb).sum(axis=0),
                                 device=device).unsqueeze(0)
                deltas[j] = full - float(head_logit(model, z)[0])
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
    ap.add_argument("--seeds", default="123,456,789")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

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
        model = AI2TALEFaithful(vocab_size=311774, k=1, tau=0.5)
        ck = os.path.join(project_root,
                          f"experiments/m1_faithful_seed{seed}/"
                          "ai2tale_best_model.pt")
        state = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model.to(device).eval()
        per_email = loo_for_seed(model, test_loader, device)
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
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("\n=== mean +/- std ===")
    for k, st in agg.items():
        print(f"  {k}: {st['mean']:.4f} +/- {st['std']:.4f}")
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
