"""M10: terminal GT range across the four lambda configurations.

Per audit line 20: report GT P@1/R@3 for lambda {1,2,3,3.5} as a terminal
RESULT RANGE, not as a selection criterion. Also computes 8k email metrics.

Input checkpoints: experiments/val_runs/m10_lambda_lam{1,2,3,3_5}.
Output: experiments/m10_gt_range.json
MPS -> run with sandbox disabled.
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.dataset import create_dataloaders
from src.evaluate_sentence_localization import evaluate_scores, load_gold
from src.models.abmil import SparseABMIL

GOLD = os.path.join(project_root,
                    "experiments/agent_annotation/exports/gold_final.csv")
OUT = os.path.join(project_root, "experiments/m10_gt_range.json")
CONFIGS = [(1.0, "m10_lambda_lam1"), (2.0, "m10_lambda_lam2"),
           (3.0, "m10_lambda_lam3"), (3.5, "m10_lambda_lam3_5")]


def main():
    gold = load_gold([GOLD])
    wanted = {eid for (eid, _) in gold["labels"]}

    import yaml
    with open(os.path.join(project_root,
                           "experiments/configs/defaults.yaml")) as f:
        config = yaml.safe_load(f)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    _, test_loader = create_dataloaders(config)

    out = {}
    for lam, d in CONFIGS:
        model = SparseABMIL(hidden_dim=64, attention_hidden=64,
                            classifier_hidden=[128, 64], use_sparse=True)
        ck = os.path.join(project_root, "experiments/val_runs", d,
                          "sparse_best_model.pt")
        state = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model.to(device).eval()

        gt_scores, email_rows = {}, []
        remaining = set(wanted)
        with torch.no_grad():
            for batch in test_loader:
                o, attn_w, _, _ = model(batch["tokens"].to(device),
                                        batch["sent_lengths"].to(device),
                                        temperature=0.1)
                p = torch.sigmoid(o).cpu().numpy()
                y = batch["label"].numpy()
                email_rows.append(np.stack([p, y], axis=1))
                for b, eid in enumerate(batch["idx"]):
                    if eid in remaining:
                        n_raw = len(batch["raw_sentences"][b])
                        gt_scores[eid] = attn_w[b, :n_raw].tolist()
                        remaining.discard(eid)

        rows = np.concatenate(email_rows, axis=0)
        scores = {int(k): np.asarray(v) for k, v in gt_scores.items()}
        loc = evaluate_scores(scores, gold, ks=(1, 3, 5), mode="strict")
        loc.pop("_per_email", None)
        email = {
            "auprc": average_precision_score(rows[:, 1], rows[:, 0]),
            "accuracy": accuracy_score(rows[:, 1], rows[:, 0] > .5),
            "f1": f1_score(rows[:, 1], rows[:, 0] > .5),
        }
        out[f"lam{lam}"] = {"localization": loc, "email": email,
                            "ckpt_epoch": state["epoch"]}
        print(f"lam{lam}: email AUPRC {email['auprc']:.4f} | "
              f"GT P@1 {loc['p@1']:.4f} R@3 {loc['r@3']:.4f}", flush=True)
        del model

    # ranges
    def rng_of(mkey, src="localization"):
        vals = [out[f"lam{l}"][src][mkey] for l, _ in CONFIGS]
        return {"min": float(min(vals)), "max": float(max(vals)),
                "values": [float(v) for v in vals]}
    summary = {f"p@{k}": rng_of(f"p@{k}") for k in (1, 3, 5)}
    for mkey in ("r@1", "r@3", "r@5"):
        summary[mkey] = rng_of(mkey)
    for mkey in ("auprc", "f1"):
        summary[mkey] = rng_of(mkey, src="email")
    out["__ranges__"] = summary
    json.dump(out, open(OUT, "w"), indent=1)

    print("\n=== terminal GT range across lambda ===")
    for k, v in summary.items():
        print(f"  {k}: [{v['min']:.4f}, {v['max']:.4f}]")
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
