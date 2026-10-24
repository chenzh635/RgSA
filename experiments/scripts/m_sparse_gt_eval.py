"""GT + full-test-set scorer for SparseABMIL checkpoints (val protocol).

Used for:
  - final 3-seed models trained on the 30k split, val-selected
    (experiments/val_runs/final_seed{s}/sparse_best_model.pt);
  - M9 fixed-corruption model (val_runs/m9_fixed_seed42/...).

Per checkpoint: full 8,000-test email metrics (AUPRC / accuracy / P / R /
F1) and GT localization (P@k / R@k / F1@k / IoU@k) against
gold_final.csv, strict mode. Multi-seed runs get mean +/- std.

Usage:
  python experiments/scripts/m_sparse_gt_eval.py --tag final --seeds 123,456,789
  python experiments/scripts/m_sparse_gt_eval.py --tag m9_fixed --seeds 42

MPS -> run with sandbox disabled.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, precision_score, recall_score)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.dataset import create_dataloaders
from src.evaluate_sentence_localization import evaluate_scores, load_gold
from src.models.abmil import SparseABMIL

GOLD = os.path.join(project_root,
                    "experiments/agent_annotation/exports/gold_final.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", default="123,456,789")
    ap.add_argument("--ks", default="1,3,5")
    ap.add_argument("--suffix", default="seed{seed}",
                    help="checkpoint dir suffix; single-seed rerun_sparse_val "
                         "runs use e.g. 'lam3' instead of 'seed{seed}'")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    ks = tuple(int(x) for x in args.ks.split(","))

    gold = load_gold([GOLD])
    wanted = {eid for (eid, _) in gold["labels"]}
    print(f"gold: {gold['n_items']} sentences over {len(wanted)} emails")

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    print("device:", device)
    import yaml
    with open(os.path.join(project_root,
                           "experiments/configs/defaults.yaml")) as f:
        config = yaml.safe_load(f)
    _, test_loader = create_dataloaders(config)

    results = {}
    email_all = {}
    gt_scores_out = {}
    for seed in seeds:
        ck = os.path.join(project_root,
                          f"experiments/val_runs/{args.tag}_"
                          f"{args.suffix.format(seed=seed)}/"
                          "sparse_best_model.pt")
        model = SparseABMIL(hidden_dim=64, attention_hidden=64,
                            classifier_hidden=[128, 64], use_sparse=True)
        state = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model.to(device).eval()

        gt_scores, email_rows = {}, []
        remaining = set(wanted)
        with torch.no_grad():
            for batch in test_loader:
                out, attn_w, _, _ = model(batch["tokens"].to(device),
                                          batch["sent_lengths"].to(device),
                                          temperature=0.1)
                probs = torch.sigmoid(out).cpu().numpy()
                labels = batch["label"].numpy()
                email_rows.append(np.stack([probs, labels], axis=1))
                for b, eid in enumerate(batch["idx"]):
                    if eid in remaining:
                        n_raw = len(batch["raw_sentences"][b])
                        gt_scores[eid] = attn_w[b, :n_raw].tolist()
                        remaining.discard(eid)
                # full pass for email metrics

        rows = np.concatenate(email_rows, axis=0)
        preds = (rows[:, 0] > 0.5).astype(int)
        email = {
            "n": int(rows.shape[0]),
            "auprc": average_precision_score(rows[:, 1], rows[:, 0]),
            "accuracy": accuracy_score(rows[:, 1], preds),
            "precision": precision_score(rows[:, 1], preds, zero_division=0),
            "recall": recall_score(rows[:, 1], preds, zero_division=0),
            "f1": f1_score(rows[:, 1], preds, zero_division=0),
            "ckpt_epoch": state.get("epoch"),
        }
        scores = {int(k): np.asarray(v) for k, v in gt_scores.items()}
        loc = evaluate_scores(scores, gold, ks=ks, mode="strict")
        loc.pop("_per_email", None)
        tag = f"{args.tag} [seed{seed}]"
        results[tag] = loc
        email_all[tag] = email
        gt_scores_out[tag] = {str(k): v for k, v in gt_scores.items()}
        print(f"[{tag}] AUPRC={email['auprc']:.4f} F1={email['f1']:.4f} "
              f"P@1={loc['p@1']:.4f} R@3={loc['r@3']:.4f} "
              f"(ckpt epoch {email['ckpt_epoch']})", flush=True)
        del model

    # mean +/- std
    summary = {}
    tags = sorted(results)
    for mk in results[tags[0]]:
        vals = [results[t][mk] for t in tags]
        if all(isinstance(v, (int, float)) for v in vals):
            summary[mk] = {"mean": float(np.mean(vals)),
                           "std": float(np.std(vals))}
    email_summary = {}
    for mk in email_all[tags[0]]:
        vals = [email_all[t][mk] for t in tags]
        if all(isinstance(v, (int, float)) for v in vals):
            email_summary[mk] = {"mean": float(np.mean(vals)),
                                 "std": float(np.std(vals))}

    out_path = os.path.join(project_root,
                            f"experiments/{args.tag}_gt_eval.json")
    with open(out_path, "w") as f:
        json.dump({"localization": results, "email": email_all,
                   "loc_seed_summary": summary,
                   "email_seed_summary": email_summary,
                   "gt_scores": gt_scores_out}, f, indent=1)
    print("\n=== loc mean +/- std ===")
    for k, st in summary.items():
        print(f"  {k}: {st['mean']:.4f} +/- {st['std']:.4f}")
    print("=== email mean +/- std ===")
    for k, st in email_summary.items():
        print(f"  {k}: {st['mean']:.4f} +/- {st['std']:.4f}")
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
