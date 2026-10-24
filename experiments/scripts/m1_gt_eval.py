"""M1: evaluate the faithful AI2TALE port on GT + full test set.

Two products, one pass over the 8,000-email test loader per seed:
  1. email-level metrics on the FULL test set: accuracy / precision / recall /
     F1 (checkpoint was selected on the training-derived validation split —
     test set invisible during training);
  2. per-sentence selection scores (sel_probs = normalized selector output
     over valid sentences) cached for the GT emails, then scored against the
     adjudicated gold_final.csv exactly like run_gt_evaluation.py:
     P@k/R@k/F1@k/IoU@k per seed, bootstrap CIs, and mean+/-std over seeds.

Usage:
  python experiments/scripts/m1_gt_eval.py --seeds 123,456,789
  python experiments/scripts/m1_gt_eval.py --scores-only   # re-score cached

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

from src.baselines.ai2tale_faithful import AI2TALEFaithful
from src.dataset import create_dataloaders
from src.evaluate_sentence_localization import (evaluate_scores, format_table,
                                                load_gold)

GOLD = os.path.join(project_root,
                    "experiments/agent_annotation/exports/gold_final.csv")
SCORES_OUT = os.path.join(project_root,
                          "experiments/m1_faithful_gt_scores.json")
RESULTS_OUT = os.path.join(project_root,
                           "experiments/m1_faithful_gt_results.json")
SEEDS = [123, 456, 789]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="123,456,789")
    ap.add_argument("--scores-only", action="store_true")
    ap.add_argument("--ks", default="1,3,5")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    ks = tuple(int(x) for x in args.ks.split(","))

    gold = load_gold([GOLD])
    wanted_ids = {eid for (eid, _) in gold["labels"]}
    print(f"gold: {gold['n_items']} sentences over {len(wanted_ids)} emails")

    store = {}
    if os.path.exists(SCORES_OUT):
        with open(SCORES_OUT) as f:
            store = json.load(f)

    if not args.scores_only:
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cpu")
        print("device:", device)
        import yaml
        with open(os.path.join(project_root,
                               "experiments/configs/defaults.yaml")) as f:
            config = yaml.safe_load(f)
        _, test_loader = create_dataloaders(config)

        for seed in seeds:
            tag = f"AI2TALE-faithful [seed{seed}]"
            ck = os.path.join(project_root,
                              f"experiments/m1_faithful_seed{seed}/"
                              "ai2tale_best_model.pt")
            model = AI2TALEFaithful(vocab_size=311774, k=1, tau=0.5)
            state = torch.load(ck, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            model.to(device).eval()

            gt_scores, email_rows = {}, []
            remaining = set(wanted_ids)
            with torch.no_grad():
                for batch in test_loader:
                    sent_lengths = batch["sent_lengths"].to(device)
                    logit, sel_probs, _ = model(batch["tokens"].to(device),
                                                sent_lengths)
                    probs = torch.sigmoid(logit).cpu().numpy()
                    labels = batch["label"].numpy()
                    email_rows.append(np.stack([probs, labels], axis=1))
                    for b, eid in enumerate(batch["idx"]):
                        if eid in remaining:
                            n_raw = len(batch["raw_sentences"][b])
                            gt_scores[eid] = sel_probs[b, :n_raw].tolist()
                            remaining.discard(eid)
                    # full pass: email metrics must cover all 8,000

            if remaining:
                print(f"WARNING: {len(remaining)} GT emails not found")

            rows = np.concatenate(email_rows, axis=0)
            preds = (rows[:, 0] > 0.5).astype(int)
            eml = {
                "n": int(rows.shape[0]),
                "auprc": average_precision_score(rows[:, 1], rows[:, 0]),
                "accuracy": accuracy_score(rows[:, 1], preds),
                "precision": precision_score(rows[:, 1], preds, zero_division=0),
                "recall": recall_score(rows[:, 1], preds, zero_division=0),
                "f1": f1_score(rows[:, 1], preds, zero_division=0),
                "ckpt_epoch": state.get("epoch"),
            }
            store[tag] = {"gt_scores": gt_scores, "email": eml}
            print(f"[{tag}] test n={eml['n']} acc={eml['accuracy']:.4f} "
                  f"f1={eml['f1']:.4f} (ckpt epoch {eml['ckpt_epoch']}); "
                  f"GT scores for {len(gt_scores)} emails")
            del model

        with open(SCORES_OUT, "w") as f:
            json.dump(store, f)
        print("saved ->", SCORES_OUT)

    # ---------------- localization scoring ----------------
    results = {}
    for tag, payload in store.items():
        scores = {int(k): np.asarray(v) for k, v in payload["gt_scores"].items()}
        res = evaluate_scores(scores, gold, ks=ks, mode="strict")
        res.pop("_per_email", None)
        res["email"] = payload["email"]
        results[tag] = res

    # mean +/- std over seeds
    metric_keys = [m for m in next(iter(results.values()))
                   if m not in ("email", "n", "bootstrap")]
    summary = {}
    tags = sorted(results)
    for mk in metric_keys:
        vals = [results[t][mk] for t in tags]
        if all(isinstance(v, (int, float)) for v in vals):
            summary[mk] = {"mean": float(np.mean(vals)),
                           "std": float(np.std(vals, ddof=0))}

    results["__seed_summary__"] = summary
    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=1)

    print("\n" + format_table(
        {t: {k: v for k, v in results[t].items() if k != "email"}
         for t in tags}, ks))
    print("\n=== mean +/- std over seeds ===")
    for mk, st in summary.items():
        print(f"  {mk:10s} {st['mean']:.4f} +/- {st['std']:.4f}")
    print("\n=== email metrics (full 8k test) ===")
    for t in tags:
        e = results[t]["email"]
        print(f"  {t}: acc={e['accuracy']:.4f} P={e['precision']:.4f} "
              f"R={e['recall']:.4f} F1={e['f1']:.4f}")
    print("\nsaved ->", RESULTS_OUT)


if __name__ == "__main__":
    main()
