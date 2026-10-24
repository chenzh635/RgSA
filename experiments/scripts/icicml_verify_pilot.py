"""Cache fidelity gate: pilot checkpoint MLP on cached embeddings vs the
stored online-extraction results.

Loads attention/classifier weights from experiments/sparse_bert/
sparse_bert_best_model.pt, runs them on cached test embeddings, and compares:
  * per-email attention vectors vs experiments/gt_scores.json
    ("Sparse BERT-ABMIL", extracted online through the frozen encoder);
  * GT metrics vs gt_localization_results_strict.json (p@1=0.6485, ...);
  * email AUPRC/F1 vs experiments/sparse_bert/results.json (.9962/.9693).

This is the ONLY ground-truth read used during development, and it reads
existing frozen pilot artifacts — no new model or hyperparameter is selected
from it.
"""

import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.evaluate_sentence_localization import evaluate_scores, load_gold  # noqa
from icicml_run_cached import CachedBERTData, CachedSparseBert  # noqa

GOLD = os.path.join(project_root,
                    "experiments/agent_annotation/exports/gold_final.csv")
CKPT = os.path.join(project_root,
                    "experiments/sparse_bert/sparse_bert_best_model.pt")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = CachedBERTData("test")
    model = CachedSparseBert().to(device)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    mlp = {k: v for k, v in ck["model_state_dict"].items()
           if k.startswith("attention.") or k.startswith("classifier.")}
    print(model.load_state_dict(mlp, strict=False))
    model.eval()

    probs, labels, weights = [], [], {}
    with torch.no_grad():
        for s in range(0, len(data.labels), 128):
            ids = list(range(s, min(s + 128, len(data.labels))))
            h, mask, _, y = data.gather(ids, device)
            logits, w, _ = model(h, mask, 0.1)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.cpu().numpy().astype(int))
            for b, gi in enumerate(ids):
                n = int(data.offsets[gi + 1] - data.offsets[gi])
                if n:
                    weights[gi] = w[b, :n].cpu().numpy().astype(float)
    probs, labels = np.concatenate(probs), np.concatenate(labels)
    auprc = average_precision_score(labels, probs)
    f1 = f1_score(labels, (probs > 0.5).astype(int))
    print(f"cached pilot: AUPRC {auprc:.4f} (pilot .9962), "
          f"F1 {f1:.4f} (pilot .9693)")

    # per-weight comparison against stored online extraction
    online = json.load(open(os.path.join(
        project_root, "experiments", "gt_scores.json")))["Sparse BERT-ABMIL"]
    diffs = []
    for k, v in online.items():
        if int(k) in weights:
            diffs.append(np.abs(weights[int(k)] - np.asarray(v)).max())
    print(f"weight max abs diff vs online: max {max(diffs):.2e}, "
          f"mean {np.mean(diffs):.2e} over {len(diffs)} GT emails")

    gold = load_gold([GOLD])
    loc = evaluate_scores(weights, gold, ks=(1, 3, 5), mode="strict")
    ref = json.load(open(os.path.join(
        project_root, "experiments", "gt_localization_results_strict.json")))
    ref = ref["results"]["Sparse BERT-ABMIL"]
    print(f"{'metric':10s} {'cached':>10s} {'pilot':>10s}")
    for m in ("p@1", "p@3", "r@3", "k@90", "mean_gold_rank"):
        print(f"{m:10s} {loc[m]:10.4f} {ref[m]:10.4f}")


if __name__ == "__main__":
    main()
