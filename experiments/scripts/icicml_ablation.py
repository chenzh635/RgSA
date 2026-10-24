"""R1 (frozen protocol 2026-10-22): BERT-side mechanism ablations.

Decomposes the bundled Sparse-BERT intervention on the frozen DistilBERT
pipeline. Architecture is held constant (CachedSparseBert); only the loss
terms and temperature schedule change:

  full         lambda_rule=1.0 lambda_sparse=0.1  annealed, inference tau 0.1
  rule_only    lambda_rule=1.0 lambda_sparse=0    annealed, inference tau 0.1
  sparse_only  lambda_rule=0   lambda_sparse=0.1  annealed, inference tau 0.1
  no_annealing lambda_rule=1.0 lambda_sparse=0.1  tau=1.0 throughout

lambda_rule=1.0 is the mechanically selected f=1.0 value (validation AUPRC,
from the first-round protocol); it is NOT re-selected here. Training/val
selection is otherwise identical to icicml_run_cached.py (10 epochs, fixed
2,000-email validation split, AUPRC with accuracy tiebreak). The `full`
variant is a parity replica of final_f1_sparse_seed*.json.

No GT is touched. Output experiments/icicml_runs/final_ablation_<v>_seed<s>.json
"""

import argparse
import importlib.util
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNER_PATH = os.path.join(project_root, "experiments", "scripts",
                          "icicml_run_cached.py")
spec = importlib.util.spec_from_file_location("icicml_run_cached", RUNNER_PATH)
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

VARIANTS = {
    # lambda_rule, lambda_sparse, anneal, inference_tau
    "full": (1.0, 0.1, True, 0.1),
    "rule_only": (1.0, 0.0, True, 0.1),
    "sparse_only": (0.0, 0.1, True, 0.1),
    "no_annealing": (1.0, 0.1, False, 1.0),
}


def ablation_loss(logits, labels, scores, rules, lambda_sparse, lambda_rule):
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    normed = torch.tanh(scores)
    if lambda_sparse > 0:
        loss = loss + lambda_sparse * torch.mean(torch.abs(normed))
    if lambda_rule > 0:
        loss = loss + lambda_rule * F.mse_loss(normed, rules * 2 - 1.0)
    return loss


def extract_weights(model, data, device, infer_tau, batch_size=128):
    """rc.extract_test_weights with an explicit inference temperature."""
    model.eval()
    weights, entropies = {}, []
    with torch.no_grad():
        for s in range(0, len(data.labels), batch_size):
            ids = list(range(s, min(s + batch_size, len(data.labels))))
            h, mask, _, _ = data.gather(ids, device)
            _, w, _ = model(h, mask, infer_tau)
            w = w.cpu().numpy()
            for b, gi in enumerate(ids):
                n = int(data.offsets[gi + 1] - data.offsets[gi])
                if n == 0:
                    continue
                wb = w[b, :n]
                weights[str(gi)] = wb.tolist()
                if n >= 2:
                    p = wb / (wb.sum() + 1e-10)
                    entropies.append(float(-(p * np.log(p + 1e-10)).sum()))
    return weights, float(np.mean(entropies))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    lam_rule, lam_sparse, anneal, infer_tau = VARIANTS[args.variant]
    rc.set_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_data = rc.CachedBERTData("train")
    test_data = rc.CachedBERTData("test")
    split = json.load(open(rc.SPLIT))
    val_ids = list(split["val_indices"])
    train_ids = list(split["train_indices"])

    model = rc.CachedSparseBert().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(args.seed)

    history = []
    best_score, best_acc, best_state, best_epoch = -1.0, -1.0, None, -1
    test_ids_all = list(range(len(test_data.labels)))

    for epoch in range(args.epochs):
        model.train()
        temp = rc.get_temperature(epoch) if anneal else 1.0
        perm = torch.randperm(len(train_ids), generator=g).tolist()
        tot_loss, tot_n = 0.0, 0
        for s in range(0, len(perm), args.batch_size):
            chunk = [train_ids[j] for j in perm[s:s + args.batch_size]]
            h, mask, rules, y = train_data.gather(chunk, device)
            opt.zero_grad()
            logits, _, scores = model(h, mask, temp)
            loss = ablation_loss(logits, y, scores, rules,
                                  lam_sparse, lam_rule)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item() * len(chunk)
            tot_n += len(chunk)

        val = rc.run_eval(model, train_data, val_ids, device, 128, infer_tau)
        history.append({"epoch": epoch, "train_loss": tot_loss / tot_n,
                        "temp": temp, "val": val})
        if (val["auprc"], val["acc"]) > (best_score, best_acc):
            best_score, best_acc = val["auprc"], val["acc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        print(f"[{args.variant} s{args.seed}] epoch {epoch} "
              f"loss {history[-1]['train_loss']:.4f} tau {temp:.3f} "
              f"val_auprc {val['auprc']:.4f} val_acc {val['acc']:.4f}"
              + ("  <- best" if best_epoch == epoch else ""), flush=True)

    model.load_state_dict(best_state)
    test_metrics = rc.run_eval(model, test_data, test_ids_all, device, 128,
                               infer_tau)
    weights, entropy = extract_weights(model, test_data, device, infer_tau)
    tag = f"final_ablation_{args.variant}_seed{args.seed}"
    out = {
        "tag": tag, "variant": args.variant, "model": "sparse",
        "ablation": {"lambda_rule": lam_rule, "lambda_sparse": lam_sparse,
                      "annealed": anneal, "inference_tau": infer_tau},
        "seed": args.seed, "lambda_rule": lam_rule,
        "select": "val_auprc", "best_epoch": best_epoch,
        "selection_value": best_score, "n_train": len(train_ids),
        "test": test_metrics, "entropy": entropy,
        "history": history, "weights": weights,
    }
    path = os.path.join(rc.RUN_DIR, f"{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"best epoch {best_epoch} val_auprc {best_score:.4f}; "
          f"test AUPRC {test_metrics['auprc']:.4f} F1 {test_metrics['f1']:.4f} "
          f"H {entropy:.4f} -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
