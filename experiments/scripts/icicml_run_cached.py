"""Cached frozen-DistilBERT trainer for the ICICML Sparse-BERT experiments.

Numerically mirrors experiments/scripts/run_sparse_bert.py, but sentence
embeddings come from the one-time cache (icicml_cache_bert.py) instead of
re-encoding every epoch:

* attention MLP 768->64->1 (Tanh), classifier 768->128->64->1, dropout 0.3;
* sparse loss identical to compute_sparse_loss(): scores masked at -1e9,
  normed=tanh(scores), L1 mean|normed|, rule MSE(normed, 2r-1);
* lambda_sparse=0.1, temperature 1.0*0.95**epoch floor 0.1;
* Adam(1e-3, wd 1e-4), grad clip 1.0, 10 epochs, batch size 32.

Selection:
  --select val_auprc  (production): best epoch by fixed 2,000-email val AUPRC,
                      accuracy tiebreak (same rule as src.train.train)
  --select test_acc   (pilot replication only): best epoch by test accuracy

Outputs experiments/icicml_runs/<tag>.json with email metrics, per-email
attention weights over the whole 8,000-email test set (terminal GT scoring
reads these), and epoch history. No GT is touched here.
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

CACHE_DIR = os.path.join(project_root, "experiments", "cache", "icicml_bert")
RUN_DIR = os.path.join(project_root, "experiments", "icicml_runs")
SPLIT = os.path.join(project_root, "experiments", "configs", "val_split_indices.json")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(s)


class CachedBERTData:
    """Flat per-occurrence embeddings + per-email offsets/rules/labels."""

    def __init__(self, split, mmap=True):
        self.h = np.load(os.path.join(CACHE_DIR, f"{split}_h.npy"),
                         mmap_mode="r" if mmap else None)
        meta = np.load(os.path.join(CACHE_DIR, f"{split}_meta.npz"))
        self.offsets = meta["offsets"]
        self.labels = meta["labels"]
        self.rules = meta["rules"]

    def gather(self, ids, device):
        """Build padded (h, mask, rules, labels) for one batch of email ids."""
        lens = [int(self.offsets[i + 1] - self.offsets[i]) for i in ids]
        max_s = max(max(lens), 1)
        h = torch.zeros(len(ids), max_s, 768)
        rules = torch.zeros(len(ids), max_s)
        mask = torch.zeros(len(ids), max_s)
        for b, (i, n) in enumerate(zip(ids, lens)):
            if n:
                a, c = int(self.offsets[i]), int(self.offsets[i + 1])
                h[b, :n] = torch.from_numpy(np.asarray(self.h[a:c]))
                rules[b, :n] = torch.from_numpy(np.asarray(self.rules[a:c]))
                mask[b, :n] = 1.0
        return h.to(device), mask.to(device), rules.to(device), \
            torch.tensor([self.labels[i] for i in ids], dtype=torch.float,
                         device=device)


class CachedSparseBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_sparse = True
        self.attention = nn.Sequential(
            nn.Linear(768, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Sequential(
            nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1))

    def forward(self, h, mask, temperature=1.0):
        scores = self.attention(h).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        w = F.softmax(scores / temperature, dim=1)
        z = torch.bmm(w.unsqueeze(1), h).squeeze(1)
        return self.classifier(z).squeeze(-1), w, scores


class CachedBertABMIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_sparse = False
        self.attention = nn.Sequential(
            nn.Linear(768, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Sequential(
            nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1))

    def forward(self, h, mask, temperature=1.0):
        scores = self.attention(h).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        w = F.softmax(scores, dim=1)
        z = torch.bmm(w.unsqueeze(1), h).squeeze(1)
        return self.classifier(z).squeeze(-1), w, scores


def sparse_loss(logits, labels, scores, rules, lambda_sparse=0.1,
                lambda_rule=3.0):
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    normed = torch.tanh(scores)
    if lambda_sparse > 0:
        loss = loss + lambda_sparse * torch.mean(torch.abs(normed))
    if lambda_rule > 0:
        targets = rules * 2 - 1.0
        loss = loss + lambda_rule * F.mse_loss(normed, targets)
    return loss


def get_temperature(epoch):
    return max(1.0 * (0.95 ** epoch), 0.1)


def run_eval(model, data, ids, device, batch_size, temperature):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for s in range(0, len(ids), batch_size):
            chunk = ids[s:s + batch_size]
            h, mask, rules, y = data.gather(chunk, device)
            logits, _, _ = model(h, mask, temperature)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(y.cpu().numpy().astype(int))
    probs, ys = np.concatenate(probs), np.concatenate(ys)
    auprc = average_precision_score(ys, probs)
    acc = float(((probs > 0.5).astype(int) == ys).mean())
    f1 = f1_score(ys, (probs > 0.5).astype(int), zero_division=0)
    return {"auprc": float(auprc), "acc": acc, "f1": float(f1)}


def extract_test_weights(model, data, device, batch_size=128):
    """Attention weights + entropy over every test email, at temp 0.1."""
    model.eval()
    weights, entropies = {}, []
    with torch.no_grad():
        for s in range(0, len(data.labels), batch_size):
            ids = list(range(s, min(s + batch_size, len(data.labels))))
            h, mask, _, _ = data.gather(ids, device)
            temp = 0.1 if model.use_sparse else 1.0
            _, w, _ = model(h, mask, temp)
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
    ap.add_argument("--model", choices=["sparse", "plain"], default="sparse")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda-rule", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--select", choices=["val_auprc", "test_acc"],
                    default="val_auprc")
    ap.add_argument("--train-ids", default="split",
                    help="split | all | path to .npy/.json of train ids")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--save-ckpt", action="store_true")
    ap.add_argument("--load-pilot-mlp", action="store_true",
                    help="init attention/classifier from the pilot checkpoint")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_data = CachedBERTData("train")
    test_data = CachedBERTData("test")

    with open(SPLIT) as f:
        split = json.load(f)
    val_ids = list(split["val_indices"])
    if args.train_ids == "split":
        train_ids = list(split["train_indices"])
    elif args.train_ids == "all":
        train_ids = list(range(len(train_data.labels)))
    else:
        path = args.train_ids
        train_ids = (np.load(path).tolist() if path.endswith(".npy")
                     else json.load(open(path)))

    model = (CachedSparseBert() if args.model == "sparse"
             else CachedBertABMIL()).to(device)

    if args.load_pilot_mlp:
        ckpt = torch.load(os.path.join(
            project_root, "experiments/sparse_bert/sparse_bert_best_model.pt"),
            map_location="cpu", weights_only=False)
        sd = {k.replace("encoder.", "XXX."): v for k, v in
              ckpt["model_state_dict"].items()}
        mlp = {k: v for k, v in sd.items()
               if k.startswith("attention.") or k.startswith("classifier.")}
        missing = model.load_state_dict(mlp, strict=False)
        print("pilot MLP loaded:", len(mlp), "tensors;", missing, flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(args.seed)

    history = []
    best_key, best_score, best_acc, best_state, best_epoch = None, -1.0, -1.0, None, -1
    test_ids_all = list(range(len(test_data.labels)))

    for epoch in range(args.epochs):
        model.train()
        temp = get_temperature(epoch)
        perm = torch.randperm(len(train_ids), generator=g).tolist()
        tot_loss, tot_n = 0.0, 0
        for s in range(0, len(perm), args.batch_size):
            chunk = [train_ids[j] for j in perm[s:s + args.batch_size]]
            h, mask, rules, y = train_data.gather(chunk, device)
            opt.zero_grad()
            logits, _, scores = model(h, mask, temp)
            if args.model == "sparse":
                loss = sparse_loss(logits, y, scores, rules,
                                   lambda_rule=args.lambda_rule)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item() * len(chunk)
            tot_n += len(chunk)

        val = run_eval(model, train_data, val_ids, device, 128,
                       0.1 if model.use_sparse else 1.0)
        row = {"epoch": epoch, "train_loss": tot_loss / tot_n,
               "temp": temp, "val": val}

        if args.select == "test_acc":
            te = run_eval(model, test_data, test_ids_all, device, 128,
                          0.1 if model.use_sparse else 1.0)
            row["test"] = te
            key, score = ("acc", te["acc"])
        else:
            key, score = ("auprc", val["auprc"])
        # val_auprc mode: validation accuracy breaks ties; test_acc mode:
        # strict improvement only, so the earliest tied epoch wins (pilot rule)
        improved = (best_state is None
                    or (score > best_score if args.select == "test_acc"
                        else (score, val["acc"]) > (best_score, best_acc)))
        if improved:
            best_score, best_acc = score, val["acc"]
            best_key, best_epoch = key, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        history.append(row)
        print(f"epoch {epoch} loss {row['train_loss']:.4f} "
              f"val_auprc {val['auprc']:.4f} val_acc {val['acc']:.4f}"
              + (f" test_acc {row['test']['acc']:.4f}" if "test" in row else "")
              + ("  <- best" if best_epoch == epoch else ""), flush=True)

    model.load_state_dict(best_state)
    test_metrics = run_eval(model, test_data, test_ids_all, device, 128,
                            0.1 if model.use_sparse else 1.0)
    weights, entropy = extract_test_weights(model, test_data, device)
    out = {
        "tag": args.tag, "model": args.model, "seed": args.seed,
        "lambda_rule": args.lambda_rule, "select": args.select,
        "best_epoch": best_epoch, "selection_key": best_key,
        "selection_value": best_score, "n_train": len(train_ids),
        "test": test_metrics, "entropy": entropy,
        "history": history, "weights": weights,
    }
    os.makedirs(RUN_DIR, exist_ok=True)
    path = os.path.join(RUN_DIR, f"{args.tag}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    if args.save_ckpt:
        torch.save({"model_state_dict": model.state_dict(), "epoch": best_epoch},
                   os.path.join(RUN_DIR, f"{args.tag}.pt"))
    print(f"best epoch {best_epoch} by {best_key}={best_score:.4f}; "
          f"test AUPRC {test_metrics['auprc']:.4f} F1 {test_metrics['f1']:.4f} "
          f"H {entropy:.4f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
