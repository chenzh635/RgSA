"""One-time cache of frozen DistilBERT CLS embeddings for the ICICML runs.

Encodes every (occurrence-specific) sentence token sequence exactly as
``BertSentenceEncoder.forward`` would produce it inside
``run_sparse_bert.py``:

* per-email dynamic length ``max_len_e = min(max_word_count + 4, 512)``;
* a sentence longer than ``max_len_e`` is encoded in its truncated form
  (deduped per (ids, max_len_e));
* untruncated sentences are deduped by full token id sequence;
* batched encoding only changes pad length, which is attention-masked,
  so outputs match the per-email online encoding (numerics permitting).

Outputs in experiments/cache/icicml_bert/ :
  {split}_h.npy       flat (N_occ, 768) float32, email order, concatenated
  {split}_meta.npz    offsets (E+1), lengths, labels, per-sentence rule score
  {split}_sents.pkl   raw decoded sentences (GT-index alignment / audit)
  cache_meta.json     encoder, counts, timing

Usage: python experiments/scripts/icicml_cache_bert.py [--device mps]
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.baselines.bert_abmil import EmailTextDataset  # noqa: E402
from src.rules import load_keywords, score_sentences  # noqa: E402

OUT_DIR = os.path.join(project_root, "experiments", "cache", "icicml_bert")
MODEL_NAME = "distilbert-base-uncased"
CLS, SEP = 101, 102


def email_max_len(sents):
    """Exact replica of BertSentenceEncoder.forward's dynamic max_len."""
    if not sents:
        return 50
    return min(max(len(s.split()) for s in sents) + 4, 512)


def build_split_records(ds, tokenizer):
    """Return per-email (sentences, max_len_e, token-id lists) and unique records.

    records: dict tuple(ids) -> row index
    occ: list (per email) of list of record keys
    """
    records = {}
    per_email = []
    t0 = time.time()
    for idx in range(len(ds)):
        sents = ds._decode_email(idx)
        max_len = email_max_len(sents)
        keys = []
        if sents:
            enc = tokenizer(sents, add_special_tokens=True)["input_ids"]
        else:
            enc = []
        for ids in enc:
            if len(ids) > max_len:
                ids = [CLS] + ids[1:-1][: max_len - 2] + [SEP]
            key = tuple(ids)
            if key not in records:
                records[key] = len(records)
            keys.append(key)
        per_email.append((sents, keys))
        if (idx + 1) % 5000 == 0:
            print(f"  tokenized {idx+1}/{len(ds)} emails, "
                  f"{len(records)} unique seqs ({time.time()-t0:.0f}s)", flush=True)
    return per_email, records


def encode_records(records, tokenizer, bert, device, batch_size=128):
    keys = [None] * len(records)
    for key, row in records.items():
        keys[row] = key
    n = len(keys)
    out = np.empty((n, bert.config.dim), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            chunk = keys[start:start + batch_size]
            maxlen = max(len(k) for k in chunk)
            input_ids = np.full((len(chunk), maxlen), tokenizer.pad_token_id,
                                dtype=np.int64)
            attn = np.zeros((len(chunk), maxlen), dtype=np.int64)
            for i, k in enumerate(chunk):
                input_ids[i, :len(k)] = k
                attn[i, :len(k)] = 1
            ii = torch.from_numpy(input_ids).to(device)
            am = torch.from_numpy(attn).to(device)
            cls = bert(input_ids=ii, attention_mask=am).last_hidden_state[:, 0, :]
            out[start:start + len(chunk)] = cls.float().cpu().numpy()
            if (start // batch_size + 1) % 200 == 0:
                done = start + len(chunk)
                rate = done / (time.time() - t0)
                eta = (n - done) / rate / 60
                print(f"  encoded {done}/{n} ({rate:.0f} rec/s, eta {eta:.1f} min)",
                      flush=True)
    return out


def process_split(name, mat_name, ds, tokenizer, bert, device, keywords):
    print(f"[{name}] decoding/tokenizing {len(ds)} emails ...", flush=True)
    per_email, records = build_split_records(ds, tokenizer)
    print(f"[{name}] {len(records)} unique token sequences; encoding ...",
          flush=True)
    emb = encode_records(records, tokenizer, bert, device)

    n_occ = sum(len(keys) for _, keys in per_email)
    flat = np.empty((n_occ, emb.shape[1]), dtype=np.float32)
    offsets = np.zeros(len(per_email) + 1, dtype=np.int64)
    rules_flat = np.empty(n_occ, dtype=np.float32)
    pos = 0
    for i, (sents, keys) in enumerate(per_email):
        rows = np.fromiter((records[k] for k in keys), dtype=np.int64,
                           count=len(keys)) if keys else np.empty(0, np.int64)
        flat[pos:pos + len(rows)] = emb[rows]
        if sents:
            rules_flat[pos:pos + len(rows)] = score_sentences(sents, keywords)
        pos += len(rows)
        offsets[i + 1] = pos
    labels = ds.y.astype(np.int64)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, f"{name}_h.npy"), flat)
    np.savez(os.path.join(OUT_DIR, f"{name}_meta.npz"),
             offsets=offsets, labels=labels, rules=rules_flat)
    with open(os.path.join(OUT_DIR, f"{name}_sents.pkl"), "wb") as f:
        pickle.dump([s for s, _ in per_email], f)
    empty = sum(1 for s, _ in per_email if not s)
    print(f"[{name}] saved {n_occ} occurrences, {len(per_email)} emails, "
          f"{empty} empty; {mat_name}", flush=True)
    return {"occurrences": int(n_occ), "emails": len(per_email),
            "unique_sequences": len(records), "empty_emails": int(empty)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    device = torch.device(args.device if args.device == "mps"
                          and torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    bert = DistilBertModel.from_pretrained(MODEL_NAME).to(device).eval()

    kws_path = os.path.join(project_root, "phishingdata", "kwords_stem.json")
    keywords = load_keywords(kws_path)
    vocab_path = os.path.join(project_root, "phishingdata", "ind_word.pickle")

    stats = {}
    t0 = time.time()
    train_ds = EmailTextDataset(
        os.path.join(project_root, "phishingdata", "x_y_tr_m.mat"), vocab_path)
    stats["train"] = process_split("train", "x_y_tr_m.mat", train_ds,
                                   tokenizer, bert, device, keywords)
    del train_ds
    test_ds = EmailTextDataset(
        os.path.join(project_root, "phishingdata", "x_y_te_m.mat"), vocab_path)
    stats["test"] = process_split("test", "x_y_te_m.mat", test_ds,
                                  tokenizer, bert, device, keywords)

    stats["encoder"] = MODEL_NAME
    stats["seconds"] = time.time() - t0
    with open(os.path.join(OUT_DIR, "cache_meta.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(f"done in {stats['seconds']/60:.1f} min -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
