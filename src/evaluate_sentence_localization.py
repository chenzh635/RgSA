"""Sentence-level localization evaluation against human gold annotations.

Non-circular replacement for Spearman(attn, rule_scores): measures whether a
model's attention actually points at human-annotated phishing sentences.

Metrics (macro-averaged over emails):
  - Precision@k (k=1/3/5): fraction of the model's top-k sentences labeled Y
  - Recall@k: fraction of gold-Y sentences captured in top-k
  - F1@k, IoU@k (top-k set vs gold set)
  - Usability: mean sentences-to-read for >=90% recall (k@90), mean rank of
    gold sentences
  - Bootstrap 95% CI over emails; paired difference vs a reference model

Edge cases (protocol v2):
  - emails with zero gold-Y sentences: excluded from P/R/F1 denominators,
    counted separately (n_no_gold)
  - '?' sentences count as N in the strict variant; the 'exclude_q' variant
    removes them from both the ranking candidates and the gold set

Gold loading: 1 CSV = single annotator; 2-3 CSVs = majority vote (ties are
left as '?' and reported).  Expected columns:
  email_id, stratum, sentence_idx, sentence_text, is_phishing_sentence, reason
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

KS_DEFAULT = (1, 3, 5)
BOOT_B = 1000
BOOT_SEED = 0
RECALL_TARGET = 0.9


# ----------------------------------------------------------------- gold
def load_gold(csv_paths: list) -> dict:
    """Return {'labels': {(eid, sidx): 'Y'|'N'|'?'},
               'strata': {eid: stratum},
               'n_raters': int, 'n_ties': int}."""
    per_rater = []
    for p in csv_paths:
        lab = {}
        strata = {}
        with open(p, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                v = (row.get("is_phishing_sentence") or "").strip()
                if v not in ("Y", "N", "?"):
                    continue
                key = (int(row["email_id"]), int(row["sentence_idx"]))
                lab[key] = v
                strata[int(row["email_id"])] = row.get("stratum", "")
        per_rater.append(lab)

    all_keys = set.intersection(*[set(r.keys()) for r in per_rater])
    labels, n_ties = {}, 0
    for k in all_keys:
        votes = [r[k] for r in per_rater]
        c = Counter(votes)
        top = c.most_common()
        if len(top) == 1 or top[0][1] > top[1][1]:
            labels[k] = top[0][0]
        else:
            labels[k] = "?"  # unresolved tie
            n_ties += 1
    # strata from first rater (all raters share the sheet)
    strata = {}
    with open(csv_paths[0], encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            strata[int(row["email_id"])] = row.get("stratum", "")
    return {"labels": labels, "strata": strata,
            "n_raters": len(per_rater), "n_ties": n_ties,
            "n_items": len(labels)}


def gold_sets(labels: dict, mode: str = "strict"):
    """Per-email gold-Y set and candidate sentence set.

    mode 'strict': '?' counts as N.
    mode 'exclude_q': '?' sentences removed from candidates and gold.
    """
    emails = sorted({eid for (eid, _) in labels})
    out = {}
    for eid in emails:
        sents = sorted(s for (e, s) in labels if e == eid)
        if mode == "exclude_q":
            sents = [s for s in sents if labels[(eid, s)] in ("Y", "N")]
        gold_y = {s for s in sents if labels[(eid, s)] == "Y"}
        out[eid] = {"candidates": sents, "gold_y": gold_y}
    return out


# ----------------------------------------------------------------- metrics
def _topk(weights: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-weights, kind="stable")[:k]


def email_metrics(weights: np.ndarray, gold_y: set, ks) -> dict:
    n = len(weights)
    entry = {"n_sentences": n, "n_gold": len(gold_y)}
    if len(gold_y) == 0:
        return entry
    ranks = np.argsort(np.argsort(-weights, kind="stable"))  # rank of each sent
    for k in ks:
        kk = min(k, n)
        top = set(int(i) for i in _topk(weights, kk))
        hits = len(top & gold_y)
        p = hits / kk
        r = hits / len(gold_y)
        entry[f"p@{k}"] = p
        entry[f"r@{k}"] = r
        entry[f"f1@{k}"] = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        union = len(top | gold_y)
        entry[f"iou@{k}"] = hits / union if union else 0.0
    # usability: smallest k reaching RECALL_TARGET recall
    order = _topk(weights, n)
    seen, k90 = 0, None
    for i, s in enumerate(order, start=1):
        if int(s) in gold_y:
            seen += 1
        if seen / len(gold_y) >= RECALL_TARGET:
            k90 = i
            break
    entry["k@90"] = k90 if k90 is not None else n
    entry["mean_gold_rank"] = float(np.mean([ranks[s] + 1 for s in gold_y]))
    return entry


def summarize(per_email: dict, ks) -> dict:
    scored = [e for e in per_email.values() if e.get("n_gold", 0) > 0]
    n_no_gold = len(per_email) - len(scored)
    out = {"n_emails": len(per_email), "n_scored": len(scored),
           "n_no_gold": n_no_gold}
    if not scored:
        return out
    for key in [f"p@{k}" for k in ks] + [f"r@{k}" for k in ks] + \
               [f"f1@{k}" for k in ks] + [f"iou@{k}" for k in ks] + \
               ["k@90", "mean_gold_rank"]:
        vals = [e[key] for e in scored if key in e]
        out[key] = float(np.mean(vals)) if vals else None
    return out


def bootstrap_ci(per_email: dict, keys, b=BOOT_B, seed=BOOT_SEED):
    """Percentile CIs over emails for the given metric keys."""
    scored = {eid: e for eid, e in per_email.items() if e.get("n_gold", 0) > 0}
    eids = sorted(scored)
    if len(eids) < 3:
        return {k: None for k in keys}
    rng = np.random.default_rng(seed)
    cis = {}
    for key in keys:
        arr = np.array([scored[e][key] for e in eids if key in scored[e]],
                       dtype=float)
        if arr.size == 0:
            cis[key] = None
            continue
        idx = rng.integers(0, arr.size, size=(b, arr.size))
        boots = arr[idx].mean(axis=1)
        cis[key] = [float(np.percentile(boots, 2.5)),
                    float(np.percentile(boots, 97.5))]
    return cis


def paired_diff_ci(per_email_a: dict, per_email_b: dict, key,
                   b=BOOT_B, seed=BOOT_SEED):
    """CI of mean(A - B) over emails where both have gold."""
    eids = sorted(e for e in per_email_a
                  if per_email_a[e].get("n_gold", 0) > 0
                  and e in per_email_b
                  and per_email_b[e].get("n_gold", 0) > 0
                  and key in per_email_a[e] and key in per_email_b[e])
    if len(eids) < 3:
        return None
    d = np.array([per_email_a[e][key] - per_email_b[e][key] for e in eids])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, d.size, size=(b, d.size))].mean(axis=1)
    return {"mean": float(d.mean()),
            "ci": [float(np.percentile(boots, 2.5)),
                   float(np.percentile(boots, 97.5))],
            "n": len(eids)}


def evaluate_scores(scores: dict, gold: dict, ks=KS_DEFAULT,
                    mode="strict", bootstrap=True) -> dict:
    """scores: {email_idx: np.ndarray over gold sentence positions}."""
    gsets = gold_sets(gold["labels"], mode=mode)
    per_email = {}
    n_missing = 0
    for eid, g in gsets.items():
        if eid not in scores:
            n_missing += 1
            continue
        cands = g["candidates"]
        w_full = np.asarray(scores[eid], dtype=float)
        w = np.zeros(len(cands))
        for j, s in enumerate(cands):
            if s < len(w_full):
                w[j] = w_full[s]
        # gold positions reindexed into candidate order
        gold_re = {j for j, s in enumerate(cands) if s in g["gold_y"]}
        per_email[eid] = email_metrics(w, gold_re, ks)
    out = summarize(per_email, ks)
    out["mode"] = mode
    out["n_missing_scores"] = n_missing
    if bootstrap and out.get("n_scored", 0) >= 3:
        out["ci95"] = bootstrap_ci(
            per_email, [f"p@{k}" for k in ks] + [f"r@{k}" for k in ks])
    out["_per_email"] = per_email
    return out


# ----------------------------------------------------------------- report
def format_table(results: dict, ks=KS_DEFAULT) -> str:
    lines = []
    hdr = (f"{'Model':<26}" + "".join(f"{'P@'+str(k):>9}" for k in ks)
           + "".join(f"{'R@'+str(k):>9}" for k in ks)
           + f"{'k@90':>7}{'mRank':>7}{'noGold':>8}")
    lines.append(hdr)
    for name, res in results.items():
        row = f"{name:<26}"
        for k in ks:
            v = res.get(f"p@{k}")
            row += f"{v:>9.3f}" if v is not None else f"{'-':>9}"
        for k in ks:
            v = res.get(f"r@{k}")
            row += f"{v:>9.3f}" if v is not None else f"{'-':>9}"
        k90 = res.get("k@90")
        mr = res.get("mean_gold_rank")
        row += (f"{k90:>7.2f}" if k90 is not None else f"{'-':>7}")
        row += (f"{mr:>7.2f}" if mr is not None else f"{'-':>7}")
        row += f"{res.get('n_no_gold', 0):>8}"
        lines.append(row)
    return "\n".join(lines)


# ----------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True,
                    help="JSON: {model_name: {email_idx: [weights...]}}")
    ap.add_argument("--gold", nargs="+", required=True,
                    help="1-3 annotation CSVs (majority vote if >1)")
    ap.add_argument("--ks", default="1,3,5")
    ap.add_argument("--mode", default="strict", choices=["strict", "exclude_q"])
    ap.add_argument("--reference", default=None,
                    help="model name for paired-diff comparisons")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ks = tuple(int(x) for x in args.ks.split(","))
    gold = load_gold(args.gold)
    print(f"gold: {gold['n_items']} sentences, {gold['n_raters']} rater(s), "
          f"{gold['n_ties']} unresolved ties")

    with open(args.scores, encoding="utf-8") as f:
        all_scores = json.load(f)

    results = {}
    per_model_per_email = {}
    for name, sc in all_scores.items():
        scores = {int(k): np.array(v) for k, v in sc.items()}
        res = evaluate_scores(scores, gold, ks=ks, mode=args.mode)
        per_model_per_email[name] = res.pop("_per_email")
        results[name] = res

    if args.reference and args.reference in results:
        ref_pe = per_model_per_email[args.reference]
        for name in results:
            if name == args.reference:
                continue
            diffs = {}
            for key in [f"p@{k}" for k in ks] + [f"r@{k}" for k in ks]:
                d = paired_diff_ci(per_model_per_email[name], ref_pe, key)
                if d is not None:
                    diffs[key] = d
            results[name]["paired_vs_reference"] = diffs

    print(format_table(results, ks))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"gold_meta": {k: v for k, v in gold.items()
                                     if k in ("n_items", "n_raters", "n_ties")},
                       "mode": args.mode, "ks": list(ks),
                       "results": results}, f, ensure_ascii=False, indent=1)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
