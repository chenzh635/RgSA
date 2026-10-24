"""M12 case-study selection (frozen caches only; no inference).

Selects two GT-annotated emails for the new case-study paragraphs:
  A) long/diffuse gold-positive email where triage ranking works;
  B) GT-adjudicated mismatch: our top-1 sentence is gold-N while gold-Y
     sentences exist (honest failure, with adjudicator reasons).

Inputs (all frozen):
  experiments/gt_scores.json            selection scores per model/email
  experiments/agent_annotation/exports/gold_final.csv

Output: experiments/m12_case_candidates.json
GT is used for terminal reporting only; this script selects illustrative
examples, not models or hyperparameters.
"""
import csv
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLD = os.path.join(ROOT, "experiments/agent_annotation/exports/gold_final.csv")
SCORES = os.path.join(ROOT, "experiments/gt_scores.json")
OUT = os.path.join(ROOT, "experiments/m12_case_candidates.json")

MIN_LEN_A = 12      # long email
MIN_GOLD_A = 2      # distributed evidence


def load_gold():
    rows = list(csv.DictReader(open(GOLD)))
    by = defaultdict(list)
    for r in rows:
        by[r["email_id"]].append(r)
    for e in by:
        by[e].sort(key=lambda r: int(r["sentence_idx"]))
    return by


def ranks(scores):
    """Rank (1-based, descending) of each sentence index under scores."""
    order = np.argsort(-np.asarray(scores, dtype=float))
    r = np.empty(len(order), dtype=int)
    r[order] = np.arange(1, len(order) + 1)
    return r


def k90(n_gold_total, n_sentences, gold_ranks):
    """Smallest k covering >=90% of gold evidence; mirrors eval metric."""
    need = np.ceil(0.9 * n_gold_total) if n_gold_total > 1 else 1
    for k in range(1, n_sentences + 1):
        if sum(rk <= k for rk in gold_ranks) >= need:
            return k
    return n_sentences


def email_record(eid, rows, scores_by_model, model_keys):
    n = len(rows)
    labels = [r["is_phishing_sentence"] for r in rows]
    gold_idx = [i for i, l in enumerate(labels) if l == "Y"]
    rec = {"email_id": eid, "n_sentences": n, "n_gold": len(gold_idx),
           "gold_idx0": gold_idx, "models": {}}
    for key in model_keys:
        sc = scores_by_model[key].get(eid)
        if sc is None or len(sc) != n:
            continue
        rk = ranks(sc)
        gold_ranks = [int(rk[i]) for i in gold_idx]
        rec["models"][key] = {
            "scores": [round(float(x), 4) for x in sc],
            "top1_idx0": int(np.argmax(sc)),
            "top1_label": labels[int(np.argmax(sc))],
            "gold_ranks": gold_ranks,
            "k90": k90(len(gold_idx), n, gold_ranks),
            "p1": 1.0 if labels[int(np.argmax(sc))] == "Y" else 0.0,
        }
    return rec


def main():
    gold = load_gold()
    scores = json.load(open(SCORES))
    ours = "Sparse ABMIL (Ours) [seed123]"
    keys = [ours, "ABMIL (CNN) [seed123]", "AI2TALE", "Rules-only",
            "BERT-ABMIL", "Sparse BERT-ABMIL"]

    pos = [e for e, rs in gold.items() if any(r["is_phishing_sentence"] == "Y" for r in rs)]

    # ---- Case A candidates: long, distributed gold, ours P@1 hit, low k@90 ----
    A = []
    for eid in pos:
        rec = email_record(eid, gold[eid], scores, keys)
        o = rec["models"].get(ours)
        if not o:
            continue
        if rec["n_sentences"] >= MIN_LEN_A and rec["n_gold"] >= MIN_GOLD_A and o["p1"] == 1.0:
            rec["k90_frac"] = o["k90"] / rec["n_sentences"]
            A.append(rec)
    A.sort(key=lambda r: (-r["n_sentences"], r["k90_frac"]))

    # ---- Case B candidates: ours top-1 = N, >=2 gold Y, no '?' at top1 ----
    B = []
    for eid in pos:
        rec = email_record(eid, gold[eid], scores, keys)
        o = rec["models"].get(ours)
        if not o or rec["n_gold"] < 2:
            continue
        top = o["top1_idx0"]
        if gold[eid][top]["is_phishing_sentence"] != "N":
            continue
        # how do comparators do on the same email?
        others = {k: v["p1"] for k, v in rec["models"].items() if k != ours}
        rec["others_p1"] = others
        # nearest gold rank under ours (mildness of failure)
        rec["best_gold_rank_ours"] = min(o["gold_ranks"])
        B.append(rec)
    B.sort(key=lambda r: (r["best_gold_rank_ours"], -r["n_sentences"]))

    def attach_text(rec):
        rec["sentences"] = [
            {"idx": int(r["sentence_idx"]), "label": r["is_phishing_sentence"],
             "reason": r.get("reason", ""), "text": r["sentence_text"]}
            for r in gold[rec["email_id"]]]
        return rec

    out = {
        "case_A_long": [attach_text(r) for r in A[:8]],
        "case_B_mismatch": [attach_text(r) for r in B[:12]],
    }
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)

    print(f"gold-positive n={len(pos)}; A candidates={len(A)}; B candidates={len(B)}")
    print("\n== A (long, distributed, ours hit) ==")
    for r in A[:8]:
        o = r["models"][ours]
        print(f"eid={r['email_id']:>5} S={r['n_sentences']:2d} gold={r['n_gold']} "
              f"gold_ranks={o['gold_ranks']} k90={o['k90']} ({o['k90']/r['n_sentences']:.0%})")
    print("\n== B (top-1 mismatch, ours) ==")
    for r in B[:12]:
        o = r["models"][ours]
        a = r["models"].get("ABMIL (CNN) [seed123]", {})
        t = r["models"].get("AI2TALE", {})
        print(f"eid={r['email_id']:>5} S={r['n_sentences']:2d} gold={r['n_gold']} "
              f"top1=idx{o['top1_idx0']} bestGoldRank={r['best_gold_rank_ours']} "
              f"ABMIL_p1={a.get('p1')} AI2TALE_p1={t.get('p1')} gold_ranks={o['gold_ranks']}")
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
