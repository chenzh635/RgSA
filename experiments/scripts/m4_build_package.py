"""M4: build review package for the 56 positive emails that contain no gold-Y.

Review issue M4: 56/258 (21.7%) positive bags have zero gold-Y sentences,
apparently violating the MIL existential assumption. Three candidate causes:
  (a) email-label noise (the email is effectively benign / mislabeled),
  (b) distributed deception (no single sentence self-sufficient; the deceptive
      act is assembled across several individually-insufficient sentences),
  (c) ensemble false negatives (a gold sentence exists but all passes missed).

This script assembles experiments/m4_review/m4_data.json:
  - gold rows per email (final labels incl. ?)
  - per-sentence pass-a/b/c votes merged from agent pass outputs
  - sparse-model attention (seed123, cached gt_scores) for top-1 / spread
  - slot for email probability (filled by m4_email_probs.py from backup ckpt)
  - heuristic initial guess a/b/c (advisory only; user makes the call)

Then m4_review.html renders a single-page keyboard-driven review UI.
"""
import csv
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLD = os.path.join(ROOT, "experiments/agent_annotation/exports/gold_final.csv")
PASS_DIR = os.path.join(ROOT, "experiments/agent_annotation/outputs")
SCORES = os.path.join(ROOT, "experiments/gt_scores.json")
OUT_DIR = os.path.join(ROOT, "experiments/m4_review")


def load_gold():
    emails = {}
    with open(GOLD, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            eid = int(row["email_id"])
            e = emails.setdefault(eid, {
                "email_id": eid, "stratum": row.get("stratum", ""),
                "sentences": []})
            e["sentences"].append({
                "idx": int(row["sentence_idx"]),
                "text": row["sentence_text"],
                "gold": row["is_phishing_sentence"],
                "reason": row.get("reason", ""),
            })
    for e in emails.values():
        e["sentences"].sort(key=lambda s: s["idx"])
    return emails


def load_pass_votes():
    """{(eid, sidx): {'a': label, 'b': ..., 'c': ...}}."""
    votes = {}
    tag = {"a": "agent_pass_a", "b": "agent_pass_b", "c": "agent_pass_c"}
    for pi, prefix in tag.items():
        for path in sorted(glob.glob(os.path.join(PASS_DIR, f"{prefix}_*.json"))):
            with open(path) as f:
                payload = json.load(f)
            for a in payload.get("annotations", []):
                votes[(int(a["email_id"]), int(a["sentence_idx"]))] = (
                    votes.get((int(a["email_id"]), int(a["sentence_idx"])), {}))
                votes[(int(a["email_id"]), int(a["sentence_idx"]))][pi] = (
                    a["label"])
    return votes


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    gold = load_gold()
    votes = load_pass_votes()
    with open(SCORES) as f:
        scores = json.load(f)
    attn = {int(k): v for k, v in scores.get(
        "Sparse ABMIL (Ours) [seed123]", {}).items()}
    try:
        with open(os.path.join(OUT_DIR, "m4_email_probs.json")) as f:
            probs = {int(k): v for k, v in json.load(f).items()}
    except FileNotFoundError:
        probs = {}

    targets = [e for e in gold.values()
               if all(s["gold"] != "Y" for s in e["sentences"])]
    targets.sort(key=lambda e: e["email_id"])

    out = []
    for e in targets:
        eid = e["email_id"]
        n_q = sum(1 for s in e["sentences"] if s["gold"] == "?")
        # attach votes
        anypass_y = 0
        patterns = {}
        for s in e["sentences"]:
            v = votes.get((eid, s["idx"]), {})
            s["votes"] = {"a": v.get("a", "?"), "b": v.get("b", "?"),
                          "c": v.get("c", "?")}
            n_y = sum(1 for x in s["votes"].values() if x == "Y")
            anypass_y += n_y > 0
            pat = "".join(s["votes"][k] for k in ("a", "b", "c"))
            patterns[pat] = patterns.get(pat, 0) + 1
        w = np.array(attn.get(eid, []), dtype=float)
        if w.size:
            top1 = int(np.argmax(w))
            top1_w = float(w[top1])
            # effective number of sentences (participation ratio)
            p = w / w.sum()
            eff_n = float(1.0 / np.sum(p ** 2))
        else:
            top1, top1_w, eff_n = None, None, None
        p_email = probs.get(eid)

        # ---- heuristic initial guess (advisory only) ----
        guess = "c"  # default: conservative ensemble
        if p_email is not None and p_email < 0.5:
            guess = "a"          # model itself doubts the phishing label
        elif w.size and eff_n > 4 and top1_w is not None and top1_w < 0.45:
            guess = "b"          # attention spread across many sentences
        elif anypass_y == 0:
            guess = "b"          # no pass saw any Y; likely distributed/no signal
        guess_note = {
            "a": "模型邮件概率<0.5：疑似标签噪声",
            "b": "注意力分散/无单句 Y：疑似分布式欺骗",
            "c": "有 pass 曾标 Y 但最终为 N：疑似集成假阴性",
        }[guess]

        out.append({
            "email_id": eid,
            "stratum": e["stratum"],
            "n_sentences": len(e["sentences"]),
            "n_question": n_q,
            "anypass_y": anypass_y,
            "vote_patterns": patterns,
            "email_prob": p_email,
            "top1": top1,
            "top1_weight": top1_w,
            "eff_n": eff_n,
            "initial_guess": guess,
            "guess_note": guess_note,
            "sentences": e["sentences"],
        })

    with open(os.path.join(OUT_DIR, "m4_data.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"{len(out)} no-gold-Y emails -> experiments/m4_review/m4_data.json")
    print("with email probs:", sum(1 for e in out if e["email_prob"] is not None))

    # self-contained HTML (data embedded so file:// needs no fetch)
    tpl_path = os.path.join(OUT_DIR, "review_template.html")
    with open(tpl_path, encoding="utf-8") as f:
        tpl = f.read()
    embedded = json.dumps(out, ensure_ascii=False)
    html = tpl.replace("/*__DATA__*/null", embedded)
    with open(os.path.join(OUT_DIR, "m4_review.html"), "w",
              encoding="utf-8") as f:
        f.write(html)
    print("experiments/m4_review/m4_review.html ready (open in browser)")


if __name__ == "__main__":
    main()
