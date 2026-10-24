"""M4: three-scenario sensitivity analysis for the 56 no-gold-Y positive bags.

Input: user verdicts from m4_review.html export
(experiments/m4_review/m4_verdicts.json): per-email a/b/c/skip.
  a = label noise (email-level label questionable)
  b = distributed deception (deceptive act spread across sentences)
  c = ensemble false negative (adjudication missed real Y)

Starting from the strict gold (202 emails), we construct gold variants:
  a  -> email excluded from positive-bag evaluation;
  b  -> add union-of-pass-Y sentences; if none, all sentences flagged Y
        (reported separately);
  c  -> add union-of-pass-Y sentences; if none, email unchanged.

Reported configurations:
  baseline        : strict gold, 202 emails;
  mixed (primary) : each email receives its user verdict's treatment;
  all-a/all-b/all-c: extreme brackets (every undecided email treated one
  way; skips follow the bracket for all-b/all-c, excluded for all-a).

Model scores are loaded from cached caches (gt_scores.json + faithful M1
scores + optional experiments/final_scores.json produced from the final
val-protocol models). Output: experiments/m4_sensitivity.json + table.

Usage: python experiments/scripts/m4_sensitivity.py
"""
import json
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.evaluate_sentence_localization import evaluate_scores, load_gold

GOLD = os.path.join(project_root,
                    "experiments/agent_annotation/exports/gold_final.csv")
VERDICTS = os.path.join(project_root,
                        "experiments/m4_review/m4_verdicts.json")
PACKAGE = os.path.join(project_root,
                       "experiments/m4_review/m4_data.json")
GT_SCORES = os.path.join(project_root, "experiments/gt_scores.json")
FAITH_SCORES = os.path.join(project_root,
                            "experiments/m1_faithful_gt_scores.json")
FINAL_SCORES = os.path.join(project_root, "experiments/final_gt_eval.json")
OUT = os.path.join(project_root, "experiments/m4_sensitivity.json")


def load_score_sources():
    """Return {display_name: {email_id: np.array}}."""
    sources = {}
    with open(GT_SCORES) as f:
        raw = json.load(f)
    # Drop the broken single-checkpoint AI2TALE row (faithful port replaces
    # it) and the old-production Ours rows (final val-protocol models replace
    # them below).
    for name, sc in raw.items():
        if name == "AI2TALE" or name.startswith("Sparse ABMIL (Ours)"):
            continue
        sources[name] = {int(k): np.asarray(v) for k, v in sc.items()}
    with open(FAITH_SCORES) as f:
        raw = json.load(f)
    for name, payload in raw.items():
        sources[name.replace("AI2TALE-faithful", "AI2TALE")] = {
            int(k): np.asarray(v) for k, v in payload["gt_scores"].items()}
    if os.path.exists(FINAL_SCORES):
        with open(FINAL_SCORES) as f:
            raw = json.load(f)["gt_scores"]
        for name, sc in raw.items():
            # 'final [seedX]' -> the canonical Ours display name
            name = name.replace("final", "Sparse ABMIL (Ours)")
            sources[name] = {int(k): np.asarray(v) for k, v in sc.items()}
    return sources


def variant_labels(gold, package, assignments):
    """Build (labels_dict, n_emails) under per-email assignments a/b/c.

    assignments: {email_id: 'a'|'b'|'c'}. Emails absent follow baseline
    (they carry no gold-Y in strict gold anyway).
    """
    labels = dict(gold["labels"])
    info = {"n_a": 0, "n_b": 0, "n_c": 0,
            "b_added_all": 0, "c_no_add": 0}
    pkg = {int(it["email_id"]): it for it in package}
    for eid, v in assignments.items():
        eid = int(eid)
        it = pkg[eid]
        if v == "a":
            # drop any gold rows for this email (email removed as positive)
            labels = {k: val for k, val in labels.items() if k[0] != eid}
            info["n_a"] += 1
            continue
        union = sorted({s["idx"] for s in it["sentences"]
                        if any(x == "Y" for x in s["votes"].values())})
        if v == "b":
            info["n_b"] += 1
            if not union:
                union = list(range(it["n_sentences"]))
                info["b_added_all"] += 1
        else:
            info["n_c"] += 1
            if not union:
                info["c_no_add"] += 1
                continue
        for j in union:
            labels[(eid, j)] = "Y"
    return labels, info


def score_config(sources, labels, base_gold):
    g = dict(base_gold)
    g["labels"] = labels
    g["n_items"] = len(labels)
    out = {}
    for name, sc in sources.items():
        r = evaluate_scores(sc, g, ks=(1, 3, 5), mode="strict",
                            bootstrap=False)
        r.pop("_per_email", None)
        out[name] = r
    return out


def main():
    gold = load_gold([GOLD])
    package = json.load(open(PACKAGE))
    pkg_ids = {int(it["email_id"]) for it in package}

    if not os.path.exists(VERDICTS):
        print("missing verdicts:", VERDICTS)
        return
    verdicts = json.load(open(VERDICTS))["verdicts"]
    user = {int(k): v.get("verdict", "skip") for k, v in verdicts.items()}
    counts = {x: sum(1 for z in user.values() if z == x)
              for x in ("a", "b", "c", "skip")}
    print("user verdict counts:", counts)
    if set(user) != pkg_ids:
        print(f"note: {len(pkg_ids - set(user))} package emails missing verdicts")

    sources = load_score_sources()
    print(f"scoring {len(sources)} model entries")

    decided = {eid: v for eid, v in user.items() if v in ("a", "b", "c")}

    configs = {"baseline": score_config(sources, gold["labels"], gold)}
    mixed_labels, info = variant_labels(gold, package, decided)
    configs["mixed"] = score_config(sources, mixed_labels, gold)
    for bracket in ("a", "b", "c"):
        assn = {eid: bracket for eid in pkg_ids}
        labels, _ = variant_labels(gold, package, assn)
        configs[f"all-{bracket}"] = score_config(sources, labels, gold)

    payload = {"verdict_counts": counts, "mixed_info": info,
               "configs": configs}
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=1)

    # compact table: P@1 / R@3 for each model across configs
    models = sorted(sources)
    print(f"\n{'model':40s} " + " ".join(f"{c:>14s}" for c in configs))
    for m in models:
        cells = []
        for c in configs:
            r = configs[c][m]
            cells.append(f"{r['p@1']:.3f}/{r['r@3']:.3f}")
        print(f"{m:40s} " + " ".join(f"{x:>14s}" for x in cells))
    print("\n(P@1/R@3); saved ->", OUT)


if __name__ == "__main__":
    main()
