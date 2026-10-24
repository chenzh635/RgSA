"""M7 follow-up: paired per-email bootstrap CIs for P@5 (strict GT, n=202).

Adds the missing P@5 significance information to the main table without any new
inference: all scores come from frozen GT caches (gt_scores.json and the
faithful AI2TALE port m1_faithful_gt_scores.json). The multi-seed reference is
the seed-averaged per-email Sparse ABMIL pseudo-model, exactly as in
run_gt_evaluation.py. GT is used for terminal reporting only.

Output: experiments/m7_p5_paired_ci.json
"""
import json
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.evaluate_sentence_localization import evaluate_scores, load_gold, paired_diff_ci

GOLD = os.path.join(project_root,
                    "experiments/agent_annotation/exports/gold_final.csv")
GT_SCORES = os.path.join(project_root, "experiments", "gt_scores.json")
M1_SCORES = os.path.join(project_root, "experiments", "m1_faithful_gt_scores.json")
OUT = os.path.join(project_root, "experiments", "m7_p5_paired_ci.json")
SEEDS = (123, 456, 789)


def per_email_for(scores, gold):
    res = evaluate_scores({int(k): np.asarray(v) for k, v in scores.items()},
                          gold, ks=(1, 3, 5), mode="strict", bootstrap=False)
    return res.pop("_per_email"), res["p@5"]


def seed_average(per_email_variants):
    common = set.intersection(*[set(p) for p in per_email_variants])
    avg = {}
    for eid in common:
        rows = [p[eid] for p in per_email_variants]
        entry = {"n_gold": rows[0].get("n_gold", 0)}
        for mkey in rows[0]:
            if mkey == "n_gold":
                continue
            vals = [r[mkey] for r in rows if mkey in r]
            if vals:
                entry[mkey] = float(np.mean(vals))
        avg[eid] = entry
    return avg


def family_ci(variants_pe, ref_pe):
    """Per-seed paired CIs (ours-minus-variant convention: comparator - ref)."""
    out = []
    for pe in variants_pe:
        d = paired_diff_ci(pe, ref_pe, "p@5")  # variant - reference
        out.append(d)
    detectable = all(d is not None and (d["ci"][0] > 0 or d["ci"][1] < 0)
                     for d in out)
    same_side = len({np.sign(d["mean"]) for d in out}) == 1
    return {"per_seed": out,
            "detectable_all_seeds": bool(detectable and same_side)}


def main():
    gold = load_gold([GOLD])
    with open(GT_SCORES) as f:
        all_scores = json.load(f)
    with open(M1_SCORES) as f:
        m1 = json.load(f)

    # reference: seed-averaged Sparse ABMIL
    ours_variants, ours_means = [], []
    for s in SEEDS:
        pe, m = per_email_for(all_scores[f"Sparse ABMIL (Ours) [seed{s}]"], gold)
        ours_variants.append(pe)
        ours_means.append(m)
    ref_pe = seed_average(ours_variants)
    print(f"ours P@5 per seed: {['%.4f' % m for m in ours_means]} "
          f"mean {np.mean(ours_means):.4f} std {np.std(ours_means):.4f}")

    report = {"mode": "strict", "n_emails": 202,
              "ours_p5": {"per_seed": ours_means,
                          "mean": float(np.mean(ours_means)),
                          "std": float(np.std(ours_means))},
              "comparators": {}}

    # cached comparators in gt_scores.json
    def add_single(label, store_key):
        pe, m = per_email_for(all_scores[store_key], gold)
        d = paired_diff_ci(pe, ref_pe, "p@5")  # comparator - ours
        star = d["ci"][0] > 0 or d["ci"][1] < 0
        report["comparators"][label] = {
            "p5": m, "diff_minus_ours": d, "star": bool(star)}
        print(f"{label:<20} P@5 {m:.4f} diff {d['mean']:+.4f} "
              f"[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] star={star}")

    add_single("Random", "Random")
    add_single("BERT-ABMIL", "BERT-ABMIL")
    add_single("Sparse BERT-ABMIL", "Sparse BERT-ABMIL")
    add_single("Rules-only", "Rules-only")
    add_single("Snorkel", "Snorkel")

    # 3-seed families: ABMIL (CNN) cached; AI2TALE faithful port in m1 cache
    for label, variants in [
            ("ABMIL (CNN)", [f"ABMIL (CNN) [seed{s}]" for s in SEEDS]),
            ("AI2TALE-faithful",
             [f"AI2TALE-faithful [seed{s}]" for s in SEEDS])]:
        pes, ms = [], []
        for tag in variants:
            scores = (all_scores[tag] if tag in all_scores
                      else m1[tag]["gt_scores"])
            pe, m = per_email_for(scores, gold)
            pes.append(pe)
            ms.append(m)
        ci = family_ci(pes, ref_pe)
        avg_pe = seed_average(pes)
        d_avg = paired_diff_ci(avg_pe, ref_pe, "p@5")
        # star convention consistent with the table's conservative reading:
        # detectable only if the paired CI on the seed-averaged model excludes
        # zero AND all three per-seed CIs exclude zero with the same sign.
        star = bool((d_avg["ci"][0] > 0 or d_avg["ci"][1] < 0)
                    and ci["detectable_all_seeds"])
        report["comparators"][label] = {
            "p5_per_seed": ms, "p5_mean": float(np.mean(ms)),
            "p5_std": float(np.std(ms)),
            "diff_minus_ours_seed_avg": d_avg,
            "per_seed_check": ci, "star": star}
        print(f"{label:<20} P@5 {np.mean(ms):.4f}+-{np.std(ms):.4f} "
              f"avgdiff {d_avg['mean']:+.4f} "
              f"[{d_avg['ci'][0]:+.3f},{d_avg['ci'][1]:+.3f}] "
              f"all-seed={ci['detectable_all_seeds']} star={star}")

    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
