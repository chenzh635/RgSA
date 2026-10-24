"""R1 terminal scoring (frozen protocol 2026-10-22, section A) plus
chance-anchor paired CIs for the two headline models (R6 folding).

Opens gold_final.csv once after all final_ablation_*.json heads exist and
computes, per ablation variant (full/rule_only/sparse_only/no_annealing):
  * 3-seed mean/std of P@1/P@3/R@3/k@90/H/email AUPRC/F1;
  * paired per-email bootstrap CI of full-minus-variant (10,000 resamples,
    seed-averaged weights -- same convention as icicml_gt_final.py);
  * parity: the `full` replica vs the first-round final_f1_sparse results.

Also computes paired bootstrap CIs of each headline model against the
per-email analytic random anchor (chance), replacing the old one-draw
permutation used for the Table stars:
  * Sparse BERT: final_f1_sparse seeds 123/456/789;
  * Sparse ABMIL (CNN): experiments/final_gt_eval.json "final [seedX]".

Output: experiments/icicml_runs/ablation_gt.json
Nothing is selected: all pre-specified variants are reported.
"""

import json
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.evaluate_sentence_localization import (  # noqa: E402
    evaluate_scores, load_gold, gold_sets, paired_diff_ci,
)

RUN_DIR = os.path.join(project_root, "experiments", "icicml_runs")
GOLD = os.path.join(project_root,
                    "experiments/agent_annotation/exports/gold_final.csv")
CNN_FINAL = os.path.join(project_root, "experiments/final_gt_eval.json")
OUT = os.path.join(RUN_DIR, "ablation_gt.json")
SEEDS = (123, 456, 789)
VARIANTS = ("full", "rule_only", "sparse_only", "no_annealing")
BOOT_B = 10000
LOC_KEYS = ("p@1", "p@3", "p@5", "r@3", "r@5", "k@90", "mean_gold_rank")
CI_KEYS = ("p@1", "p@3", "r@3", "k@90", "mean_gold_rank")


def ablation_path(v, s):
    return os.path.join(RUN_DIR, f"final_ablation_{v}_seed{s}.json")


def per_email_from_weights(weights, gold):
    res = evaluate_scores(weights, gold, ks=(1, 3, 5), mode="strict",
                          bootstrap=False)
    return res.pop("_per_email"), res


def chance_expectations(gold):
    """Per-email analytic expectations of a uniform random permutation."""
    gsets = gold_sets(gold["labels"], mode="strict")
    exp = {}
    for eid, s in gsets.items():
        n, g = len(s["candidates"]), len(s["gold_y"])
        if g == 0:
            continue
        q = int(np.ceil(0.9 * g))
        exp[eid] = {f"p@{k}": g / n for k in (1, 3, 5)}
        exp[eid].update({f"r@{k}": min(k, n) / n for k in (1, 3, 5)})
        exp[eid]["k@90"] = q * (n + 1) / (g + 1)
        exp[eid]["mean_gold_rank"] = (n + 1) / 2
        exp[eid]["n_gold"] = g
    return exp


def paired_vs_chance(per_email, exp, key, b=BOOT_B, seed=0):
    eids = sorted(e for e in per_email
                  if per_email[e].get("n_gold", 0) > 0 and key in per_email[e]
                  and e in exp)
    d = np.array([per_email[e][key] - exp[e][key] for e in eids])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, d.size, size=(b, d.size))].mean(axis=1)
    return {"mean": float(d.mean()),
            "ci": [float(np.percentile(boots, 2.5)),
                   float(np.percentile(boots, 97.5))], "n": len(eids)}


def seed_averaged_per_email(weight_sets, gold):
    """Pseudo-ensemble per-email metrics from seed-averaged weights."""
    avg = {}
    for ww in weight_sets:
        for eid, w in ww.items():
            avg[eid] = avg.get(eid, 0.0) + w / len(weight_sets)
    pe, _ = per_email_from_weights(avg, gold)
    return pe


def main():
    missing = [ablation_path(v, s) for v in VARIANTS for s in SEEDS
               if not os.path.exists(ablation_path(v, s))]
    if missing:
        print(f"REFUSING: {len(missing)} ablation runs missing, e.g.:")
        for p in missing[:8]:
            print("  ", os.path.basename(p))
        return 1

    gold = load_gold([GOLD])
    out = {"protocol": "ABLATION_PROTOCOL_2026-10-22.md#A", "seeds": list(SEEDS),
           "variants": {}, "paired_full_minus_variant": {},
           "chance_ci": {}}

    per_variant_pe, per_variant_rows = {}, {}
    for v in VARIANTS:
        seed_pe, seed_rows, seed_email, seed_h = [], [], [], []
        for s in SEEDS:
            r = json.load(open(ablation_path(v, s)))
            ww = {int(k): np.asarray(x) for k, x in r["weights"].items()}
            pe, loc = per_email_from_weights(ww, gold)
            seed_pe.append(pe)
            seed_rows.append(loc)
            seed_email.append(r["test"])
            seed_h.append(r["entropy"])
        per_variant_pe[v] = seed_pe
        agg = {}
        for k in LOC_KEYS:
            vals = [d[k] for d in seed_rows if k in d]
            if vals:
                agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                          "values": vals}
        for k in ("auprc", "f1", "acc"):
            vals = [d[k] for d in seed_email]
            agg[f"email_{k}"] = {"mean": float(np.mean(vals)),
                                 "std": float(np.std(vals)), "values": vals}
        agg["entropy"] = {"mean": float(np.mean(seed_h)),
                          "std": float(np.std(seed_h)), "values": seed_h}
        out["variants"][v] = agg

    # paired CIs: full replica minus variant, pseudo-ensemble convention
    full_pe = seed_averaged_per_email(
        [{int(k): np.asarray(x) for k, x in
          json.load(open(ablation_path("full", s)))["weights"].items()}
         for s in SEEDS], gold)
    for v in VARIANTS:
        if v == "full":
            continue
        vp = seed_averaged_per_email(
            [{int(k): np.asarray(x) for k, x in
              json.load(open(ablation_path(v, s)))["weights"].items()}
             for s in SEEDS], gold)
        out["paired_full_minus_variant"][v] = {
            k: paired_diff_ci(full_pe, vp, k, b=BOOT_B) for k in CI_KEYS}

    # parity vs the first-round published 3-seed full-model values
    gt = json.load(open(os.path.join(RUN_DIR, "gt_final.json")))
    ref = gt["fractions"]["1.0"]["sparse"]
    parity = {}
    for k in ("p@1", "r@3", "k@90"):
        parity[k] = {"round1": ref[k]["mean"],
                     "replica": out["variants"]["full"][k]["mean"],
                     "delta": out["variants"]["full"][k]["mean"]
                              - ref[k]["mean"]}
    parity["email_f1"] = {"round1": ref["email_f1"]["mean"],
                          "replica": out["variants"]["full"]["email_f1"]["mean"]}
    out["parity_vs_gt_final"] = parity

    # ---- chance CIs for the two headlines ----
    exp = chance_expectations(gold)
    bert_weights = []
    for s in SEEDS:
        r = json.load(open(os.path.join(RUN_DIR,
                                        f"final_f1_sparse_seed{s}.json")))
        bert_weights.append({int(k): np.asarray(x)
                             for k, x in r["weights"].items()})
    bert_pe = seed_averaged_per_email(bert_weights, gold)
    out["chance_ci"]["Sparse BERT"] = {
        k: paired_vs_chance(bert_pe, exp, k) for k in CI_KEYS}

    if os.path.exists(CNN_FINAL):
        cnn_raw = json.load(open(CNN_FINAL))["gt_scores"]
        cnn_weights = []
        for s in SEEDS:
            blob = cnn_raw["final [seed" + str(s) + "]"]
            cnn_weights.append({int(k): np.asarray(x)
                                for k, x in blob.items()})
        cnn_pe = seed_averaged_per_email(cnn_weights, gold)
        out["chance_ci"]["Sparse ABMIL (CNN)"] = {
            k: paired_vs_chance(cnn_pe, exp, k) for k in CI_KEYS}

    json.dump(out, open(OUT, "w"), indent=1)

    print(f"{'variant':14s} {'P@1':>14s} {'R@3':>14s} {'k@90':>14s} "
          f"{'emailF1':>8s} {'H':>7s}")
    for v in VARIANTS:
        a = out["variants"][v]
        print(f"{v:14s} {a['p@1']['mean']:.3f}±{a['p@1']['std']:.3f}  "
              f"{a['r@3']['mean']:.3f}±{a['r@3']['std']:.3f}  "
              f"{a['k@90']['mean']:.2f}±{a['k@90']['std']:.2f}   "
              f"{a['email_f1']['mean']:.4f}  {a['entropy']['mean']:.3f}")
    print("\npaired full-minus-variant CI:")
    for v, dd in out["paired_full_minus_variant"].items():
        print(f"  {v:14s} ΔP@1 {dd['p@1']['mean']:+.3f} "
              f"[{dd['p@1']['ci'][0]:+.3f},{dd['p@1']['ci'][1]:+.3f}]  "
              f"ΔR@3 {dd['r@3']['mean']:+.3f} "
              f"[{dd['r@3']['ci'][0]:+.3f},{dd['r@3']['ci'][1]:+.3f}]")
    print("\nparity:", json.dumps(parity))
    print("\nchance CIs:")
    for name, dd in out["chance_ci"].items():
        print(f"  {name:20s} P@1 {dd['p@1']['mean']:+.3f} "
              f"[{dd['p@1']['ci'][0]:+.3f},{dd['p@1']['ci'][1]:+.3f}]  "
              f"R@3 {dd['r@3']['mean']:+.3f} "
              f"[{dd['r@3']['ci'][0]:+.3f},{dd['r@3']['ci'][1]:+.3f}]  "
              f"k@90 {dd['k@90']['mean']:+.3f} "
              f"[{dd['k@90']['ci'][0]:+.3f},{dd['k@90']['ci'][1]:+.3f}]")
    print("\nsaved ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
