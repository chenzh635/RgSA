"""Terminal one-time GT evaluation for the ICICML 3-seed grid.

Runs only after every final_* run JSON exists (guard), then opens
gold_final.csv exactly once and computes, per fraction x model:
  * email-level AUPRC/F1 and attention entropy: mean +/- std over seeds;
  * GT localization (strict, ? = N): p@1/3/5, r@1/3/5, f1@3, k@90,
    mean_gold_rank, mean +/- std over seeds plus seed-pooled bootstrap 95% CI;
  * paired per-email bootstrap CI of Sparse-BERT minus BERT-ABMIL
    (10,000 resamples, seed-averaged per-email scores).

Writes experiments/icicml_runs/gt_final.json and prints paper-ready tables.
No model, epoch, or lambda is chosen from anything this script computes.
"""

import json
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.evaluate_sentence_localization import (  # noqa: E402
    evaluate_scores, load_gold, paired_diff_ci,
)

RUN_DIR = os.path.join(project_root, "experiments", "icicml_runs")
GOLD = os.path.join(project_root,
                    "experiments/agent_annotation/exports/gold_final.csv")
FRACTIONS = [0.1, 0.25, 0.5, 1.0]
SEEDS = [123, 456, 789]
BOOT_B = 10000
LOC_KEYS = ["p@1", "p@3", "p@5", "r@1", "r@3", "r@5", "f1@3",
            "k@90", "mean_gold_rank"]


def frac_tag(f):
    return ("%g" % f).replace(".", "_")


def run_path(f, model, seed):
    return os.path.join(
        RUN_DIR, f"final_f{frac_tag(f)}_{model}_seed{seed}.json")


def main():
    missing = [run_path(f, m, s) for f in FRACTIONS for m in ("sparse", "plain")
               for s in SEEDS if not os.path.exists(run_path(f, m, s))]
    if missing and "--force" not in sys.argv:
        print(f"REFUSING: {len(missing)} final runs missing, e.g.:")
        for p in missing[:8]:
            print("  ", os.path.basename(p))
        print("GT stays sealed until the grid is complete.")
        return 1

    gold = load_gold([GOLD])
    print(f"gold loaded: {gold['n_items']} items, "
          f"{len(set(e for e, _ in gold['labels']))} emails")

    out = {"boot_b": BOOT_B, "mode": "strict", "fractions": {}}
    per_email_store = {}

    for f in FRACTIONS:
        fk = str(f)
        out["fractions"][fk] = {}
        for model in ("sparse", "plain"):
            seed_loc, seed_email, seed_entropy = [], [], []
            seed_pes = []
            lambdas = []
            best_epochs = []
            for s in SEEDS:
                r = json.load(open(run_path(f, model, s)))
                scores = {int(k): np.asarray(v) for k, v in r["weights"].items()}
                res = evaluate_scores(scores, gold, ks=(1, 3, 5),
                                      mode="strict", bootstrap=False)
                pe = res.pop("_per_email")
                seed_pes.append(pe)
                seed_loc.append({k: res[k] for k in LOC_KEYS if k in res})
                seed_email.append(r["test"])
                seed_entropy.append(r["entropy"])
                lambdas.append(r["lambda_rule"])
                best_epochs.append(r["best_epoch"])

            common = set.intersection(*[set(pe) for pe in seed_pes])
            avg_pe = {}
            for eid in common:
                avg_pe[eid] = {k: float(np.mean([pe[eid][k] for pe in seed_pes]))
                               for k in seed_pes[0][eid] if k not in
                               ("n_sentences", "n_gold")}
                avg_pe[eid]["n_gold"] = seed_pes[0][eid]["n_gold"]
            per_email_store[(fk, model)] = avg_pe

            agg = {"seeds": SEEDS, "lambdas": lambdas,
                   "best_epochs": best_epochs}
            for k in LOC_KEYS:
                vals = [d[k] for d in seed_loc if k in d]
                if vals:
                    agg[k] = {"mean": float(np.mean(vals)),
                              "std": float(np.std(vals)),
                              "values": vals}
            for k in ("auprc", "f1", "acc"):
                vals = [d[k] for d in seed_email]
                agg[f"email_{k}"] = {"mean": float(np.mean(vals)),
                                     "std": float(np.std(vals)),
                                     "values": vals}
            agg["entropy"] = {"mean": float(np.mean(seed_entropy)),
                              "std": float(np.std(seed_entropy)),
                              "values": seed_entropy}
            # single-model bootstrap CI on seed-averaged per-email scores
            pseudo_scores = {}
            for s in SEEDS:
                r = json.load(open(run_path(f, model, s)))
                ww = {int(k): np.asarray(v) for k, v in r["weights"].items()}
                for eid, w in ww.items():
                    pseudo_scores[eid] = pseudo_scores.get(eid, 0.0) + w / 3
            res = evaluate_scores(pseudo_scores, gold, ks=(1, 3, 5),
                                  mode="strict", bootstrap=True)
            res.pop("_per_email", None)
            agg["bootstrap_ci95"] = res.get("ci95")
            out["fractions"][fk][model] = agg

        # paired CIs: sparse minus plain
        a, b = per_email_store[(fk, "sparse")], per_email_store[(fk, "plain")]
        diffs = {}
        for k in ["p@1", "p@3", "p@5", "r@1", "r@3", "r@5", "f1@3",
                  "k@90", "mean_gold_rank"]:
            d = paired_diff_ci(a, b, k, b=BOOT_B)
            if d:
                diffs[k] = d
        out["fractions"][fk]["paired_sparse_minus_plain"] = diffs

    path = os.path.join(RUN_DIR, "gt_final.json")
    json.dump(out, open(path, "w"), indent=1)

    # ---------------- paper-ready printout ----------------
    print("\n=== Table I/II candidate values (mean +/- std over 3 seeds) ===")
    for f in FRACTIONS:
        print(f"\nfraction {f}:")
        for model in ("plain", "sparse"):
            a = out["fractions"][str(f)][model]
            print(f"  {model:6s} AUPRC {a['email_auprc']['mean']:.4f}±"
                  f"{a['email_auprc']['std']:.4f}  F1 {a['email_f1']['mean']:.4f}"
                  f"±{a['email_f1']['std']:.4f}  P@1 {a['p@1']['mean']:.3f}±"
                  f"{a['p@1']['std']:.3f}  P@3 {a['p@3']['mean']:.3f}±"
                  f"{a['p@3']['std']:.3f}  R@3 {a['r@3']['mean']:.3f}±"
                  f"{a['r@3']['std']:.3f}  k@90 {a['k@90']['mean']:.2f} "
                  f" mRank {a['mean_gold_rank']['mean']:.2f} "
                  f" H {a['entropy']['mean']:.3f}  λ={a['lambdas']}")
        d = out["fractions"][str(f)]["paired_sparse_minus_plain"]
        for k in ("p@1", "r@3", "f1@3"):
            if k in d:
                print(f"    Δ{k}: {d[k]['mean']:+.3f} "
                      f"[{d[k]['ci'][0]:+.3f},{d[k]['ci'][1]:+.3f}]")
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
