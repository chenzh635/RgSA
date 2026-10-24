"""R5 (frozen protocol 2026-10-22): fixed-lambda cross-fraction control.

Separates the label-fraction effect from the lambda effect using the
existing seed-42 sensitivity sweeps (sweep_f{0_1,0_25,0_5,1}_lam{1,2,3,3_5},
all with full test-set weights; no retraining). Every configuration
pre-specified in the protocol is reported; nothing is selected here.

Also copies the already-published 3-seed mechanical-lambda curve from
gt_final.json into the output for side-by-side reading (no recomputation).

Output: experiments/icicml_runs/fixed_lambda_gt.json
"""

import json
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.evaluate_sentence_localization import evaluate_scores, load_gold  # noqa: E402

RUN_DIR = os.path.join(project_root, "experiments", "icicml_runs")
GOLD = os.path.join(project_root,
                    "experiments/agent_annotation/exports/gold_final.csv")
OUT = os.path.join(RUN_DIR, "fixed_lambda_gt.json")
FRACTIONS = (0.1, 0.25, 0.5, 1.0)
LAMBDAS = (1.0, 2.0, 3.0, 3.5)
LAM_TAGS = {1.0: "1", 2.0: "2", 3.0: "3", 3.5: "3_5"}


def frac_tag(f):
    return ("%g" % f).replace(".", "_")


def main():
    gold = load_gold([GOLD])
    grid = {}
    for f in FRACTIONS:
        grid[str(f)] = {}
        for lam in LAMBDAS:
            r = json.load(open(os.path.join(
                RUN_DIR, f"sweep_f{frac_tag(f)}_lam{LAM_TAGS[lam]}_seed42.json")))
            scores = {int(k): np.asarray(v) for k, v in r["weights"].items()}
            loc = evaluate_scores(scores, gold, ks=(1, 3, 5), mode="strict",
                                  bootstrap=False)
            grid[str(f)][str(lam)] = {
                "p@1": loc["p@1"], "p@3": loc["p@3"], "r@3": loc["r@3"],
                "k@90": loc["k@90"],
                "email_auprc": r["test"]["auprc"], "email_f1": r["test"]["f1"],
                "entropy": r["entropy"], "best_epoch": r["best_epoch"],
                "n_train": r["n_train"]}

    # already-published mechanical-lambda 3-seed curve (reference, unchanged)
    gt = json.load(open(os.path.join(RUN_DIR, "gt_final.json")))
    mech = {}
    for f in FRACTIONS:
        block = gt["fractions"][str(f)]
        mech[str(f)] = {
            "selected_lambda": block["sparse"]["lambdas"][0],
            "sparse_p@1_3seed": block["sparse"]["p@1"]["mean"],
            "sparse_r@3_3seed": block["sparse"]["r@3"]["mean"],
            "sparse_email_f1_3seed": block["sparse"]["email_f1"]["mean"],
            "plain_p@1_3seed": block["plain"]["p@1"]["mean"],
            "plain_r@3_3seed": block["plain"]["r@3"]["mean"]}

    out = {"protocol": "ABLATION_PROTOCOL_2026-10-22.md#B", "seed": 42,
           "fixed_lambda_grid_seed42": grid,
           "mechanical_lambda_3seed_reference": mech}
    json.dump(out, open(OUT, "w"), indent=1)

    print("P@1 matrix (rows fraction, cols lambda; seed 42):")
    print("        " + "".join(f"λ={l:<7g}" for l in LAMBDAS) + "  mech-λ 3seed")
    for f in FRACTIONS:
        row = "".join(f"{grid[str(f)][str(l)]['p@1']:<9.3f}" for l in LAMBDAS)
        print(f"f={f:<5g} {row} {mech[str(f)]['sparse_p@1_3seed']:.3f}")
    print("\nFixed λ=1.0 curve:")
    for f in FRACTIONS:
        c = grid[str(f)]["1.0"]
        print(f"  f={f:<5g} P@1 {c['p@1']:.3f}  R@3 {c['r@3']:.3f}  "
              f"F1 {c['email_f1']:.4f}  (n_train={c['n_train']})")
    print("\nMechanical-λ curve (3-seed published):")
    for f in FRACTIONS:
        c = mech[str(f)]
        print(f"  f={f:<5g} λ={c['selected_lambda']:<3g} P@1 {c['sparse_p@1_3seed']:.3f}"
              f"  R@3 {c['sparse_r@3_3seed']:.3f}  F1 {c['sparse_email_f1_3seed']:.4f}")
    print("\nsaved ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
