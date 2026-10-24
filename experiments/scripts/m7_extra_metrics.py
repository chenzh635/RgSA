"""M7: aggregate P@5 / R@5 / mean-rank from cached GT results (both modes).

Reads experiments/gt_localization_results_{strict,exclude_q}.json (produced by
run_gt_evaluation.py, ks=1,3,5 — k=5 metrics are already present) and prints
table-ready rows with 3-seed families aggregated as mean +/- std.

Paper table addition (strict): columns P@5, R@5, mRank.
"""
import json

import numpy as np

MODES = ("strict", "exclude_q")
SEED_FAMILIES = {
    "Sparse ABMIL (Ours)": "Sparse ABMIL (Ours)",
    "ABMIL (CNN)": "ABMIL (CNN)",
}
ORDER = [
    ("Random", None),
    ("ABMIL", "ABMIL (CNN)"),
    ("AI2TALE", None),
    ("BERT-ABMIL", None),
    ("Sparse BERT-ABMIL", None),
    ("Rules-only", None),
    ("Snorkel", None),
    ("Sparse ABMIL", "Sparse ABMIL (Ours)"),
]
KEYS = ("p@5", "r@5", "mean_gold_rank")


def load(mode):
    path = f"experiments/gt_localization_results_{mode}.json"
    with open(path) as f:
        return json.load(f)["results"]


def agg(results, family, key):
    if family is None:
        return results[family if family in results else key]
    variants = [n for n in results if n.startswith(family + " [seed")]
    vals = [results[n][key] for n in variants]
    return float(np.mean(vals)), float(np.std(vals, ddof=0)), len(vals)


def main():
    dump = {}
    for mode in MODES:
        results = load(mode)
        print(f"\n=== {mode} ===")
        print(f"{'model':<22}{'P@5':>14}{'R@5':>14}{'mRank':>14}")
        dump[mode] = {}
        for label, family in ORDER:
            lookup = family if family else label
            if lookup not in results and not any(
                    n.startswith(lookup + " [seed") for n in results):
                continue
            row = f"{label:<22}"
            dump[mode][label] = {}
            for key in KEYS:
                variants = [n for n in results
                            if n.startswith(lookup + " [seed")]
                if variants:
                    vals = np.array([results[n][key] for n in variants])
                    row += f"{vals.mean():>9.3f}±{vals.std(ddof=0):.3f}"
                    dump[mode][label][key] = {
                        "mean": float(vals.mean()),
                        "std": float(vals.std(ddof=0))}
                else:
                    row += f"{results[lookup][key]:>14.3f}"
                    dump[mode][label][key] = {"mean": results[lookup][key]}
            print(row)

    with open("experiments/m7_extra_metrics.json", "w") as f:
        json.dump(dump, f, indent=1)
    print("\nsaved experiments/m7_extra_metrics.json")


if __name__ == "__main__":
    main()
