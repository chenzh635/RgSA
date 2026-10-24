"""R7 (frozen protocol 2026-10-22): 56 no-gold-bag adjudication brackets for
the headline Sparse BERT model and its BERT-ABMIL backbone.

No retraining: per-email weights are read from the existing three-seed
final_f1_{sparse,plain}_seed*.json runs and seed-averaged exactly as in
icicml_gt_final.py. Bracket label construction reuses m4_sensitivity.py
variant_labels with the original user verdicts (14 a / 5 b / 37 c); nothing
is re-adjudicated here.

Configs (frozen): baseline strict 202; mixed (primary re-adjudication);
all-c (every undecided bag treated as an adjudication miss); all-a/all-b
archived for completeness. Metrics pre-specified for the paper: P@1, R@3.
Output: experiments/icicml_runs/sparse_brackets_gt.json
"""

import importlib.util
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
PACKAGE = os.path.join(project_root, "experiments/m4_review/m4_data.json")
VERDICTS = os.path.join(project_root, "experiments/m4_review/m4_verdicts.json")
CNN_SENS = os.path.join(project_root, "experiments", "m4_sensitivity.json")
OUT = os.path.join(RUN_DIR, "sparse_brackets_gt.json")
SEEDS = (123, 456, 789)
METRICS = ("p@1", "r@3")


def load_m4_module():
    path = os.path.join(project_root, "experiments", "scripts",
                        "m4_sensitivity.py")
    spec = importlib.util.spec_from_file_location("m4_sensitivity", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def per_seed_weights(model):
    """[{email: weights}] per seed, so bracket scoring aggregates exactly
    like the paper headline (evaluate per seed, then mean over seeds)."""
    outs = []
    for s in SEEDS:
        r = json.load(open(os.path.join(
            RUN_DIR, f"final_f1_{model}_seed{s}.json")))
        outs.append({int(k): np.asarray(v) for k, v in r["weights"].items()})
    return outs


def main():
    m4 = load_m4_module()
    gold = load_gold([GOLD])
    package = json.load(open(PACKAGE))
    pkg_ids = {int(it["email_id"]) for it in package}
    verdicts = json.load(open(VERDICTS))["verdicts"]
    user = {int(k): v.get("verdict", "skip") for k, v in verdicts.items()}
    decided = {eid: v for eid, v in user.items() if v in ("a", "b", "c")}
    print(f"package: {len(pkg_ids)} emails; decided: {len(decided)}")

    sources = {"Sparse BERT": per_seed_weights("sparse"),
               "BERT-ABMIL": per_seed_weights("plain")}

    mixed_labels, info = m4.variant_labels(gold, package, decided)
    label_sets = {"baseline": gold["labels"], "mixed": mixed_labels}
    for bracket in ("a", "b", "c"):
        labels, _ = m4.variant_labels(
            gold, package, {eid: bracket for eid in pkg_ids})
        label_sets[f"all-{bracket}"] = labels

    want = ("p@1", "p@3", "p@5", "r@3", "r@5", "k@90")
    configs = {}
    for cname, labels in label_sets.items():
        g = dict(gold)
        g["labels"] = labels
        g["n_items"] = len(labels)
        configs[cname] = {}
        for name, seed_weights in sources.items():
            seed_rows = []
            for scores in seed_weights:
                r = evaluate_scores(scores, g, ks=(1, 3, 5), mode="strict",
                                    bootstrap=False)
                seed_rows.append(r)
            cell = {"n_scored": seed_rows[0]["n_scored"]}
            for k in want:
                vals = [row[k] for row in seed_rows if k in row]
                if vals:
                    cell[k] = float(np.mean(vals))
                    cell[f"{k}_seed_values"] = [float(v) for v in vals]
                    cell[f"{k}_std"] = float(np.std(vals))
            configs[cname][name] = cell

    out = {"protocol": "ABLATION_PROTOCOL_2026-10-22.md#D",
           "seeds": list(SEEDS), "mode": "strict",
           "verdict_counts": {x: sum(1 for z in user.values() if z == x)
                              for x in ("a", "b", "c", "skip")},
           "mixed_info": info, "configs": configs}

    # cross-check: CNN family rows from the original M4 artifact (same code
    # path, same bracket definitions)
    if os.path.exists(CNN_SENS):
        cnn = json.load(open(CNN_SENS))["configs"]
        out["cnn_crosscheck"] = {
            c: {name: {m: row.get(m) for m in METRICS}
                for name, row in rows.items() if "Sparse ABMIL" in name}
            for c, rows in cnn.items()}

    json.dump(out, open(OUT, "w"), indent=1)

    print(f"\n{'config':10s} " + "  ".join(
        f"{n:>22s}" for n in sources))
    for cname in ("baseline", "mixed", "all-c", "all-a", "all-b"):
        cells = []
        for name in sources:
            r = configs[cname][name]
            cells.append(f"P@1 {r['p@1']:.3f} R@3 {r['r@3']:.3f} (n={r['n_scored']})")
        print(f"{cname:10s} " + "  ".join(f"{c:>22s}" for c in cells))
    print("\nsaved ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
