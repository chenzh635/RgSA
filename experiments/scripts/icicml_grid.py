"""Overnight ICICML experiment grid driver (GT stays sealed).

Stage 1 (sweep): for each label fraction f in {0.1, 0.25, 0.5, 1.0}, train
Sparse BERT at seed 42 for every lambda in {1, 2, 3, 3.5} and pick the lambda
with the highest fixed-validation-set AUPRC (same pre-registered rule as the
CNN paper protocol).

Stage 2 (final): for each fraction, train Sparse BERT (selected lambda) and
plain BERT-ABMIL at seeds 123, 456, 789.

Fraction draws are stratified (label-balanced), fixed across both models and
all seeds, and dumped to experiments/icicml_runs/fraction_ids/. Completed
tags are skipped on resume.

Usage:
  python experiments/scripts/icicml_grid.py --phase both
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN_DIR = os.path.join(project_root, "experiments", "icicml_runs")
SPLIT = os.path.join(project_root, "experiments",
                     "configs", "val_split_indices.json")
FRACTIONS = [0.1, 0.25, 0.5, 1.0]
LAMBDAS = [1.0, 2.0, 3.0, 3.5]
SEEDS = [123, 456, 789]
DRAW_SEED = 20261013
TRAIN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "icicml_run_cached.py")


def frac_tag(f):
    return ("%g" % f).replace(".", "_")


def lam_tag(l):
    return ("%g" % l).replace(".", "_")


def build_fraction_ids():
    labels = np.load(os.path.join(project_root, "experiments", "cache",
                                  "icicml_bert", "train_meta.npz"))["labels"]
    split = json.load(open(SPLIT))
    pool = np.asarray(split["train_indices"])
    os.makedirs(os.path.join(RUN_DIR, "fraction_ids"), exist_ok=True)
    rng = np.random.default_rng(DRAW_SEED)
    paths = {}
    for f in FRACTIONS:
        n = int(round(f * len(pool))) if f < 1 else len(pool)
        pos = pool[labels[pool] == 1]
        neg = pool[labels[pool] == 0]
        n_pos = int(round(f * len(pos))) if f < 1 else len(pos)
        n_neg = n - n_pos
        take_pos = rng.permutation(pos)[:n_pos]
        take_neg = rng.permutation(neg)[:n_neg]
        ids = sorted(map(int, np.concatenate([take_pos, take_neg])))
        p = os.path.join(RUN_DIR, "fraction_ids", f"f{frac_tag(f)}.json")
        json.dump(ids, open(p, "w"))
        paths[f] = p
        print(f"fraction {f}: {len(ids)} train emails "
              f"({n_pos} pos / {n_neg} neg) -> {os.path.basename(p)}")
    return paths


def run_tag(tag, model, seed, lam, ids_path, select="val_auprc"):
    out = os.path.join(RUN_DIR, f"{tag}.json")
    if os.path.exists(out):
        print(f"[skip] {tag}", flush=True)
        return
    cmd = [sys.executable, TRAIN_SCRIPT, "--model", model,
           "--seed", str(seed), "--tag", tag,
           "--train-ids", ids_path, "--select", select]
    if model == "sparse":
        cmd += ["--lambda-rule", str(lam)]
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=project_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["sweep", "final", "both"],
                    default="both")
    args = ap.parse_args()
    os.makedirs(RUN_DIR, exist_ok=True)

    id_paths = build_fraction_ids()

    # ---------------- stage 1: lambda sweep at seed 42 ----------------
    if args.phase in ("sweep", "both"):
        for f in FRACTIONS:
            for lam in LAMBDAS:
                run_tag(f"sweep_f{frac_tag(f)}_lam{lam_tag(lam)}_seed42",
                        "sparse", 42, lam, id_paths[f])

        manifest = {"draw_seed": DRAW_SEED, "lambda_grid": LAMBDAS,
                    "fractions": {}}
        for f in FRACTIONS:
            rows = []
            for lam in LAMBDAS:
                r = json.load(open(os.path.join(
                    RUN_DIR, f"sweep_f{frac_tag(f)}_lam{lam_tag(lam)}_seed42.json")))
                rows.append({"lambda": lam,
                             "val_auprc": r["history"][r["best_epoch"]]["val"]["auprc"],
                             "val_acc": r["history"][r["best_epoch"]]["val"]["acc"],
                             "best_epoch": r["best_epoch"]})
            best = max(rows, key=lambda r: (r["val_auprc"], r["val_acc"]))
            manifest["fractions"][str(f)] = {"candidates": rows,
                                              "selected_lambda": best["lambda"]}
            print(f"f={f}: selected lambda={best['lambda']} "
                  f"(val AUPRC {best['val_auprc']:.4f})", flush=True)
        json.dump(manifest, open(os.path.join(RUN_DIR, "manifest.json"), "w"),
                  indent=1)

    # ---------------- stage 2: 3-seed finals ----------------
    if args.phase in ("final", "both"):
        manifest = json.load(open(os.path.join(RUN_DIR, "manifest.json")))
        for f in FRACTIONS:
            lam = manifest["fractions"][str(f)]["selected_lambda"]
            for seed in SEEDS:
                run_tag(f"final_f{frac_tag(f)}_sparse_seed{seed}",
                        "sparse", seed, lam, id_paths[f])
                run_tag(f"final_f{frac_tag(f)}_plain_seed{seed}",
                        "plain", seed, 0.0, id_paths[f])


if __name__ == "__main__":
    main()
