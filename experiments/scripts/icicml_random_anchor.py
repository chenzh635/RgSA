"""R6 (frozen protocol 2026-10-22): correct random-ranking anchor for the
202 gold-positive terminal bags.

The paper table previously carried one fixed random permutation
(P@1 0.490, +1.77 SE lucky). This script replaces it with (1) analytic
hypergeometric expectations and (2) a Monte Carlo check with 10,000 fresh
random permutations per email, both macro-averaged over the same 202
gold-positive emails the models are scored on.

Read-only on gold_final.csv. Selects nothing. Output:
experiments/icicml_runs/random_anchor.json
"""

import json
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.evaluate_sentence_localization import load_gold, gold_sets  # noqa: E402

GOLD = os.path.join(project_root,
                    "experiments/agent_annotation/exports/gold_final.csv")
OUT = os.path.join(project_root,
                   "experiments/icicml_runs/random_anchor.json")
MC_DRAWS = 10000
MC_SEED = 20261022
KS = (1, 3, 5)
RECALL_TARGET = 0.9


def analytic_email(n, g):
    """Expected P@k/R@k and k@90 for a uniform random permutation.

    The paper's P@k is hits/kk (fraction of the top-k that is gold), not an
    indicator: E[hits] = kk*g/n, hence E[P@k] = E[R@k]*g... = g/n, and
    E[R@k] = kk/n (kk = min(k, n)).
    """
    e = {"n": n, "g": g}
    for k in KS:
        kk = min(k, n)
        e[f"p@{k}"] = g / n
        e[f"r@{k}"] = kk / n
    # position of the q-th gold in a random permutation: q*(n+1)/(g+1)
    q = int(np.ceil(RECALL_TARGET * g))
    e["k@90"] = q * (n + 1) / (g + 1)
    # mean rank of gold sentences: each gold rank is uniform on 1..n
    e["mean_gold_rank"] = (n + 1) / 2
    return e


def mc_email(rng, n, g, b):
    """Return (b,) arrays of per-draw P@1/P@3/P@5/R@3/R@5/k@90 means."""
    draws = rng.random(size=(b, n))
    order = np.argsort(-draws, axis=1, kind="stable")
    gold = np.zeros(n, dtype=bool)
    gold[:g] = True  # identity of gold sentences is exchangeable
    res = {}
    for k in KS:
        kk = min(k, n)
        top = order[:, :kk]
        hits = gold[top].sum(axis=1)
        res[f"p@{k}"] = hits / kk
        res[f"r@{k}"] = hits / g
    q = int(np.ceil(RECALL_TARGET * g))
    ranks = np.empty(n, dtype=int)
    k90 = np.empty(b)
    mrank = np.empty(b)
    for i in range(b):
        ranks[order[i]] = np.arange(n)
        gold_ranks = np.sort(ranks[gold])
        k90[i] = gold_ranks[q - 1] + 1
        mrank[i] = (gold_ranks + 1).mean()
    res["k@90"] = k90
    res["mean_gold_rank"] = mrank
    return res


def main():
    gold = load_gold([GOLD])
    gsets = gold_sets(gold["labels"], mode="strict")
    bags = sorted((eid, len(s["candidates"]), len(s["gold_y"]))
                  for eid, s in gsets.items() if s["gold_y"])
    n_bags = len(bags)
    total_sent = sum(n for _, n, _ in bags)
    total_gold = sum(g for _, _, g in bags)
    print(f"gold-positive bags: {n_bags}, sentences: {total_sent}, "
          f"gold: {total_gold}, density: {total_gold/total_sent:.3f}")

    # ---- analytic, macro-averaged over bags ----
    per = [analytic_email(n, g) for _, n, g in bags]
    keys = [f"p@{k}" for k in KS] + [f"r@{k}" for k in KS] + ["k@90",
            "mean_gold_rank"]
    analytic = {k: float(np.mean([e[k] for e in per])) for k in keys}

    # ---- Monte Carlo: macro mean within each draw, CI over draws ----
    rng = np.random.default_rng(MC_SEED)
    acc = {}
    for _, n, g in bags:
        r = mc_email(rng, n, g, MC_DRAWS)
        for k, v in r.items():
            acc.setdefault(k, []).append(v)
    macro = {k: np.stack(v, axis=1).mean(axis=1) for k, v in acc.items()}
    mc = {}
    for k, arr in macro.items():
        mc[k] = {"mean": float(arr.mean()),
                 "ci95": [float(np.percentile(arr, 2.5)),
                          float(np.percentile(arr, 97.5))]}

    agree = {k: abs(analytic[k] - mc[k]["mean"]) for k in mc}
    ok = all(v < 0.002 for v in agree.values())
    print(f"\n{'metric':6s} {'analytic':>10s} {'MC mean':>10s} {'|diff|':>8s} "
          f"MC 95% interval")
    for k in ("p@1", "p@3", "p@5", "r@3", "r@5", "k@90", "mean_gold_rank"):
        print(f"{k:6s} {analytic[k]:10.4f} {mc[k]['mean']:10.4f} "
              f"{agree[k]:8.4f}  [{mc[k]['ci95'][0]:.4f},"
              f"{mc[k]['ci95'][1]:.4f}]")
    print("\nanalytic/MC agreement <=0.002:", ok)

    out = {"protocol": "ABLATION_PROTOCOL_2026-10-22.md#C",
           "mc_draws": MC_DRAWS, "mc_seed": MC_SEED, "mode": "strict",
           "gold_positive_bags": n_bags,
           "positive_bag_sentences": total_sent, "gold_sentences": total_gold,
           "density": total_gold / total_sent,
           "analytic": analytic, "monte_carlo": mc,
           "max_analytic_minus_mc": max(agree.values()),
           "agreement_ok": bool(ok)}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("saved ->", OUT)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
