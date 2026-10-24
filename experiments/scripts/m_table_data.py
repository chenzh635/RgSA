"""Unified main-table data builder (M1/M7 consolidation).

Replaces every localization number in Table 'main_results' and regenerates
the paired-bootstrap markers vs Ours from per-email GT score caches:

  - experiments/gt_scores.json            (old caches: Random, ABMIL seeds,
                                           BERT-ABMIL, Sparse BERT, Snorkel,
                                           Rules-only; broken AI2TALE and old
                                           Ours rows are NOT used)
  - experiments/m1_faithful_gt_scores.json (faithful AI2TALE, 3 seeds)
  - experiments/final_gt_eval.json         (final Ours, 3 seeds; produced by
                                           m_sparse_gt_eval.py --tag final)

Output: experiments/main_table_data.json
  - per-row localization: P@1/P@3/P@5, R@3/R@5, k@90, mRank (strict, 202
    gold-positive emails), seed mean+/-std where applicable;
  - paired per-email bootstrap (5,000 resamples) of each row vs the
    seed-averaged Ours score map, detectable flag per metric;
  - stratum-specific P@1 (Section 'Where the proxy fails' numbers).

Pure numpy/scoring — no MPS needed.
"""
import json
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.evaluate_sentence_localization import (gold_sets, email_metrics,
                                                  summarize)

GOLD_CSV = os.path.join(project_root,
                        "experiments/agent_annotation/exports/gold_final.csv")
OLD = os.path.join(project_root, "experiments/gt_scores.json")
FAITH = os.path.join(project_root, "experiments/m1_faithful_gt_scores.json")
FINAL = os.path.join(project_root, "experiments/final_gt_eval.json")
OUT = os.path.join(project_root, "experiments/main_table_data.json")

KS = (1, 3, 5)
LOC_KEYS = ["p@1", "p@3", "p@5", "r@3", "r@5", "k@90", "mean_gold_rank"]
BOOT_B = 5000
BOOT_SEED = 42


def per_email_from_scores(score_maps, gsets):
    """score_maps: list of {eid: weights}. Returns per-email metrics of the
    per-position averaged score map, plus each constituent map's metrics."""
    eids = sorted(gsets)
    n_maps = len(score_maps)
    cand_maps = [{} for _ in range(n_maps)]
    avg_map = {}
    for eid in eids:
        g = gsets[eid]
        vectors = []
        for sm in score_maps:
            w_full = np.asarray(sm[eid], dtype=float)
            w = np.zeros(len(g["candidates"]))
            for j, s in enumerate(g["candidates"]):
                if s < len(w_full):
                    w[j] = w_full[s]
            vectors.append(w)
        for i in range(n_maps):
            cand_maps[i][eid] = vectors[i]
        avg_map[eid] = np.mean(vectors, axis=0)

    def metrics_of(mp):
        pe = {}
        for eid in eids:
            g = gsets[eid]
            gold_re = {j for j, s in enumerate(g["candidates"])
                       if s in g["gold_y"]}
            pe[eid] = email_metrics(mp[eid], gold_re, KS)
        return pe
    return metrics_of(avg_map), [metrics_of(cm) for cm in cand_maps]


def loc_summary(per_email):
    s = summarize(per_email, KS)
    return {k: s[k] for k in LOC_KEYS}


def seed_mean_std(per_email_list):
    summaries = [loc_summary(pe) for pe in per_email_list]
    out = {}
    for k in LOC_KEYS:
        vals = [s[k] for s in summaries]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                  "values": [float(v) for v in vals]}
    return out


def paired_bootstrap(pe_row, pe_ref):
    """5,000 paired per-email bootstrap differences row-minus-ref for each
    metric. Returns mean, 95% CI, detectable."""
    eids = sorted(e for e in pe_row if pe_row[e].get("n_gold", 0) > 0
                  and e in pe_ref and pe_ref[e].get("n_gold", 0) > 0)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(eids), size=(BOOT_B, len(eids)))
    out = {}
    for k in LOC_KEYS:
        a = np.array([pe_row[e][k] for e in eids])
        b = np.array([pe_ref[e][k] for e in eids])
        d = a - b
        boots = d[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[k] = {"mean": float(d.mean()), "ci": [float(lo), float(hi)],
                  "detectable": bool(lo > 0 or hi < 0), "n": len(eids)}
    return out


def main():
    from src.evaluate_sentence_localization import load_gold
    gold = load_gold([GOLD_CSV])
    gsets = gold_sets(gold["labels"], mode="strict")

    old = json.load(open(OLD))
    faith = json.load(open(FAITH))
    final = json.load(open(FINAL))["gt_scores"]

    # score-map groups
    def old_maps(prefix):
        return [{int(e): v for e, v in old[k].items()}
                for k in old if k.startswith(prefix)]

    rows = {}

    def build_row(name, maps_or_key, multi=True):
        if isinstance(maps_or_key, str):
            maps = old_maps(maps_or_key)
        else:
            maps = maps_or_key
        pe_ref_row, pe_parts = per_email_from_scores(maps, gsets)
        entry = {"per_email_avg": loc_summary(pe_ref_row), "_pe": pe_ref_row}
        if multi and len(pe_parts) > 1:
            entry["seeds"] = seed_mean_std(pe_parts)
        rows[name] = entry
        return entry

    build_row("Random", "Random", multi=False)
    build_row("ABMIL (CNN)", "ABMIL (CNN)")
    # faithful AI2TALE
    faith_maps = [[{int(e): v for e, v in
                    faith[k]["gt_scores"].items()} for k in faith]]
    build_row("AI2TALE", faith_maps[0])
    build_row("BERT-ABMIL", "BERT-ABMIL", multi=False)
    build_row("Sparse BERT-ABMIL", "Sparse BERT-ABMIL", multi=False)
    build_row("Rules-only", "Rules-only", multi=False)
    build_row("Snorkel", "Snorkel", multi=False)
    final_maps = [{int(e): v for e, v in final[k].items()} for k in final]
    build_row("Sparse ABMIL (Ours)", final_maps)

    pe_ours = rows["Sparse ABMIL (Ours)"]["_pe"]
    for name, entry in rows.items():
        entry["vs_ours"] = paired_bootstrap(entry["_pe"], pe_ours)

    # stratum-specific P@1 (150 spearman-sampled emails: high/mid/low,
    # misclassified, supplement), using the gold stratum labels
    strata = gold["strata"]
    stratum_rows = {}
    for name, entry in rows.items():
        pe = entry["_pe"]
        per_stratum = {}
        for st in sorted(set(strata.values())):
            eids = [e for e in pe if strata.get(e) == st
                    and pe[e].get("n_gold", 0) > 0]
            if eids:
                per_stratum[st] = {
                    "n": len(eids),
                    "p@1": float(np.mean([pe[e]["p@1"] for e in eids])),
                    "r@3": float(np.mean([pe[e]["r@3"] for e in eids])),
                }
        stratum_rows[name] = per_stratum

    # assemble clean output (drop per-email detail)
    clean = {}
    for name, entry in rows.items():
        clean[name] = {"loc_avg": entry["per_email_avg"],
                       "vs_ours": entry["vs_ours"]}
        if "seeds" in entry:
            clean[name]["loc_seeds"] = entry["seeds"]
    payload = {"ks": list(KS), "n_boot": BOOT_B, "rows": clean,
               "strata": stratum_rows}
    json.dump(payload, open(OUT, "w"), indent=1)

    # print compact table with markers
    hdr = f"{'row':<22}{'P@1':>16}{'P@5':>9}{'R@3':>9}{'R@5':>9}{'k@90':>8}{'mRank':>8}"
    print(hdr)
    for name, c in clean.items():
        if "loc_seeds" in c:
            s = c["loc_seeds"]
            p1 = f"{s['p@1']['mean']:.3f}±{s['p@1']['std']:.3f}"
            p5 = f"{s['p@5']['mean']:.3f}"
            r3 = f"{s['r@3']['mean']:.3f}±{s['r@3']['std']:.3f}"
            r5 = f"{s['r@5']['mean']:.3f}"
            k9 = f"{s['k@90']['mean']:.2f}"
            mr = f"{s['mean_gold_rank']['mean']:.2f}"
        else:
            s = c["loc_avg"]
            p1, p5, r3, r5, k9, mr = (f"{s['p@1']:.3f}", f"{s['p@5']:.3f}",
                                      f"{s['r@3']:.3f}", f"{s['r@5']:.3f}",
                                      f"{s['k@90']:.2f}",
                                      f"{s['mean_gold_rank']:.2f}")
        mark = "*" if c["vs_ours"]["p@1"]["detectable"] else " "
        print(f"{name:<22}{p1:>15}{mark}{p5:>9}{r3:>9}{r5:>9}{k9:>8}{mr:>8}")
    print("\nsaved ->", OUT)


if __name__ == "__main__":
    main()
