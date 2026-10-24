"""R4 archival reconstruction (2026-10-23): rebuild the round-3 (2026-09-13,
protocol v1.3) ensemble-vs-human kappa snapshot cited in the paper.

Reviewer report R4 noted that the paper's kappa = 0.381/0.380 came from an
intermediate calibration snapshot that was not archived, while recomputation
against the production export gold_final.csv gives different values (0.434/
0.439, a different ensemble). This script freezes the actual round-3
snapshot inputs and recomputes the published statistics:

  * majority-vote ensemble of the three v1.3 blind pilot passes
    (agent_pass_{a,b,c}_pilot_30_blind.json) over the 213 pilot sentences;
  * Cohen's kappa of that ensemble against each of the two human pilots
    (chan, li), the human-human ceiling kappa, and Y coverages;
  * paired bootstrap (B=10,000, resampling sentences) of the gap between
    mean ensemble-human kappa and the human-human ceiling.

Read-only with respect to every annotation file; writes one JSON next to
calibration_report.json. Run:
    python3 experiments/agent_annotation/scripts/reconstruct_ensemble_human_kappa.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.annotation_agreement import analyze, cohens_kappa, load_annotations  # noqa: E402

HA = ROOT / "experiments" / "human_annotation"
AA = HERE.parent
HUMAN_EXPORTS = [
    HA / "exports" / "annotations_chan_pilot_30_2026-09-01-15-13.json",
    HA / "exports" / "annotations_li_pilot_30_2026-09-06-14-45.json",
]
PASS_EXPORTS = [AA / "outputs" / f"agent_pass_{p}_pilot_30_blind.json"
                for p in ("a", "b", "c")]
OUT = AA / "outputs" / "ensemble_human_kappa_round3_reconstruction.json"
BOOT_B = 10_000
BOOT_SEED = 20260913
CATS = ("Y", "N", "?")


def main():
    humans = [load_annotations(p) for p in HUMAN_EXPORTS]
    passes = [load_annotations(p) for p in PASS_EXPORTS]

    keys = sorted(set(humans[0]) & set(humans[1])
                  & set(passes[0]) & set(passes[1]) & set(passes[2]))
    n = len(keys)
    chan = [humans[0][k]["label"] for k in keys]
    li = [humans[1][k]["label"] for k in keys]
    pass_labels = [[p[k]["label"] for k in keys] for p in passes]

    # majority vote of the three blind passes (213 sentences: 192 unanimous,
    # 21 split 2-1, 0 without a majority -- recorded in CALIBRATION_LOG.md)
    ens, vote_counts = [], Counter()
    for i in range(n):
        c = Counter(pass_labels[j][i] for j in range(3))
        (lab, cnt), = c.most_common(1)
        ens.append(lab)
        vote_counts[cnt] += 1

    k_ens_chan = cohens_kappa(ens, chan)
    k_ens_li = cohens_kappa(ens, li)
    k_chan_li = cohens_kappa(chan, li)
    gap = (k_ens_chan + k_ens_li) / 2 - k_chan_li

    def y_cov(labels):
        return sum(1 for x in labels if x == "Y") / len(labels)

    rng = np.random.default_rng(BOOT_SEED)
    arr = {"ens": ens, "chan": chan, "li": li}
    gaps, k_ec, k_el = [], [], []
    idx_all = np.arange(n)
    for _ in range(BOOT_B):
        idx = rng.choice(idx_all, size=n, replace=True)
        ec = cohens_kappa([arr["ens"][i] for i in idx],
                          [arr["chan"][i] for i in idx])
        el = cohens_kappa([arr["ens"][i] for i in idx],
                          [arr["li"][i] for i in idx])
        cl = cohens_kappa([arr["chan"][i] for i in idx],
                          [arr["li"][i] for i in idx])
        if np.isnan(ec) or np.isnan(el) or np.isnan(cl):
            continue
        k_ec.append(ec)
        k_el.append(el)
        gaps.append((ec + el) / 2 - cl)
    gaps = np.asarray(gaps)

    # pairwise inter-pass kappas (provenance cross-check vs report)
    pair = analyze(passes, ["a", "b", "c"])["pairwise"]

    out = {
        "snapshot": "round-3 protocol v1.3 pilot majority-vote ensemble "
                    "(CALIBRATION_LOG.md 2026-09-13)",
        "inputs": {
            "human_chan": str(HUMAN_EXPORTS[0].relative_to(ROOT)),
            "human_li": str(HUMAN_EXPORTS[1].relative_to(ROOT)),
            "passes": [str(p.relative_to(ROOT)) for p in PASS_EXPORTS],
        },
        "n_sentences": n,
        "vote_distribution": {f"{k}_of_3": v for k, v in
                              sorted(vote_counts.items(), reverse=True)},
        "kappa_ensemble_vs_chan": k_ens_chan,
        "kappa_ensemble_vs_li": k_ens_li,
        "kappa_mean_ensemble_human": (k_ens_chan + k_ens_li) / 2,
        "kappa_chan_li_ceiling": k_chan_li,
        "gap_vs_ceiling": gap,
        "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED,
                      "valid_resamples": int(gaps.size),
                      "gap_ci95": [float(np.percentile(gaps, 2.5)),
                                   float(np.percentile(gaps, 97.5))],
                      "p_ensemble_ge_ceiling": float((gaps >= 0).mean())},
        "coverage_Y": {"ensemble": y_cov(ens), "chan": y_cov(chan),
                       "li": y_cov(li)},
        "interpass_kappa_pairwise": pair,
        "paper_values_check": {
            "paper_kappa_vs_chan": 0.381,
            "paper_kappa_vs_li": 0.380,
            "paper_ceiling": 0.398,
            "paper_gap_ci95": [-0.12, 0.16]},
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)

    print(f"n={n} vote dist {dict(vote_counts)}")
    print(f"kappa ens~chan {k_ens_chan:.3f}  ens~li {k_ens_li:.3f}  "
          f"chan~li {k_chan_li:.3f}  gap {gap:+.3f}")
    print("gap bootstrap CI95 "
          f"[{np.percentile(gaps,2.5):+.2f},{np.percentile(gaps,97.5):+.2f}]"
          f"  P(ens>=ceiling) {(gaps>=0).mean():.2f}")
    print(f"Y coverage ens {y_cov(ens):.3f} chan {y_cov(chan):.3f} "
          f"li {y_cov(li):.3f}")
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
