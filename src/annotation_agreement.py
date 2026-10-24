"""Inter-annotator agreement for phishing-sentence annotation.

Reads 2-3 annotation CSVs exported from annotate.html and computes
per-sentence agreement:

  - Fleiss' kappa over all raters (= Scott's pi for 2 raters)
  - pairwise Cohen's kappa + confusion matrices
  - raw agreement, per-stratum breakdown
  - disagreements.csv (items without unanimity) + agreement_report.json

CSV columns expected:
    email_id, stratum, sentence_idx, sentence_text, is_phishing_sentence, reason

Usage:
    python src/annotation_agreement.py ann_a.csv ann_b.csv [ann_c.csv] \
        --out experiments/human_annotation/agreement_pilot
    python src/annotation_agreement.py --selftest
"""

import argparse
import csv
import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

CATS = ["Y", "N", "?"]


# ----------------------------------------------------------------- loading
def load_annotations(path: Path) -> dict:
    """Return {(email_id, sentence_idx): {'label','stratum','text'}}.

    Accepts both annotate.html export formats:
    - CSV with columns email_id, sentence_idx, sentence_text, stratum,
      is_phishing_sentence, reason
    - JSON with top-level {"annotations": [{email_id, sentence_idx,
      sentence_text, stratum, label, reason}, ...]}
    Keys are normalized to (int email_id, int sentence_idx) so raters using
    different formats can be merged.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        return _load_json_annotations(path)
    ann = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("is_phishing_sentence") or "").strip()
            if label not in CATS:
                continue  # unlabeled row (partial export)
            key = (int(row["email_id"]), int(row["sentence_idx"]))
            ann[key] = {
                "label": label,
                "stratum": row.get("stratum", ""),
                "text": row.get("sentence_text", ""),
            }
    return ann


def _load_json_annotations(path: Path) -> dict:
    import json

    ann = {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("annotations", data if isinstance(data, list) else [])
    for row in rows:
        label = str(row.get("label") or "").strip()
        if label not in CATS:
            continue  # unlabeled row (partial export)
        key = (int(row["email_id"]), int(row["sentence_idx"]))
        ann[key] = {
            "label": label,
            "stratum": row.get("stratum", ""),
            "text": row.get("sentence_text", ""),
        }
    return ann


# ----------------------------------------------------------------- metrics
def fleiss_kappa(matrix: list) -> float:
    """matrix: list over items, each a dict cat -> rater count. n = raters."""
    n_items = len(matrix)
    if n_items == 0:
        return float("nan")
    n = sum(matrix[0].values())
    if n < 2:
        return float("nan")
    p_i = []
    cat_totals = Counter()
    for row in matrix:
        p_i.append((sum(c * c for c in row.values()) - n) / (n * (n - 1)))
        for cat, c in row.items():
            cat_totals[cat] += c
    p_bar = sum(p_i) / n_items
    p_e = sum((tot / (n_items * n)) ** 2 for tot in cat_totals.values())
    if p_e >= 1.0:
        return 1.0
    return (p_bar - p_e) / (1.0 - p_e)


def cohens_kappa(labels_a: list, labels_b: list) -> float:
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    ca, cb = Counter(labels_a), Counter(labels_b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in CATS)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def raw_agreement(labels_a: list, labels_b: list) -> float:
    if not labels_a:
        return float("nan")
    return sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a)


def confusion(labels_a: list, labels_b: list) -> dict:
    m = {f"{a}>{b}": 0 for a in CATS for b in CATS}
    for a, b in zip(labels_a, labels_b):
        m[f"{a}>{b}"] += 1
    return m


# ----------------------------------------------------------------- driver
def build_item_matrix(per_rater: list, keys: list) -> list:
    matrix = []
    for k in keys:
        row = Counter(r[k]["label"] for r in per_rater)
        matrix.append(row)
    return matrix


def analyze(per_rater: list, names: list) -> dict:
    keysets = [set(r.keys()) for r in per_rater]
    common = sorted(set.intersection(*keysets))
    n_dropped = len(keysets[0]) - len(common)

    report = {
        "raters": names,
        "n_raters": len(per_rater),
        "n_items": len(common),
        "items_dropped_not_common": n_dropped,
    }
    if len(common) == 0:
        report["error"] = "no common items"
        return report

    reference = per_rater[0]
    strata = sorted({reference[k]["stratum"] for k in common})

    # overall
    report["fleiss_kappa"] = fleiss_kappa(build_item_matrix(per_rater, common))
    label_counts = Counter(reference[k]["label"] for k in common)
    report["label_distribution_ref_rater"] = dict(label_counts)
    report["pairwise"] = {}
    for i, j in combinations(range(len(per_rater)), 2):
        la = [per_rater[i][k]["label"] for k in common]
        lb = [per_rater[j][k]["label"] for k in common]
        report["pairwise"][f"{names[i]}~{names[j]}"] = {
            "cohens_kappa": cohens_kappa(la, lb),
            "raw_agreement": raw_agreement(la, lb),
            "confusion": confusion(la, lb),
        }

    # per stratum
    report["by_stratum"] = {}
    for s in strata:
        ks = [k for k in common if reference[k]["stratum"] == s]
        if not ks:
            continue
        entry = {
            "n_items": len(ks),
            "fleiss_kappa": fleiss_kappa(build_item_matrix(per_rater, ks)),
        }
        if len(per_rater) == 2:
            la = [per_rater[0][k]["label"] for k in ks]
            lb = [per_rater[1][k]["label"] for k in ks]
            entry["raw_agreement"] = raw_agreement(la, lb)
        report["by_stratum"][s] = entry

    # disagreements: no unanimous label
    disag = []
    for k in common:
        labs = [r[k]["label"] for r in per_rater]
        if len(set(labs)) > 1:
            majority = Counter(labs).most_common(2)
            maj = majority[0][0] if (len(majority) == 1 or majority[0][1] > majority[1][1]) else "TIE"
            disag.append({
                "email_id": k[0], "sentence_idx": k[1],
                "stratum": reference[k]["stratum"],
                "text": reference[k]["text"],
                **{names[i]: labs[i] for i in range(len(per_rater))},
                "majority_or_TIE": maj,
            })
    report["n_disagreements"] = len(disag)
    report["_disagreements"] = disag
    return report


def write_outputs(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    disag = report.pop("_disagreements", [])
    with open(out_dir / "agreement_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    if disag:
        fields = ["email_id", "sentence_idx", "stratum", "text",
                  "majority_or_TIE"] + report["raters"]
        with open(out_dir / "disagreements.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for d in disag:
                w.writerow({k: d.get(k, "") for k in fields})
    report["_disagreements"] = disag  # restore for console printing


def print_report(report: dict) -> None:
    print(f"raters: {report['raters']}")
    print(f"items: {report['n_items']} (dropped, not common: "
          f"{report['items_dropped_not_common']})")
    print(f"Fleiss kappa (all raters): {report.get('fleiss_kappa', float('nan')):.3f}")
    for pair, d in report.get("pairwise", {}).items():
        print(f"  {pair}: kappa={d['cohens_kappa']:.3f}  "
              f"raw agreement={d['raw_agreement']:.3f}")
    print("by stratum:")
    for s, d in report.get("by_stratum", {}).items():
        extra = f"  raw={d['raw_agreement']:.3f}" if "raw_agreement" in d else ""
        print(f"  {s}: n={d['n_items']}  kappa={d['fleiss_kappa']:.3f}{extra}")
    print(f"disagreements: {report['n_disagreements']}")
    print("label distribution (rater 1):", report.get("label_distribution_ref_rater"))


# ----------------------------------------------------------------- selftest
def selftest() -> None:
    # 1. perfect agreement -> kappa == 1
    keys = [(i, 0) for i in range(20)]
    rater = {k: {"label": "Y", "stratum": "s", "text": "t"} for k in keys}
    rep = analyze([rater, dict(rater), dict(rater)], ["a", "b", "c"])
    assert abs(rep["fleiss_kappa"] - 1.0) < 1e-9, rep["fleiss_kappa"]
    print("[PASS] perfect agreement -> kappa = 1.0")

    # 2. random labels -> kappa near 0
    rng = random.Random(0)
    r1, r2, r3 = {}, {}, {}
    for i in range(600):
        k = (i, 0)
        for r in (r1, r2, r3):
            r[k] = {"label": rng.choice(CATS), "stratum": "s", "text": "t"}
    rep = analyze([r1, r2, r3], ["a", "b", "c"])
    assert abs(rep["fleiss_kappa"]) < 0.15, rep["fleiss_kappa"]
    print(f"[PASS] random labels -> kappa = {rep['fleiss_kappa']:.3f} (near 0)")

    # 3. two raters: Fleiss (= Scott's pi, pooled marginals) ~= Cohen
    #    (separate marginals) — nearly equal, not identical by definition.
    rng = random.Random(1)
    ra, rb = {}, {}
    for i in range(300):
        k = (i, 0)
        base = rng.choice(CATS)
        ra[k] = {"label": base, "stratum": "s", "text": "t"}
        rb[k] = {"label": base if rng.random() < 0.8 else rng.choice(CATS),
                 "stratum": "s", "text": "t"}
    rep = analyze([ra, rb], ["a", "b"])
    fleiss = rep["fleiss_kappa"]
    cohen = rep["pairwise"]["a~b"]["cohens_kappa"]
    assert abs(fleiss - cohen) < 0.02, (fleiss, cohen)
    print(f"[PASS] 2-rater Fleiss (Scott's pi) ~= Cohen: "
          f"{fleiss:.4f} vs {cohen:.4f}")
    print("ALL SELFTESTS PASSED")


# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csvs", nargs="*", help="annotation CSVs (2-3 raters)")
    ap.add_argument("--names", nargs="*",
                    help="rater names (default: file stems)")
    ap.add_argument("--out", default="experiments/human_annotation/agreement",
                    help="output directory")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not (2 <= len(args.csvs) <= 3):
        ap.error("need 2-3 annotation CSVs")
    names = args.names or [Path(c).stem for c in args.csvs]
    per_rater = [load_annotations(Path(c)) for c in args.csvs]
    for name, r in zip(names, per_rater):
        print(f"loaded {name}: {len(r)} labeled sentences")
    report = analyze(per_rater, names)
    write_outputs(report, Path(args.out))
    print_report(report)
    print(f"wrote {Path(args.out) / 'agreement_report.json'} and disagreements.csv")


if __name__ == "__main__":
    main()
