"""Select the 30-email pilot batch for human annotation.

Stratified random sample from the 150-email annotation pool:
    low_spearman   10   (circular metric masks problems here — oversampled)
    misclassified   5
    mid_spearman    8
    high_spearman   7

Outputs two files in experiments/human_annotation/:
  - pilot_30.json   : BLINDED data for annotators (no attention / rule scores,
                      no pred_prob / correct / spearman — nothing that could
                      anchor judgment). Consumed by annotate.html.
  - pilot_manifest.csv : internal record (email_idx, stratum, spearman,
                      entropy, n_sentences) — do NOT send to annotators.
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANN_DIR = ROOT / "experiments" / "human_annotation"

ALLOCATION = {
    "low_spearman": 10,
    "misclassified": 5,
    "mid_spearman": 8,
    "high_spearman": 7,
}
SEED = 0


def main() -> None:
    with open(ANN_DIR / "annotation_data.json", encoding="utf-8") as f:
        data = json.load(f)
    emails = data["emails"]

    rng = random.Random(SEED)
    by_stratum: dict[str, list] = {s: [] for s in ALLOCATION}
    for e in emails:
        if e["stratum"] in by_stratum:
            by_stratum[e["stratum"]].append(e)

    picked = []
    for stratum, k in ALLOCATION.items():
        pool = sorted(by_stratum[stratum], key=lambda e: e["email_idx"])
        assert len(pool) >= k, f"not enough emails in {stratum}: {len(pool)} < {k}"
        picked.extend(rng.sample(pool, k))

    # Blinded payload: strip every model-derived or label-derived field.
    blinded = []
    for e in picked:
        blinded.append({
            "email_idx": e["email_idx"],
            "stratum": e["stratum"],
            "n_sentences": e["n_sentences"],
            "sentences": [
                {"idx": s["idx"], "text": s["text"]}
                for s in sorted(e["sentences"], key=lambda s: s["idx"])
            ],
        })
    blinded.sort(key=lambda e: e["email_idx"])

    pilot_path = ANN_DIR / "pilot_30.json"
    with open(pilot_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "batch": "pilot_30",
                "n_emails": len(blinded),
                "allocation": ALLOCATION,
                "seed": SEED,
                "blinded": True,
                "note": "No model_attention / rule_score / spearman / "
                        "pred_prob / correct by design.",
            },
            "emails": blinded,
        }, f, ensure_ascii=False, indent=1)

    manifest_path = ANN_DIR / "pilot_manifest.csv"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("email_idx,stratum,spearman,entropy,n_sentences\n")
        for e in sorted(picked, key=lambda e: e["email_idx"]):
            sp = "NA" if e["spearman"] is None else f"{e['spearman']:.4f}"
            ent = "NA" if e["entropy"] is None else f"{e['entropy']:.4f}"
            f.write(f"{e['email_idx']},{e['stratum']},"
                    f"{sp},{ent},{e['n_sentences']}\n")

    total_sents = sum(e["n_sentences"] for e in blinded)
    print(f"wrote {pilot_path} ({len(blinded)} emails, {total_sents} sentences)")
    print(f"wrote {manifest_path}")
    counts = {}
    for e in blinded:
        counts[e["stratum"]] = counts.get(e["stratum"], 0) + 1
    print("allocation check:", counts)


if __name__ == "__main__":
    main()
