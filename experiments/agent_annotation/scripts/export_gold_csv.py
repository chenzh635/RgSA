"""Bridge: gold_final.json (fallback gold_majority.json) -> CSV consumable by
src.evaluate_sentence_localization.load_gold.

Stratum for --by-stratum comes from the coordinator sidecar tasks/_key_stratum.json
(spearman strata for prod150; supplement emails get 'supplement').
Also writes a failure_mode-stratified variant (stratum field inside gold annotations).
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repository root
BASE = ROOT / "experiments/agent_annotation"
OUT = BASE / "exports"
OUT.mkdir(exist_ok=True)

final = BASE / "outputs/gold_final.json"
src_path = final if final.exists() else BASE / "outputs/gold_majority.json"
gold = json.loads(src_path.read_text())
side = json.loads((BASE / "tasks/_key_stratum.json").read_text())
print("source:", src_path.name, "|", len(gold["annotations"]), "annotations")

def stratum_for(eid):
    v = side.get(str(eid), {})
    return v.get("stratum") or "supplement"

HEADER = ["email_id", "stratum", "sentence_idx", "sentence_text",
          "is_phishing_sentence", "reason"]

def write_csv(path, stratum_fn):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for a in sorted(gold["annotations"], key=lambda x: (x["email_id"], x["sentence_idx"])):
            w.writerow([a["email_id"], stratum_fn(a), a["sentence_idx"],
                        a["sentence_text"], a["label"], a.get("reason", "")])
    print("written", path)

stem = "gold_final" if src_path is final else "gold_majority"
csv_path = OUT / f"{stem}.csv"
write_csv(csv_path, lambda a: stratum_for(a["email_id"]))
write_csv(OUT / f"{stem}_by_failure_mode.csv", lambda a: a["stratum"])

# round-trip verification through the official loader
sys.path.insert(0, str(ROOT))
from src.evaluate_sentence_localization import load_gold  # noqa: E402

g = load_gold([str(csv_path)])
lab = Counter(g["labels"].values())
print("load_gold:", g["n_items"], "items |", g["n_raters"], "rater |", dict(lab))
exp = Counter(a["label"] for a in gold["annotations"])
assert g["n_items"] == len(gold["annotations"]) and g["n_raters"] == 1
assert lab == exp, (lab, exp)
strata = Counter(g["strata"].values())
print("strata per email:", dict(strata), "| emails:", len(g["strata"]))
assert sum(strata.values()) == gold.get("n_emails", 258)
print("ALL CHECKS PASSED")
