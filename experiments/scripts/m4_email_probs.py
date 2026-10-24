"""M4 helper: fill email probabilities for the 56 no-gold-Y emails.

Runs inference with the sparse seed123 checkpoint from the 2026-08-23 backup
(checkpoints were deleted from the working tree) and writes
experiments/m4_review/m4_email_probs.json {email_id: p(phishing)}.

MPS -> run with sandbox disabled.
"""
import json
import os
import sys

import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_faithfulness import build_model, load_config, load_state
from src.dataset import create_dataloaders

BACKUP_CKPT = os.environ.get("SPARSE_CKPT",
    "experiments/multi_seed_sparse_seed123/sparse_best_model.pt")  # checkpoints not shipped; set SPARSE_CKPT to your copy
OUT = os.path.join(project_root, "experiments/m4_review/m4_email_probs.json")
DATA = os.path.join(project_root, "experiments/m4_review/m4_data.json")


def main():
    with open(DATA) as f:
        wanted = {e["email_id"] for e in json.load(f)}

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)
    model = build_model("cnn_sparse")
    model.load_state_dict(load_state(BACKUP_CKPT), strict=False)
    model.to(device).eval()

    _, test_loader = create_dataloaders(load_config())
    probs = {}
    remaining = set(wanted)
    with torch.no_grad():
        for batch in test_loader:
            sent_lengths = batch["sent_lengths"].to(device)
            out, _, _, _ = model(batch["tokens"].to(device), sent_lengths,
                                 temperature=0.1)
            p = torch.sigmoid(out).cpu().numpy()
            for b, eid in enumerate(batch["idx"]):
                if eid in remaining:
                    probs[eid] = float(p[b])
                    remaining.discard(eid)
            if not remaining:
                break
    with open(OUT, "w") as f:
        json.dump(probs, f)
    print(f"saved {len(probs)} probs -> {OUT}")


if __name__ == "__main__":
    main()
