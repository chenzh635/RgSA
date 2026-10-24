"""Quick check: accuracy of a faithful-port checkpoint on the full 8k test set
(the distribution the reference implementation validates/reports on)."""
import os
import sys

import torch
from torch.utils.data import DataLoader

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.baselines.ai2tale_faithful import AI2TALEFaithful
from src.dataset import PhishingEmailDataset
from m1_run_faithful import _collate

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "experiments/m1_faithful_seed123/ai2tale_best_model.pt"

test_mat = os.path.join(project_root, "phishingdata/x_y_te_m.mat")
vocab = os.path.join(project_root, "phishingdata/ind_word.pickle")
ds = PhishingEmailDataset(test_mat, vocab)
loader = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=_collate)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = AI2TALEFaithful(vocab_size=311774, k=1, tau=0.5).to(device)
state = torch.load(os.path.join(project_root, CKPT), map_location=device,
                   weights_only=False)
model.load_state_dict(state["model_state_dict"], strict=False)
model.eval()

correct = seen = 0
with torch.no_grad():
    for batch in loader:
        logit, _, _ = model(batch["tokens"].to(device),
                            batch["sent_lengths"].to(device))
        preds = (logit > 0).long().cpu()
        labels = batch["label"]
        correct += (preds == labels).sum().item()
        seen += labels.shape[0]
print(f"test-set acc: {correct/seen:.4f} ({correct}/{seen}); ckpt epoch {state.get('epoch')}")
