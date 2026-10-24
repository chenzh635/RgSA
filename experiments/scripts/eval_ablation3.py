"""Evaluate the lambda_rule=3.0 ablation reruns + ruleflip20 (review S5/B7).

Usage: python3 experiments/scripts/eval_ablation3.py [name ...]
(default: ablation3_no_rule ablation3_no_l1 ablation3_no_temp ruleflip20)

Loads experiments/{name}/sparse_best_model.pt, runs src.evaluate.evaluate on
the test set (MPS), merges into experiments/ablation3_results.json.
"""

import json
import os
import sys

import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'experiments/scripts'))

from eval_faithfulness import build_model, load_config, load_state
from src.dataset import create_dataloaders
from src.evaluate import evaluate

DEFAULT_NAMES = ['ablation3_no_rule', 'ablation3_no_l1', 'ablation3_no_temp', 'ruleflip20']


def main():
    names = sys.argv[1:] or DEFAULT_NAMES
    config = load_config()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    _, test_loader = create_dataloaders(config)

    out_path = os.path.join(project_root, 'experiments/ablation3_results.json')
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)

    for name in names:
        ckpt_path = os.path.join(project_root, 'experiments', name, 'sparse_best_model.pt')
        if not os.path.exists(ckpt_path):
            print(f'SKIP {name}: no checkpoint at {ckpt_path}')
            continue
        model = build_model('cnn_sparse')
        ck = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(load_state(ckpt_path), strict=False)
        model.to(device)
        model.eval()
        r = evaluate(model, test_loader, config, device)
        entry = {k: float(v) for k, v in r.items()}
        entry['_epoch'] = ck.get('epoch')
        entry['_ckpt_test_acc'] = float(ck.get('test_acc', float('nan')))
        results[name] = entry
        print(f'\n== {name} ==')
        print(json.dumps(entry, indent=1))
        del model

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=1)
    print(f'\nsaved -> {out_path}')


if __name__ == '__main__':
    main()
