"""Verify restored checkpoints: reload and recompute headline metrics.

Expected values (from multi_seed_summary.json / individual results.json):
  multi_seed_sparse_seed123: spearman 0.7251, f1 0.9827, auprc 0.9982
  multi_seed_sparse_seed456: spearman 0.7001, f1 0.9839, auprc 0.9987
  multi_seed_sparse_seed789: spearman 0.7147, f1 0.9817, auprc 0.9980
  archive/sparse_best_model: spearman ~0.459 (old lambda_rule=0.5 model)
"""
import os
import sys

import torch
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.dataset import create_dataloaders
from src.evaluate import compute_email_auprc, compute_email_f1, compute_spearman_correlation
from src.models.abmil import ABMILBaseline, SparseABMIL

device = torch.device('cpu')


def load_config():
    with open(os.path.join(project_root, 'experiments/configs/defaults.yaml')) as f:
        return yaml.safe_load(f)


def load_model(ckpt_path, use_sparse):
    if use_sparse:
        model = SparseABMIL(hidden_dim=64, attention_hidden=64,
                            classifier_hidden=[128, 64], use_sparse=True)
    else:
        model = ABMILBaseline(hidden_dim=64, attention_hidden=64,
                              classifier_hidden=[128, 64])
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    config = load_config()
    _, test_loader = create_dataloaders(config)
    print(f"Test emails: {len(test_loader.dataset)}")

    targets = [
        ('multi_seed_sparse_seed123/sparse_best_model.pt', True, 0.7251),
        ('multi_seed_sparse_seed456/sparse_best_model.pt', True, 0.7001),
        ('multi_seed_sparse_seed789/sparse_best_model.pt', True, 0.7147),
        ('archive/sparse_best_model.pt', True, 0.4591),
    ]

    for rel, use_sparse, expected in targets:
        path = os.path.join(project_root, 'experiments', rel)
        if not os.path.exists(path):
            print(f"MISSING {rel}")
            continue
        model = load_model(path, use_sparse)
        spearman = compute_spearman_correlation(model, test_loader, config, device)
        f1 = compute_email_f1(model, test_loader, config, device)
        status = "MATCH" if abs(spearman - expected) < 0.02 else "MISMATCH"
        print(f"{rel}: spearman={spearman:.4f} (expected ~{expected}, {status}), f1={f1:.4f}")


if __name__ == '__main__':
    main()
