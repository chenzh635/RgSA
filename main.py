#!/usr/bin/env python3
"""Sparse ABMIL for Phishing Sentence Localization."""

import argparse
import os
import sys

import yaml
import torch

from src.dataset import create_dataloaders
from src.models.abmil import SparseABMIL, ABMILBaseline
from src.train import train
from src.evaluate import evaluate


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Sparse ABMIL for Phishing Sentence Localization')
    parser.add_argument('--config', type=str, default='experiments/configs/defaults.yaml', help='Config file path')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'both'], help='Run mode')
    parser.add_argument('--model', type=str, default='sparse', choices=['abmil', 'sparse'], help='Model type')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()

    config = load_config(args.config)

    # Set random seed if specified
    if args.seed is not None:
        import numpy as np
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Random seed set to: {args.seed}")

    # Set device
    device = torch.device(config.get('device', 'mps'))
    if not torch.backends.mps.is_available():
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(config)
    print(f"Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")

    # Create model
    if args.model == 'sparse':
        model = SparseABMIL(
            hidden_dim=config.get('hidden_dim', 64),
            attention_hidden=config.get('attention_hidden', 64),
            classifier_hidden=config.get('classifier_hidden', [128, 64]),
            use_sparse=config.get('use_sparse', True),
        )
    else:
        model = ABMILBaseline(
            hidden_dim=config.get('hidden_dim', 64),
            attention_hidden=config.get('attention_hidden', 64),
            classifier_hidden=config.get('classifier_hidden', [128, 64]),
        )
    model = model.to(device)
    print(f"Model: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.mode in ['train', 'both']:
        train(model, train_loader, test_loader, config, device)

    if args.mode in ['eval', 'both']:
        # Load best checkpoint if available
        ckpt_path = os.path.join(config['save_dir'], 'best_model.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded checkpoint from {ckpt_path}")

        results = evaluate(model, test_loader, config, device)
        print("\n=== Evaluation Results ===")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
