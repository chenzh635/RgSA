"""Sweep lambda_rule hyperparameter for Sparse ABMIL."""
import json, os, yaml, torch, time
from src.dataset import create_dataloaders
from src.models.abmil import SparseABMIL
from src.train import train
from src.evaluate import evaluate

configs = [
    ('0.1', 'experiments/configs/sparse_lambda_0.1.yaml'),
    ('0.3', 'experiments/configs/sparse_lambda_0.3.yaml'),
    ('0.5', 'experiments/configs/defaults.yaml'),
    ('1.0', 'experiments/configs/sparse_lambda_1.0.yaml'),
    ('2.0', 'experiments/configs/sparse_lambda_2.0.yaml'),
]

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}\n")

results_all = {}
train_loader, test_loader = None, None

for label, cfg_path in configs:
    print(f"\n{'='*60}")
    print(f"lambda_rule = {label}")
    print(f"{'='*60}")

    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    if train_loader is None:
        train_loader, test_loader = create_dataloaders(config)
        print(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

    model = SparseABMIL(
        hidden_dim=config.get('hidden_dim', 64),
        attention_hidden=config.get('attention_hidden', 64),
        classifier_hidden=config.get('classifier_hidden', [128, 64]),
        use_sparse=True,
    ).to(device)

    start = time.time()
    train(model, train_loader, test_loader, config, device)
    elapsed = time.time() - start

    # Load best and evaluate
    ckpt_path = os.path.join(config['save_dir'], 'sparse_best_model.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

    results = evaluate(model, test_loader, config, device)
    results['training_time_sec'] = round(elapsed, 1)
    results_all[label] = results

    print(f"\n  lambda_rule={label}: ", end='')
    for k, v in results.items():
        if k != 'training_time_sec':
            print(f"{k}={v:.4f} ", end='')
    print(f"  ({elapsed/60:.1f} min)")

# Summary table
print(f"\n{'='*80}")
print(f"{'lambda_rule':>12} {'AUPRC':>8} {'F1':>8} {'Spearman':>10} {'Entropy':>10} {'Time':>8}")
print(f"{'='*80}")
for label in ['0.1', '0.3', '0.5', '1.0', '2.0']:
    r = results_all[label]
    print(f"{label:>12} {r['email_auprc']:>8.4f} {r['email_f1']:>8.4f} {r['spearman_corr']:>10.4f} {r['attention_entropy']:>10.4f} {r['training_time_sec']/60:>7.1f}m")
print(f"{'='*80}")

with open('experiments/lambda_sweep_results.json', 'w') as f:
    json.dump(results_all, f, indent=2)
print("Saved to experiments/lambda_sweep_results.json")
