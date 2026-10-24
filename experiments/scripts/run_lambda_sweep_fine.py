"""Fine-grained lambda_rule sweep: 1.1 to 3.0 in 0.1 steps."""
import json, os, yaml, torch, time
from src.dataset import create_dataloaders
from src.models.abmil import SparseABMIL
from src.train import train
from src.evaluate import evaluate

# Load existing results
existing = {}
if os.path.exists('experiments/lambda_sweep_results.json'):
    with open('experiments/lambda_sweep_results.json') as f:
        existing = json.load(f)

configs = [
    ('1.1', 'experiments/configs/sparse_lambda_1_1.yaml'),
    ('1.2', 'experiments/configs/sparse_lambda_1_2.yaml'),
    ('1.3', 'experiments/configs/sparse_lambda_1_3.yaml'),
    ('1.4', 'experiments/configs/sparse_lambda_1_4.yaml'),
    ('1.5', 'experiments/configs/sparse_lambda_1_5.yaml'),
    ('1.6', 'experiments/configs/sparse_lambda_1_6.yaml'),
    ('1.7', 'experiments/configs/sparse_lambda_1_7.yaml'),
    ('1.8', 'experiments/configs/sparse_lambda_1_8.yaml'),
    ('1.9', 'experiments/configs/sparse_lambda_1_9.yaml'),
    ('2.1', 'experiments/configs/sparse_lambda_2_1.yaml'),
    ('2.2', 'experiments/configs/sparse_lambda_2_2.yaml'),
    ('2.3', 'experiments/configs/sparse_lambda_2_3.yaml'),
    ('2.4', 'experiments/configs/sparse_lambda_2_4.yaml'),
    ('2.5', 'experiments/configs/sparse_lambda_2_5.yaml'),
    ('2.6', 'experiments/configs/sparse_lambda_2_6.yaml'),
    ('2.7', 'experiments/configs/sparse_lambda_2_7.yaml'),
    ('2.8', 'experiments/configs/sparse_lambda_2_8.yaml'),
    ('2.9', 'experiments/configs/sparse_lambda_2_9.yaml'),
    ('3.0', 'experiments/configs/sparse_lambda_3_0.yaml'),
]

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}\n")

results_all = dict(existing)
train_loader, test_loader = None, None
t0 = time.time()

for i, (label, cfg_path) in enumerate(configs):
    if label in results_all:
        r = results_all[label]
        print(f"Skip lambda={label}: already done (Spearman={r['spearman_corr']:.4f})")
        continue

    elapsed_so_far = time.time() - t0
    remaining = (len(configs) - i) * (elapsed_so_far / max(i, 1))
    print(f"\n{'='*60}")
    print(f"lambda_rule = {label}  [{i+1}/{len(configs)}]  ~{remaining/3600:.1f}h remaining")
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

    ckpt_path = os.path.join(config['save_dir'], 'sparse_best_model.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

    results = evaluate(model, test_loader, config, device)
    results['training_time_sec'] = round(elapsed, 1)
    results_all[label] = results

    print(f"\n  lambda_rule={label}: auprc={results['email_auprc']:.4f} f1={results['email_f1']:.4f} spearman={results['spearman_corr']:.4f} entropy={results['attention_entropy']:.4f}  ({elapsed/60:.1f} min)")

    # Save incrementally
    with open('experiments/lambda_sweep_fine_results.json', 'w') as f:
        json.dump(results_all, f, indent=2)

# Summary table
print(f"\n{'='*90}")
print(f"{'lambda':>7} {'AUPRC':>8} {'F1':>8} {'Spearman':>10} {'Entropy':>10} {'Time':>7}")
print(f"{'='*90}")
for label in sorted(results_all.keys(), key=lambda x: float(x)):
    r = results_all[label]
    t = f"{r['training_time_sec']/60:.0f}m" if 'training_time_sec' in r else '-'
    print(f"{label:>7} {r['email_auprc']:>8.4f} {r['email_f1']:>8.4f} {r['spearman_corr']:>10.4f} {r['attention_entropy']:>10.4f} {t:>7}")
print(f"{'='*90}")

with open('experiments/lambda_sweep_fine_results.json', 'w') as f:
    json.dump(results_all, f, indent=2)
print("Saved to experiments/lambda_sweep_fine_results.json")
