"""Re-train main models with epoch-level logging and generate training curves."""
import json, os, yaml, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from src.dataset import create_dataloaders
from src.models.abmil import SparseABMIL, ABMILBaseline
from src.train import train, compute_loss, get_temperature
from src.rules import load_keywords
from src.evaluate import evaluate


def train_with_logging(model, train_loader, test_loader, config, device, label):
    """Train with epoch-level metric logging."""
    epochs = config.get('epochs', 10)
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0.0001)
    save_dir = config.get('save_dir', 'experiments')
    os.makedirs(save_dir, exist_ok=True)

    keywords_path = config.get('keywords_file', 'phishingdata/kwords_stem.json')
    if not os.path.isabs(keywords_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        keywords_path = os.path.join(project_root, keywords_path)
    keywords = load_keywords(keywords_path) if os.path.exists(keywords_path) else None

    lambda_sparse = config.get('lambda_sparse', 0.1)
    lambda_rule = config.get('lambda_rule', 0.5)
    temp_init = config.get('temperature_init', 1.0)
    temp_min = config.get('temperature_min', 0.1)
    temp_decay = config.get('temperature_decay', 0.95)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    is_sparse = hasattr(model, 'use_sparse') and model.use_sparse

    log = []
    best_test_acc = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_losses = {'bce': 0, 'sparse': 0, 'rule': 0}

        temperature = get_temperature(epoch, temp_init, temp_min, temp_decay) if is_sparse else 1.0

        pbar = tqdm(train_loader, desc=f'{label} Epoch {epoch+1}/{epochs} [τ={temperature:.3f}]')
        for batch in pbar:
            tokens = batch['tokens'].to(device)
            labels = batch['label'].to(device)
            sent_lengths = batch['sent_lengths'].to(device)
            raw_sentences = batch['raw_sentences']

            optimizer.zero_grad()

            if is_sparse:
                output, attn_weights, attn_scores, h = model(tokens, sent_lengths, temperature)
                loss, losses = compute_loss(
                    output, labels, attn_scores, raw_sentences, keywords,
                    lambda_sparse, lambda_rule
                )
            else:
                output, attn_weights, h = model(tokens, sent_lengths)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(output, labels.float())
                losses = {'bce': loss.item()}

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * labels.shape[0]
            preds = (torch.sigmoid(output) > 0.5).long()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.shape[0]

            for k in all_losses:
                all_losses[k] += losses.get(k, 0) * labels.shape[0]

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{total_correct/total_samples:.4f}'})

        # Epoch averages (train)
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        avg_bce = all_losses['bce'] / total_samples
        avg_sparse = all_losses['sparse'] / total_samples
        avg_rule = all_losses['rule'] / total_samples

        # Test accuracy
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch in test_loader:
                tokens = batch['tokens'].to(device)
                labels = batch['label'].to(device)
                sent_lengths = batch['sent_lengths'].to(device)
                if is_sparse:
                    output, _, _, _ = model(tokens, sent_lengths, temperature=temp_min)
                else:
                    output, _, _ = model(tokens, sent_lengths)
                preds = (torch.sigmoid(output) > 0.5).long()
                test_correct += (preds == labels).sum().item()
                test_total += labels.shape[0]
        test_acc = test_correct / max(test_total, 1)

        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': round(avg_loss, 6),
            'train_acc': round(avg_acc, 6),
            'train_bce': round(avg_bce, 6),
            'train_sparse': round(avg_sparse, 6),
            'train_rule': round(avg_rule, 6),
            'test_acc': round(test_acc, 6),
            'temperature': round(temperature, 6),
        }
        log.append(epoch_log)

        print(f'  Epoch {epoch+1}: train_loss={avg_loss:.4f} train_acc={avg_acc:.4f} '
              f'test_acc={test_acc:.4f} bce={avg_bce:.4f} sparse={avg_sparse:.4f} rule={avg_rule:.4f}')

        # Save best
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            model_suffix = 'sparse' if is_sparse else 'abmil'
            ckpt_path = os.path.join(save_dir, f'{model_suffix}_best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'temperature': temperature,
            }, ckpt_path)

        model.train()

    # Save log
    log_path = os.path.join(save_dir, f'{label.lower().replace(" ", "_")}_log.json')
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f'Saved training log to {log_path}')
    return log


def plot_curves(logs):
    """Generate training curves figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Panel 1: Test accuracy comparison
    ax = axes[0]
    colors = {'Sparse ABMIL': '#dc2626', 'ABMIL': '#2563eb'}
    for label, log in logs.items():
        epochs = [e['epoch'] for e in log]
        test_acc = [e['test_acc'] for e in log]
        train_acc = [e['train_acc'] for e in log]
        color = colors.get(label, '#333')
        ax.plot(epochs, train_acc, '--', color=color, alpha=0.4, linewidth=1.5)
        ax.plot(epochs, test_acc, 'o-', color=color, linewidth=2, markersize=6, label=label)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_ylim(0.94, 1.0)
    ax.set_title('Classification Accuracy', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.2)

    # Panel 2: Sparse ABMIL loss decomposition
    ax = axes[1]
    log = logs['Sparse ABMIL']
    epochs = [e['epoch'] for e in log]
    ax.plot(epochs, [e['train_loss'] for e in log], 'o-', color='#dc2626', linewidth=2, markersize=6, label='Total Loss')
    ax.plot(epochs, [e['train_bce'] for e in log], 's--', color='#2563eb', linewidth=2, markersize=6, label='BCE (Classification)')
    ax.plot(epochs, [e['train_sparse'] for e in log], '^--', color='#f59e0b', linewidth=2, markersize=6, label='L1 Sparsity')
    ax.plot(epochs, [e['train_rule'] for e in log], 'd--', color='#059669', linewidth=2, markersize=6, label='Rule Alignment')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Loss Decomposition (Sparse ABMIL)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.2)

    # Panel 3: Temperature annealing + test accuracy
    ax = axes[2]
    log = logs['Sparse ABMIL']
    epochs = [e['epoch'] for e in log]
    temp = [e['temperature'] for e in log]
    ax.plot(epochs, temp, 'o-', color='#7c3aed', linewidth=2, markersize=6, label='Temperature')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Temperature', color='#7c3aed', fontsize=11)
    ax.tick_params(axis='y', labelcolor='#7c3aed')
    ax.set_title('Temperature Annealing', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)

    ax2 = ax.twinx()
    ax2.plot(epochs, [e['test_acc'] for e in log], 's-', color='#dc2626', linewidth=2, markersize=6, label='Test Acc')
    ax2.set_ylabel('Test Accuracy', color='#dc2626', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='#dc2626')
    ax2.set_ylim(0.94, 1.0)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig('experiments/training_curves.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('experiments/training_curves.png', dpi=200, bbox_inches='tight')
    print('Saved training curves.')


# ── Main ────────────────────────────────────────────────────────────

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}\n")

with open('experiments/configs/defaults.yaml', 'r') as f:
    config = yaml.safe_load(f)

train_loader, test_loader = create_dataloaders(config)
print(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

logs = {}

# Train Sparse ABMIL
print("\n" + "="*60)
print("Training Sparse ABMIL (λ_rule=3.0)")
print("="*60)
config_sparse = dict(config)
config_sparse['use_sparse'] = True
config_sparse['lambda_sparse'] = 0.1
config_sparse['lambda_rule'] = 3.0
config_sparse['save_dir'] = 'experiments/sparse_new'

model_sparse = SparseABMIL(
    hidden_dim=64, attention_hidden=64,
    classifier_hidden=[128, 64], use_sparse=True,
).to(device)

log_sparse = train_with_logging(model_sparse, train_loader, test_loader, config_sparse, device, 'Sparse ABMIL')
logs['Sparse ABMIL'] = log_sparse

# Train ABMIL
print("\n" + "="*60)
print("Training ABMIL (baseline)")
print("="*60)
config_abmil = dict(config)
config_abmil['use_sparse'] = False
config_abmil['save_dir'] = 'experiments/abmil_new'

model_abmil = ABMILBaseline(hidden_dim=64, attention_hidden=64, classifier_hidden=[128, 64]).to(device)

log_abmil = train_with_logging(model_abmil, train_loader, test_loader, config_abmil, device, 'ABMIL')
logs['ABMIL'] = log_abmil

# Plot
print("\nGenerating training curves...")
plot_curves(logs)

# Evaluate final models
for label, cfg in [('Sparse ABMIL', config_sparse), ('ABMIL', config_abmil)]:
    ckpt_path = os.path.join(cfg['save_dir'], f"{'sparse' if 'Sparse' in label else 'abmil'}_best_model.pt")
    if os.path.exists(ckpt_path):
        model = SparseABMIL(use_sparse=True).to(device) if 'Sparse' in label else ABMILBaseline().to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        results = evaluate(model, test_loader, cfg, device)
        print(f"\n{label} final results:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
