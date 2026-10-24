"""Plot lambda sweep results for the paper (full range 0.1-3.0)."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load results
coarse = json.load(open('experiments/lambda_sweep_results.json'))
fine = json.load(open('experiments/lambda_sweep_fine_results.json'))

# Combine all results
all_lambdas = {}
for d in [coarse, fine]:
    for k, v in d.items():
        all_lambdas[float(k)] = v

# Sort by lambda
lambdas = sorted(all_lambdas.keys())
spearman = [all_lambdas[l]['spearman_corr'] for l in lambdas]
f1 = [all_lambdas[l]['email_f1'] for l in lambdas]
auprc = [all_lambdas[l]['email_auprc'] for l in lambdas]
entropy = [all_lambdas[l]['attention_entropy'] for l in lambdas]

# Mark optimal
opt_lambda = 3.0
opt_idx = lambdas.index(opt_lambda)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Panel 1: Spearman + F1 (dual axis)
ax1 = axes[0]
color1 = '#2563eb'
color2 = '#dc2626'
ax1.plot(lambdas, spearman, 'o-', color=color1, linewidth=2, markersize=5, label='Spearman')
ax1.set_xlabel(r'$\lambda_{\mathrm{rule}}$', fontsize=12)
ax1.set_ylabel('Spearman Rank Correlation', color=color1, fontsize=11)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0.25, 0.75)
ax1.axvline(x=opt_lambda, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax1.scatter([opt_lambda], [spearman[opt_idx]], s=120, color=color1, zorder=5, edgecolors='white', linewidths=2)
ax1.annotate(f'Optimal\nλ={opt_lambda}', xy=(opt_lambda, spearman[opt_idx]),
             xytext=(opt_lambda - 0.5, spearman[opt_idx] - 0.06),
             fontsize=10, color=color1,
             arrowprops=dict(arrowstyle='->', color=color1, lw=1.5))

ax2 = ax1.twinx()
ax2.plot(lambdas, f1, 's--', color=color2, linewidth=2, markersize=5, label='F1')
ax2.set_ylabel('Email-Level F1', color=color2, fontsize=11)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0.978, 0.991)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=9)
ax1.set_title('Classification vs. Localization Trade-off', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.2)

# Panel 2: Entropy
ax3 = axes[1]
ax3.plot(lambdas, entropy, 'o-', color='#7c3aed', linewidth=2, markersize=5)
ax3.scatter([opt_lambda], [entropy[opt_idx]], s=120, color='#7c3aed', zorder=5, edgecolors='white', linewidths=2)
ax3.axvline(x=opt_lambda, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax3.set_xlabel(r'$\lambda_{\mathrm{rule}}$', fontsize=12)
ax3.set_ylabel('Attention Entropy', fontsize=11)
ax3.set_title('Attention Focus (lower = sharper)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.2)
ax3.annotate(f'λ={opt_lambda}\nH={entropy[opt_idx]:.3f}', xy=(opt_lambda, entropy[opt_idx]),
             xytext=(opt_lambda - 0.5, entropy[opt_idx] - 0.05),
             fontsize=10, color='#7c3aed',
             arrowprops=dict(arrowstyle='->', color='#7c3aed', lw=1.5))

# Panel 3: AUPRC
ax4 = axes[2]
ax4.plot(lambdas, auprc, 'o-', color='#059669', linewidth=2, markersize=5)
ax4.scatter([opt_lambda], [auprc[opt_idx]], s=120, color='#059669', zorder=5, edgecolors='white', linewidths=2)
ax4.axvline(x=opt_lambda, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax4.set_xlabel(r'$\lambda_{\mathrm{rule}}$', fontsize=12)
ax4.set_ylabel('Email-Level AUPRC', fontsize=11)
ax4.set_title('Classification Performance', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.2)
ax4.set_ylim(0.9970, 0.9995)
ax4.annotate(f'λ={opt_lambda}\nAUPRC={auprc[opt_idx]:.4f}', xy=(opt_lambda, auprc[opt_idx]),
             xytext=(opt_lambda - 0.5, auprc[opt_idx] - 0.0003),
             fontsize=10, color='#059669',
             arrowprops=dict(arrowstyle='->', color='#059669', lw=1.5))

plt.tight_layout()
plt.savefig('experiments/lambda_sweep_plot.pdf', dpi=300, bbox_inches='tight')
plt.savefig('experiments/lambda_sweep_plot.png', dpi=200, bbox_inches='tight')
print('Saved lambda sweep plot (full range).')
