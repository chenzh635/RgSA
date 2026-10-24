"""Plot baseline comparison radar chart for the paper (GT-metric version).

Axes: AUPRC, F1 (email-level), GT P@1, GT R@3 (localization vs ground truth),
k@90 and Entropy (both inverted so that outward = better).
Replaces the old Spearman-axis version (Spearman demoted to appendix proxy).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

models = ['Rules-only', 'Snorkel', 'ABMIL', 'BERT-ABMIL', 'AI2TALE',
          'Sparse BERT', 'Sparse ABMIL (Ours)']

# Sources: experiments/main_table_data.json (loc_seeds mean for 3-seed rows,
# loc_avg for single runs; strict mode, 202 gold-positive emails) and
# experiments/m_proxy_refresh.json (entropy). λ_rule=1.0 validation-selected.
auprc   = [0.7860, 0.6383, 0.9992, 0.9984, 0.9951, 0.9962, 0.9986]
f1      = [0.6734, 0.5113, 0.9904, 0.9803, 0.9845, 0.9693, 0.9826]
p_at_1  = [0.658,  0.663,  0.439,  0.475,  0.457,  0.649,  0.507]
r_at_3  = [0.715,  0.729,  0.619,  0.681,  0.659,  0.754,  0.668]
k90_raw = [5.47,   5.59,   6.32,   5.72,   6.02,   5.34,   5.77]
ent_raw = [1.3684, 1.3801, 1.306,  1.0672, 1.478,  0.5234, 0.513]

def inv(vals, lo, hi):
    """Invert and scale to [0,1]: lower raw -> higher norm."""
    return [min(1.0, max(0.0, 1.0 - (v - lo) / (hi - lo))) for v in vals]

k90_norm = inv(k90_raw, 5.0, 7.0)
ent_norm = inv(ent_raw, 0.4, 1.5)

data = {
    'AUPRC':        auprc,
    'F1':           f1,
    'GT P@1':       p_at_1,
    'GT R@3':       r_at_3,
    'k@90 (inv)':   k90_norm,
    'Entropy (inv)': ent_norm,
}

n_vars = len(data)
angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(4.8, 4.8), subplot_kw=dict(projection='polar'))

colors = ['#6b7280', '#f59e0b', '#2563eb', '#059669', '#8b5cf6',
          '#06b6d4', '#dc2626']
linewidths = [1.2, 1.2, 1.5, 1.5, 1.2, 1.5, 2.5]

for i, (model, color, lw) in enumerate(zip(models, colors, linewidths)):
    values = [data[k][i] for k in data.keys()]
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=lw, label=model)
    if 'Ours' in model:
        ax.fill(angles, values, color=color, alpha=0.12)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(list(data.keys()), fontsize=9)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
ax.grid(True, alpha=0.3)

legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
                   fontsize=7.5, ncol=3, frameon=False)
for lh in legend.legend_handles:
    lh.set_linewidth(1.5)

plt.tight_layout()
plt.savefig('experiments/baseline_radar.pdf', dpi=300, bbox_inches='tight')
plt.savefig('experiments/baseline_radar.png', dpi=200, bbox_inches='tight')
print('Saved GT-based baseline radar chart.')
