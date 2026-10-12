"""
Generate hyper-parameter sensitivity analysis figure for MOSI dataset.

Reads per-seed CSVs from result/ablation/<param>_<value>/normal/mosi.csv,
averages across 3 seeds, and produces a 2×2 subplot figure with F1 and MAE
on dual y-axes — styled for publication.

Default (non-swept) values use the SARM baseline provided by the user:
  F1=71.76, MAE=111.41 (already ×100 scale in CSVs)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# =========================================================================
# 0. Publication-quality RC params
# =========================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.1,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4.5,
    'ytick.major.size': 4.5,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
})

# =========================================================================
# 1. Define sweeps
# =========================================================================
# Each entry: (param_name, display_label, {value: dir_name_or_None}, default_value)
#   dir_name = None  →  use baseline SARM result
BASELINE_F1  = 71.76
BASELINE_MAE = 111.41   # ×100 scale

ABLATION_ROOT = os.path.join(os.path.dirname(__file__))

sweeps = [
    {
        'param': 'lambda_cpa',
        'label': r'$\lambda_{\mathrm{CPA}}$',
        'values': [0, 0.05, 0.1, 0.2],
        'default': 0.1,
        'dirs': {
            0:    'lambda_cpa_0',
            0.05: 'lambda_cpa_0.05',
            0.2:  'lambda_cpa_0.2',
        },
    },
    {
        'param': 'nmb_prototypes',
        'label': r'$K_p$',
        'values': [3, 5, 7, 9],
        'default': 7,
        'dirs': {
            3: 'nmb_prototypes_3',
            5: 'nmb_prototypes_5',
            9: 'nmb_prototypes_9',
        },
    },
    {
        'param': 'tau_c',
        'label': r'$\tau$',
        'values': [0.25, 0.5, 1, 2],
        'default': 1,
        'dirs': {
            0.25: 'tau_c_0.25',
            0.5:  'tau_c_0.5',
            2:    'tau_c_2',
        },
    },
    {
        'param': 'K_neighbors',
        'label': r'$K_n$',
        'values': [5, 10, 20, 50],
        'default': 10,
        'dirs': {
            5:  'K_neighbors_5',
            20: 'K_neighbors_20',
            50: 'K_neighbors_50',
        },
    },
]

# =========================================================================
# 2. Collect data  (mean across seeds)
# =========================================================================
def load_mean(dir_name):
    """Load a sweep CSV and return (mean_f1, mean_mae)."""
    csv_path = os.path.join(ABLATION_ROOT, dir_name, 'normal', 'mosi.csv')
    df = pd.read_csv(csv_path)
    f1  = df['F1_score Mean'].mean()
    mae = df['MAE Mean'].mean()
    return f1, mae


sweep_data = []
for sw in sweeps:
    f1_list, mae_list = [], []
    for v in sw['values']:
        if v == sw['default']:
            f1_list.append(BASELINE_F1)
            mae_list.append(BASELINE_MAE)
        else:
            f1_val, m = load_mean(sw['dirs'][v])
            f1_list.append(round(f1_val, 2))
            mae_list.append(round(m, 2))
    sweep_data.append({
        'label': sw['label'],
        'values': sw['values'],
        'f1': f1_list,
        'mae': mae_list,
    })

# Print summary table
print("=" * 70)
print("Hyper-parameter sensitivity — averages across 3 seeds")
print("=" * 70)
for sd in sweep_data:
    print(f"\n{sd['label']}")
    for v, f, m in zip(sd['values'], sd['f1'], sd['mae']):
        print(f"  {v:>6}  F1={f:6.2f}   MAE={m:6.2f}")

# =========================================================================
# 3. Colour palette  (from Tableau 10 used in the project)
# =========================================================================
COLOR_F1  = '#F28E2B'   # Orange  (DecAlign accent)
COLOR_MAE = '#4E79A7'   # Steel-blue (SARM accent)

# =========================================================================
# 4. Plot
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), dpi=300)
axes = axes.flatten()

for idx, (ax, sd) in enumerate(zip(axes, sweep_data)):
    x_vals = sd['values']
    x_labels = [str(v) for v in x_vals]
    x_pos = np.arange(len(x_vals))

    # ---- F1 (left y-axis) ----
    ax.set_xlabel(sd['label'], fontsize=16, labelpad=6)
    ln1 = ax.plot(
        x_pos, sd['f1'],
        color=COLOR_F1, marker='o', linewidth=2.4, markersize=8,
        markeredgecolor='white', markeredgewidth=1.3,
        label='F1', zorder=5,
    )
    ax.tick_params(axis='y', labelcolor='black', labelsize=13)
    ax.tick_params(axis='x', labelsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)

    f1_arr = np.array(sd['f1'])
    f1_min, f1_max = f1_arr.min(), f1_arr.max()
    pad = max((f1_max - f1_min) * 0.6, 2.5)
    ax.set_ylim(f1_min - pad, f1_max + pad)

    # ---- MAE (right y-axis, inverted so lower=better appears higher) ----
    ax2 = ax.twinx()
    mae_display = [m / 100.0 for m in sd['mae']]
    ln2 = ax2.plot(
        x_pos, mae_display,
        color=COLOR_MAE, marker='D', linewidth=2.4, markersize=7,
        markeredgecolor='white', markeredgewidth=1.3,
        label='MAE', zorder=4, linestyle='-',
    )
    ax2.tick_params(axis='y', labelcolor='black', labelsize=13)

    mae_arr = np.array(mae_display)
    mae_min, mae_max = mae_arr.min(), mae_arr.max()
    mae_pad = max((mae_max - mae_min) * 0.6, 0.02)
    ax2.set_ylim(mae_min - mae_pad, mae_max + mae_pad)
    ax2.invert_yaxis()

    # ---- "F1" top-left, "MAE" top-right — above the axes frame ----
    ax.text(0.02, 1.02, 'F1', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=15, color='black')
    ax2.text(0.98, 1.02, 'MAE', transform=ax2.transAxes,
             ha='right', va='bottom', fontsize=15, color='black')

    # ---- Legend ----
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper right', frameon=True, fontsize=12,
              framealpha=0.92, edgecolor='#CCCCCC',
              borderpad=0.5, labelspacing=0.3, handlelength=1.6)

    ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)

plt.tight_layout(w_pad=1.5, h_pad=2.0)
fig.subplots_adjust(bottom=0.09, top=0.94)

# =========================================================================
# 5. Save
# =========================================================================
out_dir = os.path.dirname(__file__)
png_path = os.path.join(out_dir, 'hp_sensitivity_mosi.png')
pdf_path = os.path.join(out_dir, 'hp_sensitivity_mosi.pdf')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')
plt.close()
print(f'\nSaved: {png_path}')
print(f'Saved: {pdf_path}')
