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
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

ACL_COL_W = 3.03   # inch, 1 cột ACL

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,            # tránh Type 3
    'ps.fonttype': 42,
    'font.size': 7,
    'axes.labelsize': 8,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 7,
    'axes.linewidth': 0.7,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.pad': 2,
    'ytick.major.pad': 2,
    'lines.linewidth': 1.4,
    'lines.markersize': 4.5,
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
        'default': sw['default'],      # <-- thêm dòng này
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
fig, axes = plt.subplots(2, 2, figsize=(ACL_COL_W, 2.5), layout='constrained')
fig.get_layout_engine().set(
    w_pad=0.01,
    h_pad=0.01,
    wspace=0.05,
    hspace=0.05
)
axes = axes.flatten()

# Span chung cho mọi panel -> độ dốc so sánh được giữa các panel
F1_SPAN  = max(max(np.ptp(sd['f1']) for sd in sweep_data) * 1.8, 3.0)
MAE_SPAN = max(max(np.ptp(sd['mae']) for sd in sweep_data) / 100.0 * 1.8, 0.03)

handles = None
for ax, sd in zip(axes, sweep_data):
    n = len(sd['values'])
    x_pos = np.arange(n)
    d_idx = sd['values'].index(sd['default'])

    # default marker line
    ax.axvline(d_idx, color='0.7', linestyle='--', linewidth=0.7, zorder=1)

    # ---- F1 (left) ----
    ln1 = ax.plot(x_pos, sd['f1'], color=COLOR_F1, marker='o',
                  markeredgecolor='white', markeredgewidth=0.6,
                  label='F1', zorder=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(v) for v in sd['values']])
    ax.set_xlim(-0.3, n - 1 + 0.3)
    ax.set_xlabel(sd['label'], labelpad=2)

    mid = (max(sd['f1']) + min(sd['f1'])) / 2
    ax.set_ylim(mid - F1_SPAN / 2, mid + F1_SPAN / 2)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.grid(axis='y', linestyle=':', linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title('F1', loc='left', fontsize=7, pad=2)

    # ---- MAE (right, inverted) ----
    ax2 = ax.twinx()
    mae = [m / 100.0 for m in sd['mae']]
    ln2 = ax2.plot(x_pos, mae, color=COLOR_MAE, marker='D', markersize=4,
                   markeredgecolor='white', markeredgewidth=0.6,
                   label='MAE', zorder=4)
    mid2 = (max(mae) + min(mae)) / 2
    ax2.set_ylim(mid2 + MAE_SPAN / 2, mid2 - MAE_SPAN / 2)   # inverted
    ax2.yaxis.set_major_locator(MaxNLocator(4))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.set_title('MAE', loc='right', fontsize=7, pad=2)

    if handles is None:
        handles = ln1 + ln2

# ---- 1 legend chung ----
fig.legend(handles, [h.get_label() for h in handles],
           loc='outside upper center', ncol=2, frameon=False,
           handlelength=1.8, columnspacing=1.5)

# =========================================================================
# 5. Save
# =========================================================================
out_dir = os.path.dirname(__file__)
png_path = os.path.join(out_dir, 'hp_sensitivity_mosi.png')
pdf_path = os.path.join(out_dir, 'hp_sensitivity_mosi.pdf')
fig.savefig(pdf_path)
fig.savefig(png_path, dpi=600)
plt.close(fig)