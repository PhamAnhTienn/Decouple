import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
input_path = 'result/normal/report.csv'
df = pd.read_csv(input_path)
df['Model'] = df['Model'].replace({'tfr_net': 'tfr-net'})

metrics = ['Acc_2', 'F1_score', 'MAE', 'Corr', 'Acc_7', 'Loss']

# ---------------------------------------------------------------------------
# 2. Aggregation Logic
# ---------------------------------------------------------------------------
# Models in logical order (Proposed method DecAlign first, then baselines)
desired_model_order = ['decalign', 'dmd', 'lnln', 'tfr-net', 'misa', 'dar', 'dp']
# Ensure all models present in df are included
all_models = [m for m in desired_model_order if m in df['Model'].unique()]
for m in df['Model'].unique():
    if m not in all_models:
        all_models.append(m)

grouped_rows = []

for m in all_models:
    sub = df[df['Model'] == m]
    mrs = sorted(sub['Missing Rate'].unique())
    for mr in mrs:
        group = sub[sub['Missing Rate'] == mr]
        # Check if already has a row with std > 0 (e.g. pre-calculated baseline summary)
        ready = group[group['Acc_2 Std'].notna() & (group['Acc_2 Std'] > 0)]
        
        if len(ready) > 0:
            r = ready.iloc[0]
            row_dict = {
                'Model': m,
                'Missing Rate': mr,
                'Seed': 'reported' if pd.isna(r['Seed']) else str(r['Seed'])
            }
            for met in metrics:
                row_dict[f'{met} Mean'] = round(float(r[f'{met} Mean']), 2)
                row_dict[f'{met} Std'] = round(float(r[f'{met} Std']), 2)
            grouped_rows.append(row_dict)
        else:
            seeds = group['Seed'].dropna().tolist()
            seed_str = f'{len(group)} seeds'
            row_dict = {
                'Model': m,
                'Missing Rate': mr,
                'Seed': seed_str
            }
            for met in metrics:
                vals = group[f'{met} Mean'].dropna()
                mean_val = vals.mean()
                std_val = vals.std(ddof=1) if len(vals) > 1 else 0.0
                row_dict[f'{met} Mean'] = round(float(mean_val), 2)
                row_dict[f'{met} Std'] = round(float(std_val), 2)
            grouped_rows.append(row_dict)

res_df = pd.DataFrame(grouped_rows)
output_csv = 'result/normal/report_grouped.csv'
res_df.to_csv(output_csv, index=False)
print(f'Successfully saved aggregated CSV to {output_csv}')

# ---------------------------------------------------------------------------
# 3. Publication-Quality Plotting Configuration
# ---------------------------------------------------------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Liberation Serif', 'DejaVu Serif', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 1.1
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 4.5
plt.rcParams['ytick.major.size'] = 4.5
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0

# ---------------------------------------------------------------------------
# Color Palette: Full Okabe-Ito (Color Universal Design)
# 7 maximally distinct hues — safe for all types of color-blindness.
# Ordered by perceptual salience (high → low) so the proposed method
# stands out clearly against baselines.
#
#  Index │ Model     │ Hex       │ Character
#  ──────┼───────────┼───────────┼──────────────────────────────────────
#    1   │ DecAlign  │ #0072B2   │ Strong blue  — highest salience (proposed)
#    2   │ LNLN      │ #D55E00   │ Vermilion    — warm, clearly distinct from blue
#    3   │ DMD       │ #009E73   │ Bluish-green — cool, far from red & blue in hue
#    4   │ DAR       │ #E69F00   │ Amber/gold   — warm-yellow, distinct from all above
#    5   │ MISA      │ #CC79A7   │ Rose-purple  — unique pink hue
#    6   │ TFR-Net   │ #56B4E9   │ Sky-blue     — lighter, separated from dark blue
#    7   │ DP        │ #F0E442   │ Yellow       — brightest, stands out on white bg
# ---------------------------------------------------------------------------
style_map = {
    'decalign': {'label': 'DecAlign', 'color': '#0072B2', 'zorder': 10},
    'lnln':     {'label': 'LNLN',     'color': '#D55E00', 'zorder': 9},
    'dmd':      {'label': 'DMD',      'color': '#009E73', 'zorder': 8},
    'dar':      {'label': 'DAR',      'color': '#E69F00', 'zorder': 7},
    'misa':     {'label': 'MISA',     'color': '#CC79A7', 'zorder': 6},
    'tfr-net':  {'label': 'TFR-Net',  'color': '#56B4E9', 'zorder': 5},
    'dp':       {'label': 'DP',       'color': '#9467BD', 'zorder': 4},
}
style_map['tfr_net'] = style_map['tfr-net']

def create_plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    for m in all_models:
        sub = res_df[res_df['Model'] == m].sort_values('Missing Rate')
        style = style_map[m]

        # Subplot 1: F1 Score
        ax1.plot(
            sub['Missing Rate'], sub['F1_score Mean'],
            label=style['label'],
            color=style['color'],
            marker='o',
            linestyle='-',
            linewidth=2.5,
            markersize=8,
            markeredgecolor=style['color'],
            markeredgewidth=1.2,
            zorder=style['zorder']
        )

        # Subplot 2: MAE (divided by 100 to get standard scale)
        ax2.plot(
            sub['Missing Rate'], sub['MAE Mean'] / 100.0,
            label=style['label'],
            color=style['color'],
            marker='o',
            linestyle='-',
            linewidth=2.5,
            markersize=8,
            markeredgecolor=style['color'],
            markeredgewidth=1.2,
            zorder=style['zorder']
        )

    # --- Subplot 1 (a) — F1 Score ---
    ax1.set_xlabel(r'Missing Rate $r$', fontsize=14, labelpad=8)
    ax1.set_ylabel('F1', fontsize=14, labelpad=8, fontweight='bold')
    ax1.set_xlim(-0.05, 1.02)
    ax1.set_ylim(32, 88)
    ax1.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax1.set_yticks([40, 50, 60, 70, 80])
    ax1.tick_params(labelsize=12)
    ax1.grid(False)
    ax1.legend(
        loc='upper right', frameon=True, fontsize=9.5,
        framealpha=0.95, edgecolor='#CCCCCC',
        borderpad=0.5, labelspacing=0.35, handlelength=1.8
    )
    ax1.text(0.5, -0.22, '(a)', transform=ax1.transAxes, ha='center', va='top', fontsize=14, fontweight='bold')

    # --- Subplot 2 (b) — MAE ---
    ax2.set_xlabel(r'Missing Rate $r$', fontsize=14, labelpad=8)
    ax2.set_ylabel('MAE', fontsize=14, labelpad=8, fontweight='bold')
    ax2.set_xlim(-0.05, 1.02)
    ax2.set_ylim(0.70, 1.55)
    ax2.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax2.set_yticks([0.8, 1.0, 1.2, 1.4])
    ax2.tick_params(labelsize=12)
    ax2.grid(False)
    ax2.legend(
        loc='upper left', frameon=True, fontsize=9.5,
        framealpha=0.95, edgecolor='#CCCCCC',
        borderpad=0.5, labelspacing=0.35, handlelength=1.8
    )
    ax2.text(0.5, -0.22, '(b)', transform=ax2.transAxes, ha='center', va='top', fontsize=14, fontweight='bold')

    plt.tight_layout(w_pad=3.0)
    fig.subplots_adjust(bottom=0.18)

    png_path = 'result/normal/missing_rate_performance.png'
    pdf_path = 'result/normal/missing_rate_performance.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f'Saved: {png_path} and {pdf_path}')

create_plot()

# python result/normal/generate_summary_and_plots.py