import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Load data from per-dataset CSVs
# ---------------------------------------------------------------------------
mosi_df = pd.read_csv('result/normal/mosi-final.csv')
mosei_df = pd.read_csv('result/normal/mosei-final.csv')

# Normalize model names
for df in [mosi_df, mosei_df]:
    df['Model'] = df['Model'].replace({'tfr_net': 'tfr-net'})

# ---------------------------------------------------------------------------
# 1b. Mock SIMS data (no real CSV yet)
# ---------------------------------------------------------------------------
_rng = np.random.RandomState(42)
_missing_rates = np.arange(0.0, 0.8 + 1e-9, 0.1).round(2)
_sims_rows = []
for model in ['decalign', 'dmd', 'lnln', 'tfr-net', 'misa', 'dar', 'dp']:
    for mr in _missing_rates:
        # Baseline F1 and MAE that degrade with missing rate; model-specific offsets
        offsets = {
            'decalign': (82, 52), 'dp': (80, 54), 'lnln': (78, 56),
            'dmd': (76, 58), 'dar': (74, 60), 'misa': (72, 62),
            'tfr-net': (70, 64),
        }
        f1_base, mae_base = offsets[model]
        f1_val = f1_base - 18 * mr + _rng.normal(0, 1.0)
        mae_val = mae_base + 15 * mr + _rng.normal(0, 0.8)
        _sims_rows.append({
            'Model': model,
            'Missing Rate': mr,
            'F1_score Mean': f1_val,
            'MAE Mean': mae_val,
        })
sims_df = pd.DataFrame(_sims_rows)

# ---------------------------------------------------------------------------
# 2. Model order and color palette (unchanged from original)
# ---------------------------------------------------------------------------
# SARM uses the dark blue #4e79a7
# ---------------------------------------------------------------------------
desired_model_order = [
    'decalign', 'dmd', 'lnln', 'tfr-net', 'misa', 'dar', 'dp'
]

style_map = {
    'decalign': {'label': 'DecAlign', 'color': '#f28e2b', 'zorder': 10},
    'dmd':      {'label': 'DMD',      'color': '#e15759', 'zorder': 8},
    'lnln':     {'label': 'LNLN',     'color': '#59a14f', 'zorder': 9},
    'tfr-net':  {'label': 'TFR-Net',  'color': '#76b7b2', 'zorder': 5},
    'misa':     {'label': 'MISA',     'color': '#b07aa1', 'zorder': 6},
    'dar':      {'label': 'DAR',      'color': '#ababab', 'zorder': 7},
    'dp':       {'label': 'SARM',     'color': '#4e79a7', 'zorder': 11},
}

# ---------------------------------------------------------------------------
# 3. Aggregate: mean across seeds per (Model, Missing Rate)
# ---------------------------------------------------------------------------
def aggregate_data(df):
    """Group by Model and Missing Rate, compute mean of '*_Mean' columns."""
    all_models = [m for m in desired_model_order if m in df['Model'].unique()]
    for m in df['Model'].unique():
        if m not in all_models:
            all_models.append(m)

    rows = []
    for m in all_models:
        sub = df[df['Model'] == m]
        for mr in sorted(sub['Missing Rate'].unique()):
            group = sub[sub['Missing Rate'] == mr]
            row = {'Model': m, 'Missing Rate': mr}
            for met in ['F1_score', 'MAE']:
                vals = group[f'{met} Mean'].dropna()
                row[f'{met} Mean'] = vals.mean()
            rows.append(row)
    return pd.DataFrame(rows), all_models


mosi_agg, mosi_models = aggregate_data(mosi_df)
mosei_agg, mosei_models = aggregate_data(mosei_df)
sims_agg, sims_models = aggregate_data(sims_df)

# ---------------------------------------------------------------------------
# 4. CVPR two-column (narrow) configuration from paper_style.py
#    CVPR \columnwidth = (6.875 - 0.3125) / 2 = 3.28125 in
# ---------------------------------------------------------------------------
# COL_W = (6.875 - 0.3125) / 2          # ≈ 3.28 in (CVPR narrow)
# FIG_H = COL_W * 0.7 * 3 + 0.5         # 3 rows of panels (compact)

COL_W = (6.875 - 0.3125) / 2
# --- các lề tính bằng inch ---
EDGE   = 0.02    # khe giữa legend và mép trên figure
LEG_H  = 0.34    # chiều cao legend (2 hàng, font 7pt)
GAP    = 0.03    # khe giữa legend và title hàng 1
TITLE  = 0.15    # chiều cao title (8pt + pad)
BOT    = 0.28    # tick label + xlabel hàng dưới cùng

HSPACE = 0.14    # khoảng cách giữa các hàng (tỉ lệ theo chiều cao axes)
WSPACE = 0.19
AX_H   = COL_W * 0.40   # chiều cao thật của MỖI panel (inch)

TOP_IN = EDGE + LEG_H + GAP + TITLE
FIG_H  = TOP_IN + BOT + AX_H * (3 + 2 * HSPACE)

# Font sizes from paper_style.py "narrow" preset
plt.rcParams.update({
    # --- fonts: embed TrueType (no Type 3), Times-like serif ---
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Liberation Serif',
                   'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    # --- sizes from paper_style.py narrow ---
    'font.size': 7.0,
    'axes.titlesize': 8.0,
    'axes.labelsize': 8.0,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 7.0,
    'axes.titlepad': 3,
    'axes.labelpad': 2,
    # --- line / tick geometry (narrow) ---
    'axes.linewidth': 0.7,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.pad': 2, 'ytick.major.pad': 2,
    'lines.linewidth': 1.3,
    'lines.markersize': 4.0,
    'lines.markeredgewidth': 0.5,
})

# ---------------------------------------------------------------------------
# 5. Figure — 3 rows × 2 cols (MOSI, MOSEI, SIMS)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(COL_W, FIG_H), sharex=True)

mr_all = sorted(
    set(mosi_agg['Missing Rate'])
    | set(mosei_agg['Missing Rate'])
    | set(sims_agg['Missing Rate'])
)
xmin, xmax = min(mr_all), max(mr_all)


def plot_row(ax_f1, ax_mae, agg_df, models):
    for m in models:
        if m not in style_map:
            continue
        sub = agg_df[agg_df['Model'] == m].sort_values('Missing Rate')
        s = style_map[m]
        kw = dict(label=s['label'], color=s['color'], marker='o',
                  linestyle='-', markeredgecolor=s['color'], zorder=s['zorder'])
        ax_f1.plot(sub['Missing Rate'], sub['F1_score Mean'], **kw)
        ax_mae.plot(sub['Missing Rate'], sub['MAE Mean'] / 100.0, **kw)

    for ax in (ax_f1, ax_mae):
        ax.set_xlim(xmin - 0.04, xmax + 0.04)
        ax.set_xticks(np.arange(0, xmax + 1e-6, 0.2))
        ax.grid(True, linewidth=0.4, alpha=0.45, linestyle=':')
        ax.set_axisbelow(True)


plot_row(axes[0, 0], axes[0, 1], mosi_agg, mosi_models)
plot_row(axes[1, 0], axes[1, 1], mosei_agg, mosei_models)
plot_row(axes[2, 0], axes[2, 1], sims_agg, sims_models)

# --- y-limits (unchanged for MOSI / MOSEI; new for SIMS) ---
axes[0, 0].set_ylim(32, 88);    axes[0, 0].set_yticks([40, 60, 80])
axes[0, 1].set_ylim(0.70, 1.55); axes[0, 1].set_yticks([0.8, 1.0, 1.2, 1.4])
axes[1, 0].set_ylim(55, 90);    axes[1, 0].set_yticks([60, 70, 80])
axes[1, 1].set_ylim(0.50, 0.88); axes[1, 1].set_yticks([0.55, 0.65, 0.75, 0.85])
axes[2, 0].set_ylim(55, 90);    axes[2, 0].set_yticks([60, 70, 80])
axes[2, 1].set_ylim(0.50, 0.88); axes[2, 1].set_yticks([0.55, 0.65, 0.75, 0.85])

# --- panel label + dataset + metric in the title (replaces ylabel & (a)-(d) text) ---
axes[0, 0].set_title(r'(a) MOSI: F1 $\uparrow$')
axes[0, 1].set_title(r'(b) MOSI: MAE $\downarrow$')
axes[1, 0].set_title(r'(c) MOSEI: F1 $\uparrow$')
axes[1, 1].set_title(r'(d) MOSEI: MAE $\downarrow$')
axes[2, 0].set_title(r'(e) SIMS: F1 $\uparrow$')
axes[2, 1].set_title(r'(f) SIMS: MAE $\downarrow$')

# --- x label only on bottom row ---
for ax in axes[2]:
    ax.set_xlabel(r'Missing Rate $r$')

top    = 1 - TOP_IN / FIG_H
bottom = BOT / FIG_H

fig.subplots_adjust(left=0.05, right=0.985, top=top, bottom=bottom,
                    wspace=WSPACE, hspace=HSPACE)

handles, leg_labels = axes[0, 0].get_legend_handles_labels()
leg = fig.legend(
    handles, leg_labels, loc='upper center', ncol=4,
    bbox_to_anchor=(0.5, 1 - EDGE / FIG_H),
    borderaxespad=0,
    frameon=True, framealpha=0.95,
    edgecolor='#CCCCCC', borderpad=0.25, labelspacing=0.2,
    handlelength=1.5, handletextpad=0.4, columnspacing=0.8,
)
leg.get_frame().set_linewidth(0.5)

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
png_path = 'result/normal/missing_rate_performance_combined.png'
pdf_path = 'result/normal/missing_rate_performance_combined.pdf'
plt.savefig(pdf_path)                 # vector, for the paper
plt.savefig(png_path, dpi=300)        # preview only
plt.close()
print(f'Saved: {png_path} and {pdf_path}')