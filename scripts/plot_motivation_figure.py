"""
Motivation Figure: Publication-quality grouped bar chart showing
shared vs specific uncertainty dynamics.

Subfigure (a): MR=0.0, Sample 642 — shared metrics dominate (u_con >> u_nei)
Subfigure (b): Text-only missing at 0.5, Sample 354, Ckpt=0.0 — specific neighbor dominates

Colors: Audio #CE4256, Text #CE4256, Visual #027313
"""

import sys
sys.path.insert(0, '/home/tien/NCKH/DecAlign')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from pathlib import Path

# ─── Configuration ───
ROOT = Path('/home/tien/NCKH/DecAlign')
SCRATCH = Path('/home/tien/.gemini/antigravity-ide/brain/7630bf75-42e9-479d-82ef-795adcea4994/scratch')
FIG_DIR = ROOT / 'result' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODALITIES = ['l', 'v', 'a']
MOD_LABELS = {'l': 'Text (T)', 'v': 'Visual (V)', 'a': 'Audio (A)'}
MOD_SHORT = {'l': 'T', 'v': 'V', 'a': 'A'}
# User-specified colors
MOD_COLORS = {'l': '#CE4256', 'v': '#027313', 'a': '#CE4256'}

# Metric bar colors and styles
METRIC_STYLES = {
    'u_con': {'color': '#90CAF9', 'edgecolor': '#1565C0', 'hatch': '///', 'label': r'$u^{\mathrm{con}}$'},
    'u_cy_c': {'color': '#42A5F5', 'edgecolor': '#0D47A1', 'hatch': '',    'label': r'$u^{c,\mathrm{cy}}$'},
    'u_nei': {'color': '#FFAB91', 'edgecolor': '#D84315', 'hatch': '\\\\\\', 'label': r'$u^{\mathrm{nei}}$'},
    'u_cy_s': {'color': '#FF7043', 'edgecolor': '#BF360C', 'hatch': '',    'label': r'$u^{s,\mathrm{cy}}$'},
}

# RC Params
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'figure.dpi': 300,
})


def load_subfig_a_data():
    """Load MR=0.0 uncertainty data."""
    path = ROOT / 'result' / 'uncertainty' / 'uncertainty_mosi_mr0.0_seed1111.pt'
    return torch.load(path, weights_only=False)


def load_subfig_b_data():
    """Load single-modality-missing inference data."""
    path = SCRATCH / 'motivation_all_data.pt'
    all_data = torch.load(path, weights_only=False)
    return all_data['subfig_b']


def get_sample_metrics(data, idx):
    """Extract the 4 raw uncertainty metrics for a given sample index."""
    metrics = {}
    for m in MODALITIES:
        metrics[m] = {
            'u_con': data[f'u_{m}_shared_consensus'][idx].item(),
            'u_cy_c': data[f'u_{m}_shared_cycle'][idx].item(),
            'u_nei': data[f'u_{m}_specific_neighbor'][idx].item(),
            'u_cy_s': data[f'u_{m}_specific_cycle'][idx].item(),
        }
    return metrics


def compute_weights_for_sample(metrics, tau_c=1.0, tau_s=1.0, r_min=0.1):
    """Compute r_shared, r_specific from the 4 metrics."""
    weights = {}
    for m in MODALITIES:
        u_c = 0.5 * (metrics[m]['u_con'] + metrics[m]['u_cy_c'])
        u_s = 0.5 * (metrics[m]['u_nei'] + metrics[m]['u_cy_s'])
        r_c = max(r_min, min(1.0, np.exp(-u_c / tau_c)))
        r_s = max(r_min, min(1.0, np.exp(-u_s / tau_s)))
        weights[m] = {'r_shared': r_c, 'r_specific': r_s}
    return weights


def plot_motivation_figure():
    """Create the 2-subfigure motivation figure."""
    
    # Load data
    data_a = load_subfig_a_data()
    data_b_all = load_subfig_b_data()
    
    # Selected samples
    IDX_A = 642  # MR=0.0: high u_con, low u_nei, error=0.009
    IDX_B = 354  # Text-stressed, ckpt=0.0: u_nei=1.23 >> u_con=0.75
    
    data_b = data_b_all['l_ckpt0.0']  # Text stressed, checkpoint 0.0
    
    metrics_a = get_sample_metrics(data_a, IDX_A)
    metrics_b = get_sample_metrics(data_b, IDX_B)
    weights_a = compute_weights_for_sample(metrics_a)
    weights_b = compute_weights_for_sample(metrics_b)
    
    error_a = abs(data_a['labels'][IDX_A].item() - data_a['predictions'][IDX_A].item())
    error_b = abs(data_b['labels'][IDX_B].item() - data_b['predictions'][IDX_B].item())
    
    # ─── Figure Layout ───
    fig = plt.figure(figsize=(14, 5.5))
    gs = gridspec.GridSpec(1, 2, wspace=0.30, left=0.06, right=0.97, top=0.85, bottom=0.12)
    
    metric_keys = ['u_con', 'u_cy_c', 'u_nei', 'u_cy_s']
    n_metrics = len(metric_keys)
    n_mods = len(MODALITIES)
    x_mods = np.arange(n_mods)
    bar_w = 0.18
    
    # ════════════════════════════════════════════════════════════════
    # Subfigure (a): No Missing — Shared metrics dominate
    # ════════════════════════════════════════════════════════════════
    ax_a = fig.add_subplot(gs[0])
    
    for j, mk in enumerate(metric_keys):
        style = METRIC_STYLES[mk]
        vals = [metrics_a[m][mk] for m in MODALITIES]
        offset = (j - 1.5) * bar_w
        bars = ax_a.bar(
            x_mods + offset, vals, bar_w,
            color=style['color'], edgecolor=style['edgecolor'],
            hatch=style['hatch'], linewidth=0.8, label=style['label'],
            zorder=3,
        )
        for bar, val in zip(bars, vals):
            ax_a.text(
                bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7.5,
                fontweight='bold', color=style['edgecolor'],
            )
    
    # Annotate: u_con is dominant
    max_u_con_m = max(MODALITIES, key=lambda m: metrics_a[m]['u_con'])
    max_u_con_val = metrics_a[max_u_con_m]['u_con']
    max_u_con_x = MODALITIES.index(max_u_con_m) + (0 - 1.5) * bar_w
    
    ax_a.annotate(
        'Shared consensus\nis high',
        xy=(max_u_con_x, max_u_con_val),
        xytext=(max_u_con_x + 0.45, max_u_con_val + 0.15),
        ha='center', fontsize=8, color='#1565C0', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5),
    )
    
    # Annotate: u_nei is low for audio
    min_u_nei_m = min(MODALITIES, key=lambda m: metrics_a[m]['u_nei'])
    min_u_nei_val = metrics_a[min_u_nei_m]['u_nei']
    min_u_nei_x = MODALITIES.index(min_u_nei_m) + (2 - 1.5) * bar_w
    
    ax_a.annotate(
        'Neighbor uncertainty\nis low',
        xy=(min_u_nei_x, min_u_nei_val + 0.03),
        xytext=(min_u_nei_x - 0.3, min_u_nei_val + 0.35),
        ha='center', fontsize=8, color='#D84315', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#D84315', lw=1.5),
    )
    
    ax_a.set_xticks(x_mods)
    ax_a.set_xticklabels([MOD_LABELS[m] for m in MODALITIES], fontsize=10)
    ax_a.set_ylabel('Uncertainty Score', fontsize=11)
    ax_a.set_title('(a) No Missing (MR = 0.0)', fontsize=13, fontweight='bold', pad=12)
    ax_a.legend(fontsize=8.5, loc='upper left', ncol=2, framealpha=0.92, edgecolor='grey',
                handlelength=1.8, handletextpad=0.5)
    ax_a.set_ylim(0, max(max(metrics_a[m][mk] for mk in metric_keys) for m in MODALITIES) * 1.45)
    ax_a.grid(axis='y', alpha=0.3, zorder=0)
    ax_a.set_axisbelow(True)
    
    # Weight summary box
    weight_text_a = (
        f'$r^c$:  T={weights_a["l"]["r_shared"]:.2f}  V={weights_a["v"]["r_shared"]:.2f}  A={weights_a["a"]["r_shared"]:.2f}\n'
        f'$r^s$:  T={weights_a["l"]["r_specific"]:.2f}  V={weights_a["v"]["r_specific"]:.2f}  A={weights_a["a"]["r_specific"]:.2f}\n'
        f'Error: {error_a:.3f}'
    )
    ax_a.text(
        0.98, 0.98, weight_text_a,
        transform=ax_a.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', alpha=0.9, edgecolor='#90CAF9'),
        family='monospace',
    )
    
    # ════════════════════════════════════════════════════════════════
    # Subfigure (b): Text Missing — Specific neighbor dominates
    # ════════════════════════════════════════════════════════════════
    ax_b = fig.add_subplot(gs[1])
    
    for j, mk in enumerate(metric_keys):
        style = METRIC_STYLES[mk]
        vals = [metrics_b[m][mk] for m in MODALITIES]
        offset = (j - 1.5) * bar_w
        bars = ax_b.bar(
            x_mods + offset, vals, bar_w,
            color=style['color'], edgecolor=style['edgecolor'],
            hatch=style['hatch'], linewidth=0.8, label=style['label'],
            zorder=3,
        )
        for bar, val in zip(bars, vals):
            ax_b.text(
                bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7.5,
                fontweight='bold', color=style['edgecolor'],
            )
    
    # Highlight Text as the stressed modality with a background band
    ax_b.axvspan(-0.5, 0.5, alpha=0.08, color='#CE4256', zorder=0)
    ax_b.text(0, -0.08, '▼ Missing', ha='center', va='top', fontsize=7.5,
              color='#CE4256', fontweight='bold', transform=ax_b.get_xaxis_transform())
    
    # Annotate: u_nei for Text is very high
    u_nei_text_val = metrics_b['l']['u_nei']
    u_nei_text_x = 0 + (2 - 1.5) * bar_w  # modality 0, metric 2
    
    ax_b.annotate(
        f'Neighbor uncertainty\nskyrockets ({u_nei_text_val:.2f})',
        xy=(u_nei_text_x, u_nei_text_val),
        xytext=(u_nei_text_x + 0.55, u_nei_text_val + 0.12),
        ha='center', fontsize=8, color='#D84315', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#D84315', lw=1.5),
    )
    
    # Annotate: u_con for Text is lower than u_nei
    u_con_text_val = metrics_b['l']['u_con']
    u_con_text_x = 0 + (0 - 1.5) * bar_w
    
    ax_b.annotate(
        f'Consensus\nstays moderate',
        xy=(u_con_text_x, u_con_text_val),
        xytext=(u_con_text_x - 0.3, u_con_text_val + 0.35),
        ha='center', fontsize=8, color='#1565C0', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5),
    )
    
    ax_b.set_xticks(x_mods)
    ax_b.set_xticklabels([MOD_LABELS[m] for m in MODALITIES], fontsize=10)
    ax_b.set_ylabel('Uncertainty Score', fontsize=11)
    ax_b.set_title('(b) Text Missing (MR = 0.5, single modality)', fontsize=13, fontweight='bold', pad=12)
    ax_b.legend(fontsize=8.5, loc='upper left', ncol=2, framealpha=0.92, edgecolor='grey',
                handlelength=1.8, handletextpad=0.5)
    ax_b.set_ylim(0, max(max(metrics_b[m][mk] for mk in metric_keys) for m in MODALITIES) * 1.35)
    ax_b.grid(axis='y', alpha=0.3, zorder=0)
    ax_b.set_axisbelow(True)
    
    # Weight summary box
    weight_text_b = (
        f'$r^c$:  T={weights_b["l"]["r_shared"]:.2f}  V={weights_b["v"]["r_shared"]:.2f}  A={weights_b["a"]["r_shared"]:.2f}\n'
        f'$r^s$:  T={weights_b["l"]["r_specific"]:.2f}  V={weights_b["v"]["r_specific"]:.2f}  A={weights_b["a"]["r_specific"]:.2f}\n'
        f'Error: {error_b:.3f}'
    )
    ax_b.text(
        0.98, 0.98, weight_text_b,
        transform=ax_b.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF3E0', alpha=0.9, edgecolor='#FFAB91'),
        family='monospace',
    )
    
    # ─── Suptitle ───
    fig.suptitle(
        'Motivation: Shared vs. Specific Uncertainty Under Missing Modality',
        fontsize=14, fontweight='bold', y=0.97,
    )
    
    # Save
    fig_path = FIG_DIR / 'motivation_figure.pdf'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig_path_png = FIG_DIR / 'motivation_figure.png'
    fig.savefig(fig_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {fig_path}')
    print(f'Saved: {fig_path_png}')
    
    # Also print the selected data for paper reference
    print("\n=== SUBFIGURE (a): Sample 642, MR=0.0 ===")
    for m in MODALITIES:
        print(f"  {MOD_LABELS[m]}: u_con={metrics_a[m]['u_con']:.4f} u_cy_c={metrics_a[m]['u_cy_c']:.4f} "
              f"u_nei={metrics_a[m]['u_nei']:.4f} u_cy_s={metrics_a[m]['u_cy_s']:.4f} "
              f"| r_c={weights_a[m]['r_shared']:.4f} r_s={weights_a[m]['r_specific']:.4f}")
    print(f"  Error: {error_a:.4f}")
    
    print(f"\n=== SUBFIGURE (b): Sample 354, Text missing at 0.5, Ckpt=0.0 ===")
    for m in MODALITIES:
        print(f"  {MOD_LABELS[m]}: u_con={metrics_b[m]['u_con']:.4f} u_cy_c={metrics_b[m]['u_cy_c']:.4f} "
              f"u_nei={metrics_b[m]['u_nei']:.4f} u_cy_s={metrics_b[m]['u_cy_s']:.4f} "
              f"| r_c={weights_b[m]['r_shared']:.4f} r_s={weights_b[m]['r_specific']:.4f}")
    print(f"  Error: {error_b:.4f}")


if __name__ == '__main__':
    plot_motivation_figure()
