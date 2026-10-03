"""
Redundancy vs Complementarity Analysis of u^{c,con} and u^{s,nei}.

Experiment 1:
    Compute JS divergence and Spearman correlation between the shared-consensus
    uncertainty (u^{c,con}) and the specific-neighbor uncertainty (u^{s,nei})
    across missing rates [0.0, 0.5, 0.7, 0.9].  Plot both metrics vs missing rate.

Experiment 2:
    Classify each sample into one of four quadrants based on whether its
    shared-branch and specific-branch uncertainty scores are above or below
    the median threshold:
        N1 — shared HIGH, specific LOW   (shared branch catches failure)
        N2 — shared LOW,  specific HIGH  (specific branch catches failure)
        N3 — both HIGH                   (both branches flag issue)
        N4 — both LOW                    (neither branch flags issue)
    Repeat this analysis under two settings:
        (A) Without u_cycle:  shared = u^{c,con},         specific = u^{s,nei}
        (B) With u_cycle:     shared = 0.5*(u_con+u_cy_c), specific = 0.5*(u_nei+u_cy_s)
    If N1 + N2 is large, the two metrics are complementary (both needed);
    if N3 + N4 dominates, they are redundant.

Usage:
    conda activate baseline
    python scripts/analyze_redundancy_complementarity.py

Produces figures in result/figures/redundancy_analysis/
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import jensenshannon
import seaborn as sns

# ─── Configuration ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
UNCERTAINTY_DIR = ROOT / 'result' / 'uncertainty'
FIG_DIR = ROOT / 'result' / 'figures' / 'redundancy_analysis'
FIG_DIR.mkdir(parents=True, exist_ok=True)

MISSING_RATES = [0.0, 0.5, 0.7, 0.9]
SEED = 1111
MODALITIES = ['l', 'v', 'a']
MOD_NAMES = {'l': 'Language', 'v': 'Vision', 'a': 'Audio'}

# Aesthetics
sns.set_theme(style='whitegrid', font_scale=1.15)
PALETTE = {'l': '#4CAF50', 'v': '#2196F3', 'a': '#FF9800'}
MOD_MARKERS = {'l': 'o', 'v': 's', 'a': '^'}


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════
def load_uncertainty(mr: float):
    path = UNCERTAINTY_DIR / f'uncertainty_mosi_mr{mr:.1f}_seed{SEED}.pt'
    if not path.exists():
        raise FileNotFoundError(f'Missing file: {path}')
    return torch.load(path, weights_only=False)


def load_all():
    datasets = {}
    for mr in MISSING_RATES:
        datasets[mr] = load_uncertainty(mr)
    return datasets


# ═══════════════════════════════════════════════════════════════════════
# Utility: histogram-based JS divergence
# ═══════════════════════════════════════════════════════════════════════
def js_divergence_hist(u1: np.ndarray, u2: np.ndarray, n_bins: int = 50):
    """
    Compute Jensen-Shannon divergence between two 1-D distributions
    via shared-bin histograms.
    """
    # Common support range
    lo = min(u1.min(), u2.min())
    hi = max(u1.max(), u2.max())
    bins = np.linspace(lo - 1e-8, hi + 1e-8, n_bins + 1)

    hist1, _ = np.histogram(u1, bins=bins, density=True)
    hist2, _ = np.histogram(u2, bins=bins, density=True)

    # Normalize to proper probability distributions
    hist1 = hist1 / (hist1.sum() + 1e-12)
    hist2 = hist2 / (hist2.sum() + 1e-12)

    return jensenshannon(hist1, hist2) ** 2  # JSD (squared version = true JSD)


# ═══════════════════════════════════════════════════════════════════════
# Experiment 1: JS divergence and Spearman of u^{c,con} vs u^{s,nei}
# ═══════════════════════════════════════════════════════════════════════
def experiment1_js_spearman(datasets):
    """
    For each modality and missing rate, compute:
      - JSD(u^{c,con}_m, u^{s,nei}_m)
      - Spearman(u^{c,con}_m, u^{s,nei}_m)
    Plot both as line charts across missing rate.
    """
    print('[Exp 1] JS divergence & Spearman: u^{c,con} vs u^{s,nei}')

    results = {m: {'js': [], 'spearman': [], 'pval': []} for m in MODALITIES}

    for mr in MISSING_RATES:
        data = datasets[mr]
        for m in MODALITIES:
            u_con = data[f'u_{m}_shared_consensus'].numpy()
            u_nei = data[f'u_{m}_specific_neighbor'].numpy()

            js = js_divergence_hist(u_con, u_nei)
            rho, pval = stats.spearmanr(u_con, u_nei)

            results[m]['js'].append(js)
            results[m]['spearman'].append(rho)
            results[m]['pval'].append(pval)

    # ─── Figure: 2 subplots side by side ──────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        r'Experiment 1: Divergence & Correlation between $u^{c,\mathrm{con}}$ and $u^{s,\mathrm{nei}}$',
        fontsize=15, y=1.02
    )

    for m in MODALITIES:
        ax1.plot(MISSING_RATES, results[m]['js'], marker=MOD_MARKERS[m],
                 color=PALETTE[m], linewidth=2, markersize=8,
                 label=MOD_NAMES[m])
        ax2.plot(MISSING_RATES, results[m]['spearman'], marker=MOD_MARKERS[m],
                 color=PALETTE[m], linewidth=2, markersize=8,
                 label=MOD_NAMES[m])

    ax1.set_xlabel('Missing Rate')
    ax1.set_ylabel('Jensen-Shannon Divergence')
    ax1.set_title(r'JSD($u^{c,\mathrm{con}}_m$, $u^{s,\mathrm{nei}}_m$)')
    ax1.legend()
    ax1.set_xticks(MISSING_RATES)

    ax2.set_xlabel('Missing Rate')
    ax2.set_ylabel('Spearman ρ')
    ax2.set_title(r'Spearman($u^{c,\mathrm{con}}_m$, $u^{s,\mathrm{nei}}_m$)')
    ax2.legend()
    ax2.set_xticks(MISSING_RATES)
    ax2.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'exp1_js_spearman_vs_mr.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'exp1_js_spearman_vs_mr.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> Saved exp1_js_spearman_vs_mr.png/pdf')

    # Print numeric table
    print('\n  ┌──────────┬────────┬────────────┬────────────┬──────────┐')
    print('  │  MR      │ Modal  │  JSD       │  Spearman  │  p-value │')
    print('  ├──────────┼────────┼────────────┼────────────┼──────────┤')
    for i, mr in enumerate(MISSING_RATES):
        for m in MODALITIES:
            js = results[m]['js'][i]
            rho = results[m]['spearman'][i]
            pv = results[m]['pval'][i]
            sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
            print(f'  │  {mr:.1f}     │  {m.upper()}     │  {js:.6f}  │  {rho:+.4f}   │  {pv:.2e} {sig}│')
    print('  └──────────┴────────┴────────────┴────────────┴──────────┘')

    return results


# ═══════════════════════════════════════════════════════════════════════
# Experiment 2: Quadrant Classification (N1-N4)
# ═══════════════════════════════════════════════════════════════════════
def _classify_quadrants(u_shared: np.ndarray, u_specific: np.ndarray,
                        threshold_type: str = 'median'):
    """
    Classify samples into 4 quadrants based on uncertainty scores.
    
    Args:
        u_shared: (N,) shared-branch uncertainty
        u_specific: (N,) specific-branch uncertainty
        threshold_type: 'median' or 'mean'
    
    Returns:
        labels: (N,) array with values in {1,2,3,4} for N1-N4
        thresholds: (thresh_shared, thresh_specific)
    """
    if threshold_type == 'median':
        t_s = np.median(u_shared)
        t_sp = np.median(u_specific)
    else:
        t_s = np.mean(u_shared)
        t_sp = np.mean(u_specific)

    high_shared = u_shared >= t_s
    high_specific = u_specific >= t_sp

    labels = np.zeros(len(u_shared), dtype=int)
    labels[high_shared & ~high_specific] = 1   # N1: shared HIGH, spec LOW
    labels[~high_shared & high_specific] = 2   # N2: shared LOW, spec HIGH
    labels[high_shared & high_specific]  = 3   # N3: both HIGH
    labels[~high_shared & ~high_specific] = 4  # N4: both LOW

    return labels, (t_s, t_sp)


def experiment2_quadrants(datasets):
    """
    Classify samples into N1-N4 quadrants under:
      (A) Without u_cycle: shared = u^{c,con}, specific = u^{s,nei}
      (B) With u_cycle:    shared = 0.5*(u_con + u_cy_c), specific = 0.5*(u_nei + u_cy_s)
    
    Plot stacked bar charts and scatter quadrant visualizations.
    """
    print('\n[Exp 2] Quadrant Classification (N1–N4)')

    quadrant_names = {1: 'N1 (S↑ Sp↓)', 2: 'N2 (S↓ Sp↑)',
                      3: 'N3 (both↑)', 4: 'N4 (both↓)'}
    quadrant_colors = {1: '#E91E63', 2: '#9C27B0', 3: '#FF5722', 4: '#607D8B'}

    settings = {
        'Without $u_{cycle}$': 'no_cycle',
        'With $u_{cycle}$': 'with_cycle',
    }

    all_counts = {}  # (setting, mr, mod) -> {1: count, 2: count, ...}

    for setting_label, setting_key in settings.items():
        for mr in MISSING_RATES:
            data = datasets[mr]
            for m in MODALITIES:
                if setting_key == 'no_cycle':
                    u_sh = data[f'u_{m}_shared_consensus'].numpy()
                    u_sp = data[f'u_{m}_specific_neighbor'].numpy()
                else:
                    # With cycle: aggregated scores (already 0.5*(con+cy_c) and 0.5*(nei+cy_s))
                    u_sh = data[f'u_{m}_shared'].numpy()
                    u_sp = data[f'u_{m}_specific'].numpy()

                labels, thresholds = _classify_quadrants(u_sh, u_sp)
                counts = {q: int((labels == q).sum()) for q in [1, 2, 3, 4]}
                all_counts[(setting_key, mr, m)] = counts

    # ─── Figure 2a: Stacked bar chart ─────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'Experiment 2a: Sample Quadrant Distribution (N1–N4)\n'
        r'N1: $u_{shared}$↑ $u_{specific}$↓  |  '
        r'N2: $u_{shared}$↓ $u_{specific}$↑  |  '
        r'N3: both↑  |  N4: both↓',
        fontsize=14, y=1.01
    )

    for row, (setting_label, setting_key) in enumerate(settings.items()):
        for col, m in enumerate(MODALITIES):
            ax = axes[row, col]

            n1_pcts, n2_pcts, n3_pcts, n4_pcts = [], [], [], []
            for mr in MISSING_RATES:
                counts = all_counts[(setting_key, mr, m)]
                total = sum(counts.values())
                n1_pcts.append(counts[1] / total * 100)
                n2_pcts.append(counts[2] / total * 100)
                n3_pcts.append(counts[3] / total * 100)
                n4_pcts.append(counts[4] / total * 100)

            x = np.arange(len(MISSING_RATES))
            width = 0.55

            ax.bar(x, n4_pcts, width, label='N4 (both↓)',
                   color=quadrant_colors[4], alpha=0.85)
            ax.bar(x, n3_pcts, width, bottom=n4_pcts,
                   label='N3 (both↑)', color=quadrant_colors[3], alpha=0.85)
            bottom_n2 = [a + b for a, b in zip(n4_pcts, n3_pcts)]
            ax.bar(x, n2_pcts, width, bottom=bottom_n2,
                   label='N2 (S↓ Sp↑)', color=quadrant_colors[2], alpha=0.85)
            bottom_n1 = [a + b for a, b in zip(bottom_n2, n2_pcts)]
            ax.bar(x, n1_pcts, width, bottom=bottom_n1,
                   label='N1 (S↑ Sp↓)', color=quadrant_colors[1], alpha=0.85)

            # Annotate N1+N2 percentage
            for i, mr in enumerate(MISSING_RATES):
                complementary = n1_pcts[i] + n2_pcts[i]
                ax.text(i, 102, f'{complementary:.1f}%',
                        ha='center', va='bottom', fontsize=9, fontweight='bold',
                        color='#D32F2F')

            ax.set_xticks(x)
            ax.set_xticklabels([f'{mr}' for mr in MISSING_RATES])
            ax.set_xlabel('Missing Rate')
            ax.set_ylabel('% of Samples')
            ax.set_ylim(0, 115)
            ax.set_title(f'{setting_label} — {MOD_NAMES[m]}', fontsize=12)
            if row == 0 and col == 2:
                ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.35, 1.0))

    # Add row label: N1+N2 = complementary ratio (red annotation)
    fig.text(0.5, -0.02,
             'Red numbers above bars: N1+N2 (complementary %) — higher means metrics are more complementary',
             ha='center', fontsize=11, color='#D32F2F', style='italic')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(FIG_DIR / 'exp2a_quadrant_bars.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'exp2a_quadrant_bars.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> Saved exp2a_quadrant_bars.png/pdf')

    # ─── Figure 2b: Scatter quadrant visualization ────────────────
    # 4 columns (missing rates) x 2 rows (with/without cycle), pick one modality = avg
    for m in MODALITIES:
        fig, axes = plt.subplots(2, len(MISSING_RATES), figsize=(5 * len(MISSING_RATES), 10))
        fig.suptitle(
            f'Experiment 2b: Quadrant Scatter — {MOD_NAMES[m]}\n'
            r'$x = u_{shared}$, $y = u_{specific}$  (dashed = median threshold)',
            fontsize=14, y=1.02
        )

        for row, (setting_label, setting_key) in enumerate(settings.items()):
            for col, mr in enumerate(MISSING_RATES):
                ax = axes[row, col]
                data = datasets[mr]
                error = np.abs(data['labels'].numpy() - data['predictions'].numpy())

                if setting_key == 'no_cycle':
                    u_sh = data[f'u_{m}_shared_consensus'].numpy()
                    u_sp = data[f'u_{m}_specific_neighbor'].numpy()
                else:
                    u_sh = data[f'u_{m}_shared'].numpy()
                    u_sp = data[f'u_{m}_specific'].numpy()

                labels, (t_s, t_sp) = _classify_quadrants(u_sh, u_sp)

                # Scatter colored by quadrant
                for q in [4, 3, 1, 2]:  # draw N4 first (background), then N1/N2 on top
                    mask = labels == q
                    ax.scatter(u_sh[mask], u_sp[mask], s=14, alpha=0.6,
                               color=quadrant_colors[q], label=quadrant_names[q],
                               edgecolors='none')

                # Threshold lines
                ax.axvline(x=t_s, color='black', linestyle='--', linewidth=1, alpha=0.7)
                ax.axhline(y=t_sp, color='black', linestyle='--', linewidth=1, alpha=0.7)

                counts = all_counts[(setting_key, mr, m)]
                total = sum(counts.values())
                complementary_pct = (counts[1] + counts[2]) / total * 100

                ax.set_xlabel(r'$u_{shared}$')
                ax.set_ylabel(r'$u_{specific}$')
                ax.set_title(
                    f'{setting_label}\nMR={mr}  |  N1+N2={complementary_pct:.1f}%',
                    fontsize=10
                )
                if row == 0 and col == len(MISSING_RATES) - 1:
                    ax.legend(fontsize=7, loc='upper right')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(FIG_DIR / f'exp2b_quadrant_scatter_{m}.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(FIG_DIR / f'exp2b_quadrant_scatter_{m}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  -> Saved exp2b_quadrant_scatter_{m}.png/pdf')

    # ─── Figure 2c: Prediction error by quadrant ──────────────────
    fig, axes = plt.subplots(2, len(MISSING_RATES), figsize=(5 * len(MISSING_RATES), 10))
    fig.suptitle(
        'Experiment 2c: Mean |Prediction Error| by Quadrant\n'
        r'(averaged across modalities L/V/A)',
        fontsize=14, y=1.02
    )

    for row, (setting_label, setting_key) in enumerate(settings.items()):
        for col, mr in enumerate(MISSING_RATES):
            ax = axes[row, col]
            data = datasets[mr]
            error = np.abs(data['labels'].numpy() - data['predictions'].numpy())

            # Average quadrant assignment across modalities
            quad_errors = {q: [] for q in [1, 2, 3, 4]}
            quad_counts = {q: 0 for q in [1, 2, 3, 4]}

            for m in MODALITIES:
                if setting_key == 'no_cycle':
                    u_sh = data[f'u_{m}_shared_consensus'].numpy()
                    u_sp = data[f'u_{m}_specific_neighbor'].numpy()
                else:
                    u_sh = data[f'u_{m}_shared'].numpy()
                    u_sp = data[f'u_{m}_specific'].numpy()

                labels_q, _ = _classify_quadrants(u_sh, u_sp)
                for q in [1, 2, 3, 4]:
                    mask = labels_q == q
                    if mask.sum() > 0:
                        quad_errors[q].append(error[mask].mean())
                        quad_counts[q] += int(mask.sum())

            mean_errors = []
            quadrant_labels_plot = []
            colors_plot = []
            for q in [1, 2, 3, 4]:
                if quad_errors[q]:
                    mean_errors.append(np.mean(quad_errors[q]))
                else:
                    mean_errors.append(0)
                quadrant_labels_plot.append(quadrant_names[q])
                colors_plot.append(quadrant_colors[q])

            bars = ax.bar(range(4), mean_errors, color=colors_plot, alpha=0.85)
            for bar, err in zip(bars, mean_errors):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{err:.3f}', ha='center', va='bottom', fontsize=9)

            ax.set_xticks(range(4))
            ax.set_xticklabels([quadrant_names[q] for q in [1, 2, 3, 4]],
                               fontsize=8, rotation=15)
            ax.set_ylabel('Mean |Prediction Error|')
            ax.set_title(f'{setting_label}\nMR={mr}', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / 'exp2c_error_by_quadrant.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'exp2c_error_by_quadrant.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> Saved exp2c_error_by_quadrant.png/pdf')

    # ─── Print summary table ──────────────────────────────────────
    print('\n  Quadrant Distribution Summary:')
    print('  ' + '=' * 90)
    print(f'  {"Setting":<22s} {"MR":<5s} {"Mod":<5s} '
          f'{"N1(%)":<8s} {"N2(%)":<8s} {"N3(%)":<8s} {"N4(%)":<8s} '
          f'{"N1+N2(%)":<10s}')
    print('  ' + '-' * 90)

    for setting_label, setting_key in settings.items():
        for mr in MISSING_RATES:
            for m in MODALITIES:
                counts = all_counts[(setting_key, mr, m)]
                total = sum(counts.values())
                pcts = {q: counts[q] / total * 100 for q in [1, 2, 3, 4]}
                comp = pcts[1] + pcts[2]
                label = 'No cycle' if setting_key == 'no_cycle' else 'With cycle'
                print(f'  {label:<22s} {mr:<5.1f} {m.upper():<5s} '
                      f'{pcts[1]:<8.1f} {pcts[2]:<8.1f} {pcts[3]:<8.1f} {pcts[4]:<8.1f} '
                      f'{comp:<10.1f}')
            print()
    print('  ' + '=' * 90)

    return all_counts


# ═══════════════════════════════════════════════════════════════════════
# Experiment 2d: Delta analysis — effect of adding u_cycle
# ═══════════════════════════════════════════════════════════════════════
def experiment2d_cycle_effect(datasets):
    """
    Analyze how adding u_cycle changes the quadrant membership.
    Show the shift in N1+N2 (complementary ratio) with vs without cycle.
    """
    print('\n[Exp 2d] Effect of u_cycle on complementarity')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        r'Effect of Adding $u_{cycle}$ on Complementarity (N1+N2 %)',
        fontsize=14, y=1.03
    )

    for col, m in enumerate(MODALITIES):
        ax = axes[col]
        comp_no_cycle = []
        comp_with_cycle = []

        for mr in MISSING_RATES:
            data = datasets[mr]

            # Without cycle
            u_sh_nc = data[f'u_{m}_shared_consensus'].numpy()
            u_sp_nc = data[f'u_{m}_specific_neighbor'].numpy()
            labels_nc, _ = _classify_quadrants(u_sh_nc, u_sp_nc)
            n1n2_nc = ((labels_nc == 1) | (labels_nc == 2)).sum() / len(labels_nc) * 100

            # With cycle
            u_sh_wc = data[f'u_{m}_shared'].numpy()
            u_sp_wc = data[f'u_{m}_specific'].numpy()
            labels_wc, _ = _classify_quadrants(u_sh_wc, u_sp_wc)
            n1n2_wc = ((labels_wc == 1) | (labels_wc == 2)).sum() / len(labels_wc) * 100

            comp_no_cycle.append(n1n2_nc)
            comp_with_cycle.append(n1n2_wc)

        x = np.arange(len(MISSING_RATES))
        width = 0.3

        ax.bar(x - width/2, comp_no_cycle, width,
               label=r'Without $u_{cycle}$', color='#42A5F5', alpha=0.85)
        ax.bar(x + width/2, comp_with_cycle, width,
               label=r'With $u_{cycle}$', color='#EF5350', alpha=0.85)

        # Annotate delta
        for i in range(len(MISSING_RATES)):
            delta = comp_with_cycle[i] - comp_no_cycle[i]
            y_pos = max(comp_no_cycle[i], comp_with_cycle[i]) + 1
            ax.text(i, y_pos, f'Δ={delta:+.1f}%',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([f'{mr}' for mr in MISSING_RATES])
        ax.set_xlabel('Missing Rate')
        ax.set_ylabel('N1+N2 (%)')
        ax.set_title(f'{MOD_NAMES[m]}', fontsize=12)
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'exp2d_cycle_effect.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'exp2d_cycle_effect.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> Saved exp2d_cycle_effect.png/pdf')


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    print('Loading uncertainty data ...')
    datasets = load_all()

    # Verify data integrity
    for mr in MISSING_RATES:
        if 'predictions' not in datasets[mr]:
            print(f'  WARNING: Missing predictions for MR={mr}.')
            sys.exit(1)

    print(f'Figures will be saved to: {FIG_DIR}\n')

    # Experiment 1
    exp1_results = experiment1_js_spearman(datasets)

    # Experiment 2
    exp2_counts = experiment2_quadrants(datasets)

    # Experiment 2d: Cycle effect analysis
    experiment2d_cycle_effect(datasets)

    print(f'\nAll experiments complete. Figures saved to: {FIG_DIR}')


if __name__ == '__main__':
    main()
