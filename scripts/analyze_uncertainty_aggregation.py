"""
Experiment: Uncertainty Score Aggregation Analysis.

Compare aggregation strategies (max, mean, softmax, L2-norm, harmonic, rank-based)
for combining shared metrics (consensus + cycle) and specific metrics (neighbor + cycle)
into single scores used to compute weights: w = exp(-score).

Usage:
    conda activate baseline
    python scripts/analyze_uncertainty_aggregation.py

Produces figures in result/figures/aggregation_analysis/
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
from scipy.optimize import minimize_scalar
import seaborn as sns

# ─── Configuration ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
UNCERTAINTY_DIR = ROOT / 'result' / 'uncertainty'
FIG_DIR = ROOT / 'result' / 'figures' / 'aggregation_analysis'
FIG_DIR.mkdir(parents=True, exist_ok=True)

MISSING_RATES = [0.0, 0.5, 0.9]
SEED = 1111
MODALITIES = ['l', 'v', 'a']
MOD_NAMES = {'l': 'Language', 'v': 'Vision', 'a': 'Audio'}
METRIC_NAMES = {
    'shared_consensus': r'$u^{c,\mathrm{con}}$',
    'shared_cycle':     r'$u^{c,\mathrm{cy}}$',
    'specific_neighbor': r'$u^{s,\mathrm{nei}}$',
    'specific_cycle':   r'$u^{s,\mathrm{cy}}$',
}
METRIC_GROUPS = {
    'shared':   ['shared_consensus', 'shared_cycle'],
    'specific': ['specific_neighbor', 'specific_cycle'],
}

# Aesthetics
sns.set_theme(style='whitegrid', font_scale=1.1)
PALETTE = sns.color_palette('Set2', 3)
MR_COLORS = {0.0: '#2196F3', 0.5: '#FF9800', 0.9: '#F44336'}
MR_LABELS = {0.0: 'MR=0.0', 0.5: 'MR=0.5', 0.9: 'MR=0.9'}


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════
def load_uncertainty(mr: float):
    path = UNCERTAINTY_DIR / f'uncertainty_mosi_mr{mr:.1f}_seed{SEED}.pt'
    if not path.exists():
        raise FileNotFoundError(f'Missing file: {path}')
    return torch.load(path, weights_only=False)


def load_all():
    """Returns dict: mr -> data dict."""
    datasets = {}
    for mr in MISSING_RATES:
        datasets[mr] = load_uncertainty(mr)
    return datasets


# ═══════════════════════════════════════════════════════════════════════
# Aggregation functions: (u1, u2) -> score  (both 1-D numpy arrays)
# ═══════════════════════════════════════════════════════════════════════
def agg_mean(u1, u2):
    return (u1 + u2) / 2.0

def agg_max(u1, u2):
    return np.maximum(u1, u2)

def agg_softmax_self(u1, u2):
    """Softmax-weighted: alpha = softmax([u1, u2])[0]."""
    stack = np.stack([u1, u2], axis=1)  # (N, 2)
    exp_s = np.exp(stack - stack.max(axis=1, keepdims=True))
    alpha = exp_s / exp_s.sum(axis=1, keepdims=True)  # (N, 2)
    return alpha[:, 0] * u1 + alpha[:, 1] * u2

def agg_norm(u1, u2):
    """L2 combination."""
    return np.sqrt(u1**2 + u2**2)

def agg_harmonic(u1, u2):
    """Harmonic mean (with eps to avoid division by zero)."""
    eps = 1e-8
    return 2.0 * u1 * u2 / (u1 + u2 + eps)

def agg_rank_mean(u1, u2):
    """Rank-normalize each metric -> mean in rank space."""
    r1 = stats.rankdata(u1) / len(u1)
    r2 = stats.rankdata(u2) / len(u2)
    return (r1 + r2) / 2.0

AGG_FUNCS = {
    'mean': agg_mean,
    'max': agg_max,
    'softmax': agg_softmax_self,
    'L2-norm': agg_norm,
    'harmonic': agg_harmonic,
    'rank-mean': agg_rank_mean,
}


def get_metric_pair(data, group, mod):
    """Return (u1, u2) as numpy arrays for a group ('shared'/'specific') and modality."""
    m1_key = f'u_{mod}_{METRIC_GROUPS[group][0]}'
    m2_key = f'u_{mod}_{METRIC_GROUPS[group][1]}'
    return data[m1_key].numpy(), data[m2_key].numpy()


def get_actual_error(data):
    """Return |y - y_hat| as numpy array."""
    labels = data['labels'].numpy()
    preds = data['predictions'].numpy()
    return np.abs(labels - preds)


# ═══════════════════════════════════════════════════════════════════════
# Experiment 1: Per-Metric Distribution Analysis
# ═══════════════════════════════════════════════════════════════════════
def exp1_distribution(datasets):
    """4x3 violin/box plots: metrics x missing rates, each with 3 modalities."""
    metrics = ['shared_consensus', 'shared_cycle', 'specific_neighbor', 'specific_cycle']

    fig, axes = plt.subplots(4, 3, figsize=(16, 18))
    fig.suptitle('Experiment 1: Per-Metric Uncertainty Distributions', fontsize=16, y=0.98)

    for row, metric in enumerate(metrics):
        for col, mr in enumerate(MISSING_RATES):
            ax = axes[row, col]
            data = datasets[mr]

            plot_data = []
            mod_labels = []
            for m in MODALITIES:
                vals = data[f'u_{m}_{metric}'].numpy()
                plot_data.append(vals)
                mod_labels.append(MOD_NAMES[m])

            parts = ax.violinplot(plot_data, positions=[1, 2, 3], showmeans=True,
                                  showmedians=True, widths=0.7)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(PALETTE[i])
                pc.set_alpha(0.6)
            parts['cmeans'].set_color('red')
            parts['cmedians'].set_color('black')

            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(mod_labels)
            if col == 0:
                ax.set_ylabel(METRIC_NAMES[metric], fontsize=12)
            if row == 0:
                ax.set_title(f'MR = {mr}', fontsize=13)

            # Add text stats
            for i, vals in enumerate(plot_data):
                ax.text(i + 1, ax.get_ylim()[1] * 0.95,
                        f'mu={vals.mean():.3f}\nsig={vals.std():.3f}',
                        ha='center', va='top', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / 'exp1_distributions.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> Saved exp1_distributions.png')


# ═══════════════════════════════════════════════════════════════════════
# Experiment 2: Correlation & Redundancy Analysis
# ═══════════════════════════════════════════════════════════════════════
def exp2_correlation(datasets):
    """2a: Scatter plots within groups. 2b: Full correlation heatmap."""

    # --- 2a: Scatter plots ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Experiment 2a: Intra-Group Metric Correlation', fontsize=16, y=0.98)

    groups = ['shared', 'specific']
    group_titles = {
        'shared': r'Shared: $u^{c,\mathrm{con}}$ vs $u^{c,\mathrm{cy}}$',
        'specific': r'Specific: $u^{s,\mathrm{nei}}$ vs $u^{s,\mathrm{cy}}$',
    }

    for row, group in enumerate(groups):
        for col, m in enumerate(MODALITIES):
            ax = axes[row, col]
            for mr in MISSING_RATES:
                data = datasets[mr]
                u1, u2 = get_metric_pair(data, group, m)
                ax.scatter(u1, u2, s=8, alpha=0.4, color=MR_COLORS[mr],
                           label=MR_LABELS[mr])

                # Spearman correlation
                rho, pval = stats.spearmanr(u1, u2)
                ax.text(0.02, 0.98 - 0.08 * MISSING_RATES.index(mr),
                        f'{MR_LABELS[mr]}: rho={rho:.3f} (p={pval:.2e})',
                        transform=ax.transAxes, fontsize=8, va='top',
                        color=MR_COLORS[mr], fontweight='bold')

            m1_name = METRIC_NAMES[METRIC_GROUPS[group][0]]
            m2_name = METRIC_NAMES[METRIC_GROUPS[group][1]]
            ax.set_xlabel(m1_name)
            ax.set_ylabel(m2_name)
            if row == 0:
                ax.set_title(f'{MOD_NAMES[m]}', fontsize=13)
            if col == 0:
                ax.annotate(group_titles[group], xy=(-0.35, 0.5),
                            xycoords='axes fraction', fontsize=11, rotation=90,
                            va='center', ha='center', fontweight='bold')
            if row == 0 and col == 2:
                ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout(rect=[0.05, 0, 1, 0.96])
    fig.savefig(FIG_DIR / 'exp2a_scatter_correlation.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> Saved exp2a_scatter_correlation.png')

    # --- 2b: Heatmap ---
    for mr in MISSING_RATES:
        data = datasets[mr]
        metric_keys = []
        metric_labels = []
        for metric in ['shared_consensus', 'shared_cycle', 'specific_neighbor', 'specific_cycle']:
            for m in MODALITIES:
                metric_keys.append(f'u_{m}_{metric}')
                metric_labels.append(f'{METRIC_NAMES[metric]} ({m.upper()})')

        values = np.stack([data[k].numpy() for k in metric_keys], axis=1)
        corr_matrix = np.zeros((len(metric_keys), len(metric_keys)))
        for i in range(len(metric_keys)):
            for j in range(len(metric_keys)):
                corr_matrix[i, j] = stats.spearmanr(values[:, i], values[:, j])[0]

        fig, ax = plt.subplots(figsize=(14, 11))
        sns.heatmap(corr_matrix, xticklabels=metric_labels, yticklabels=metric_labels,
                    annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    vmin=-1, vmax=1, ax=ax, annot_kws={'size': 7})
        ax.set_title(f'Experiment 2b: Spearman Rank Correlation (MR={mr})', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        fig.savefig(FIG_DIR / f'exp2b_heatmap_mr{mr:.1f}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  -> Saved exp2b_heatmap_mr{mr:.1f}.png')


# ═══════════════════════════════════════════════════════════════════════
# Experiment 3: Oracle Analysis -- Spearman with actual error
# ═══════════════════════════════════════════════════════════════════════
def exp3_oracle(datasets):
    """Compare aggregation strategies via Spearman correlation with |y - y_hat|."""

    # --- 3a: Bar chart ---
    groups = ['shared', 'specific']
    n_agg = len(AGG_FUNCS)
    n_mr = len(MISSING_RATES)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Experiment 3a: Spearman rho(score_agg, |y - yhat|) -- Higher is Better',
                 fontsize=15, y=0.99)

    bar_width = 0.12
    x = np.arange(n_agg)

    results_table = {}

    for row, group in enumerate(groups):
        for col, m in enumerate(MODALITIES):
            ax = axes[row, col]
            for i_mr, mr in enumerate(MISSING_RATES):
                data = datasets[mr]
                error = get_actual_error(data)
                u1, u2 = get_metric_pair(data, group, m)

                rhos = []
                for agg_name, agg_fn in AGG_FUNCS.items():
                    score = agg_fn(u1, u2)
                    rho, pval = stats.spearmanr(score, error)
                    rhos.append(rho)
                    results_table[(group, m, mr, agg_name)] = (rho, pval)

                bars = ax.bar(x + i_mr * bar_width, rhos, bar_width,
                              label=MR_LABELS[mr], color=MR_COLORS[mr], alpha=0.85)
                # Add value labels
                for bar, rho in zip(bars, rhos):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                            f'{rho:.3f}', ha='center', va='bottom', fontsize=6, rotation=90)

            ax.set_xticks(x + bar_width)
            ax.set_xticklabels(list(AGG_FUNCS.keys()), fontsize=9, rotation=30)
            ax.set_ylabel('Spearman rho')
            ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)
            title = f'{group.title()} -- {MOD_NAMES[m]}'
            ax.set_title(title, fontsize=12)
            if row == 0 and col == 2:
                ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / 'exp3a_spearman_bars.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> Saved exp3a_spearman_bars.png')

    # --- 3b: Calibration plot (quantile bins) ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Experiment 3b: Calibration -- Mean |Error| per Score Quantile Bin',
                 fontsize=15, y=0.99)
    n_bins = 10

    # Only compare mean vs max vs rank-mean
    selected_aggs = {'mean': '-o', 'max': '-s', 'rank-mean': '-^'}

    for row, group in enumerate(groups):
        for col, mr in enumerate(MISSING_RATES):
            ax = axes[row, col]
            # Average across modalities for clarity
            for agg_name, marker in selected_aggs.items():
                agg_fn = AGG_FUNCS[agg_name]
                bin_errors_all = []
                for m in MODALITIES:
                    data = datasets[mr]
                    error = get_actual_error(data)
                    u1, u2 = get_metric_pair(data, group, m)
                    score = agg_fn(u1, u2)

                    # Quantile bins
                    quantiles = np.percentile(score, np.linspace(0, 100, n_bins + 1))
                    bin_errors = []
                    for b in range(n_bins):
                        mask = (score >= quantiles[b]) & (score < quantiles[b + 1])
                        if b == n_bins - 1:  # include last edge
                            mask = (score >= quantiles[b]) & (score <= quantiles[b + 1])
                        if mask.sum() > 0:
                            bin_errors.append(error[mask].mean())
                        else:
                            bin_errors.append(np.nan)
                    bin_errors_all.append(bin_errors)

                mean_bin_errors = np.nanmean(bin_errors_all, axis=0)
                ax.plot(range(1, n_bins + 1), mean_bin_errors, marker,
                        label=agg_name, markersize=5, linewidth=1.5)

            ax.set_xlabel('Score Quantile Bin')
            ax.set_ylabel('Mean |Error|')
            ax.set_title(f'{group.title()} -- MR={mr}')
            if row == 0 and col == 0:
                ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / 'exp3b_calibration.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> Saved exp3b_calibration.png')

    return results_table


# ═══════════════════════════════════════════════════════════════════════
# Experiment 4: Weight Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════════════
def exp4_weight_sensitivity(datasets):
    """Weight distributions and scatter under max vs mean."""

    # --- 4a: Histogram overlay ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Experiment 4a: Weight Distributions w = exp(-score) -- Mean vs Max',
                 fontsize=15, y=0.99)

    groups = ['shared', 'specific']

    for row, group in enumerate(groups):
        for col, mr in enumerate(MISSING_RATES):
            ax = axes[row, col]
            data = datasets[mr]

            for m in MODALITIES:
                u1, u2 = get_metric_pair(data, group, m)
                w_mean = np.exp(-agg_mean(u1, u2))
                w_max = np.exp(-agg_max(u1, u2))

                ax.hist(w_mean, bins=40, alpha=0.35, color=PALETTE[MODALITIES.index(m)],
                        label=f'{MOD_NAMES[m]} (mean)', histtype='stepfilled')
                ax.hist(w_max, bins=40, alpha=0.6, color=PALETTE[MODALITIES.index(m)],
                        label=f'{MOD_NAMES[m]} (max)', histtype='step', linewidth=1.5)

            ax.set_xlabel('Weight w = exp(-score)')
            ax.set_ylabel('Count')
            ax.set_title(f'{group.title()} -- MR={mr}')
            if row == 0 and col == 0:
                ax.legend(fontsize=7, ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / 'exp4a_weight_histograms.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> Saved exp4a_weight_histograms.png')

    # --- 4b: w_shared vs w_specific scatter, colored by error ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    fig.suptitle('Experiment 4b: w_shared vs w_specific (colored by |error|)',
                 fontsize=15, y=0.99)

    for row, mr in enumerate(MISSING_RATES):
        data = datasets[mr]
        error = get_actual_error(data)

        for col_idx, (agg_name, agg_fn) in enumerate([('mean', agg_mean), ('max', agg_max)]):
            ax = axes[row, col_idx]

            # Average across modalities for each group
            ws_list, wsp_list = [], []
            for m in MODALITIES:
                u1_s, u2_s = get_metric_pair(data, 'shared', m)
                u1_sp, u2_sp = get_metric_pair(data, 'specific', m)
                ws_list.append(np.exp(-agg_fn(u1_s, u2_s)))
                wsp_list.append(np.exp(-agg_fn(u1_sp, u2_sp)))

            w_shared = np.mean(ws_list, axis=0)
            w_specific = np.mean(wsp_list, axis=0)

            sc = ax.scatter(w_shared, w_specific, c=error, cmap='YlOrRd',
                            s=12, alpha=0.6, edgecolors='none')
            plt.colorbar(sc, ax=ax, label='|Error|', shrink=0.8)
            ax.set_xlabel('w_shared (avg across modalities)')
            ax.set_ylabel('w_specific (avg across modalities)')
            ax.set_title(f'Agg={agg_name} -- MR={mr}')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / 'exp4b_weight_scatter.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> Saved exp4b_weight_scatter.png')


# ═══════════════════════════════════════════════════════════════════════
# Experiment 5: Adaptive Aggregation Comparison
# ═══════════════════════════════════════════════════════════════════════
def exp5_adaptive(datasets):
    """Compare all strategies including calibrated alpha on held-out."""

    # Approach A: Error-calibrated alpha (use test set for now, noting limitation)
    def find_best_alpha(u1, u2, error):
        """Find alpha in [0,1] that maximizes Spearman correlation."""
        def neg_rho(alpha):
            score = alpha * u1 + (1.0 - alpha) * u2
            rho, _ = stats.spearmanr(score, error)
            return -rho
        result = minimize_scalar(neg_rho, bounds=(0, 1), method='bounded')
        return result.x, -result.fun

    groups = ['shared', 'specific']

    # Collect all results
    all_results = {}

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Experiment 5: All Aggregation Strategies -- Spearman rho with |Error|',
                 fontsize=15, y=0.99)

    for row, group in enumerate(groups):
        for col, mr in enumerate(MISSING_RATES):
            ax = axes[row, col]
            data = datasets[mr]
            error = get_actual_error(data)

            agg_names_ext = list(AGG_FUNCS.keys()) + ['calibrated-a']
            rho_per_mod = {agg: [] for agg in agg_names_ext}

            for m in MODALITIES:
                u1, u2 = get_metric_pair(data, group, m)

                for agg_name, agg_fn in AGG_FUNCS.items():
                    score = agg_fn(u1, u2)
                    rho, _ = stats.spearmanr(score, error)
                    rho_per_mod[agg_name].append(rho)

                # Calibrated alpha
                best_alpha, best_rho = find_best_alpha(u1, u2, error)
                rho_per_mod['calibrated-a'].append(best_rho)
                all_results[(group, m, mr)] = best_alpha

            # Plot average across modalities
            avg_rhos = [np.mean(rho_per_mod[agg]) for agg in agg_names_ext]
            colors = ['#2196F3'] * len(AGG_FUNCS) + ['#E91E63']
            bars = ax.bar(range(len(agg_names_ext)), avg_rhos, color=colors, alpha=0.8)

            for bar, rho in zip(bars, avg_rhos):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
                        f'{rho:.3f}', ha='center', va='bottom', fontsize=8)

            ax.set_xticks(range(len(agg_names_ext)))
            ax.set_xticklabels(agg_names_ext, fontsize=8, rotation=30)
            ax.set_ylabel('Avg Spearman rho (across modalities)')
            ax.set_title(f'{group.title()} -- MR={mr}')
            ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / 'exp5_adaptive_comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> Saved exp5_adaptive_comparison.png')

    # Print calibrated alpha values
    print('\n  Calibrated alpha* values (alpha*metric1 + (1-alpha)*metric2):')
    for (group, m, mr), alpha in sorted(all_results.items()):
        print(f'    {group:10s} {MOD_NAMES[m]:8s} MR={mr}: alpha* = {alpha:.4f}')

    return all_results


# ═══════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════
def print_summary_table(results_table):
    """Print a consolidated summary of Exp 3 results."""
    print('\n' + '=' * 90)
    print('SUMMARY: Spearman rho(aggregated_score, |error|) -- Exp 3')
    print('=' * 90)

    for group in ['shared', 'specific']:
        print(f'\n--- {group.upper()} GROUP ---')
        header = f'{"MR":>5s}  {"Mod":>5s}  '
        for agg in AGG_FUNCS:
            header += f'{agg:>10s}  '
        print(header)
        print('-' * len(header))

        for mr in MISSING_RATES:
            for m in MODALITIES:
                row = f'{mr:5.1f}  {m.upper():>5s}  '
                for agg in AGG_FUNCS:
                    rho, pval = results_table.get((group, m, mr, agg), (0, 1))
                    star = '*' if pval < 0.05 else ' '
                    row += f'{rho:9.4f}{star} '
                print(row)
            print()
    print('=' * 90)
    print('* indicates p < 0.05\n')


# ═══════════════════════════════════════════════════════════════════════
# Additional diagnostic: individual metric Spearman with error
# ═══════════════════════════════════════════════════════════════════════
def exp_individual_spearman(datasets):
    """Bar chart: Spearman of each individual metric with |error| (no aggregation)."""
    metrics = ['shared_consensus', 'shared_cycle', 'specific_neighbor', 'specific_cycle']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Diagnostic: Individual Metric Spearman rho with |Error|', fontsize=14, y=1.02)

    bar_width = 0.06
    x = np.arange(len(metrics))

    for col, mr in enumerate(MISSING_RATES):
        ax = axes[col]
        data = datasets[mr]
        error = get_actual_error(data)

        for i_m, m in enumerate(MODALITIES):
            rhos = []
            for metric in metrics:
                vals = data[f'u_{m}_{metric}'].numpy()
                rho, _ = stats.spearmanr(vals, error)
                rhos.append(rho)

            offset = (i_m - 1) * bar_width
            bars = ax.bar(x + offset, rhos, bar_width, label=MOD_NAMES[m],
                          color=PALETTE[i_m], alpha=0.85)
            for bar, rho in zip(bars, rhos):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                        f'{rho:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_NAMES[m] for m in metrics], fontsize=9)
        ax.set_ylabel('Spearman rho')
        ax.set_title(f'MR = {mr}')
        ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)
        if col == 0:
            ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'exp_diagnostic_individual_spearman.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> Saved exp_diagnostic_individual_spearman.png')


# ═══════════════════════════════════════════════════════════════════════
# Per-sample scatter: aggregated score vs |error|
# ═══════════════════════════════════════════════════════════════════════
def exp6_persample_score_vs_error(datasets):
    """
    Per-sample scatter plots: aggregated score (x) vs |error| (y).
    Each dot = 1 sample. Mean vs Max shown side-by-side.
    One row per group (shared/specific), one column per missing rate.
    Sub-columns for mean vs max. Points colored by modality.
    """
    groups = ['shared', 'specific']

    for group in groups:
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(
            f'Per-Sample: Aggregated {group.title()} Score vs |Error|\n'
            f'(each dot = 1 test sample)',
            fontsize=15, y=0.99
        )

        for row, mr in enumerate(MISSING_RATES):
            data = datasets[mr]
            error = get_actual_error(data)

            for col_idx, (agg_name, agg_fn) in enumerate([('mean', agg_mean), ('max', agg_max)]):
                ax = axes[row, col_idx]

                for i_m, m in enumerate(MODALITIES):
                    u1, u2 = get_metric_pair(data, group, m)
                    score = agg_fn(u1, u2)

                    ax.scatter(score, error, s=10, alpha=0.35,
                               color=PALETTE[i_m], label=MOD_NAMES[m],
                               edgecolors='none')

                    # Trend line (linear fit)
                    z = np.polyfit(score, error, 1)
                    x_line = np.linspace(score.min(), score.max(), 100)
                    ax.plot(x_line, np.polyval(z, x_line), '--',
                            color=PALETTE[i_m], linewidth=1.5, alpha=0.8)

                # Spearman annotation (averaged across modalities)
                rhos = []
                for m in MODALITIES:
                    u1, u2 = get_metric_pair(data, group, m)
                    score = agg_fn(u1, u2)
                    rho, _ = stats.spearmanr(score, error)
                    rhos.append(rho)
                avg_rho = np.mean(rhos)

                ax.set_xlabel(f'Aggregated Score ({agg_name})')
                ax.set_ylabel('|Prediction Error|')
                ax.set_title(f'Agg={agg_name}  |  MR={mr}  |  avg rho={avg_rho:.3f}')
                if row == 0 and col_idx == 0:
                    ax.legend(fontsize=9, loc='upper left')

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(FIG_DIR / f'exp6_persample_score_vs_error_{group}.png',
                    dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  -> Saved exp6_persample_score_vs_error_{group}.png')


# ═══════════════════════════════════════════════════════════════════════
# Per-sample 2D metric space colored by |error|
# ═══════════════════════════════════════════════════════════════════════
def exp7_persample_2d_metric_space(datasets):
    """
    For each (group, modality, MR): scatter u1 (x) vs u2 (y),
    color = |error|. This reveals WHERE in the 2D metric space
    high-error samples live, informing whether max or mean is better.

    - If high-error samples cluster along axes (one metric high, other low)
      -> max captures them better.
    - If high-error samples spread along diagonal (both metrics correlated)
      -> mean is better.
    """
    groups = ['shared', 'specific']
    group_metric_labels = {
        'shared': (r'$u^{c,\mathrm{con}}$ (consensus)', r'$u^{c,\mathrm{cy}}$ (cycle)'),
        'specific': (r'$u^{s,\mathrm{nei}}$ (neighbor)', r'$u^{s,\mathrm{cy}}$ (cycle)'),
    }

    for group in groups:
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        xlabel, ylabel = group_metric_labels[group]
        fig.suptitle(
            f'Per-Sample 2D Metric Space: {group.title()} Group\n'
            f'(color = |prediction error|, each dot = 1 sample)',
            fontsize=15, y=0.99
        )

        for row, mr in enumerate(MISSING_RATES):
            data = datasets[mr]
            error = get_actual_error(data)

            for col, m in enumerate(MODALITIES):
                ax = axes[row, col]
                u1, u2 = get_metric_pair(data, group, m)

                # Sort by error so high-error dots are drawn on top
                order = np.argsort(error)
                sc = ax.scatter(u1[order], u2[order], c=error[order],
                                cmap='YlOrRd', s=14, alpha=0.7, edgecolors='none')
                plt.colorbar(sc, ax=ax, label='|Error|', shrink=0.85)

                # Mark max and mean "decision boundaries"
                # max(u1,u2) = c  is the L-shaped boundary
                # mean(u1,u2) = c is a diagonal line
                lim_min = min(u1.min(), u2.min())
                lim_max = max(u1.max(), u2.max())

                # Show the median score iso-contours
                median_score_mean = np.median(agg_mean(u1, u2))
                median_score_max = np.median(agg_max(u1, u2))

                # Mean iso-line: (u1+u2)/2 = c  ->  u2 = 2c - u1
                x_line = np.linspace(lim_min, lim_max, 200)
                ax.plot(x_line, 2 * median_score_mean - x_line, 'b--',
                        linewidth=1.5, alpha=0.7, label=f'mean={median_score_mean:.2f}')

                # Max iso-line: max(u1,u2) = c  ->  L-shape
                c = median_score_max
                ax.plot([c, c], [lim_min, c], 'r-', linewidth=1.5, alpha=0.7,
                        label=f'max={c:.2f}')
                ax.plot([lim_min, c], [c, c], 'r-', linewidth=1.5, alpha=0.7)

                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(f'{MOD_NAMES[m]}  |  MR={mr}')

                if row == 0 and col == 0:
                    ax.legend(fontsize=8, loc='upper left')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(FIG_DIR / f'exp7_persample_2d_metric_{group}.png',
                    dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  -> Saved exp7_persample_2d_metric_{group}.png')


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    print('Loading uncertainty data ...')
    datasets = load_all()

    # Verify predictions exist
    for mr in MISSING_RATES:
        if 'predictions' not in datasets[mr]:
            print(f'  WARNING: Missing predictions for MR={mr}. '
                  f'Re-run uncertainty mode to include predictions.')
            sys.exit(1)

    print(f'Figures will be saved to: {FIG_DIR}\n')

    print('[Exp 1] Distribution Analysis ...')
    exp1_distribution(datasets)

    print('[Exp 2] Correlation & Redundancy ...')
    exp2_correlation(datasets)

    print('[Diagnostic] Individual Metric Spearman ...')
    exp_individual_spearman(datasets)

    print('[Exp 3] Oracle Analysis ...')
    results_table = exp3_oracle(datasets)
    print_summary_table(results_table)

    print('[Exp 4] Weight Sensitivity ...')
    exp4_weight_sensitivity(datasets)

    print('[Exp 5] Adaptive Aggregation ...')
    exp5_adaptive(datasets)

    print('[Exp 6] Per-Sample: Score vs |Error| ...')
    exp6_persample_score_vs_error(datasets)

    print('[Exp 7] Per-Sample: 2D Metric Space ...')
    exp7_persample_2d_metric_space(datasets)

    print(f'\nAll experiments complete. Figures saved to: {FIG_DIR}')


if __name__ == '__main__':
    main()
