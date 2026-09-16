"""
Case Study & Fusion Weight Analysis.

Computes fusion weights (r_m^c, r_m^s) from saved uncertainty metrics,
finds failure/success case studies, and produces publication-quality figures:

  A. Qualitative Case Study panels (similar to CICA paper Figure)
  B. Fusion Weight vs Missing Rate trend chart
  C. Per-sample weight distribution violin plots

Usage:
    conda activate baseline
    python scripts/analyze_case_study.py [--tau_c 1.0] [--tau_s 1.0] [--r_min 0.1]

Output: result/figures/case_study/
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import torch
import seaborn as sns

# ─── Configuration ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
UNCERTAINTY_DIR = ROOT / 'result' / 'uncertainty'
FIG_DIR = ROOT / 'result' / 'figures' / 'case_study'
FIG_DIR.mkdir(parents=True, exist_ok=True)

MISSING_RATES = [round(x * 0.1, 1) for x in range(10)]
SEED = 1111
MODALITIES = ['l', 'v', 'a']
MOD_NAMES = {'l': 'Text (T)', 'v': 'Video (V)', 'a': 'Audio (A)'}
MOD_NAMES_SHORT = {'l': 'T', 'v': 'V', 'a': 'A'}
MOD_COLORS = {'l': '#4CAF50', 'v': '#2196F3', 'a': '#F44336'}

# Aesthetics
sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.titleweight'] = 'bold'


# ═══════════════════════════════════════════════════════════════════════
# Data loading & weight computation
# ═══════════════════════════════════════════════════════════════════════
def load_uncertainty(mr: float, uncertainty_dir=None):
    if uncertainty_dir is None:
        uncertainty_dir = UNCERTAINTY_DIR
    path = uncertainty_dir / f'uncertainty_mosi_mr{mr:.1f}_seed{SEED}.pt'
    if not path.exists():
        raise FileNotFoundError(f'Missing file: {path}')
    return torch.load(path, weights_only=False)


def compute_weights_from_data(data, tau_c=1.0, tau_s=1.0, r_min=0.1):
    """
    Compute fusion weights from raw uncertainty metrics in a data dict.
    Works on pre-existing .pt files that may not have fusion weights yet.

    Returns updated data dict with added keys:
        u_{m}_shared, u_{m}_specific, r_{m}_shared, r_{m}_specific
    """
    for m in MODALITIES:
        # Aggregated uncertainties
        u_c = 0.5 * (data[f'u_{m}_shared_consensus'] + data[f'u_{m}_shared_cycle'])
        u_s = 0.5 * (data[f'u_{m}_specific_neighbor'] + data[f'u_{m}_specific_cycle'])

        # Fusion weights
        r_c = torch.clamp(torch.exp(-u_c / tau_c), min=r_min, max=1.0)
        r_s = torch.clamp(torch.exp(-u_s / tau_s), min=r_min, max=1.0)

        data[f'u_{m}_shared'] = u_c
        data[f'u_{m}_specific'] = u_s
        data[f'r_{m}_shared'] = r_c
        data[f'r_{m}_specific'] = r_s

    return data


def load_all(tau_c=1.0, tau_s=1.0, r_min=0.1, missing_rates=None, uncertainty_dir=None):
    """Load and compute weights for all missing rates."""
    if missing_rates is None:
        missing_rates = MISSING_RATES
    datasets = {}
    for mr in missing_rates:
        data = load_uncertainty(mr, uncertainty_dir=uncertainty_dir)
        data = compute_weights_from_data(data, tau_c, tau_s, r_min)
        datasets[mr] = data
    return datasets


def get_error(data):
    """Per-sample absolute prediction error."""
    return (data['labels'] - data['predictions']).abs().numpy()


# ═══════════════════════════════════════════════════════════════════════
# A. Qualitative Case Study (CICA-style 3-panel)
# ═══════════════════════════════════════════════════════════════════════
def find_case_studies(data, n_failure=3, n_conflict=3):
    """
    Find case studies:
      - conflict (CICA-style success): Unimodal conflict, Avg Fusion gives wrong sentiment sign
        or large error, but DecAlign is correct because it suppresses corrupted modalities.
      - failure: Samples where DecAlign has the highest prediction error.
    """
    error = get_error(data)
    preds = data['predictions'].numpy()
    labels = data['labels'].numpy()

    # Failure cases: top-N highest error
    failure_idx = np.argsort(-error)[:n_failure]

    # Conflict cases (CICA-style input conflict)
    has_uni = 'pred_l' in data and 'pred_v' in data and 'pred_a' in data
    if has_uni:
        pred_l = data['pred_l'].numpy()
        pred_v = data['pred_v'].numpy()
        pred_a = data['pred_a'].numpy()
        pred_avg = (pred_l + pred_v + pred_a) / 3.0

        err_avg = np.abs(labels - pred_avg)
        improvement = err_avg - error

        signs = np.sign(np.stack([pred_l, pred_v, pred_a], axis=1))
        conflict_mask = (signs.min(axis=1) * signs.max(axis=1) < 0)
        gt_sign = np.sign(labels)
        model_sign_correct = (np.sign(preds) == gt_sign) & (np.abs(labels) > 0.2)
        avg_sign_wrong = (np.sign(pred_avg) != gt_sign)
        ideal_mask = conflict_mask & model_sign_correct & avg_sign_wrong

        if ideal_mask.sum() >= n_conflict:
            cand = np.where(ideal_mask)[0]
            conflict_idx = cand[np.argsort(-improvement[cand])][:n_conflict]
        else:
            cand = np.where(conflict_mask)[0] if conflict_mask.sum() > 0 else np.arange(len(labels))
            conflict_idx = cand[np.argsort(-improvement[cand])][:n_conflict]
    else:
        low_error_mask = error < np.percentile(error, 30)
        r_all = np.stack([
            data[f'r_{m}_shared'].numpy() for m in MODALITIES
        ] + [
            data[f'r_{m}_specific'].numpy() for m in MODALITIES
        ], axis=1)
        weight_range = r_all.max(axis=1) - r_all.min(axis=1)
        weight_range[~low_error_mask] = -1
        conflict_idx = np.argsort(-weight_range)[:n_conflict]

    return failure_idx, conflict_idx


def plot_case_study_panel(data, sample_idx, case_type, fig_path, mr=0.5, model_name='DecAlign'):
    """
    Plot a single case study sample in CICA-style 3-panel layout.

    Panel (a): The Input Conflict (uni_pred)
    Panel (b): Perception & Decision Metrics
    Panel (c): Sentiment Outcome Comparison (Ground Truth vs Avg Fusion vs Model)
    """
    labels = data['labels'].numpy()
    preds = data['predictions'].numpy()
    error = get_error(data)

    i = sample_idx
    sample_id = data['ids'][i] if data['ids'] else f'Sample {i}'
    sample_id = str(sample_id).replace('$_$', '_').replace('$', '')
    label_val = labels[i]
    pred_val = preds[i]
    err_val = error[i]

    has_uni = 'pred_l' in data and 'pred_v' in data and 'pred_a' in data
    if has_uni:
        pred_l = data['pred_l'][i].item()
        pred_v = data['pred_v'][i].item()
        pred_a = data['pred_a'][i].item()
        uni_preds = {'l': pred_l, 'v': pred_v, 'a': pred_a}
        avg_pred = (pred_l + pred_v + pred_a) / 3.0
    else:
        avg_pred = pred_val

    # Collect per-modality metrics for this sample
    u_shared = {m: data[f'u_{m}_shared'][i].item() for m in MODALITIES}
    u_specific = {m: data[f'u_{m}_specific'][i].item() for m in MODALITIES}
    r_shared = {m: data[f'r_{m}_shared'][i].item() for m in MODALITIES}
    r_specific = {m: data[f'r_{m}_specific'][i].item() for m in MODALITIES}

    # --- Figure ---
    fig = plt.figure(figsize=(19, 5.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.35, 0.9], wspace=0.32)

    # ── Panel (a): The Input Conflict (uni_pred) ──
    ax_a = fig.add_subplot(gs[0])
    x_mods = np.arange(len(MODALITIES))
    bar_w = 0.52

    if has_uni:
        uni_vals = [uni_preds[m] for m in MODALITIES]
        bar_colors_a = ['#C8E6C9', '#FFCDD2', '#BBDEFB']
        edge_colors_a = ['#2E7D32', '#C62828', '#1565C0']
        hatches_a = ['///', '///', 'xxx']

        bars_a = ax_a.bar(x_mods, uni_vals, bar_w, color=bar_colors_a,
                          edgecolor=edge_colors_a, hatch=hatches_a, linewidth=1.5)

        for bar, val, m in zip(bars_a, uni_vals, MODALITIES):
            tag = MOD_NAMES_SHORT[m]
            if abs(val) < 0.2:
                s_label = 'Neutral'
            elif val > 0:
                s_label = 'Positive!' if val > 0.8 else 'Positive'
            else:
                s_label = 'Negative!' if val < -0.8 else 'Negative'

            y_pos = val + 0.08 if val >= 0 else val - 0.20
            ax_a.text(bar.get_x() + bar.get_width() / 2., y_pos,
                      f'($\\hat{{y}}_{{{tag}}}$): {val:+.3f}\n({s_label})',
                      ha='center', va='bottom' if val >= 0 else 'top',
                      fontsize=8.5, fontweight='bold')

        ax_a.set_xticks(x_mods)
        ax_a.set_xticklabels([MOD_NAMES[m] for m in MODALITIES], fontsize=10.5)
        ax_a.set_ylabel('Unimodal Sentiment Prediction', fontsize=11)
        ax_a.set_title('(a) The Input Conflict (uni_pred)', fontsize=12.5, fontweight='bold')
        ax_a.axhline(y=0, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
        y_a_min = min(min(uni_vals), 0) - 0.60
        y_a_max = max(max(uni_vals), 0) + 0.60
        ax_a.set_ylim(y_a_min, y_a_max)
    else:
        # Fallback to raw uncertainty metrics if uni_preds not available
        u_shared_con = {m: data[f'u_{m}_shared_consensus'][i].item() for m in MODALITIES}
        u_shared_cy  = {m: data[f'u_{m}_shared_cycle'][i].item() for m in MODALITIES}
        u_spec_nei   = {m: data[f'u_{m}_specific_neighbor'][i].item() for m in MODALITIES}
        u_spec_cy    = {m: data[f'u_{m}_specific_cycle'][i].item() for m in MODALITIES}
        metric_vals = [
            [u_shared_con[m] for m in MODALITIES],
            [u_shared_cy[m] for m in MODALITIES],
            [u_spec_nei[m] for m in MODALITIES],
            [u_spec_cy[m] for m in MODALITIES],
        ]
        metrics_labels = [r'$u^{c,\mathrm{con}}$', r'$u^{c,\mathrm{cy}}$',
                          r'$u^{s,\mathrm{nei}}$', r'$u^{s,\mathrm{cy}}$']
        metric_colors = ['#81D4FA', '#42A5F5', '#FFAB91', '#FF7043']
        hatches = ['///', '', '\\\\\\', '']
        w = 0.18
        for j, (vals, label, color, hatch) in enumerate(
                zip(metric_vals, metrics_labels, metric_colors, hatches)):
            offset = (j - 1.5) * w
            bars = ax_a.bar(x_mods + offset, vals, w, label=label,
                            color=color, hatch=hatch, edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax_a.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                          f'{val:.2f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
        ax_a.set_xticks(x_mods)
        ax_a.set_xticklabels([MOD_NAMES[m] for m in MODALITIES], fontsize=10)
        ax_a.set_ylabel('Uncertainty Score', fontsize=10)
        ax_a.set_title('(a) Raw Uncertainty Metrics', fontsize=12, fontweight='bold')
        ax_a.legend(fontsize=8, loc='upper left', ncol=2, framealpha=0.9)
        ax_a.set_ylim(0, max(max(v) for v in metric_vals) * 1.35)

    # ── Panel (b): DecAlign Perception & Decision Metrics ──
    ax_b = fig.add_subplot(gs[1])
    width_b = 0.18
    vals_u_shared = [u_shared[m] for m in MODALITIES]
    vals_u_specific = [u_specific[m] for m in MODALITIES]
    vals_r_shared = [r_shared[m] for m in MODALITIES]
    vals_r_specific = [r_specific[m] for m in MODALITIES]

    bars_us = ax_b.bar(x_mods - 1.5 * width_b, vals_u_shared, width_b,
                       label=r'Shared Unc. ($u_m^c$)', color='#90CAF9',
                       hatch='///', edgecolor='#1976D2', linewidth=0.8)
    bars_usp = ax_b.bar(x_mods - 0.5 * width_b, vals_u_specific, width_b,
                        label=r'Specific Unc. ($u_m^s$)', color='#FFE082',
                        hatch='\\\\\\', edgecolor='#FFA000', linewidth=0.8)
    bars_rs = ax_b.bar(x_mods + 0.5 * width_b, vals_r_shared, width_b,
                       label=r'Shared Wt. ($r_m^c$)', color='#81C784',
                       edgecolor='#2E7D32', linewidth=0.8)
    bars_rsp = ax_b.bar(x_mods + 1.5 * width_b, vals_r_specific, width_b,
                        label=r'Specific Wt. ($r_m^s$)', color='#FF8A65',
                        edgecolor='#D84315', linewidth=0.8)

    for bars, vals in [(bars_us, vals_u_shared), (bars_usp, vals_u_specific),
                       (bars_rs, vals_r_shared), (bars_rsp, vals_r_specific)]:
        for bar, val in zip(bars, vals):
            ax_b.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.015,
                      f'{val:.2f}', ha='center', va='bottom', fontsize=8,
                      fontweight='bold')

    # Annotate high uncertainty, suppressed, amplified
    all_unc = [(vals_u_shared[k], 'c', k) for k in range(3)] + [(vals_u_specific[k], 's', k) for k in range(3)]
    max_u_val, max_u_type, max_u_mod = max(all_unc, key=lambda x: x[0])
    offset_u = -1.5 * width_b if max_u_type == 'c' else -0.5 * width_b
    ax_b.annotate(f'High uncertainty\n{max_u_val:.2f}',
                  xy=(max_u_mod + offset_u, max_u_val),
                  xytext=(max_u_mod + offset_u, max_u_val + 0.22),
                  ha='center', fontsize=7.5, color='#D32F2F', fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color='#757575', lw=1.8))

    all_w = [(vals_r_shared[k], 'c', k) for k in range(3)] + [(vals_r_specific[k], 's', k) for k in range(3)]
    min_w_val, min_w_type, min_w_mod = min(all_w, key=lambda x: x[0])
    offset_w = 0.5 * width_b if min_w_type == 'c' else 1.5 * width_b
    ax_b.annotate(f'suppressed\n{min_w_val:.2f}',
                  xy=(min_w_mod + offset_w, min_w_val),
                  xytext=(min_w_mod + offset_w, min_w_val + 0.22),
                  ha='center', fontsize=7.5, color='#D32F2F', fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color='#757575', lw=1.8))

    max_w_val, max_w_type, max_w_mod = max(all_w, key=lambda x: x[0])
    offset_max_w = 0.5 * width_b if max_w_type == 'c' else 1.5 * width_b
    ax_b.annotate(f'amplified\n{max_w_val:.2f}',
                  xy=(max_w_mod + offset_max_w, max_w_val),
                  xytext=(max_w_mod + offset_max_w, max_w_val + 0.22),
                  ha='center', fontsize=7.5, color='#2E7D32', fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color='#757575', lw=1.8))

    ax_b.set_xticks(x_mods)
    ax_b.set_xticklabels([MOD_NAMES[m] for m in MODALITIES], fontsize=10.5)
    ax_b.set_ylabel('Score / Weight', fontsize=11)
    ax_b.set_title(f'(b) {model_name} Perception & Decision Metrics', fontsize=12.5, fontweight='bold')
    ax_b.legend(fontsize=7.5, loc='upper left', ncol=2, framealpha=0.9, edgecolor='grey')
    all_vals_b = vals_u_shared + vals_u_specific + vals_r_shared + vals_r_specific
    ax_b.set_ylim(0, max(all_vals_b) * 1.55)

    # ── Panel (c): Sentiment Outcome Comparison ──
    ax_c = fig.add_subplot(gs[2])

    if has_uni:
        bar_model_label = f'{model_name} (Ours)' if model_name == 'DecAlign' else model_name
        bar_labels = ['Ground Truth', 'Avg Fusion', bar_model_label]
        bar_vals = [label_val, avg_pred, pred_val]
        bar_colors = ['#FFCDD2', '#E0E0E0', '#FF8A65']
        bar_edges = ['#C62828', '#757575', '#D84315']
        bar_hatches = ['///', '\\\\\\', '']
    else:
        bar_labels = ['Ground Truth', 'Model Output']
        bar_vals = [label_val, pred_val]
        bar_colors = ['#E8EAF6', '#FFCDD2']
        bar_edges = ['#3F51B5', '#E53935']
        bar_hatches = ['', '']

    bars_c = ax_c.bar(range(len(bar_labels)), bar_vals, color=bar_colors,
                      edgecolor=bar_edges, hatch=bar_hatches, linewidth=1.5,
                      width=0.55)

    for bar, val in zip(bars_c, bar_vals):
        if abs(val) < 0.2:
            sentiment = 'Neutral'
        elif val > 0:
            sentiment = 'Positive'
        else:
            sentiment = 'Negative'

        y_pos = bar.get_height() + 0.08 if val >= 0 else bar.get_height() - 0.22
        ax_c.text(bar.get_x() + bar.get_width() / 2., y_pos,
                  f'{sentiment}\n({val:+.3f})', ha='center',
                  va='bottom' if val >= 0 else 'top',
                  fontsize=9, fontweight='bold')

    ax_c.set_xticks(range(len(bar_labels)))
    ax_c.set_xticklabels(bar_labels, fontsize=9.5, fontweight='bold')
    ax_c.set_ylabel('Final Sentiment Score', fontsize=11)
    ax_c.set_title('(c) Sentiment Outcome Comparison', fontsize=12.5, fontweight='bold')
    ax_c.axhline(y=0, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)

    y_min = min(0, min(bar_vals)) - 0.65
    y_max = max(0, max(bar_vals)) + 0.65
    ax_c.set_ylim(y_min, y_max)

    # --- Suptitle ---
    fig.suptitle(
        f'{case_type} Case Study  |  ID: {sample_id}  |  '
        f'Error = {err_val:.3f}  |  MR = {mr}',
        fontsize=13, fontweight='bold', y=0.98,
        color='#D32F2F' if case_type == 'Failure' else '#2E7D32'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fig_path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def run_case_studies(datasets, mr=0.5, n_failure=3, n_conflict=3, fig_dir=None, model_name='DecAlign'):
    """Find and plot case study panels."""
    if fig_dir is None:
        fig_dir = FIG_DIR
    data = datasets[mr]
    failure_idx, conflict_idx = find_case_studies(data, n_failure=n_failure, n_conflict=n_conflict)

    print(f'\n  Conflict / Resolution cases (CICA-style, MR={mr}):')
    error = get_error(data)
    for rank, idx in enumerate(conflict_idx):
        pred_l_str = f"T={data['pred_l'][idx]:.2f}, V={data['pred_v'][idx]:.2f}, A={data['pred_a'][idx]:.2f}" if 'pred_l' in data else ""
        print(f'    #{rank+1}: sample {idx} | ID={data["ids"][idx]} | '
              f'label={data["labels"][idx]:.3f} | pred={data["predictions"][idx]:.3f} | '
              f'error={error[idx]:.3f} | {pred_l_str}')
        fig_path = fig_dir / f'case_conflict_{rank+1}_mr{mr:.1f}.png'
        plot_case_study_panel(data, idx, 'Conflict Resolution', fig_path, mr=mr, model_name=model_name)
        print(f'    -> Saved {fig_path.name}')

    print(f'\n  Failure cases (MR={mr}):')
    for rank, idx in enumerate(failure_idx):
        pred_l_str = f"T={data['pred_l'][idx]:.2f}, V={data['pred_v'][idx]:.2f}, A={data['pred_a'][idx]:.2f}" if 'pred_l' in data else ""
        print(f'    #{rank+1}: sample {idx} | ID={data["ids"][idx]} | '
              f'label={data["labels"][idx]:.3f} | pred={data["predictions"][idx]:.3f} | '
              f'error={error[idx]:.3f} | {pred_l_str}')
        fig_path = fig_dir / f'case_failure_{rank+1}_mr{mr:.1f}.png'
        plot_case_study_panel(data, idx, 'Failure', fig_path, mr=mr, model_name=model_name)
        print(f'    -> Saved {fig_path.name}')


# ═══════════════════════════════════════════════════════════════════════
# B. Weight vs Missing Rate Trend Chart
# ═══════════════════════════════════════════════════════════════════════
def plot_weight_vs_missing_rate(datasets, missing_rates=None, fig_dir=None):
    """
    Line chart: mean fusion weight vs missing rate.
    6 lines: 3 modalities × 2 types (shared solid, specific dashed).
    Shows how shared and specific weights diverge under increasing corruption.
    """
    if fig_dir is None:
        fig_dir = FIG_DIR
    if missing_rates is None:
        missing_rates = sorted(datasets.keys())

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # --- Panel 1: Shared & Specific weights together ---
    ax = axes[0]
    mrs = np.array(missing_rates)

    for m in MODALITIES:
        color = MOD_COLORS[m]
        # Shared weight
        r_shared_means = [datasets[mr][f'r_{m}_shared'].mean().item() for mr in missing_rates]
        r_shared_stds = [datasets[mr][f'r_{m}_shared'].std().item() for mr in missing_rates]

        # Specific weight
        r_specific_means = [datasets[mr][f'r_{m}_specific'].mean().item() for mr in missing_rates]
        r_specific_stds = [datasets[mr][f'r_{m}_specific'].std().item() for mr in missing_rates]

        ax.plot(mrs, r_shared_means, 'o-', color=color, linewidth=2.2, markersize=7,
                label=f'{MOD_NAMES_SHORT[m]} shared ($r_m^c$)')
        ax.fill_between(mrs,
                         np.array(r_shared_means) - np.array(r_shared_stds),
                         np.array(r_shared_means) + np.array(r_shared_stds),
                         alpha=0.12, color=color)

        ax.plot(mrs, r_specific_means, 's--', color=color, linewidth=2.2, markersize=7,
                label=f'{MOD_NAMES_SHORT[m]} specific ($r_m^s$)')
        ax.fill_between(mrs,
                         np.array(r_specific_means) - np.array(r_specific_stds),
                         np.array(r_specific_means) + np.array(r_specific_stds),
                         alpha=0.08, color=color)

    ax.set_xlabel('Missing Rate', fontsize=12)
    ax.set_ylabel('Mean Fusion Weight', fontsize=12)
    ax.set_title('Fusion Weights vs Missing Rate', fontsize=13, fontweight='bold')
    ax.set_xticks(mrs)
    ax.legend(fontsize=8, ncol=2, loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Gap between shared and specific ---
    ax2 = axes[1]

    for m in MODALITIES:
        color = MOD_COLORS[m]
        gaps = []
        for mr in missing_rates:
            r_c_mean = datasets[mr][f'r_{m}_shared'].mean().item()
            r_s_mean = datasets[mr][f'r_{m}_specific'].mean().item()
            gaps.append(r_c_mean - r_s_mean)

        ax2.plot(mrs, gaps, 'D-', color=color, linewidth=2.2, markersize=7,
                 label=f'{MOD_NAMES_SHORT[m]}: $r_m^c - r_m^s$')

    ax2.axhline(y=0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.set_xlabel('Missing Rate', fontsize=12)
    ax2.set_ylabel('Weight Gap (shared − specific)', fontsize=12)
    ax2.set_title('Shared vs Specific Weight Divergence', fontsize=13, fontweight='bold')
    ax2.set_xticks(mrs)
    ax2.legend(fontsize=9, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = fig_dir / 'weight_vs_missing_rate.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  -> Saved {fig_path.name}')

    # Also print numeric table
    print('\n  Mean Fusion Weights Table:')
    print(f'  {"MR":>5}  ', end='')
    for m in MODALITIES:
        print(f' {MOD_NAMES_SHORT[m]}_r^c  {MOD_NAMES_SHORT[m]}_r^s  ', end='')
    print()
    print('  ' + '-' * 55)
    for mr in missing_rates:
        print(f'  {mr:5.1f}  ', end='')
        for m in MODALITIES:
            rc = datasets[mr][f'r_{m}_shared'].mean().item()
            rs = datasets[mr][f'r_{m}_specific'].mean().item()
            print(f' {rc:.4f}  {rs:.4f}  ', end='')
        print()


# ═══════════════════════════════════════════════════════════════════════
# C. Per-Sample Weight Distribution (Violin Plots)
# ═══════════════════════════════════════════════════════════════════════
def plot_weight_distributions(datasets, mr_list=None, fig_dir=None):
    """
    Violin plots showing distribution of r_m^c and r_m^s per modality,
    across different missing rates.
    """
    if fig_dir is None:
        fig_dir = FIG_DIR
    if mr_list is None:
        all_mrs = sorted(datasets.keys())
        if len(all_mrs) <= 5:
            mr_list = all_mrs
        else:
            mr_list = [0.0, 0.3, 0.5, 0.7, 0.9]
            mr_list = [mr for mr in mr_list if mr in datasets]

    n_cols = len(mr_list)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.8 * n_cols, 10), squeeze=False)
    fig.suptitle('Per-Sample Fusion Weight Distributions',
                 fontsize=15, fontweight='bold', y=0.99)

    weight_types = [('shared', r'$r_m^c$ (Shared Weight)'),
                    ('specific', r'$r_m^s$ (Specific Weight)')]

    for row, (wtype, title) in enumerate(weight_types):
        for col, mr in enumerate(mr_list):
            ax = axes[row, col]
            data = datasets[mr]

            plot_data = []
            positions = []
            colors = []
            tick_labels = []

            for i, m in enumerate(MODALITIES):
                vals = data[f'r_{m}_{wtype}'].numpy()
                plot_data.append(vals)
                positions.append(i)
                colors.append(MOD_COLORS[m])
                tick_labels.append(MOD_NAMES[m])

            parts = ax.violinplot(plot_data, positions=positions,
                                  showmeans=True, showmedians=True,
                                  widths=0.65)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.6)
                pc.set_edgecolor('white')
            parts['cmeans'].set_color('#D32F2F')
            parts['cmeans'].set_linewidth(2)
            parts['cmedians'].set_color('black')

            # Overlay box plot for quartiles
            bp = ax.boxplot(plot_data, positions=positions, widths=0.15,
                            patch_artist=True, showfliers=False,
                            medianprops={'color': 'black', 'linewidth': 1.5},
                            boxprops={'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'grey'},
                            whiskerprops={'color': 'grey'},
                            capprops={'color': 'grey'})

            # Add mean value annotation
            for i, vals in enumerate(plot_data):
                mean_val = vals.mean()
                std_val = vals.std()
                ax.text(positions[i], ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.05,
                        f'μ={mean_val:.3f}\nσ={std_val:.3f}',
                        ha='center', va='top', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  alpha=0.85, edgecolor='grey'))

            ax.set_xticks(positions)
            ax.set_xticklabels(tick_labels, fontsize=10)
            ax.set_ylim(-0.05, 1.15)
            if col == 0:
                ax.set_ylabel(title, fontsize=11)
            ax.set_title(f'Missing Rate = {mr}', fontsize=12, fontweight='bold')
            ax.axhline(y=0.1, color='grey', linestyle=':', linewidth=0.8,
                       alpha=0.5, label=r'$r_{\min}$' if (row == 0 and col == 0) else '')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = fig_dir / 'weight_distributions.png'
    fig.savefig(fig_path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  -> Saved {fig_path.name}')


# ═══════════════════════════════════════════════════════════════════════
# D. Comprehensive 4-metric decomposition per missing rate
# ═══════════════════════════════════════════════════════════════════════
def plot_uncertainty_decomposition(datasets, mr_list=None, fig_dir=None):
    """
    Stacked bar chart showing how each of the 4 raw uncertainty metrics
    contributes to the aggregated shared/specific uncertainty, per modality,
    across missing rates. Helps reviewer see which metric drives the weight change.
    """
    if fig_dir is None:
        fig_dir = FIG_DIR
    if mr_list is None:
        all_mrs = sorted(datasets.keys())
        if len(all_mrs) <= 5:
            mr_list = all_mrs
        else:
            mr_list = [0.0, 0.3, 0.5, 0.7, 0.9]
            mr_list = [mr for mr in mr_list if mr in datasets]

    n_cols = len(mr_list)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.8 * n_cols, 5.5), squeeze=False)
    fig.suptitle('Uncertainty Metric Decomposition by Missing Rate',
                 fontsize=14, fontweight='bold', y=1.02)

    metric_labels = [
        (r'$u^{c,\mathrm{con}}$', 'shared_consensus', '#BBDEFB'),
        (r'$u^{c,\mathrm{cy}}$', 'shared_cycle', '#42A5F5'),
        (r'$u^{s,\mathrm{nei}}$', 'specific_neighbor', '#FFCCBC'),
        (r'$u^{s,\mathrm{cy}}$', 'specific_cycle', '#FF7043'),
    ]

    for col, mr in enumerate(mr_list):
        ax = axes[0, col]
        data = datasets[mr]

        x = np.arange(len(MODALITIES))
        width = 0.18

        for j, (label, metric_key, color) in enumerate(metric_labels):
            vals = [data[f'u_{m}_{metric_key}'].mean().item() for m in MODALITIES]
            offset = (j - 1.5) * width
            bars = ax.bar(x + offset, vals, width, label=label, color=color,
                          edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels([MOD_NAMES[m] for m in MODALITIES], fontsize=10)
        ax.set_ylabel('Mean Uncertainty Score', fontsize=10)
        ax.set_title(f'Missing Rate = {mr}', fontsize=12, fontweight='bold')
        if col == 0:
            ax.legend(fontsize=8, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    fig_path = fig_dir / 'uncertainty_decomposition.png'
    fig.savefig(fig_path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  -> Saved {fig_path.name}')


# ═══════════════════════════════════════════════════════════════════════
# E. Print detailed weight summary
# ═══════════════════════════════════════════════════════════════════════
def print_weight_summary(datasets, missing_rates=None):
    """Print comprehensive summary of fusion weights across all missing rates."""
    if missing_rates is None:
        missing_rates = sorted(datasets.keys())
    print('\n' + '=' * 80)
    print('FUSION WEIGHT SUMMARY')
    print('=' * 80)

    for mr in missing_rates:
        data = datasets[mr]
        print(f'\n--- Missing Rate = {mr} ---')
        print(f'  {"Mod":>5}  '
              f'{"u_c":>8}  {"u_s":>8}  '
              f'{"r_c":>8}  {"r_s":>8}  '
              f'{"r_c-r_s":>8}')
        print('  ' + '-' * 55)

        for m in MODALITIES:
            u_c = data[f'u_{m}_shared'].mean().item()
            u_s = data[f'u_{m}_specific'].mean().item()
            r_c = data[f'r_{m}_shared'].mean().item()
            r_s = data[f'r_{m}_specific'].mean().item()
            gap = r_c - r_s
            print(f'  {MOD_NAMES_SHORT[m]:>5}  '
                  f'{u_c:8.4f}  {u_s:8.4f}  '
                  f'{r_c:8.4f}  {r_s:8.4f}  '
                  f'{gap:+8.4f}')

    print('\n' + '=' * 80)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Case Study & Fusion Weight Analysis')
    parser.add_argument('--model', type=str, default='dp', choices=['dp', 'dmd'],
                        help='Model to analyze: dp or dmd (default: dp)')
    parser.add_argument('--uncertainty_dir', type=str, default=None,
                        help='Directory containing uncertainty .pt files (default: auto)')
    parser.add_argument('--fig_dir', type=str, default=None,
                        help='Directory to save output figures (default: auto)')
    parser.add_argument('--tau_c', type=float, default=1.0,
                        help='Temperature for shared fusion weight')
    parser.add_argument('--tau_s', type=float, default=1.0,
                        help='Temperature for specific fusion weight')
    parser.add_argument('--r_min', type=float, default=0.1,
                        help='Minimum fusion weight (clamp floor)')
    parser.add_argument('--case_mr', type=float, default=0.5,
                        help='Missing rate for case study selection')
    parser.add_argument('--n_failure', type=int, default=3,
                        help='Number of failure cases to plot')
    parser.add_argument('--n_conflict', '--n_success', dest='n_conflict', type=int, default=3,
                        help='Number of conflict / resolution cases to plot')
    parser.add_argument('--missing_rates', type=float, nargs='+', default=None,
                        help='Missing rates to analyze (default: 0.0 to 0.9)')
    args = parser.parse_args()

    missing_rates = args.missing_rates if args.missing_rates is not None else MISSING_RATES

    model_key = args.model.lower()
    if args.uncertainty_dir is not None:
        uncertainty_dir = Path(args.uncertainty_dir)
    else:
        uncertainty_dir = ROOT / 'result' / ('uncertainty_dmd' if model_key == 'dmd' else 'uncertainty')

    if args.fig_dir is not None:
        fig_dir = Path(args.fig_dir)
    else:
        fig_dir = ROOT / 'result' / 'figures' / ('case_study_dmd' if model_key == 'dmd' else 'case_study')
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_name = 'DMD' if model_key == 'dmd' else 'DecAlign'

    print(f'Model: {model_name} ({model_key})')
    print(f'Uncertainty dir: {uncertainty_dir}')
    print(f'Configuration: tau_c={args.tau_c}, tau_s={args.tau_s}, r_min={args.r_min}')
    print(f'Missing rates: {missing_rates}')
    print(f'Figures will be saved to: {fig_dir}\n')

    print('Loading uncertainty data ...')
    datasets = load_all(tau_c=args.tau_c, tau_s=args.tau_s, r_min=args.r_min, missing_rates=missing_rates, uncertainty_dir=uncertainty_dir)

    # Verify predictions exist
    for mr in missing_rates:
        if 'predictions' not in datasets[mr]:
            print(f'  WARNING: Missing predictions for MR={mr}. '
                  f'Re-run uncertainty mode to include predictions.')
            sys.exit(1)

    # Print numeric summary
    print_weight_summary(datasets, missing_rates=missing_rates)

    # A. Case studies
    print(f'\n[A] Qualitative Case Studies (MR={args.case_mr}) ...')
    run_case_studies(datasets, mr=args.case_mr,
                     n_failure=args.n_failure, n_conflict=args.n_conflict,
                     fig_dir=fig_dir, model_name=model_name)

    # B. Weight vs Missing Rate
    print('\n[B] Weight vs Missing Rate ...')
    plot_weight_vs_missing_rate(datasets, missing_rates=missing_rates, fig_dir=fig_dir)

    # C. Weight distributions
    print('\n[C] Weight Distributions ...')
    plot_weight_distributions(datasets, fig_dir=fig_dir)

    # D. Uncertainty decomposition
    print('\n[D] Uncertainty Decomposition ...')
    plot_uncertainty_decomposition(datasets, fig_dir=fig_dir)

    print(f'\nAll figures saved to: {fig_dir}')


if __name__ == '__main__':
    main()
