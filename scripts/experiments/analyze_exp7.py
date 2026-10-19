#!/usr/bin/env python3
"""
Experiment 7 Analysis: Joint gate γ — why normalized reliability alone isn't enough.

Reads trained models for Full SARM and No-γ variants across multiple
high missing rates, collects:
  - Downstream F1/MAE
  - Mean γ^c per missing rate

Generates:
  1. Line plot of F1 or MAE vs missing rate (Full SARM vs No γ)
  2. Line plot of mean γ^c vs missing rate
  3. Summary performance table
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure project root is importable
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import get_config_regression
from data_loader import MMDataLoader
from models import build_model
from trains.ATIO import ATIO
from utils import setup_seed

logger = logging.getLogger('DecAlign')


def collect_gamma(model, dataloader, args):
    """Collect gamma_c values from forward passes on test set."""
    model.eval()
    gamma_c_list = []

    with torch.no_grad():
        for batch_data in dataloader:
            text = batch_data['text'].to(args.device)
            text_m = batch_data['text_m'].to(args.device)
            audio = batch_data['audio'].to(args.device)
            audio_m = batch_data['audio_m'].to(args.device)
            vision = batch_data['vision'].to(args.device)
            vision_m = batch_data['vision_m'].to(args.device)

            output = model(text_m, audio_m, vision_m, text, audio, vision)
            gamma_c_list.append(output['gamma_c'].cpu())

    return torch.cat(gamma_c_list, dim=0)  # (N,)


def main():
    parser = argparse.ArgumentParser(description='Experiment 7 Analysis')
    parser.add_argument('--input-root', type=str, required=True,
                        help='Root directory containing variant results')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory for output figures and tables')
    parser.add_argument('--dataset', type=str, default='mosi')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1111])
    parser.add_argument('--missing-rates', type=float, nargs='+',
                        default=[0.5, 0.7, 0.8, 0.9])
    parser.add_argument('--config-file', type=str, default='')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = {
        'full_sarm': {'disable_gamma': False, 'label': 'Full SARM'},
        'no_gamma': {'disable_gamma': True, 'label': r'$\gamma = 1$'},
    }

    all_results = []

    for seed in args.seeds:
        setup_seed(seed)

        for rate in args.missing_rates:
            for cond_key, cond_info in conditions.items():
                run_dir = Path(args.input_root) / cond_key / f"mr{rate}"
                model_dir = run_dir / "models"

                model_path = list(model_dir.glob("*.pth"))
                if not model_path:
                    print(f"WARNING: No model found for {cond_key}, rate={rate}, seed={seed}")
                    continue
                model_path = model_path[0]

                # Build config
                config_file = args.config_file or str(ROOT_DIR / "config" / f"dp_{args.dataset}_config.json")
                cfg = get_config_regression('dp', args.dataset, config_file,
                                            data_dir=args.data_dir)
                cfg.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
                cfg.missing_rate_eval_test = rate
                cfg.data_missing = True
                cfg.disable_gamma = cond_info['disable_gamma']
                cfg.cur_seed = seed
                cfg.model_save_path = str(model_path)

                # Load model + data
                dataloader = MMDataLoader(cfg, num_workers=4)
                model = build_model(cfg).to(cfg.device)
                model.load_state_dict(torch.load(model_path, map_location=cfg.device))
                model.update_train_bank(dataloader['train'], cfg.device)

                # Get metrics
                trainer = ATIO().getTrain(cfg)
                eval_results = trainer.do_test(model, dataloader['test'], mode="TEST")

                # Collect gamma
                # For gamma collection, temporarily enable gamma computation
                # (even for no_gamma variant, we want to see what gamma WOULD be)
                orig_disable = model.disable_gamma
                model.disable_gamma = False  # compute true gamma
                gamma_c_all = collect_gamma(model, dataloader['test'], cfg)
                model.disable_gamma = orig_disable  # restore

                result_row = {
                    'condition': cond_key,
                    'condition_label': cond_info['label'],
                    'missing_rate': rate,
                    'seed': seed,
                    'mean_gamma_c': gamma_c_all.mean().item(),
                    'std_gamma_c': gamma_c_all.std().item(),
                    'median_gamma_c': gamma_c_all.median().item(),
                }
                # Add standard metrics
                for k, v in eval_results.items():
                    if isinstance(v, (int, float)):
                        result_row[k] = v
                all_results.append(result_row)

                del model
                torch.cuda.empty_cache()

    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / 'exp7_results.csv', index=False)
    print(f"\nResults saved to {output_dir / 'exp7_results.csv'}")
    print(df.to_string(index=False))

    # ================================================================
    # Generate Figures
    # ================================================================
    try:
        sys.path.insert(0, str(ROOT_DIR / 'scripts'))
        from paper_style import apply_style, new_fig, save_fig
        apply_style('acl', 'wide')
    except Exception:
        import matplotlib
        matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    colors = {
        'full_sarm': '#2196F3',   # Blue
        'no_gamma': '#F44336',     # Red
    }
    markers = {
        'full_sarm': 'o',
        'no_gamma': 's',
    }

    # --- Determine primary metric (MAE for regression, F1 for classification) ---
    metric_name = 'MAE' if 'MAE' in df.columns else 'Loss'
    if args.dataset in ['iemocap']:
        metric_name = 'WAF1' if 'WAF1' in df.columns else metric_name

    # Group by condition and missing_rate
    grouped = df.groupby(['condition', 'missing_rate']).agg(
        {col: ['mean', 'std'] for col in df.columns
         if col not in ['condition', 'condition_label', 'missing_rate', 'seed']}
    ).reset_index()

    # --- Figure: Performance vs Missing Rate (2 subplots side by side) ---
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), layout='constrained')

    # Panel (a): Downstream metric
    ax = axes[0]
    for cond_key, cond_info in conditions.items():
        cond_data = grouped[grouped['condition'] == cond_key].sort_values('missing_rate')
        if cond_data.empty:
            continue
        rates = cond_data['missing_rate'].values
        means = cond_data[(metric_name, 'mean')].values
        stds = cond_data[(metric_name, 'std')].values

        ax.plot(rates, means, marker=markers[cond_key], color=colors[cond_key],
                label=cond_info['label'], linewidth=1.5, markersize=5)
        ax.fill_between(rates, means - stds, means + stds,
                        alpha=0.15, color=colors[cond_key])

    ax.set_xlabel('Missing Rate', fontsize=7)
    ax.set_ylabel(metric_name, fontsize=7)
    ax.set_title(f'{metric_name} vs Missing Rate', fontsize=8)
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_xticks(args.missing_rates)

    # Panel (b): Mean γ^c vs Missing Rate
    ax = axes[1]
    for cond_key, cond_info in conditions.items():
        cond_data = grouped[grouped['condition'] == cond_key].sort_values('missing_rate')
        if cond_data.empty:
            continue
        rates = cond_data['missing_rate'].values
        means = cond_data[('mean_gamma_c', 'mean')].values
        stds = cond_data[('mean_gamma_c', 'std')].values

        ax.plot(rates, means, marker=markers[cond_key], color=colors[cond_key],
                label=cond_info['label'], linewidth=1.5, markersize=5)
        ax.fill_between(rates, means - stds, means + stds,
                        alpha=0.15, color=colors[cond_key])

    # Add reference line at γ=1
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6,
               label=r'$\gamma = 1$ (no suppression)')
    ax.set_xlabel('Missing Rate', fontsize=7)
    ax.set_ylabel(r'Mean $\gamma^c$', fontsize=7)
    ax.set_title(r'Joint Gate $\gamma^c$ vs Missing Rate', fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc='best')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_xticks(args.missing_rates)
    ax.set_ylim(0, 1.1)

    fig.savefig(output_dir / 'exp7_gamma_gate.pdf', bbox_inches='tight')
    fig.savefig(output_dir / 'exp7_gamma_gate.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Figure saved to {output_dir / 'exp7_gamma_gate.pdf'}")

    # --- Table: Performance comparison ---
    pivot_cols = ['missing_rate']
    metric_cols_all = [c for c in df.columns if c not in
                       ['condition', 'condition_label', 'missing_rate', 'seed']]

    # Build a clean comparison table
    table_rows = []
    for rate in sorted(args.missing_rates):
        row = {'Missing Rate': rate}
        for cond_key, cond_info in conditions.items():
            cond_rate = df[(df['condition'] == cond_key) & (df['missing_rate'] == rate)]
            if cond_rate.empty:
                continue
            for col in ['mean_gamma_c', metric_name, 'Loss']:
                if col in cond_rate.columns:
                    mean_val = cond_rate[col].mean()
                    row[f'{cond_info["label"]} {col}'] = f'{mean_val:.4f}'
        table_rows.append(row)

    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(output_dir / 'exp7_performance_table.csv', index=False)
    print(f"\nPerformance table:")
    print(table_df.to_string(index=False))

    # LaTeX table
    latex_lines = [
        r'\begin{tabular}{c|cc|cc}',
        r'\toprule',
        r'\multirow{2}{*}{Missing Rate} & \multicolumn{2}{c|}{Full SARM} & \multicolumn{2}{c}{$\gamma = 1$} \\',
        r'& ' + metric_name + r' & $\bar{\gamma}^c$ & ' + metric_name + r' & $\bar{\gamma}^c$ \\',
        r'\midrule',
    ]

    for rate in sorted(args.missing_rates):
        parts = [f'{rate:.1f}']
        for cond_key in ['full_sarm', 'no_gamma']:
            cond_rate = df[(df['condition'] == cond_key) & (df['missing_rate'] == rate)]
            if not cond_rate.empty:
                metric_val = cond_rate[metric_name].mean()
                gamma_val = cond_rate['mean_gamma_c'].mean()
                parts.extend([f'{metric_val:.4f}', f'{gamma_val:.4f}'])
            else:
                parts.extend(['--', '--'])
        latex_lines.append(' & '.join(parts) + r' \\')

    latex_lines.extend([r'\bottomrule', r'\end{tabular}'])
    latex_str = '\n'.join(latex_lines)
    (output_dir / 'exp7_table.tex').write_text(latex_str)
    print(f"\nLaTeX table saved to {output_dir / 'exp7_table.tex'}")
    print(latex_str)


if __name__ == '__main__':
    main()
