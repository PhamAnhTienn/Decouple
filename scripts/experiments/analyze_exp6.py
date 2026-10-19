#!/usr/bin/env python3
"""
Experiment 6 Analysis: Specific reliability — conditional-locality assumption.

Reads trained models for each neighbor variant (complement, self, random),
runs diagnostics to collect LocalCoh and downstream metrics, then generates:
  1. Violin / box plot of per-sample specific cosine similarity per variant
  2. Table (CSV + LaTeX) of downstream performance
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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


def collect_diagnostics(model, dataloader, args, K=10):
    """
    Collect z_s, z_c and compute LocalCoh + per-sample neighbor cosine
    for all three modalities.
    """
    model.eval()
    z_s_all = {'l': [], 'v': [], 'a': []}
    z_c_all = {'l': [], 'v': [], 'a': []}

    with torch.no_grad():
        for batch_data in dataloader:
            text = batch_data['text'].to(args.device)
            text_m = batch_data['text_m'].to(args.device)
            audio = batch_data['audio'].to(args.device)
            audio_m = batch_data['audio_m'].to(args.device)
            vision = batch_data['vision'].to(args.device)
            vision_m = batch_data['vision_m'].to(args.device)

            output = model(text_m, audio_m, vision_m, text, audio, vision)

            for m in ['l', 'v', 'a']:
                z_s_all[m].append(output['z_s'][m].cpu())
                z_c_all[m].append(output['z_c'][m].cpu())

    for m in ['l', 'v', 'a']:
        z_s_all[m] = torch.cat(z_s_all[m], dim=0)
        z_c_all[m] = torch.cat(z_c_all[m], dim=0)

    return z_s_all, z_c_all


def compute_local_coherence(z_s_all, z_c_all, neighbor_mode, K=10):
    """
    Compute LocalCoh^m = (1/NK) Σ_i Σ_{j∈N_K(i)} cos(z_s^{m,i}, z_s^{m,j})

    Also returns per-sample neighbor cosine values for violin plots.
    """
    local_coh = {}
    per_sample_cosines = {}  # m → (N, K)

    for m in ['l', 'v', 'a']:
        z_s_m = z_s_all[m]
        N = z_s_m.shape[0]
        K_actual = min(K, N - 1)

        if K_actual <= 0:
            local_coh[m] = 0.0
            per_sample_cosines[m] = torch.zeros(N, 1)
            continue

        # Determine neighbor query based on mode
        if neighbor_mode == 'complement':
            others = [n for n in ['l', 'v', 'a'] if n != m]
            z_query = F.normalize(sum(z_c_all[n] for n in others), dim=1)
        elif neighbor_mode == 'self':
            z_query = z_c_all[m]
        else:  # random
            z_query = None

        if z_query is not None:
            sim = z_query @ z_query.t()
            sim.fill_diagonal_(-float('inf'))
            _, topk_idx = sim.topk(K_actual, dim=1)
        else:
            topk_idx = torch.stack([
                torch.randperm(N)[:K_actual] for _ in range(N)
            ], dim=0)

        z_s_neighbors = z_s_m[topk_idx]  # (N, K, D)
        cos_vals = F.cosine_similarity(
            z_s_m.unsqueeze(1), z_s_neighbors, dim=2
        )  # (N, K)

        local_coh[m] = cos_vals.mean().item()
        per_sample_cosines[m] = cos_vals

    return local_coh, per_sample_cosines


def main():
    parser = argparse.ArgumentParser(description='Experiment 6 Analysis')
    parser.add_argument('--input-root', type=str, required=True,
                        help='Root directory containing variant results')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory for output figures and tables')
    parser.add_argument('--dataset', type=str, default='mosi')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1111])
    parser.add_argument('--missing-rate', type=float, default=0.5)
    parser.add_argument('--config-file', type=str, default='')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = ['complement', 'self', 'random']
    variant_labels = {
        'complement': 'Complement\n(SARM)',
        'self': 'Self-\nconditioned',
        'random': 'Random',
    }

    # Collect results across seeds and variants
    all_results = []
    all_cosines = {v: {'l': [], 'v': [], 'a': []} for v in variants}

    for seed in args.seeds:
        setup_seed(seed)

        for variant in variants:
            run_dir = Path(args.input_root) / variant / f"mr{args.missing_rate}"
            model_dir = run_dir / "models"

            # Find model file
            model_path = list(model_dir.glob("*.pth"))
            if not model_path:
                print(f"WARNING: No model found for variant={variant}, seed={seed}")
                continue
            model_path = model_path[0]

            # Build config
            config_file = args.config_file or str(ROOT_DIR / "config" / f"dp_{args.dataset}_config.json")
            cfg = get_config_regression('dp', args.dataset, config_file,
                                        data_dir=args.data_dir)
            cfg.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
            cfg.missing_rate_eval_test = args.missing_rate
            cfg.data_missing = True
            cfg.neighbor_mode = variant
            cfg.cur_seed = seed
            cfg.model_save_path = str(model_path)

            # Load model + data
            dataloader = MMDataLoader(cfg, num_workers=4)
            model = build_model(cfg).to(cfg.device)
            model.load_state_dict(torch.load(model_path, map_location=cfg.device))
            model.update_train_bank(dataloader['train'], cfg.device)

            # Get trainer for metrics
            trainer = ATIO().getTrain(cfg)
            eval_results = trainer.do_test(model, dataloader['test'], mode="TEST")

            # Collect diagnostics
            z_s_all, z_c_all = collect_diagnostics(model, dataloader['test'], cfg, K=args.K)
            local_coh, per_sample_cosines = compute_local_coherence(
                z_s_all, z_c_all, variant, K=args.K
            )

            # Store results
            result_row = {
                'variant': variant,
                'seed': seed,
                'missing_rate': args.missing_rate,
                **{f'LocalCoh_{m}': local_coh[m] for m in ['l', 'v', 'a']},
                'LocalCoh_mean': np.mean([local_coh[m] for m in ['l', 'v', 'a']]),
            }
            # Add standard metrics
            for k, v in eval_results.items():
                if isinstance(v, (int, float)):
                    result_row[k] = v
            all_results.append(result_row)

            # Store per-sample cosines for violin plot
            for m in ['l', 'v', 'a']:
                all_cosines[variant][m].append(
                    per_sample_cosines[m].mean(dim=1).numpy()  # (N,)
                )

            del model
            torch.cuda.empty_cache()

    # Create results DataFrame
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / 'exp6_results.csv', index=False)
    print(f"\nResults saved to {output_dir / 'exp6_results.csv'}")
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

    import matplotlib.pyplot as plt

    # --- Figure 1: Violin/Box plot of neighbor cosine similarity ---
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), layout='constrained')
    modality_names = {'l': 'Language', 'v': 'Vision', 'a': 'Audio'}
    colors = {
        'complement': '#2196F3',  # Blue
        'self': '#FF9800',         # Orange
        'random': '#9E9E9E',       # Gray
    }

    for idx, m in enumerate(['l', 'v', 'a']):
        ax = axes[idx]
        data_for_plot = []
        labels_for_plot = []
        colors_for_plot = []

        for variant in variants:
            if all_cosines[variant][m]:
                combined = np.concatenate(all_cosines[variant][m])
                data_for_plot.append(combined)
                labels_for_plot.append(variant_labels[variant])
                colors_for_plot.append(colors[variant])

        if data_for_plot:
            parts = ax.violinplot(data_for_plot, showmeans=True, showmedians=True)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors_for_plot[i])
                pc.set_alpha(0.7)
            for key in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
                if key in parts:
                    parts[key].set_color('#333333')
                    parts[key].set_linewidth(0.8)

            ax.set_xticks(range(1, len(labels_for_plot) + 1))
            ax.set_xticklabels(labels_for_plot, fontsize=6.5)

        ax.set_title(modality_names[m], fontsize=8)
        ax.set_ylabel('Neighbor cosine sim' if idx == 0 else '', fontsize=7)
        ax.grid(True, alpha=0.3, linestyle=':')

    fig.suptitle('Specific Cosine Similarity by Neighbor Selection Strategy',
                 fontsize=9, y=1.02)

    fig.savefig(output_dir / 'exp6_violin_plot.pdf', bbox_inches='tight')
    fig.savefig(output_dir / 'exp6_violin_plot.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Violin plot saved to {output_dir / 'exp6_violin_plot.pdf'}")

    # --- Figure 2: LocalCoh bar plot ---
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 2.5), layout='constrained')

    # Group by variant, average over seeds
    grouped = df.groupby('variant').agg({
        'LocalCoh_l': ['mean', 'std'],
        'LocalCoh_v': ['mean', 'std'],
        'LocalCoh_a': ['mean', 'std'],
    })

    x = np.arange(3)  # modalities
    width = 0.25
    modality_keys = ['LocalCoh_l', 'LocalCoh_v', 'LocalCoh_a']
    modality_labels_short = ['L', 'V', 'A']

    for i, variant in enumerate(variants):
        if variant in grouped.index:
            means = [grouped.loc[variant, (k, 'mean')] for k in modality_keys]
            stds = [grouped.loc[variant, (k, 'std')] for k in modality_keys]
            ax.bar(x + i * width, means, width, yerr=stds,
                   label=variant.capitalize(), color=colors[variant], alpha=0.85,
                   capsize=3, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(modality_labels_short)
    ax.set_ylabel('LocalCoh', fontsize=7)
    ax.set_title('Local Coherence by Modality', fontsize=8)
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')

    fig.savefig(output_dir / 'exp6_local_coh.pdf', bbox_inches='tight')
    fig.savefig(output_dir / 'exp6_local_coh.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"LocalCoh bar plot saved to {output_dir / 'exp6_local_coh.pdf'}")

    # --- Table: Downstream performance ---
    metric_cols = [c for c in df.columns if c not in
                   ['variant', 'seed', 'missing_rate',
                    'LocalCoh_l', 'LocalCoh_v', 'LocalCoh_a', 'LocalCoh_mean']]
    perf_table = df.groupby('variant')[['LocalCoh_mean'] + metric_cols].agg(['mean', 'std'])
    perf_table.to_csv(output_dir / 'exp6_performance_table.csv')

    # LaTeX table
    latex_lines = []
    latex_lines.append(r'\begin{tabular}{l' + 'c' * (len(metric_cols) + 1) + '}')
    latex_lines.append(r'\toprule')
    header = 'Variant & LocalCoh$^\\text{mean}$'
    for c in metric_cols:
        header += f' & {c}'
    header += r' \\'
    latex_lines.append(header)
    latex_lines.append(r'\midrule')

    for variant in variants:
        if variant in perf_table.index:
            row = variant.capitalize()
            coh_mean = perf_table.loc[variant, ('LocalCoh_mean', 'mean')]
            row += f' & {coh_mean:.4f}'
            for c in metric_cols:
                val = perf_table.loc[variant, (c, 'mean')]
                row += f' & {val:.4f}'
            row += r' \\'
            latex_lines.append(row)

    latex_lines.append(r'\bottomrule')
    latex_lines.append(r'\end{tabular}')

    latex_str = '\n'.join(latex_lines)
    (output_dir / 'exp6_table.tex').write_text(latex_str)
    print(f"\nLaTeX table saved to {output_dir / 'exp6_table.tex'}")
    print(latex_str)


if __name__ == '__main__':
    main()
