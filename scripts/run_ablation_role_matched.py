"""
Ablation 9: Role-matched Interaction Ablation Runner & Visualizer

Iterates through all 5 interaction options at moderate (p=0.5) and severe (p=0.8) missing rates:
  1. full_sarm:       Shared consensus + Specific directional (Proposed Full SARM)
  2. generic:         Shared weighted + Specific weighted (Generic baseline)
  3. consensus_all:   Shared consensus + Specific consensus (Consensus-only)
  4. directional_all: Shared directional + Specific directional (Directional-only)
  5. role_swapped:    Shared directional + Specific consensus (Role-swapped counterfactual)

At the end of execution, this script automatically aggregates all results, computes
performance drops (delta) relative to Full SARM, and displays/saves a clear comparison table.

Can be run directly via bash/cmd or in a single Kaggle notebook cell.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

VARIANTS = ['full_sarm', 'generic', 'consensus_all', 'directional_all', 'role_swapped']

VARIANT_DESCRIPTIONS = {
    'full_sarm':        ('Consensus', 'Directional', 'Proposed Full SARM (Role-matched)'),
    'generic':          ('Weighted',  'Weighted',    'Generic Weighted Fusion'),
    'consensus_all':    ('Consensus', 'Consensus',   'Consensus for All'),
    'directional_all':  ('Directional','Directional', 'Directional for All'),
    'role_swapped':     ('Directional','Consensus',   'Role-swapped (Counterfactual)'),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Role-matched Interaction Ablation")
    parser.add_argument('--dataset', type=str, default='mosi', choices=['mosi', 'mosei', 'sims'], help='Dataset name')
    parser.add_argument('--config_file', type=str, default='config/dp_mosi_config.json', help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='', help='Path to dataset directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1111], help='Random seeds')
    parser.add_argument('--missing_rates', type=float, nargs='+', default=[0.5, 0.8], help='Missing rates (Moderate p=0.5, Severe p=0.8)')
    parser.add_argument('--variants', type=str, nargs='+', default=VARIANTS, help='Ablation variants to run')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='GPU IDs')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--base_save_dir', type=str, default='./ablation_role_matched', help='Base directory for logs and models')
    parser.add_argument('--skip_train', action='store_true', help='Skip training and only generate summary from existing results')
    return parser.parse_args()


def generate_summary(csv_file, output_dir, dataset_name, variants, missing_rates):
    if not csv_file.is_file():
        print(f"\n[Notice] Result CSV file not found at: {csv_file}. Cannot generate summary table.")
        return

    df = pd.read_csv(csv_file)
    print("\n" + "=" * 90)
    print("                     ABLATION 9: ROLE-MATCHED INTERACTION SUMMARY                     ")
    print("=" * 90)

    # Detect metric columns (prioritize Acc_2, F1, MAE, Corr, Acc_7)
    target_metrics = []
    preferred_metrics = [
        'Acc_2 Mean', 'F1_score Mean', 'MAE Mean', 'Corr Mean', 'Acc_7 Mean',
        'Non0_acc_2 Mean', 'Non0_F1_score Mean', 'Has0_acc_2 Mean', 'Has0_F1_score Mean'
    ]
    for m in preferred_metrics:
        if m in df.columns:
            target_metrics.append(m)

    if not target_metrics:
        # Fallback to any Mean column
        target_metrics = [c for c in df.columns if c.endswith('Mean')]

    rows = []
    for var in variants:
        model_name = f"dp_{var}"
        shared_role, spec_role, desc = VARIANT_DESCRIPTIONS.get(var, ('-', '-', var))
        
        row_dict = {
            'Variant': var,
            'Shared (H)': shared_role,
            'Specific (P)': spec_role,
        }

        for mr in missing_rates:
            # Query matching row in dataframe
            sub = df[(df['Model'] == model_name) & (np.isclose(df['Missing Rate'], mr, atol=1e-4))]
            if not sub.empty:
                last_record = sub.iloc[-1]
                for tm in target_metrics:
                    short_name = tm.replace(' Mean', '')
                    col_key = f"{short_name} (p={mr})"
                    row_dict[col_key] = last_record[tm]
            else:
                for tm in target_metrics:
                    short_name = tm.replace(' Mean', '')
                    col_key = f"{short_name} (p={mr})"
                    row_dict[col_key] = np.nan

        rows.append(row_dict)

    summary_df = pd.DataFrame(rows)

    # Calculate Deltas relative to full_sarm
    if 'full_sarm' in summary_df['Variant'].values:
        baseline_idx = summary_df[summary_df['Variant'] == 'full_sarm'].index[0]
        baseline = summary_df.loc[baseline_idx]
        delta_rows = []
        for idx, row in summary_df.iterrows():
            d_dict = {'Variant': row['Variant'], 'Shared (H)': row['Shared (H)'], 'Specific (P)': row['Specific (P)']}
            for col in summary_df.columns:
                if col not in ['Variant', 'Shared (H)', 'Specific (P)']:
                    val = row[col]
                    b_val = baseline[col]
                    if pd.notna(val) and pd.notna(b_val):
                        diff = val - b_val
                        sign = "+" if diff > 0 else ""
                        d_dict[col] = f"{val:.2f} ({sign}{diff:.2f})"
                    else:
                        d_dict[col] = "-"
            delta_rows.append(d_dict)
        display_df = pd.DataFrame(delta_rows)
    else:
        display_df = summary_df

    # Print to console as Markdown
    print(display_df.to_markdown(index=False))

    # Save to CSV and Markdown
    out_csv = Path(output_dir) / f"ablation_9_{dataset_name}_summary.csv"
    out_md = Path(output_dir) / f"ablation_9_{dataset_name}_summary.md"
    summary_df.to_csv(out_csv, index=False)
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(f"# Ablation 9: Role-matched Interaction Ablation ({dataset_name.upper()})\n\n")
        f.write(display_df.to_markdown(index=False))
        f.write("\n\n*Numbers in parentheses indicate performance gap (delta) relative to Full SARM.*\n")

    print("\n" + "=" * 90)
    print(f"Summary table saved to:")
    print(f"  - Markdown: {out_md}")
    print(f"  - CSV:      {out_csv}")
    print("=" * 90)


def main():
    args = parse_args()
    base_dir = Path(args.base_save_dir)
    model_save_dir = base_dir / "pt"
    res_save_dir = base_dir / "result"
    log_dir = base_dir / "log"

    model_save_dir.mkdir(parents=True, exist_ok=True)
    res_save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_file = res_save_dir / "normal" / f"{args.dataset}.csv"

    print("=" * 75)
    print("ROLE-MATCHED INTERACTION ABLATION EXPERIMENT RUNNER (ABLATION 9)")
    print(f"Dataset:       {args.dataset.upper()}")
    print(f"Config:        {args.config_file}")
    print(f"Variants:      {args.variants}")
    print(f"Missing Rates: {args.missing_rates} (Moderate p=0.5, Severe p=0.8)")
    print(f"Seeds:         {args.seeds}")
    print(f"Result Dir:    {res_save_dir}")
    print("=" * 75)

    if not args.skip_train:
        for variant in args.variants:
            print(f"\n>>>>>>> RUNNING VARIANT: [{variant.upper()}] <<<<<<<")
            for mr in args.missing_rates:
                for seed in args.seeds:
                    print(f"\n[Variant: {variant} | Missing Rate: {mr} | Seed: {seed}]")
                    
                    cmd = [
                        sys.executable, "main.py",
                        "--data_missing",
                        "--model", "dp",
                        "--dataset", args.dataset,
                        "--config_file", args.config_file,
                        "--mode", "train",
                        "--seeds", str(seed),
                        "--missing_rate_eval_test", str(mr),
                        "--interaction_ablation", variant,
                        "--gpu_ids"
                    ] + [str(g) for g in args.gpu_ids] + [
                        "--num_workers", str(args.num_workers),
                        "--model_save_dir", str(model_save_dir),
                        "--res_save_dir", str(res_save_dir),
                        "--log_dir", str(log_dir / variant),
                    ]

                    if args.data_dir:
                        cmd += ["--data_dir", args.data_dir]

                    print(f"Command: {' '.join(cmd)}")
                    ret = subprocess.run(cmd)
                    if ret.returncode != 0:
                        print(f"WARNING: Run failed for variant={variant}, mr={mr}, seed={seed}")

    # Generate summary report across all variants and missing rates
    generate_summary(csv_file, base_dir, args.dataset, args.variants, args.missing_rates)


if __name__ == '__main__':
    main()
