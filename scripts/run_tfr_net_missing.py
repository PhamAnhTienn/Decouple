"""
Sequential Benchmark Runner for TFRNet across multiple seeds and missing rates.
Sequence: Seed-by-seed (1111 -> 1112 -> 1113 -> 1114 -> 1115), with missing rates 0.0 -> 0.9.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def get_default_python():
    """Detect python executable, preferring decouple_env if available."""
    conda_env_python = Path(r"C:\Users\PC\miniconda3\envs\decouple_env\python.exe")
    if conda_env_python.exists():
        return str(conda_env_python)
    return sys.executable


def parse_args():
    parser = argparse.ArgumentParser(description="Run TFRNet across seeds and missing rates")
    parser.add_argument("--dataset", type=str, default="mosi", help="Dataset name (default: mosi)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--config_file", type=str, default="", help="Path to config JSON (default: config/tfr_net_{dataset}_config.json)")
    parser.add_argument("--gpu_ids", "--gpu", dest="gpu_ids", type=int, nargs="+", default=[0], help="GPU ID(s) to use (default: 0)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1111, 1112, 1113, 1114, 1115], help="Seeds to test")
    parser.add_argument(
        "--missing_rates",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Missing rates to evaluate",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size (recommended: 8 or 12 for 4GB VRAM)")
    parser.add_argument("--num_workers", type=int, default=0 if os.name == 'nt' else 4, help="Number of workers (default: 0 on Windows, 4 on Linux)")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override number of epochs if desired")
    parser.add_argument(
        "--config_override",
        action="append",
        default=[],
        help="Override any config value with key=value. Can be repeated.",
    )
    parser.add_argument("--resume", action="store_true", default=True, help="Skip completed runs (default: True)")
    parser.add_argument("--no_resume", action="store_false", dest="resume", help="Do not skip completed runs")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument("--python_exec", type=str, default=None, help="Python interpreter to use")
    return parser.parse_args()


def aggregate_summary(decouple_root, dataset, seeds, missing_rates):
    """Aggregate results across seeds for each missing rate and save a summary CSV."""
    results_dir = decouple_root / "result"
    summary_data = []

    for rate in missing_rates:
        rate_metrics = {}
        for seed in seeds:
            candidates = [
                results_dir / f"seed_{seed}" / f"{dataset}.csv",
                results_dir / f"seed_{seed}" / "normal" / f"{dataset}.csv",
            ]
            csv_path = next((p for p in candidates if p.is_file()), None)
            if not csv_path:
                continue
            try:
                df = pd.read_csv(csv_path)
                match = df[np.isclose(df["Missing Rate"].astype(float), float(rate), atol=1e-4)]
                if not match.empty:
                    row = match.iloc[-1]
                    for col in df.columns:
                        if col in ["Model", "Missing Rate"]:
                            continue
                        # Use Mean column if available
                        metric_val = row[col]
                        if pd.notna(metric_val):
                            rate_metrics.setdefault(col, []).append(float(metric_val))
            except Exception as e:
                print(f"[Warning] Error reading {csv_path}: {e}")

        if rate_metrics:
            row_summary = {"Model": "tfr_net", "Missing Rate": rate}
            for metric_name, values in rate_metrics.items():
                if values:
                    mean_val = round(float(np.mean(values)), 2)
                    std_val = round(float(np.std(values)), 2)
                    row_summary[f"{metric_name}"] = f"{mean_val} ± {std_val}"
            summary_data.append(row_summary)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        out_path = results_dir / f"tfr_net_{dataset}_missing_summary.csv"
        summary_df.to_csv(out_path, index=False)
        print("\n" + "=" * 60)
        print(f"BENCHMARK SUMMARY SAVED TO: {out_path}")
        print("=" * 60)
        try:
            print(summary_df.to_markdown(index=False))
        except Exception:
            print(summary_df.to_string(index=False))


def main():
    args = parse_args()
    decouple_root = Path(__file__).resolve().parent.parent
    python_exec = args.python_exec or get_default_python()

    if not args.config_file:
        config_path = decouple_root / "config" / f"tfr_net_{args.dataset}_config.json"
        if not config_path.is_file():
            config_path = decouple_root / "config" / "tfr_net_mosi_config.json"
    else:
        config_path = Path(args.config_file).resolve()

    total_runs = len(args.seeds) * len(args.missing_rates)
    current_run = 0

    print("=" * 70)
    print(f"TFRNet Sequential Benchmark: {len(args.seeds)} Seeds x {len(args.missing_rates)} Missing Rates")
    print(f"Seeds: {args.seeds}")
    print(f"Missing Rates: {args.missing_rates}")
    print(f"Python Executable: {python_exec}")
    print(f"Config File: {config_path}")
    print("=" * 70)

    for seed in args.seeds:
        for rate in args.missing_rates:
            current_run += 1
            rate_str = f"{rate:.1f}"
            ckpt_path = decouple_root / "pt" / f"seed_{seed}" / f"tfr_net-{args.dataset}-missing{rate_str}.pth"

            print(f"\n>>> [{current_run}/{total_runs}] Starting Seed: {seed} | Missing Rate: {rate_str}")

            if args.resume and ckpt_path.is_file():
                print(f"[Skip] Checkpoint {ckpt_path} already exists. Skipping this run.")
                continue

            cmd = [
                python_exec,
                str(decouple_root / "main.py"),
                "--model", "tfr_net",
                "--dataset", args.dataset,
                "--config_file", str(config_path),
                "--data_missing",
                "--missing_rate_eval_test", rate_str,
                "--seeds", str(seed),
                "--gpu_ids", *[str(g) for g in args.gpu_ids],
                "--data_dir", args.data_dir,
                "--model_save_dir", str(decouple_root / "pt" / f"seed_{seed}"),
                "--res_save_dir", str(decouple_root / "result" / f"seed_{seed}"),
                "--log_dir", str(decouple_root / "log" / f"seed_{seed}"),
                "--num_workers", str(args.num_workers),
            ]

            if args.batch_size is not None:
                cmd.extend(["--batch_size", str(args.batch_size)])

            if args.num_epochs is not None:
                cmd.extend(["--num_epochs", str(args.num_epochs)])

            for override in args.config_override:
                cmd.extend(["--config_override", override])

            print("Command:", " ".join(cmd))

            if args.dry_run:
                continue

            try:
                ret = subprocess.run(cmd, cwd=str(decouple_root), check=True)
            except subprocess.CalledProcessError as e:
                print(f"[Error] Run failed for seed {seed}, missing rate {rate_str} with exit code {e.returncode}")
                try:
                    choice = input("Continue with next run? [Y/n]: ").strip().lower()
                    if choice == "n":
                        print("Aborting benchmark.")
                        sys.exit(1)
                except (EOFError, KeyboardInterrupt):
                    print("Non-interactive session detected. Aborting benchmark.")
                    sys.exit(1)

    if not args.dry_run:
        aggregate_summary(decouple_root, args.dataset, args.seeds, args.missing_rates)


if __name__ == "__main__":
    main()
