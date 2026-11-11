"""Run LNLN on SIMS for five seeds and missing rates 0.0 through 0.9."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LNLN on SIMS across seeds and missing rates"
    )
    parser.add_argument("--data_dir", default="./data", help="Dataset directory")
    parser.add_argument("--config_file", default="", help="Config JSON path")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1111, 1112, 1113, 1114, 1115])
    parser.add_argument(
        "--missing_rates",
        type=float,
        nargs="+",
        default=[i / 10 for i in range(10)],
        help="Evaluation missing rates (default: 0.0 to 0.9)",
    )
    parser.add_argument("--python_exec", default=sys.executable)
    parser.add_argument("--num_workers", type=int, default=0 if os.name == "nt" else 4)
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    config = Path(args.config_file).resolve() if args.config_file else repo_root / "config" / "lnln_sims_config.json"
    if not config.is_file():
        raise FileNotFoundError(f"LNLN config not found: {config}")

    total = len(args.seeds) * len(args.missing_rates)
    run_number = 0
    for seed in args.seeds:
        for rate in args.missing_rates:
            run_number += 1
            rate_text = f"{rate:.1f}"
            seed_dir = f"seed_{seed}"
            command = [
                args.python_exec,
                str(repo_root / "main.py"),
                "--model", "lnln",
                "--dataset", "sims",
                "--config_file", str(config),
                "--data_dir", args.data_dir,
                "--data_missing",
                "--missing_rate_eval_test", rate_text,
                "--seeds", str(seed),
                "--gpu_ids", *map(str, args.gpu_ids),
                "--num_workers", str(args.num_workers),
                "--model_save_dir", str(repo_root / "pt" / seed_dir),
                "--res_save_dir", str(repo_root / "result" / seed_dir),
                "--log_dir", str(repo_root / "log" / seed_dir),
            ]
            print(f"\n>>> [{run_number}/{total}] seed={seed}, missing_rate={rate_text}")
            print(" ".join(f'"{part}"' if " " in part else part for part in command))
            if not args.dry_run:
                subprocess.run(command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
