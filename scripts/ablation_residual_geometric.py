#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/ablation_residual_geometric.py
======================================
Train DP/MIRRORS model under 4 Residual Geometric Calibration modes:
  1. uncalibrated   (Proj: x, Soft Norm: x)
  2. only_proj      (Proj: v, Soft Norm: x)
  3. only_soft_norm (Proj: x, Soft Norm: v)
  4. full           (Proj: v, Soft Norm: v)

Usage:
  python scripts/ablation_residual_geometric.py --dataset mosi --seeds 1111
  python scripts/ablation_residual_geometric.py --modes uncalibrated only_proj only_soft_norm full
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# MODES = ["uncalibrated", "only_proj", "only_soft_norm", "full"]
MODES = ["uncalibrated", "only_proj", "only_soft_norm"]

def parse_args():
    parser = argparse.ArgumentParser(description="Ablation: Residual Geometric Calibration")
    parser.add_argument("--model", type=str, default="dp", help="Model name (default: dp)")
    parser.add_argument("--dataset", type=str, default="mosi", help="Dataset name (mosi, mosei, sims)")
    parser.add_argument("--config_file", type=str, default="", help="Path to config file (optional)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1111], help="Random seeds (default: 1111)")
    parser.add_argument("--missing_rates", type=float, nargs="+", default=[0.5],
                        help="Missing rates to evaluate/test on (default: 0.5)")
    parser.add_argument("--modes", type=str, nargs="+", default=MODES, choices=MODES,
                        help="Calibration modes to train (default: all 4)")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0], help="GPU ID(s)")
    parser.add_argument("--data_missing", action="store_true", default=True,
                        help="Enable missing modality simulation during test/eval (default: True)")
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[1]

    # Resolve default config path if not provided
    if not args.config_file:
        config_path = root_dir / "config" / f"{args.model}_{args.dataset}_config.json"
        if not config_path.exists():
            # Fallback for mosi
            config_path = root_dir / "config" / "dp_mosi_config.json"
    else:
        config_path = Path(args.config_file)

    print("=" * 80)
    print("Ablation Study: Residual Geometric Calibration")
    print(f"Modes: {args.modes}")
    print(f"Dataset: {args.dataset} | Seeds: {args.seeds} | Missing Rates: {args.missing_rates}")
    print(f"Config: {config_path}")
    print("=" * 80)

    for mode in args.modes:
        print(f"\n>>>> [Calibration Mode: {mode.upper()}] <<<<")
        model_save_dir = root_dir / "pt" / "ablation" / f"calib_{mode}"
        res_save_dir = root_dir / "result" / "ablation" / f"calib_{mode}"
        log_dir = root_dir / "log" / "ablation" / f"calib_{mode}"

        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs(res_save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        for mr in args.missing_rates:
            for seed in args.seeds:
                cmd = [
                    sys.executable,
                    str(root_dir / "main.py"),
                    "--model", args.model,
                    "--dataset", args.dataset,
                    "--config_file", str(config_path),
                    "--mode", "train",
                    "--seeds", str(seed),
                    "--gpu_ids", *[str(g) for g in args.gpu_ids],
                    "--missing_rate_eval_test", str(mr),
                    "--residual_calib_mode", mode,
                    "--model_save_dir", str(model_save_dir),
                    "--res_save_dir", str(res_save_dir),
                    "--log_dir", str(log_dir),
                ]
                if args.data_missing:
                    cmd.append("--data_missing")

                print(f"Executing: {' '.join(cmd)}")
                ret = subprocess.run(cmd, cwd=str(root_dir))
                if ret.returncode != 0:
                    print(f"Error during training mode={mode}, mr={mr}, seed={seed}. Exiting...")
                    sys.exit(ret.returncode)

    print("\n" + "=" * 80)
    print("All residual geometric ablation experiments finished successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

# python scripts/ablation_residual_geometric.py --dataset mosi --seeds 1111 --missing_rates 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
