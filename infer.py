"""
Inference script: evaluate a single checkpoint across multiple missing rates.

Usage examples:
  # Infer on all missing rates 0.0 to 0.9 using auto-detected checkpoint
  python infer.py --model dp --dataset mosi --seeds 1111

  # Infer using a specific checkpoint file
  python infer.py --model dp --dataset mosi --seeds 1111 \
      --checkpoint ./pt/dp-mosi-missing0.0.pth

  # Infer on specific missing rates only
  python infer.py --model dp --dataset mosi --seeds 1111 \
      --missing_rates 0.0 0.5 0.9

  # Infer with DMD model (uses main-dmd.py style model building)
  python infer.py --model dmd --dataset mosi --seeds 1111 \
      --checkpoint ./pt/dmd-mosi-missing0.0.pth

Notes:
  - Training masking is random (np.random.uniform) and does NOT depend on
    missing_rate_eval_test, so a model trained at any missing_rate produces
    the same weights for a given seed. Only eval/test masking changes.
  - This script loads the checkpoint once per seed and evaluates it on
    every requested missing rate, saving results to the same CSV format
    as the training pipeline.
"""

import argparse
import gc
import logging
import os
from pathlib import Path

import ast
import numpy as np
import pandas as pd
import torch

from config import get_config_regression, normalize_dataset_name
from data_loader import MMDataLoader
from trains.ATIO import ATIO
from utils import assign_gpu, count_parameters, setup_seed
from models import build_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
logger = logging.getLogger('DecAlign')


def _parse_override_value(value):
    lowered = value.lower()
    if lowered == 'true':
        return True
    if lowered == 'false':
        return False
    if lowered == 'none':
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _parse_config_overrides(items):
    overrides = {}
    for item in items or []:
        if '=' not in item:
            raise ValueError(f"Invalid --config_override '{item}'. Expected key=value.")
        key, value = item.split('=', 1)
        overrides[key] = _parse_override_value(value)
    return overrides


def _set_logger(log_dir, model_name, dataset_name, verbose_level):
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}-infer.log"
    logger = logging.getLogger('DecAlign')
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def _run_single_test(args, model, trainer, num_workers):
    """Build dataloader with current missing_rate_eval_test and run test."""
    dataloader = MMDataLoader(args, num_workers)
    results = trainer.do_test(model, dataloader['test'], mode="TEST")
    return results


def infer_all_missing_rates(
    model_name, dataset_name, checkpoint_path=None, config_file="",
    seeds=None, missing_rates=None, data_dir="", model_save_dir="",
    res_save_dir="", log_dir="", gpu_ids=None, num_workers=4,
    verbose_level=1, config=None,
):
    if gpu_ids is None:
        gpu_ids = [0]
    if seeds is None:
        seeds = [1111]
    if missing_rates is None:
        missing_rates = [round(x * 0.1, 1) for x in range(10)]  # 0.0 to 0.9

    model_name = model_name.lower()
    dataset_name = normalize_dataset_name(dataset_name)

    # Resolve config file
    if config_file != "":
        config_file = Path(config_file)
    else:
        config_map = {
            "mosi": {
                "dmd": "dmd_mosi_config.json",
                "dp": "dp_mosi_config.json",
            },
            "mosei": {
                "dmd": "dmd_mosei_config.json",
            },
            "iemocap": {},
            "sims": {},
        }
        dataset_configs = config_map.get(dataset_name, {})
        config_name = dataset_configs.get(model_name)
        if config_name is None:
            # Try model-specific pattern first, then fall back to dataset defaults
            specific = f"{model_name}_{dataset_name}_config.json"
            specific_path = Path(__file__).parent / "config" / specific
            if specific_path.is_file():
                config_name = specific
            elif dataset_name == "mosi":
                config_name = "dp_mosi_config.json"
            elif dataset_name == "mosei":
                config_name = "dec_mosei_config.json"
            elif dataset_name == "iemocap":
                config_name = "iemocap_decalign_config.json"
            elif dataset_name == "sims":
                config_name = "dec_sims_config.json"
            else:
                config_name = "dec_config.json"
        config_file = Path(__file__).parent / "config" / config_name
    if not config_file.is_file():
        raise ValueError(f"Config file {str(config_file)} not found.")

    if model_save_dir == "":
        model_save_dir = "./pt"
    if res_save_dir == "":
        res_save_dir = "./result"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)
    if log_dir == "":
        log_dir = "./log"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    _set_logger(log_dir, model_name, dataset_name, verbose_level)

    # Build base config (use first missing_rate just to init; we'll override per-loop)
    args = get_config_regression(model_name, dataset_name, config_file,
                                 data_dir=data_dir if data_dir else None)
    args.mode = 'test'
    args.device = assign_gpu(gpu_ids)
    args.feature_T = ""
    args.feature_A = ""
    args.feature_V = ""

    # Apply config overrides
    if config:
        for key, value in config.items():
            setattr(args, key, value)

    # Ensure data_missing is enabled (required for masking)
    args.data_missing = True

    # Results CSV setup
    res_save_path = Path(res_save_dir) / "normal"
    res_save_path.mkdir(parents=True, exist_ok=True)
    csv_file = res_save_path / f"{dataset_name}.csv"

    # Get trainer
    trainer = ATIO().getTrain(args)

    # Resolve checkpoint path (same for all missing rates & seeds)
    if checkpoint_path:
        ckpt = checkpoint_path
    else:
        base = Path(model_save_dir)
        candidate = base / f"{model_name}-{dataset_name}-missing0.0.pth"
        if candidate.is_file():
            ckpt = str(candidate)
        else:
            pattern = f"{model_name}-{dataset_name}-missing*.pth"
            matches = sorted(base.glob(pattern))
            if matches:
                ckpt = str(matches[0])
            else:
                raise FileNotFoundError(
                    f"No checkpoint found for {model_name}-{dataset_name} in {model_save_dir}/. "
                    f"Searched pattern: {pattern}. "
                    f"Please specify --checkpoint explicitly."
                )

    print(f"\n{'='*60}")
    print(f"Inference: {model_name} on {dataset_name}")
    print(f"Seeds: {seeds}")
    print(f"Missing rates: {missing_rates}")
    print(f"Checkpoint: {ckpt}")
    print(f"Results: {csv_file}")
    print(f"{'='*60}\n")

    args.model_save_path = ckpt

    # Collect results: {missing_rate: [result_seed1, result_seed2, ...]}
    mr_results = {mr: [] for mr in missing_rates}

    for seed in seeds:
        setup_seed(seed)
        args.cur_seed = seed

        # Build model once per seed, reuse across missing rates
        model = build_model(args)
        model = model.cuda()
        state_dict = torch.load(ckpt, map_location=args.device)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info(f"Loaded checkpoint: {ckpt} (seed={seed})")

        for missing_rate in missing_rates:
            setup_seed(seed)  # Reset seed so masking is reproducible
            args.missing_rate_eval_test = missing_rate

            logger.info(f"Testing missing_rate={missing_rate:.1f}, seed={seed}")
            print(f"  Testing missing_rate={missing_rate:.1f}, seed={seed} ...")
            results = _run_single_test(args, model, trainer, num_workers)
            mr_results[missing_rate].append(results)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Save all results to CSV (same format as training pipeline)
    for missing_rate in missing_rates:
        seed_results = mr_results[missing_rate]
        criterions = list(seed_results[0].keys())

        if csv_file.is_file():
            df = pd.read_csv(csv_file)
        else:
            columns = ["Model", "Missing Rate", "Seeds"]
            for c in criterions:
                columns.append(f"{c} Mean")
                columns.append(f"{c} Std")
            df = pd.DataFrame(columns=columns)

        expected_columns = ["Model", "Missing Rate", "Seeds"]
        for c in criterions:
            expected_columns.extend([f"{c} Mean", f"{c} Std"])

        for column in expected_columns:
            if column not in df.columns:
                df[column] = np.nan

        res = {
            "Model": model_name,
            "Missing Rate": missing_rate,
            "Seeds": str(seeds),
        }

        for c in criterions:
            values = [r[c] for r in seed_results]
            mean = float(round(np.mean(values) * 100, 2))
            std = float(round(np.std(values) * 100, 2))
            res[f"{c} Mean"] = mean
            res[f"{c} Std"] = std

        df.loc[len(df)] = [res.get(column, np.nan) for column in df.columns]
        df.to_csv(csv_file, index=None)

        # Print summary for this missing rate
        acc_key = "Acc_2 Mean" if "Acc_2 Mean" in res else None
        mae_key = "MAE Mean" if "MAE Mean" in res else None
        summary_parts = [f"missing_rate={missing_rate:.1f}"]
        if acc_key:
            summary_parts.append(f"Acc2={res[acc_key]}")
        if mae_key:
            summary_parts.append(f"MAE={res[mae_key]}")
        print(f"  -> {', '.join(summary_parts)}")

    logger.info(f"All results saved to {csv_file}.")
    print(f"\nDone! Results saved to {csv_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Infer from a single checkpoint across multiple missing rates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--model', type=str, required=True,
                        help="Model name (dp, misa, midas, dar, dmd, lnln, cenet, self_mm, decalign)")
    parser.add_argument('--dataset', type=str, default='mosi',
                        choices=['mosi', 'mosei', 'sims', 'chsims', 'ch-sims', 'ch_sims', 'iemocap'],
                        help='Dataset name')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file. If not set, auto-detects from model_save_dir.')
    parser.add_argument('--config_file', type=str, default='',
                        help='Path to model config JSON')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--model_save_dir', type=str, default='./pt',
                        help='Directory to search for checkpoints when --checkpoint is not specified (default: ./pt)')
    parser.add_argument('--res_save_dir', type=str, default='./result',
                        help='Directory to save results')
    parser.add_argument('--log_dir', type=str, default='./log',
                        help='Directory to save logs')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1111],
                        help='Random seeds')
    parser.add_argument('--missing_rates', type=float, nargs='+', default=None,
                        help='Missing rates to evaluate (default: 0.0 0.1 ... 0.9)')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0],
                        help='GPU IDs to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument(
        '--config_override',
        action='append',
        default=[],
        help='Override any config value with key=value. Can be repeated.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_overrides = _parse_config_overrides(args.config_override)

    infer_all_missing_rates(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        config_file=args.config_file,
        data_dir=args.data_dir,
        model_save_dir=args.model_save_dir,
        res_save_dir=args.res_save_dir,
        log_dir=args.log_dir,
        seeds=args.seeds,
        missing_rates=args.missing_rates,
        gpu_ids=args.gpu_ids,
        num_workers=args.num_workers,
        config=config_overrides if config_overrides else None,
    )


if __name__ == '__main__':
    main()

# python infer.py \
#     --model dp \
#     --dataset mosi \
#     --config_file ./config/dp_mosi_config.json \
#     --checkpoint ./pt/dp-mosi-missing0.0.pth \
#     --seeds 1111 \
#     --missing_rates 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
#     --data_dir ./data \
#     --res_save_dir ./result \
#     --log_dir ./log \
#     --gpu_ids 0 \
#     --num_workers 4