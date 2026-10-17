"""
Experiment 4 – Reliability Validity
====================================
Validate that reliability scores (rᶜ, rˢ) genuinely reflect representation
degradation, rather than being mere geometric heuristics.

For each test sample we have both its **complete** and **corrupted** (missing)
counterpart.  We compute representation distortion using symmetric normalized Frobenius error:

    δᵠ(m,i) = ||Xᵠ(m,i; comp) − Xᵠ(m,i; miss)||_F / (||Xᵠ(m,i; comp)||_F + ||Xᵠ(m,i; miss)||_F + eps),  q ∈ {c, s}

Then measure Spearman ρ between:
    ρ(uᵠ, δᵠ) > 0   (uncertainty should increase with distortion)
    ρ(rᵠ, δᵠ) < 0   (reliability should decrease with distortion)

Output
------
* Console table: Dataset × score (rᶜ / rˢ) × Spearman ρ
* Appendix table: per-modality breakdown
* Saved results as .pt and .csv in result/experiments/exp04/
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import get_config_regression
from data_loader import MMDataLoader
from models import build_model
from utils import assign_gpu, setup_seed

logger = logging.getLogger("DecAlign")
MODALITIES = ["l", "v", "a"]
MOD_NAMES = {"l": "Language", "v": "Vision", "a": "Audio"}


# =====================================================================
# Core: collect representations for BOTH complete and corrupted inputs
# =====================================================================

def collect_paired_representations(model, dataloader, device):
    """
    Run the model twice per batch:
      1. **Corrupted** pass: text_m / audio_m / vision_m  (standard inference)
      2. **Complete**  pass: text   / audio   / vision    (no missing)

    For each sample return:
      delta_c_m,    delta_s_m     — symmetric normalized distortion: ||X_comp - X_miss||_F / (||X_comp||_F + ||X_miss||_F)
      z_c_m^{miss}, z_s_m^{miss}  — from corrupted pass (mean pooled, normalized)
      z_c_m^{comp}, z_s_m^{comp}  — from complete pass (mean pooled, normalized)
      r_shared_m,   r_specific_m  — reliability from corrupted pass
    """
    model.eval()

    acc = {
        # Symmetric normalized distortion
        **{f"delta_c_{m}": [] for m in MODALITIES},
        **{f"delta_s_{m}": [] for m in MODALITIES},
        # corrupted pooled
        **{f"z_c_{m}_miss": [] for m in MODALITIES},
        **{f"z_s_{m}_miss": [] for m in MODALITIES},
        # complete pooled
        **{f"z_c_{m}_comp": [] for m in MODALITIES},
        **{f"z_s_{m}_comp": [] for m in MODALITIES},
        # reliability & uncertainty (from corrupted pass)
        **{f"r_shared_{m}": [] for m in MODALITIES},
        **{f"r_specific_{m}": [] for m in MODALITIES}
    }
    labels_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Exp04] Collecting paired representations"):
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            text_m = batch.get("text_m", batch["text"]).to(device)
            audio_m = batch.get("audio_m", batch["audio"]).to(device)
            vision_m = batch.get("vision_m", batch["vision"]).to(device)
            labels = batch["labels"]["M"].to(device)

            # ── corrupted pass ──
            out_miss = model(text_m, audio_m, vision_m, text, audio, vision)

            # ── complete pass  ──
            out_comp = model(text, audio, vision, text, audio, vision)

            for m in MODALITIES:
                # Shared reps: c_m is (T, B, D) in output
                c_miss = out_miss[f"c_{m}"]  # (T, B, D)
                c_comp = out_comp[f"c_{m}"]
                s_miss = out_miss[f"s_{m}"]
                s_comp = out_comp[f"s_{m}"]

                # Frobenius norms across (T, D) for each sample in batch
                norm_c_comp = torch.norm(c_comp, p="fro", dim=(0, 2))  # (B,)
                norm_c_miss = torch.norm(c_miss, p="fro", dim=(0, 2))
                diff_c = torch.norm(c_comp - c_miss, p="fro", dim=(0, 2))
                both_zero_c = (norm_c_comp < 1e-6) & (norm_c_miss < 1e-6)
                delta_c = diff_c / (norm_c_comp + norm_c_miss + 1e-8)
                delta_c[both_zero_c] = 0.0

                norm_s_comp = torch.norm(s_comp, p="fro", dim=(0, 2))
                norm_s_miss = torch.norm(s_miss, p="fro", dim=(0, 2))
                diff_s = torch.norm(s_comp - s_miss, p="fro", dim=(0, 2))
                both_zero_s = (norm_s_comp < 1e-6) & (norm_s_miss < 1e-6)
                delta_s = diff_s / (norm_s_comp + norm_s_miss + 1e-8)
                delta_s[both_zero_s] = 0.0

                acc[f"delta_c_{m}"].append(delta_c.cpu())
                acc[f"delta_s_{m}"].append(delta_s.cpu())

                acc[f"z_c_{m}_miss"].append(
                    F.normalize(c_miss.mean(dim=0), dim=1).cpu()
                )
                acc[f"z_c_{m}_comp"].append(
                    F.normalize(c_comp.mean(dim=0), dim=1).cpu()
                )
                acc[f"z_s_{m}_miss"].append(
                    F.normalize(s_miss.mean(dim=0), dim=1).cpu()
                )
                acc[f"z_s_{m}_comp"].append(
                    F.normalize(s_comp.mean(dim=0), dim=1).cpu()
                )

                # Reliability & uncertainty from corrupted pass
                acc[f"r_shared_{m}"].append(out_miss["r_shared"][m].cpu())
                acc[f"r_specific_{m}"].append(out_miss["r_specific"][m].cpu())

            labels_list.append(labels.cpu())

    # Concatenate
    data = {k: torch.cat(v, dim=0) for k, v in acc.items()}
    data["labels"] = torch.cat(labels_list, dim=0).squeeze()
    return data


# =====================================================================
# Compute distortion + Spearman correlations
# =====================================================================

def compute_distortion_correlations(data):
    """
    δᵠ(m,i) = ||X_comp - X_miss||_F / (||X_comp||_F + ||X_miss||_F + eps)
    Then Spearman ρ(rᵠ, δᵠ) and ρ(uᵠ, δᵠ).
    """
    results = {}

    for m in MODALITIES:
        # ── Shared distortion ──
        delta_c = data[f"delta_c_{m}"].numpy()

        # ── Specific distortion ──
        delta_s = data[f"delta_s_{m}"].numpy()

        r_c = data[f"r_shared_{m}"].numpy()
        r_s = data[f"r_specific_{m}"].numpy()

        # Uncertainty u = 1 - r as a monotone proxy
        u_c = 1.0 - r_c
        u_s = 1.0 - r_s

        # Spearman correlations
        rho_rc_dc, p_rc_dc = stats.spearmanr(r_c, delta_c)
        rho_rs_ds, p_rs_ds = stats.spearmanr(r_s, delta_s)
        rho_uc_dc, p_uc_dc = stats.spearmanr(u_c, delta_c)
        rho_us_ds, p_us_ds = stats.spearmanr(u_s, delta_s)

        results[m] = {
            "delta_c_mean": float(np.mean(delta_c)),
            "delta_c_std": float(np.std(delta_c)),
            "delta_s_mean": float(np.mean(delta_s)),
            "delta_s_std": float(np.std(delta_s)),
            # ρ(rᶜ, δᶜ) — expect negative
            "rho_rc_dc": float(rho_rc_dc),
            "p_rc_dc": float(p_rc_dc),
            # ρ(rˢ, δˢ) — expect negative
            "rho_rs_ds": float(rho_rs_ds),
            "p_rs_ds": float(p_rs_ds),
            # ρ(uᶜ, δᶜ) — expect positive
            "rho_uc_dc": float(rho_uc_dc),
            "p_uc_dc": float(p_uc_dc),
            # ρ(uˢ, δˢ) — expect positive
            "rho_us_ds": float(rho_us_ds),
            "p_us_ds": float(p_us_ds),
        }

    return results


# =====================================================================
# Pretty-print tables
# =====================================================================

def print_main_table(all_results, global_results=None):
    """Table 3 style: Dataset × score × Spearman ρ."""
    print("\n" + "=" * 80)
    print("TABLE 3: Reliability Validity — Spearman ρ(score, distortion)")
    print("=" * 80)
    header = f"{'Dataset':<22} {'Score':<6} {'ρ':<10} {'p-value':<12} {'Expected':<10}"
    print(header)
    print("-" * 80)

    for dataset, mr_results in all_results.items():
        for mr, mod_results in mr_results.items():
            rho_rc_list = [mod_results[m]["rho_rc_dc"] for m in MODALITIES]
            rho_rs_list = [mod_results[m]["rho_rs_ds"] for m in MODALITIES]
            p_rc_list = [mod_results[m]["p_rc_dc"] for m in MODALITIES]
            p_rs_list = [mod_results[m]["p_rs_ds"] for m in MODALITIES]
            label = f"{dataset} (mr={mr})"

            if all(np.isnan(x) for x in rho_rc_list):
                mean_rho_rc = np.nan
                mean_p_rc = np.nan
            else:
                mean_rho_rc = np.nanmean(rho_rc_list)
                mean_p_rc = np.nanmean(p_rc_list)

            if all(np.isnan(x) for x in rho_rs_list):
                mean_rho_rs = np.nan
                mean_p_rs = np.nan
            else:
                mean_rho_rs = np.nanmean(rho_rs_list)
                mean_p_rs = np.nanmean(p_rs_list)

            if np.isnan(mean_rho_rc):
                print(f"{label:<22} {'rᶜ':<6} {'N/A (clean)':<10} {'N/A':<12} {'Baseline'}")
                print(f"{'':<22} {'rˢ':<6} {'N/A (clean)':<10} {'N/A':<12} {'Baseline'}")
            else:
                print(
                    f"{label:<22} {'rᶜ':<6} {mean_rho_rc:>+.4f}    "
                    f"{mean_p_rc:.2e}    {'ρ<0 ✓' if mean_rho_rc < 0 else 'ρ<0 ✗'}"
                )
                print(
                    f"{'':<22} {'rˢ':<6} {mean_rho_rs:>+.4f}    "
                    f"{mean_p_rs:.2e}    {'ρ<0 ✓' if mean_rho_rs < 0 else 'ρ<0 ✗'}"
                )

        if global_results and dataset in global_results:
            print("-" * 80)
            g = global_results[dataset]
            g_rc = np.mean([g[m]["rho_rc_dc"] for m in MODALITIES])
            g_rs = np.mean([g[m]["rho_rs_ds"] for m in MODALITIES])
            g_p_rc = np.mean([g[m]["p_rc_dc"] for m in MODALITIES])
            g_p_rs = np.mean([g[m]["p_rs_ds"] for m in MODALITIES])

            g_label = f"{dataset} (Global Pooled)"
            print(
                f"{g_label:<22} {'rᶜ':<6} {g_rc:>+.4f}    "
                f"{g_p_rc:.2e}    {'ρ<0 ✓' if g_rc < 0 else 'ρ<0 ✗'}"
            )
            print(
                f"{'':<22} {'rˢ':<6} {g_rs:>+.4f}    "
                f"{g_p_rs:.2e}    {'ρ<0 ✓' if g_rs < 0 else 'ρ<0 ✗'}"
            )
    print("=" * 80)


def print_appendix_table(all_results, global_results=None):
    """Appendix: per-modality breakdown."""
    print("\n" + "=" * 90)
    print("APPENDIX: Per-Modality Reliability Validity")
    print("=" * 90)
    header = (
        f"{'Dataset':<12} {'MR':<7} {'Mod':<10} "
        f"{'ρ(rᶜ,δᶜ)':<12} {'p':<10} "
        f"{'ρ(rˢ,δˢ)':<12} {'p':<10} "
        f"{'δᶜ mean':<10} {'δˢ mean':<10}"
    )
    print(header)
    print("-" * 90)

    for dataset, mr_results in all_results.items():
        for mr, mod_results in mr_results.items():
            for m in MODALITIES:
                r = mod_results[m]
                rho_rc_str = f"{r['rho_rc_dc']:>+.4f}" if not np.isnan(r['rho_rc_dc']) else "   N/A   "
                rho_rs_str = f"{r['rho_rs_ds']:>+.4f}" if not np.isnan(r['rho_rs_ds']) else "   N/A   "
                p_rc_str = f"{r['p_rc_dc']:.2e}" if not np.isnan(r['p_rc_dc']) else "   N/A   "
                p_rs_str = f"{r['p_rs_ds']:.2e}" if not np.isnan(r['p_rs_ds']) else "   N/A   "
                print(
                    f"{dataset:<12} {mr:<7} {MOD_NAMES[m]:<10} "
                    f"{rho_rc_str}      {p_rc_str}  "
                    f"{rho_rs_str}      {p_rs_str}  "
                    f"{r['delta_c_mean']:.4f}    {r['delta_s_mean']:.4f}"
                )

        if global_results and dataset in global_results:
            print("-" * 90)
            for m in MODALITIES:
                g = global_results[dataset][m]
                print(
                    f"{dataset:<12} {'Global':<7} {MOD_NAMES[m]:<10} "
                    f"{g['rho_rc_dc']:>+.4f}      {g['p_rc_dc']:.2e}  "
                    f"{g['rho_rs_ds']:>+.4f}      {g['p_rs_ds']:.2e}  "
                    f"{g['delta_c_mean']:.4f}    {g['delta_s_mean']:.4f}"
                )
    print("=" * 90)


# =====================================================================
# Main
# =====================================================================

def run_single(args, checkpoint_path, missing_rate, num_workers=4):
    """Run experiment for a single checkpoint + missing rate."""
    args.missing_rate_eval_test = missing_rate
    args.data_missing = True
    args.cur_seed = getattr(args, "cur_seed", 1111)

    setup_seed(args.cur_seed)

    dataloader = MMDataLoader(args, num_workers)
    model = build_model(args).to(args.device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))

    # Build train bank (needed for neighbor uncertainty)
    logger.info("Building train-set KNN bank...")
    model.update_train_bank(
        model_forward_fn=model,
        train_dataloader=dataloader["train"],
        device=args.device,
    )

    # Collect paired representations on test set
    data = collect_paired_representations(model, dataloader["test"], args.device)

    # Compute correlations
    correlations = compute_distortion_correlations(data)

    del model
    torch.cuda.empty_cache()

    return correlations, data


def main():
    parser = argparse.ArgumentParser(description="Exp04: Reliability Validity")
    parser.add_argument("--dataset", type=str, default="mosi", choices=["mosi", "mosei", "sims"])
    parser.add_argument("--config_file", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint",
                        help="Directory containing dp-{dataset}-missing{rate}.pth checkpoints")
    parser.add_argument("--missing_rates", type=float, nargs="+",
                        default=[0.0, 0.4, 0.6, 0.8, 0.9],
                        help="Missing rates to evaluate")
    parser.add_argument("--save_dir", type=str, default="./result/experiments/exp04")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1111)
    args = parser.parse_args()

    # Setup
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    if args.config_file == "":
        config_file = str(Path(__file__).resolve().parents[2] / "config" / f"dp_{args.dataset}_config.json")
    else:
        config_file = args.config_file

    model_args = get_config_regression("dp", args.dataset, config_file, data_dir=args.data_dir)
    model_args.device = assign_gpu(args.gpu_ids)
    model_args.cur_seed = args.seed
    model_args.data_missing = True
    model_args.mode = "test"

    all_results = {args.dataset: {}}
    all_data_list = []

    for mr in args.missing_rates:
        ckpt_path = os.path.join(args.checkpoint_dir, f"dp-{args.dataset}-missing{mr}.pth")
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Checkpoint not found: {ckpt_path}, skipping mr={mr}")
            continue

        print(f"\n{'='*60}")
        print(f"  Dataset={args.dataset}  Missing Rate={mr}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        correlations, data = run_single(model_args, ckpt_path, mr, args.num_workers)
        all_results[args.dataset][mr] = correlations
        all_data_list.append(data)

        # Save raw data
        torch.save(
            {"correlations": correlations, "raw_data": data},
            os.path.join(args.save_dir, f"exp04_{args.dataset}_mr{mr}.pt"),
        )

    # Compute Global Pooled correlations across all evaluated missing rates
    global_results = {args.dataset: {}}
    if len(all_data_list) > 1:
        for m in MODALITIES:
            pooled_rc = np.concatenate([d[f"r_shared_{m}"].numpy() for d in all_data_list])
            pooled_rs = np.concatenate([d[f"r_specific_{m}"].numpy() for d in all_data_list])
            pooled_dc = np.concatenate([d[f"delta_c_{m}"].numpy() for d in all_data_list])
            pooled_ds = np.concatenate([d[f"delta_s_{m}"].numpy() for d in all_data_list])

            rho_rc, p_rc = stats.spearmanr(pooled_rc, pooled_dc)
            rho_rs, p_rs = stats.spearmanr(pooled_rs, pooled_ds)

            global_results[args.dataset][m] = {
                "delta_c_mean": float(np.mean(pooled_dc)),
                "delta_c_std": float(np.std(pooled_dc)),
                "delta_s_mean": float(np.mean(pooled_ds)),
                "delta_s_std": float(np.std(pooled_ds)),
                "rho_rc_dc": float(rho_rc),
                "p_rc_dc": float(p_rc),
                "rho_rs_ds": float(rho_rs),
                "p_rs_ds": float(p_rs),
            }

    # Print tables
    print_main_table(all_results, global_results)
    print_appendix_table(all_results, global_results)

    # Save summary JSON
    summary_path = os.path.join(args.save_dir, f"exp04_{args.dataset}_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"per_mr": all_results, "global": global_results}, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
