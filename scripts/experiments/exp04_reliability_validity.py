"""
Experiment 4 – Reliability Validity
====================================
Validate that reliability scores (rᶜ, rˢ) genuinely reflect representation
degradation, rather than being mere geometric heuristics.

For each test sample we have both its **complete** and **corrupted** (missing)
counterpart.  We compute representation distortion:

    δᵠ(m,i) = 1 − cos(zᵠ(m,i; complete), zᵠ(m,i; missing)),  q ∈ {c, s}

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
      z_c_m^{miss}, z_s_m^{miss}  — from corrupted pass
      z_c_m^{comp}, z_s_m^{comp}  — from complete pass
      r_shared_m,   r_specific_m  — reliability from corrupted pass
      u_consensus_m, u_neighbor_m — uncertainties from corrupted pass
    """
    model.eval()

    acc = {
        # corrupted
        **{f"z_c_{m}_miss": [] for m in MODALITIES},
        **{f"z_s_{m}_miss": [] for m in MODALITIES},
        # complete
        **{f"z_c_{m}_comp": [] for m in MODALITIES},
        **{f"z_s_{m}_comp": [] for m in MODALITIES},
        # reliability & uncertainty (from corrupted pass)
        **{f"r_shared_{m}": [] for m in MODALITIES},
        **{f"r_specific_{m}": [] for m in MODALITIES},
        **{f"u_shared_{m}": [] for m in MODALITIES},
        **{f"u_specific_{m}": [] for m in MODALITIES},
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
                acc[f"z_c_{m}_miss"].append(
                    F.normalize(c_miss.mean(dim=0), dim=1).cpu()
                )
                acc[f"z_c_{m}_comp"].append(
                    F.normalize(c_comp.mean(dim=0), dim=1).cpu()
                )

                # Specific reps: s_m is (T, B, D) in output
                s_miss = out_miss[f"s_{m}"]
                s_comp = out_comp[f"s_{m}"]
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
    δᵠ(m,i) = 1 − cos(z_comp, z_miss)
    Then Spearman ρ(rᵠ, δᵠ) and ρ(uᵠ, δᵠ).
    """
    results = {}

    for m in MODALITIES:
        # ── Shared distortion ──
        delta_c = 1.0 - F.cosine_similarity(
            data[f"z_c_{m}_comp"], data[f"z_c_{m}_miss"], dim=1
        ).numpy()

        # ── Specific distortion ──
        delta_s = 1.0 - F.cosine_similarity(
            data[f"z_s_{m}_comp"], data[f"z_s_{m}_miss"], dim=1
        ).numpy()

        r_c = data[f"r_shared_{m}"].numpy()
        r_s = data[f"r_specific_{m}"].numpy()

        # Uncertainty u = -log(r) * tau  (invert the mapping)
        # Or we can just compute 1 - r as a monotone proxy
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

def print_main_table(all_results):
    """Table 3 style: Dataset × score × Spearman ρ."""
    print("\n" + "=" * 80)
    print("TABLE 3: Reliability Validity — Spearman ρ(score, distortion)")
    print("=" * 80)
    header = f"{'Dataset':<12} {'Score':<6} {'ρ':<10} {'p-value':<12} {'Expected':<10}"
    print(header)
    print("-" * 80)

    for dataset, mr_results in all_results.items():
        for mr, mod_results in mr_results.items():
            # Average across modalities for the main table
            rho_rc_list = [mod_results[m]["rho_rc_dc"] for m in MODALITIES]
            rho_rs_list = [mod_results[m]["rho_rs_ds"] for m in MODALITIES]
            p_rc_list = [mod_results[m]["p_rc_dc"] for m in MODALITIES]
            p_rs_list = [mod_results[m]["p_rs_ds"] for m in MODALITIES]

            label = f"{dataset} (mr={mr})"
            print(
                f"{label:<20} {'rᶜ':<6} {np.mean(rho_rc_list):>+.4f}    "
                f"{np.mean(p_rc_list):.2e}    {'ρ<0 ✓' if np.mean(rho_rc_list) < 0 else 'ρ<0 ✗'}"
            )
            print(
                f"{'':<20} {'rˢ':<6} {np.mean(rho_rs_list):>+.4f}    "
                f"{np.mean(p_rs_list):.2e}    {'ρ<0 ✓' if np.mean(rho_rs_list) < 0 else 'ρ<0 ✗'}"
            )
    print("=" * 80)


def print_appendix_table(all_results):
    """Appendix: per-modality breakdown."""
    print("\n" + "=" * 90)
    print("APPENDIX: Per-Modality Reliability Validity")
    print("=" * 90)
    header = (
        f"{'Dataset':<12} {'MR':<5} {'Mod':<5} "
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
                print(
                    f"{dataset:<12} {mr:<5} {MOD_NAMES[m]:<5} "
                    f"{r['rho_rc_dc']:>+.4f}      {r['p_rc_dc']:.2e}  "
                    f"{r['rho_rs_ds']:>+.4f}      {r['p_rs_ds']:.2e}  "
                    f"{r['delta_c_mean']:.4f}    {r['delta_s_mean']:.4f}"
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
                        default=[0.0, 0.2, 0.5, 0.7],
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

        # Save raw data
        torch.save(
            {"correlations": correlations, "raw_data": data},
            os.path.join(args.save_dir, f"exp04_{args.dataset}_mr{mr}.pt"),
        )

    # Print tables
    print_main_table(all_results)
    print_appendix_table(all_results)

    # Save summary JSON
    summary_path = os.path.join(args.save_dir, f"exp04_{args.dataset}_summary.json")
    # Convert numpy types for JSON serialization
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
