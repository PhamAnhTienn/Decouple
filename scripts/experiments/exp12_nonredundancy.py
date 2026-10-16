"""
Experiment 12 – Non-redundancy of uᶜ and uˢ
=============================================
Show that shared uncertainty uᶜ and specific uncertainty uˢ capture
**different** aspects of reliability, justifying the two-branch design.

Measurements:
  1. **Spearman ρ(uᶜ, uˢ)** per modality and missing rate.
     Low ρ → the two uncertainties are largely independent.

  2. **Median-split into 4 states** (LL, LH, HL, HH):
     - LL: both uᶜ and uˢ below median  (both confident)
     - LH: uᶜ low, uˢ high             (shared OK, specific degraded)
     - HL: uᶜ high, uˢ low             (shared degraded, specific OK)
     - HH: both above median            (both degraded)
     A large proportion of LH + HL indicates that the two uncertainties
     often disagree — evidence of non-redundancy.

Output
------
* Heatmap: Spearman ρ (rows = modality, cols = missing rate)
* 100% stacked bar: fraction of LL / LH / HL / HH per modality × missing rate
* Saved to result/experiments/exp12/
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import get_config_regression
from data_loader import MMDataLoader
from models import build_model
from utils import assign_gpu, setup_seed

logger = logging.getLogger("DecAlign")
MODALITIES = ["l", "v", "a"]
MOD_NAMES = {"l": "Language", "v": "Vision", "a": "Audio"}


# =====================================================================
# Collect uncertainties
# =====================================================================

def collect_uncertainties(model, dataloader, device):
    """
    Run model and collect uᶜ (shared uncertainty) and uˢ (specific uncertainty)
    per modality per sample.

    uᶜ is derived from r_shared via u = 1 − r  (monotone transform)
    uˢ is derived from r_specific via u = 1 − r
    """
    model.eval()
    u_shared = {m: [] for m in MODALITIES}
    u_specific = {m: [] for m in MODALITIES}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Exp12] Collecting uncertainties"):
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            text_m = batch.get("text_m", batch["text"]).to(device)
            audio_m = batch.get("audio_m", batch["audio"]).to(device)
            vision_m = batch.get("vision_m", batch["vision"]).to(device)

            output = model(text_m, audio_m, vision_m, text, audio, vision)

            for m in MODALITIES:
                # Convert reliability to uncertainty: u = 1 − r
                u_shared[m].append((1.0 - output["r_shared"][m]).cpu())
                u_specific[m].append((1.0 - output["r_specific"][m]).cpu())

    u_shared = {m: torch.cat(u_shared[m]).numpy() for m in MODALITIES}
    u_specific = {m: torch.cat(u_specific[m]).numpy() for m in MODALITIES}
    return u_shared, u_specific


# =====================================================================
# Compute Spearman ρ and median-split states
# =====================================================================

def compute_metrics(u_shared, u_specific):
    """
    For each modality:
      - Spearman ρ(uᶜ, uˢ)
      - Fraction of LL, LH, HL, HH states
    """
    results = {}
    for m in MODALITIES:
        uc = u_shared[m]
        us = u_specific[m]

        # Spearman
        rho, pval = stats.spearmanr(uc, us)

        # Median split
        med_c = np.median(uc)
        med_s = np.median(us)
        low_c = uc <= med_c
        low_s = us <= med_s

        n = len(uc)
        ll = np.sum(low_c & low_s) / n
        lh = np.sum(low_c & ~low_s) / n
        hl = np.sum(~low_c & low_s) / n
        hh = np.sum(~low_c & ~low_s) / n

        results[m] = {
            "spearman_rho": float(rho),
            "spearman_p": float(pval),
            "frac_LL": float(ll),
            "frac_LH": float(lh),
            "frac_HL": float(hl),
            "frac_HH": float(hh),
            "disagree_frac": float(lh + hl),  # LH + HL
        }

    return results


# =====================================================================
# Visualization
# =====================================================================


def plot_heatmap_and_bars(all_results, missing_rates, save_dir, dataset):
    """
    Save two separate figures:
      Figure 1: Heatmap of Spearman rho
      Figure 2: 100% stacked bar of 4 states
    """
    n_rates = len(missing_rates)

    # Build matrices (unchanged)
    rho_matrix = np.zeros((3, n_rates))
    state_matrix = np.zeros((3, n_rates, 4))

    for j, mr in enumerate(missing_rates):
        if mr not in all_results:
            continue
        for i, m in enumerate(MODALITIES):
            r = all_results[mr][m]
            rho_matrix[i, j] = r["spearman_rho"]
            state_matrix[i, j] = [
                r["frac_LL"],
                r["frac_LH"],
                r["frac_HL"],
                r["frac_HH"],
            ]

    # ================================================================
    # Figure 1: Spearman rho heatmap
    # ================================================================
    fig1, ax1 = plt.subplots(figsize=(8, 5))

    im = ax1.imshow(
        rho_matrix,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-1,
        vmax=1,
    )

    ax1.set_yticks(range(3))
    ax1.set_yticklabels([MOD_NAMES[m] for m in MODALITIES])
    ax1.set_xticks(range(n_rates))
    ax1.set_xticklabels([f"{r:.1f}" for r in missing_rates], rotation=45)
    ax1.set_xlabel("Missing Rate")
    ax1.set_title("Spearman ρ(uᶜ, uˢ)")

    for i in range(3):
        for j in range(n_rates):
            ax1.text(
                j, i, f"{rho_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(rho_matrix[i, j]) > 0.5 else "black",
            )

    fig1.colorbar(im, ax=ax1)
    fig1.tight_layout()

    heatmap_path = os.path.join(
        save_dir, f"exp12_nonredundancy_heatmap_{dataset}.png"
    )
    fig1.savefig(heatmap_path, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    print(f"Heatmap saved to {heatmap_path}")

    # ================================================================
    # Figure 2: 100% stacked bar
    # ================================================================
    fig2, ax2 = plt.subplots(figsize=(12, 5))

    x_labels = []
    ll_vals, lh_vals, hl_vals, hh_vals = [], [], [], []

    for j, mr in enumerate(missing_rates):
        for i, m in enumerate(MODALITIES):
            x_labels.append(f"{MOD_NAMES[m][0]}\nmr={mr:.1f}")
            ll_vals.append(state_matrix[i, j, 0])
            lh_vals.append(state_matrix[i, j, 1])
            hl_vals.append(state_matrix[i, j, 2])
            hh_vals.append(state_matrix[i, j, 3])

    x = np.arange(len(x_labels))
    width = 0.7

    ax2.bar(
        x, ll_vals, width,
        label="LL (both low)",
        color="#4CAF50", alpha=0.85,
    )
    ax2.bar(
        x, lh_vals, width,
        bottom=ll_vals,
        label="LH (uᶜ low, uˢ high)",
        color="#FF9800", alpha=0.85,
    )

    bottoms_2 = [a + b for a, b in zip(ll_vals, lh_vals)]
    ax2.bar(
        x, hl_vals, width,
        bottom=bottoms_2,
        label="HL (uᶜ high, uˢ low)",
        color="#2196F3", alpha=0.85,
    )

    bottoms_3 = [a + b for a, b in zip(bottoms_2, hl_vals)]
    ax2.bar(
        x, hh_vals, width,
        bottom=bottoms_3,
        label="HH (both high)",
        color="#F44336", alpha=0.85,
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=7)
    ax2.set_ylabel("Fraction")
    ax2.set_title("Median-Split States (LL / LH / HL / HH)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_ylim(0, 1.05)

    fig2.tight_layout()

    bars_path = os.path.join(
        save_dir, f"exp12_nonredundancy_states_{dataset}.png"
    )
    fig2.savefig(bars_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    print(f"Stacked bar saved to {bars_path}")


def print_table(all_results, missing_rates):
    print("\n" + "=" * 90)
    print("EXPERIMENT 12: Non-redundancy of uᶜ and uˢ")
    print("=" * 90)
    header = f"{'MR':<6} {'Mod':<8} {'ρ(uᶜ,uˢ)':<12} {'p-value':<12} {'LL':<8} {'LH':<8} {'HL':<8} {'HH':<8} {'LH+HL':<8}"
    print(header)
    print("-" * 90)

    for mr in missing_rates:
        if mr not in all_results:
            continue
        for m in MODALITIES:
            r = all_results[mr][m]
            print(
                f"{mr:<6.1f} {MOD_NAMES[m]:<8} {r['spearman_rho']:>+.4f}     "
                f"{r['spearman_p']:.2e}   "
                f"{r['frac_LL']:.3f}  {r['frac_LH']:.3f}  {r['frac_HL']:.3f}  {r['frac_HH']:.3f}  "
                f"{r['disagree_frac']:.3f}"
            )
    print("=" * 90)


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Exp12: Non-redundancy of uᶜ and uˢ")
    parser.add_argument("--dataset", type=str, default="mosi", choices=["mosi", "mosei", "sims"])
    parser.add_argument("--config_file", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint")
    parser.add_argument("--missing_rates", type=float, nargs="+",
                        default=[0.0, 0.3, 0.5, 0.7],
                        help="Missing rates to evaluate")
    parser.add_argument("--save_dir", type=str, default="./result/experiments/exp12")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1111)
    args = parser.parse_args()

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

    all_results = {}

    for mr in args.missing_rates:
        ckpt_path = os.path.join(args.checkpoint_dir, f"dp-{args.dataset}-missing{mr}.pth")
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Checkpoint not found: {ckpt_path}, skipping mr={mr}")
            continue

        print(f"\n{'='*60}")
        print(f"  Dataset={args.dataset}  Missing Rate={mr}")
        print(f"{'='*60}")

        model_args.missing_rate_eval_test = mr
        setup_seed(args.seed)

        dataloader = MMDataLoader(model_args, args.num_workers)
        model = build_model(model_args).to(model_args.device)
        model.load_state_dict(torch.load(ckpt_path, map_location=model_args.device))

        # Build train bank
        logger.info("Building train-set KNN bank...")
        model.update_train_bank(model, dataloader["train"], model_args.device)

        # Collect
        u_shared, u_specific = collect_uncertainties(model, dataloader["test"], model_args.device)

        # Metrics
        metrics = compute_metrics(u_shared, u_specific)
        all_results[mr] = metrics

        # Save per-rate raw data
        torch.save(
            {"u_shared": u_shared, "u_specific": u_specific, "metrics": metrics},
            os.path.join(args.save_dir, f"exp12_{args.dataset}_mr{mr}.pt"),
        )

        del model
        torch.cuda.empty_cache()

    # Print table
    print_table(all_results, args.missing_rates)

    # Plot
    plot_heatmap_and_bars(all_results, args.missing_rates, args.save_dir, args.dataset)

    # Save summary
    summary_path = os.path.join(args.save_dir, f"exp12_{args.dataset}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
