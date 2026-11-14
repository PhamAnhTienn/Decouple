#!/usr/bin/env python3
"""
Plot sample-level relationship between Shared Reliability (r^c) and Unimodal Label Disagreement
under missing scenarios (Clean vs. MR=0.5 vs. MR=0.8) on CH-SIMS.

Follows CVPR style guidelines via scripts/experiments/paper_style.py:
- Two-column wide layout (width = 6.875 in).
- Times-like serif fonts, STIX math, PDF fonttype 42.
- Layout 'constrained', no tight_layout() or bbox_inches='tight'.
- Saves vector PDF and PNG preview.
"""

import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "experiments"))
from paper_style import apply_style, new_fig, save_fig, shared_legend, FS

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


def load_sims_raw_labels(data_path):
    with open(data_path, "rb") as f:
        raw = pickle.load(f)

    y_l = np.array(raw["test"]["regression_labels_T"])
    y_a = np.array(raw["test"]["regression_labels_A"])
    y_v = np.array(raw["test"]["regression_labels_V"])
    y_m = np.array(raw["test"]["regression_labels"])

    sign_l = np.sign(y_l)
    sign_v = np.sign(y_v)
    sign_a = np.sign(y_a)

    is_agree = (sign_l == sign_v) & (sign_v == sign_a) & (sign_l != 0)
    is_conflict = (sign_l * sign_v < 0) | (sign_l * sign_a < 0) | (sign_v * sign_a < 0)
    is_partial = (~is_agree) & (~is_conflict)

    pair_diff = (np.abs(y_l - y_v) + np.abs(y_l - y_a) + np.abs(y_v - y_a)) / 3.0

    return {
        "y_l": y_l, "y_v": y_v, "y_a": y_a, "y_m": y_m,
        "is_agree": is_agree, "is_partial": is_partial, "is_conflict": is_conflict,
        "pair_diff": pair_diff
    }


def load_exp04_mr(pt_path):
    d = torch.load(pt_path, map_location="cpu")
    rd = d["raw_data"]
    rc_l = rd["r_shared_l"].numpy()
    rc_v = rd["r_shared_v"].numpy()
    rc_a = rd["r_shared_a"].numpy()
    rc_mean = (rc_l + rc_v + rc_a) / 3.0

    rs_l = rd["r_specific_l"].numpy()
    rs_v = rd["r_specific_v"].numpy()
    rs_a = rd["r_specific_a"].numpy()
    rs_mean = (rs_l + rs_v + rs_a) / 3.0

    dc_mean = ((rd["delta_c_l"] + rd["delta_c_v"] + rd["delta_c_a"]) / 3.0).numpy()
    ds_mean = ((rd["delta_s_l"] + rd["delta_s_v"] + rd["delta_s_a"]) / 3.0).numpy()

    return {
        "rc_l": rc_l, "rc_v": rc_v, "rc_a": rc_a, "rc_mean": rc_mean,
        "rs_l": rs_l, "rs_v": rs_v, "rs_a": rs_a, "rs_mean": rs_mean,
        "dc_mean": dc_mean, "ds_mean": ds_mean
    }


def plot_sample_level_missing_comparison(labels_dict, mr_data_dict, save_stem):
    venue = "cvpr"
    kind = "wide"
    apply_style(venue=venue, kind=kind)

    # 1 row, 3 columns: Clean (0.0), Moderate Missing (0.5), Severe Missing (0.8)
    fig, axes = new_fig(nrows=1, ncols=3, height=2.3)
    fs = FS[kind]

    pair_diff = labels_dict["pair_diff"]
    is_agree = labels_dict["is_agree"]
    is_partial = labels_dict["is_partial"]
    is_conflict = labels_dict["is_conflict"]

    mr_configs = [
        ("0.0", "(a) Clean Baseline (MR = 0.0)", axes[0]),
        ("0.5", "(b) Moderate Missing (MR = 0.5)", axes[1]),
        ("0.8", "(c) Severe Missing (MR = 0.8)", axes[2]),
    ]

    x_vals = np.linspace(0.0, pair_diff.max(), 100)
    y_min, y_max = 0.22, 0.55

    for mr_key, title, ax in mr_configs:
        data = mr_data_dict[mr_key]
        rc_mean = data["rc_mean"]

        h1 = ax.scatter(
            pair_diff[is_agree], rc_mean[is_agree],
            s=11, alpha=0.6, color="#1f77b4", marker="o", label="Full Agreement ($N=226$)", edgecolors="none"
        )
        h2 = ax.scatter(
            pair_diff[is_partial], rc_mean[is_partial],
            s=11, alpha=0.5, color="#7f7f7f", marker="s", label="Weak / Neutral ($N=145$)", edgecolors="none"
        )
        h3 = ax.scatter(
            pair_diff[is_conflict], rc_mean[is_conflict],
            s=14, alpha=0.8, color="#d62728", marker="^", label="Polarity Conflict ($N=86$)", edgecolors="black", linewidths=0.3
        )

        slope, intercept, _, _, _ = stats.linregress(pair_diff, rc_mean)
        rho, p_rho = stats.spearmanr(pair_diff, rc_mean)
        h_line, = ax.plot(
            x_vals, intercept + slope * x_vals,
            color="black", linestyle="--", linewidth=1.1, label="Linear Fit"
        )

        ax.set_title(title, fontsize=fs["panel"], pad=4)
        ax.set_xlabel(r"Disagreement $\Delta_{\mathrm{pair}}$", fontsize=fs["label"])
        ax.set_xlim(-0.05, 1.7)
        ax.set_ylim(y_min, y_max)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(True, linestyle=":", alpha=0.45, linewidth=0.4)

        if ax == axes[0]:
            ax.set_ylabel(r"Shared Reliability $\bar{r}^c$", fontsize=fs["label"])
        else:
            ax.tick_params(labelleft=False)

        # Statistical callout box
        p_str = f"{p_rho:.2f}" if p_rho >= 0.01 else f"{p_rho:.1e}"
        annot_text = (
            f"Mean: {rc_mean.mean():.3f} " + r"$\pm$" + f" {rc_mean.std():.3f}\n"
            f"Spearman " + r"$\rho = " + f"{rho:+.3f}$\n"
            f"($p = {p_str}$)"
        )
        ax.text(
            0.96, 0.94, annot_text,
            transform=ax.transAxes,
            fontsize=fs["annot"] - 0.5,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.9, linewidth=0.5)
        )

    # Place shared legend at the top
    handles = [h1, h2, h3, h_line]
    labels = ["Full Agreement ($N=226$)", "Weak / Neutral ($N=145$)", "Polarity Conflict ($N=86$)", "Linear Fit"]
    shared_legend(fig, handles, labels, ncol=4)

    save_fig(fig, save_stem)
    plt.close(fig)
    print(f"[Done] Saved sample-level comparison to {save_stem}.pdf / .png")


def plot_decoupling_dynamics(labels_dict, mr_dict, save_stem):
    venue = "cvpr"
    kind = "wide"
    apply_style(venue=venue, kind=kind)

    # 1 row, 2 columns: (a) r^c vs r^s over missing rates; (b) Tier comparison across missing rates
    fig, axes = new_fig(nrows=1, ncols=2, height=2.3)
    ax1, ax2 = axes[0], axes[1]
    fs = FS[kind]

    mr_rates = [0.0, 0.4, 0.5, 0.6, 0.8, 0.9]
    rc_means = [mr_dict[f"{mr:.1f}"]["rc_mean"].mean() for mr in mr_rates]
    rc_stds = [mr_dict[f"{mr:.1f}"]["rc_mean"].std() for mr in mr_rates]

    rs_means = [mr_dict[f"{mr:.1f}"]["rs_mean"].mean() for mr in mr_rates]
    rs_stds = [mr_dict[f"{mr:.1f}"]["rs_mean"].std() for mr in mr_rates]

    rs_l = [mr_dict[f"{mr:.1f}"]["rs_l"].mean() for mr in mr_rates]
    rs_v = [mr_dict[f"{mr:.1f}"]["rs_v"].mean() for mr in mr_rates]
    rs_a = [mr_dict[f"{mr:.1f}"]["rs_a"].mean() for mr in mr_rates]

    # Panel (a): Evolution of r^c vs r^s
    ax1.plot(mr_rates, rc_means, "o-", color="#1f77b4", linewidth=1.5, markersize=5, label=r"Shared $\bar{r}^c$ (Consensus)")
    ax1.fill_between(mr_rates, np.array(rc_means) - np.array(rc_stds), np.array(rc_means) + np.array(rc_stds), color="#1f77b4", alpha=0.15)

    ax1.plot(mr_rates, rs_means, "s--", color="#d62728", linewidth=1.5, markersize=5, label=r"Specific $\bar{r}^s$ (Mean)")
    ax1.fill_between(mr_rates, np.array(rs_means) - np.array(rs_stds), np.array(rs_means) + np.array(rs_stds), color="#d62728", alpha=0.12)

    ax1.plot(mr_rates, rs_v, "^:", color="#2ca02c", linewidth=1.1, markersize=4, label=r"Specific $r_v^s$ (Vision)")
    ax1.plot(mr_rates, rs_a, "v:", color="#ff7f0e", linewidth=1.1, markersize=4, label=r"Specific $r_a^s$ (Audio)")
    ax1.plot(mr_rates, rs_l, "d:", color="#9467bd", linewidth=1.1, markersize=4, label=r"Specific $r_l^s$ (Language)")

    ax1.set_title(r"(a) Degradation Sensitivity: $\bar{r}^c$ vs. $r^s$", fontsize=fs["panel"], pad=4)
    ax1.set_xlabel("Missing Rate (MR)", fontsize=fs["label"])
    ax1.set_ylabel("Reliability Score", fontsize=fs["label"])
    ax1.set_xlim(-0.03, 0.93)
    ax1.set_ylim(0.15, 0.55)
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax1.grid(True, linestyle=":", alpha=0.45, linewidth=0.4)
    ax1.legend(loc="lower left", fontsize=fs["legend"] - 0.8, frameon=True, facecolor="#f8f9fa", edgecolor="#cccccc", framealpha=0.9)

    # Panel (b): Tier Breakdown across Missing Rates (Grouped Bar Chart)
    is_agree = labels_dict["is_agree"]
    is_partial = labels_dict["is_partial"]
    is_conflict = labels_dict["is_conflict"]

    eval_mrs = ["0.0", "0.5", "0.8"]
    tier_labels = ["Full Agree", "Weak / Neu.", "Conflict"]
    x = np.arange(len(eval_mrs))
    bar_width = 0.22

    # Plot r^c across tiers for MR=0.0, 0.5, 0.8
    rc_agree = [mr_dict[mr]["rc_mean"][is_agree].mean() for mr in eval_mrs]
    rc_partial = [mr_dict[mr]["rc_mean"][is_partial].mean() for mr in eval_mrs]
    rc_conflict = [mr_dict[mr]["rc_mean"][is_conflict].mean() for mr in eval_mrs]

    rs_agree = [mr_dict[mr]["rs_mean"][is_agree].mean() for mr in eval_mrs]
    rs_conflict = [mr_dict[mr]["rs_mean"][is_conflict].mean() for mr in eval_mrs]

    rects1 = ax2.bar(x - bar_width, rc_agree, bar_width, label=r"$\bar{r}^c$: Full Agree", color="#1f77b4", alpha=0.85)
    rects2 = ax2.bar(x, rc_partial, bar_width, label=r"$\bar{r}^c$: Weak / Neu.", color="#7f7f7f", alpha=0.75)
    rects3 = ax2.bar(x + bar_width, rc_conflict, bar_width, label=r"$\bar{r}^c$: Polarity Conflict", color="#d62728", alpha=0.85)

    # Overlay rs lines or points for contrast
    ax2.plot(x - bar_width, rs_agree, "k+", markersize=6, mew=1.2, label=r"$\bar{r}^s$ Agree")
    ax2.plot(x + bar_width, rs_conflict, "kx", markersize=6, mew=1.2, label=r"$\bar{r}^s$ Conflict")

    ax2.set_title(r"(b) Shared $\bar{r}^c$ Stability Across Disagreement Tiers", fontsize=fs["panel"], pad=4)
    ax2.set_xlabel("Evaluation Scenario", fontsize=fs["label"])
    ax2.set_ylabel("Reliability Score", fontsize=fs["label"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Clean (0.0)", "MR = 0.5", "MR = 0.8"])
    ax2.set_ylim(0.18, 0.58)
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2.grid(True, linestyle=":", alpha=0.45, linewidth=0.4)
    ax2.legend(loc="upper right", fontsize=fs["legend"] - 1.0, ncol=2, frameon=True, facecolor="#f8f9fa", edgecolor="#cccccc", framealpha=0.9, columnspacing=0.8, handletextpad=0.3)

    save_fig(fig, save_stem)
    plt.close(fig)
    print(f"[Done] Saved decoupling dynamics to {save_stem}.pdf / .png")


def main():
    data_path = PROJECT_ROOT / "data" / "SIMS" / "unaligned_39.pkl"
    labels_dict = load_sims_raw_labels(data_path)

    mr_rates = ["0.0", "0.4", "0.5", "0.6", "0.8", "0.9"]
    mr_dict = {}
    for mr in mr_rates:
        pt_path = PROJECT_ROOT / "result" / "experiments" / "exp04" / f"exp04_sims_mr{mr}.pt"
        if not pt_path.exists():
            print(f"[WARN] File not found: {pt_path}")
            continue
        mr_dict[mr] = load_exp04_mr(pt_path)

    fig_dir = PROJECT_ROOT / "result" / "experiments" / "exp04" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig1_stem = fig_dir / "exp04_sims_sample_level_rc_missing_comparison"
    plot_sample_level_missing_comparison(labels_dict, mr_dict, fig1_stem)

    fig2_stem = fig_dir / "exp04_sims_rc_rs_decoupling_dynamics"
    plot_decoupling_dynamics(labels_dict, mr_dict, fig2_stem)


if __name__ == "__main__":
    main()
