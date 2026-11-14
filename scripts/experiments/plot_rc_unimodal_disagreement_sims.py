#!/usr/bin/env python3
"""
Plot relation between Shared Reliability (r^c) and Unimodal Label Disagreement on CH-SIMS
at individual sample resolution (no binning).

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
from paper_style import apply_style, new_fig, save_fig, FS

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


def load_sims_data_and_scores(pt_path, data_path):
    with open(data_path, "rb") as f:
        raw = pickle.load(f)

    y_l = np.array(raw["test"]["regression_labels_T"])
    y_a = np.array(raw["test"]["regression_labels_A"])
    y_v = np.array(raw["test"]["regression_labels_V"])
    y_m = np.array(raw["test"]["regression_labels"])

    d = torch.load(pt_path, map_location="cpu")
    rc_l = d["raw_data"]["r_shared_l"].numpy()
    rc_v = d["raw_data"]["r_shared_v"].numpy()
    rc_a = d["raw_data"]["r_shared_a"].numpy()
    rc_mean = (rc_l + rc_v + rc_a) / 3.0
    rc_sum = rc_l + rc_v + rc_a
    gamma_c = 1.0 - (1.0 - rc_l) * (1.0 - rc_v) * (1.0 - rc_a)

    w_l = rc_l / rc_sum
    w_v = rc_v / rc_sum
    w_a = rc_a / rc_sum

    sign_l = np.sign(y_l)
    sign_v = np.sign(y_v)
    sign_a = np.sign(y_a)

    is_agree = (sign_l == sign_v) & (sign_v == sign_a) & (sign_l != 0)
    is_conflict = (sign_l * sign_v < 0) | (sign_l * sign_a < 0) | (sign_v * sign_a < 0)
    is_partial = (~is_agree) & (~is_conflict)

    pair_diff = (np.abs(y_l - y_v) + np.abs(y_l - y_a) + np.abs(y_v - y_a)) / 3.0

    delta_y_l = np.abs(y_l - (y_a + y_v) / 2.0)
    delta_y_v = np.abs(y_v - (y_l + y_a) / 2.0)
    delta_y_a = np.abs(y_a - (y_l + y_v) / 2.0)

    return {
        "y_l": y_l, "y_v": y_v, "y_a": y_a, "y_m": y_m,
        "rc_l": rc_l, "rc_v": rc_v, "rc_a": rc_a, "rc_mean": rc_mean, "gamma_c": gamma_c,
        "w_l": w_l, "w_v": w_v, "w_a": w_a,
        "delta_y_l": delta_y_l, "delta_y_v": delta_y_v, "delta_y_a": delta_y_a,
        "is_agree": is_agree, "is_partial": is_partial, "is_conflict": is_conflict,
        "pair_diff": pair_diff
    }


def make_plot(data, save_stem):
    venue = "cvpr"
    kind = "wide"
    apply_style(venue=venue, kind=kind)

    # 1 row, 2 columns, width 6.875 in, height 2.6 in
    fig, axes = new_fig(nrows=1, ncols=2, height=2.6)
    ax1, ax2 = axes[0], axes[1]
    fs = FS[kind]

    # Scaling factor for r^c (displayed as x 10^-2)
    scale = 100.0

    # -------------------------------------------------------------
    # Panel (a): Per-Modality Sample Discrepancy vs. r_m^c
    # -------------------------------------------------------------
    mods = [
        ("Language", data["delta_y_l"], data["rc_l"] * scale, "#1f77b4", "o"),
        ("Vision",   data["delta_y_v"], data["rc_v"] * scale, "#2ca02c", "s"),
        ("Audio",    data["delta_y_a"], data["rc_a"] * scale, "#ff7f0e", "^"),
    ]

    for name, delta_m, rc_m, color, marker in mods:
        ax1.scatter(
            delta_m, rc_m, s=12, alpha=0.35, color=color,
            marker=marker, edgecolors="none", zorder=2
        )
        # OLS regression trendline
        slope, intercept, r_val, p_val, _ = stats.linregress(delta_m, rc_m)
        rho, p_rho = stats.spearmanr(delta_m, rc_m)
        x_grid = np.linspace(0.0, delta_m.max(), 100)
        p_star = "*" if p_rho < 0.05 else ""
        lbl = f"{name} ($\\rho={rho:+.2f}${p_star}, $p={p_rho:.2f}$)"
        ax1.plot(x_grid, intercept + slope * x_grid, color=color, linewidth=1.2, label=lbl, zorder=3)

    ax1.set_xlabel("Unimodal Label Deviation $\\Delta y_m = |y_m - \\bar{y}_{-m}|$")
    ax1.set_ylabel("Shared Reliability ($r_m^c \\times 10^{-2}$)")
    ax1.set_title("(a) Modality Discrepancy vs. $r_m^c$ (Sample-level)", loc="left", fontweight="bold", pad=4)
    ax1.set_ylim(0.0, 5.5)
    ax1.set_xlim(-0.05, 2.05)
    ax1.yaxis.set_major_locator(MaxNLocator(5))
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax1.grid(True, zorder=0)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.4, handletextpad=0.4)

    # -------------------------------------------------------------
    # Panel (b): Sample-Level Pairwise Disagreement vs. Mean r^c
    # -------------------------------------------------------------
    diff = data["pair_diff"]
    rc_mean_scaled = data["rc_mean"] * scale

    c_agree = "#1f77b4"
    c_partial = "#7f7f7f"
    c_conflict = "#d62728"

    ax2.scatter(
        diff[data["is_agree"]], rc_mean_scaled[data["is_agree"]],
        s=14, alpha=0.50, color=c_agree, marker="o", label="Full Agreement ($N=226$)", edgecolors="none", zorder=2
    )
    ax2.scatter(
        diff[data["is_partial"]], rc_mean_scaled[data["is_partial"]],
        s=14, alpha=0.50, color=c_partial, marker="s", label="Weak / Neutral ($N=145$)", edgecolors="none", zorder=2
    )
    ax2.scatter(
        diff[data["is_conflict"]], rc_mean_scaled[data["is_conflict"]],
        s=18, alpha=0.75, color=c_conflict, marker="^", label="Polarity Conflict ($N=86$)",
        edgecolors="black", linewidths=0.3, zorder=3
    )

    # Linear trendline with 95% confidence interval
    slope, intercept, r_val, p_val, std_err = stats.linregress(diff, rc_mean_scaled)
    rho_b, p_rho_b = stats.spearmanr(diff, rc_mean_scaled)
    x_grid_b = np.linspace(0.0, diff.max(), 100)
    y_fit = intercept + slope * x_grid_b

    # 95% CI band
    n = len(diff)
    t_val = stats.t.ppf(0.975, df=n - 2)
    s_err = np.sqrt(np.sum((rc_mean_scaled - (intercept + slope * diff)) ** 2) / (n - 2))
    ci = t_val * s_err * np.sqrt(1.0 / n + (x_grid_b - np.mean(diff)) ** 2 / np.sum((diff - np.mean(diff)) ** 2))

    trend_lbl = f"Overall Trend ($\\rho={rho_b:+.2f}$, $p={p_rho_b:.2f}$)"
    ax2.plot(x_grid_b, y_fit, color="black", linestyle="--", linewidth=1.1, label=trend_lbl, zorder=4)
    ax2.fill_between(x_grid_b, y_fit - ci, y_fit + ci, color="black", alpha=0.08, zorder=1)

    ax2.set_xlabel("Pairwise Label Disagreement $\\Delta_{\\mathrm{pair}} = \\frac{1}{3}\\sum |y_i - y_j|$")
    ax2.set_ylabel("Mean Shared Reliability ($\\bar{r}^c \\times 10^{-2}$)")
    ax2.set_title("(b) Sample Disagreement vs. $\\bar{r}^c$ (Sample-level)", loc="left", fontweight="bold", pad=4)
    ax2.set_ylim(0.0, 4.0)
    ax2.set_xlim(-0.05, 1.55)
    ax2.yaxis.set_major_locator(MaxNLocator(5))
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax2.grid(True, zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.4, handletextpad=0.4)

    # Save output
    save_fig(fig, save_stem)
    plt.close(fig)


def main():
    pt_path = PROJECT_ROOT / "result" / "experiments" / "exp04" / "exp04_sims_mr0.0.pt"
    data_path = PROJECT_ROOT / "data" / "SIMS" / "unaligned_39.pkl"
    out_dir = PROJECT_ROOT / "result" / "experiments" / "exp04" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_stem = out_dir / "exp04_sims_rc_unimodal_disagreement_samplewise"

    data = load_sims_data_and_scores(pt_path, data_path)
    make_plot(data, save_stem)
    print(f"Figure saved to {save_stem}.pdf and {save_stem}.png")


if __name__ == "__main__":
    main()
