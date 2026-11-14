#!/usr/bin/env python3
"""
Plot sample-level relation between Shared Reliability (r^c) and Unimodal Label Disagreement on CH-SIMS.

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

    sign_l = np.sign(y_l)
    sign_v = np.sign(y_v)
    sign_a = np.sign(y_a)

    is_agree = (sign_l == sign_v) & (sign_v == sign_a) & (sign_l != 0)
    is_conflict = (sign_l * sign_v < 0) | (sign_l * sign_a < 0) | (sign_v * sign_a < 0)
    is_partial = (~is_agree) & (~is_conflict)

    pair_diff = (np.abs(y_l - y_v) + np.abs(y_l - y_a) + np.abs(y_v - y_a)) / 3.0

    dev_l = np.abs(y_l - (y_a + y_v) / 2.0)
    dev_v = np.abs(y_v - (y_l + y_a) / 2.0)
    dev_a = np.abs(y_a - (y_l + y_v) / 2.0)

    return {
        "y_l": y_l, "y_v": y_v, "y_a": y_a, "y_m": y_m,
        "rc_l": rc_l, "rc_v": rc_v, "rc_a": rc_a, "rc_mean": rc_mean,
        "is_agree": is_agree, "is_partial": is_partial, "is_conflict": is_conflict,
        "pair_diff": pair_diff,
        "dev_l": dev_l, "dev_v": dev_v, "dev_a": dev_a
    }


def make_plot(data, save_stem):
    venue = "cvpr"
    kind = "wide"
    apply_style(venue=venue, kind=kind)

    # 1 row, 2 columns, width 6.875 in
    fig, axes = new_fig(nrows=1, ncols=2, height=2.5)
    ax1, ax2 = axes[0], axes[1]
    fs = FS[kind]

    # -------------------------------------------------------------
    # Panel (a): Sample-Level Disagreement vs. Mean r^c
    # -------------------------------------------------------------
    pair_diff = data["pair_diff"]
    rc_mean = data["rc_mean"]

    ax1.scatter(
        pair_diff[data["is_agree"]], rc_mean[data["is_agree"]],
        s=12, alpha=0.6, color="#1f77b4", marker="o", label="Full Agreement ($N=226$)", edgecolors="none"
    )
    ax1.scatter(
        pair_diff[data["is_partial"]], rc_mean[data["is_partial"]],
        s=12, alpha=0.5, color="#7f7f7f", marker="s", label="Weak / Neutral ($N=145$)", edgecolors="none"
    )
    ax1.scatter(
        pair_diff[data["is_conflict"]], rc_mean[data["is_conflict"]],
        s=15, alpha=0.8, color="#d62728", marker="^", label="Polarity Conflict ($N=86$)", edgecolors="black", linewidths=0.3
    )

    slope, intercept, r_val, p_val, _ = stats.linregress(pair_diff, rc_mean)
    rho_a, p_rho_a = stats.spearmanr(pair_diff, rc_mean)
    x_vals = np.linspace(0.0, pair_diff.max(), 100)
    ax1.plot(
        x_vals, intercept + slope * x_vals,
        color="black", linestyle="--", linewidth=1.1,
        label=f"Trend ($\\rho={rho_a:+.3f}$, $p={p_rho_a:.2f}$)"
    )

    ax1.set_xlabel(r"Sample Disagreement $\Delta_{\mathrm{pair}} = \frac{1}{3}\sum |y_i - y_j|$")
    ax1.set_ylabel(r"Mean Shared Reliability ($\bar{r}^c$)")
    ax1.set_title(r"(a) Sample Disagreement vs. $\bar{r}^c$", loc="left", fontweight="bold", pad=4)
    ax1.set_ylim(0.24, 0.54)
    ax1.yaxis.set_major_locator(MaxNLocator(5))
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax1.grid(True, zorder=0)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.2, handletextpad=0.3, fontsize=6.5)

    # -------------------------------------------------------------
    # Panel (b): Per-Modality Deviation vs. Modality-Specific r_m^c
    # -------------------------------------------------------------
    mods = [
        ("Language ($r_l^c$)", data["dev_l"], data["rc_l"], "#1f77b4", "o"),
        ("Vision ($r_v^c$)", data["dev_v"], data["rc_v"], "#2ca02c", "s"),
        ("Audio ($r_a^c$)", data["dev_a"], data["rc_a"], "#ff7f0e", "^")
    ]

    for label, dev, rc, col, mkr in mods:
        rho_m, p_m = stats.spearmanr(dev, rc)
        ax2.scatter(dev, rc, s=10, alpha=0.45, color=col, marker=mkr, edgecolors="none")
        slope_m, inc_m, _, _, _ = stats.linregress(dev, rc)
        x_m = np.linspace(0.0, dev.max(), 100)
        sig_star = r"$^*$" if p_m < 0.05 else ""
        ax2.plot(
            x_m, inc_m + slope_m * x_m, color=col, linestyle="-", linewidth=1.2,
            label=f"{label}: $\\rho={rho_m:+.3f}${sig_star}"
        )

    ax2.set_xlabel(r"Modality Label Discrepancy $\Delta y_m = |y_m - \bar{y}_{-m}|$")
    ax2.set_ylabel(r"Modality Shared Reliability ($r_m^c$)")
    ax2.set_title(r"(b) Modality Discrepancy vs. $r_m^c$", loc="left", fontweight="bold", pad=4)
    ax2.set_ylim(0.22, 0.58)
    ax2.yaxis.set_major_locator(MaxNLocator(5))
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2.grid(True, zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.2, handletextpad=0.3, fontsize=6.5)

    save_fig(fig, save_stem)
    plt.close(fig)


def main():
    pt_path = PROJECT_ROOT / "result" / "experiments" / "exp04" / "exp04_sims_mr0.0.pt"
    data_path = PROJECT_ROOT / "data" / "SIMS" / "unaligned_39.pkl"
    out_dir = PROJECT_ROOT / "result" / "experiments" / "exp04" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_stem = out_dir / "exp04_sims_sample_level_rc_disagreement"

    data = load_sims_data_and_scores(pt_path, data_path)
    make_plot(data, save_stem)
    print(f"Figure saved to {save_stem}.pdf and {save_stem}.png")


if __name__ == "__main__":
    main()
