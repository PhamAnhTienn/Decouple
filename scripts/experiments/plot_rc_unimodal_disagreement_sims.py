#!/usr/bin/env python3
"""
Plot relation between Shared Reliability (r^c) and Unimodal Label Disagreement on CH-SIMS.

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
    # 1. Load ground truth unimodal labels
    with open(data_path, "rb") as f:
        raw = pickle.load(f)

    y_l = np.array(raw["test"]["regression_labels_T"])
    y_a = np.array(raw["test"]["regression_labels_A"])
    y_v = np.array(raw["test"]["regression_labels_V"])
    y_m = np.array(raw["test"]["regression_labels"])

    # 2. Load exp04 representations & reliability
    d = torch.load(pt_path, map_location="cpu")
    rc_l = d["raw_data"]["r_shared_l"].numpy()
    rc_v = d["raw_data"]["r_shared_v"].numpy()
    rc_a = d["raw_data"]["r_shared_a"].numpy()
    rc_mean = (rc_l + rc_v + rc_a) / 3.0

    rs_l = d["raw_data"]["r_specific_l"].numpy()
    rs_v = d["raw_data"]["r_specific_v"].numpy()
    rs_a = d["raw_data"]["r_specific_a"].numpy()

    # 3. Disagreement metrics
    sign_l = np.sign(y_l)
    sign_v = np.sign(y_v)
    sign_a = np.sign(y_a)

    is_agree = (sign_l == sign_v) & (sign_v == sign_a) & (sign_l != 0)
    is_conflict = (sign_l * sign_v < 0) | (sign_l * sign_a < 0) | (sign_v * sign_a < 0)
    is_partial = (~is_agree) & (~is_conflict)

    pair_diff = (np.abs(y_l - y_v) + np.abs(y_l - y_a) + np.abs(y_v - y_a)) / 3.0

    return {
        "y_l": y_l, "y_v": y_v, "y_a": y_a, "y_m": y_m,
        "rc_l": rc_l, "rc_v": rc_v, "rc_a": rc_a, "rc_mean": rc_mean,
        "rs_l": rs_l, "rs_v": rs_v, "rs_a": rs_a,
        "is_agree": is_agree, "is_partial": is_partial, "is_conflict": is_conflict,
        "pair_diff": pair_diff
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
    # Panel (a): Distribution of r^c across Agreement Tiers
    # -------------------------------------------------------------
    tiers = [
        ("Full Agreement\n($N=226$)", data["is_agree"]),
        ("Weak Disagreement\n($N=145$)", data["is_partial"]),
        ("Polarity Conflict\n($N=86$)", data["is_conflict"])
    ]

    tier_labels = [t[0] for t in tiers]
    x_pos = np.arange(len(tiers))
    width = 0.20

    mod_keys = [("rc_l", "Language ($r_l^c$)", "#1f77b4"),
                ("rc_v", "Vision ($r_v^c$)", "#2ca02c"),
                ("rc_a", "Audio ($r_a^c$)", "#ff7f0e")]

    for i, (key, label, color) in enumerate(mod_keys):
        offset = (i - 1) * width
        means = [np.mean(data[key][mask]) for _, mask in tiers]
        stds = [np.std(data[key][mask]) for _, mask in tiers]
        
        ax1.bar(
            x_pos + offset, means, yerr=stds, width=width,
            label=label, color=color, alpha=0.85, edgecolor="black",
            linewidth=0.6, capsize=2.5, error_kw={"elinewidth": 0.8, "capthick": 0.8}
        )

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(tier_labels)
    ax1.set_ylabel("Shared Reliability ($r_m^c$)")
    ax1.set_title("(a) $r^c$ across Unimodal Agreement Tiers", loc="left", fontweight="bold", pad=4)
    ax1.set_ylim(0.20, 0.48)
    ax1.yaxis.set_major_locator(MaxNLocator(5))
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax1.grid(axis="y", zorder=0)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.2, handletextpad=0.4)

    # -------------------------------------------------------------
    # Panel (b): Sample-Level Disagreement vs. Mean r^c
    # -------------------------------------------------------------
    diff = data["pair_diff"]
    rc_mean = data["rc_mean"]

    # Colors for points
    c_agree = "#1f77b4"
    c_partial = "#7f7f7f"
    c_conflict = "#d62728"

    ax2.scatter(
        diff[data["is_agree"]], rc_mean[data["is_agree"]],
        s=12, alpha=0.55, color=c_agree, label="Full Agreement", edgecolors="none"
    )
    ax2.scatter(
        diff[data["is_partial"]], rc_mean[data["is_partial"]],
        s=12, alpha=0.55, color=c_partial, label="Weak / Neutral", edgecolors="none"
    )
    ax2.scatter(
        diff[data["is_conflict"]], rc_mean[data["is_conflict"]],
        s=16, alpha=0.75, color=c_conflict, marker="^", label="Polarity Conflict", edgecolors="black", linewidths=0.3
    )

    # Linear regression trendline
    slope, intercept, r_val, p_val, _ = stats.linregress(diff, rc_mean)
    rho, p_rho = stats.spearmanr(diff, rc_mean)
    x_vals = np.linspace(0.0, diff.max(), 100)
    trend_label = f"Trend (rho = {rho:+.3f}, p = {p_rho:.2f})"
    ax2.plot(
        x_vals, intercept + slope * x_vals,
        color="black", linestyle="--", linewidth=1.1,
        label=trend_label
    )

    ax2.set_xlabel("Unimodal Label Disagreement $\\Delta_{\\mathrm{pair}} = \\frac{1}{3}\\sum |y_i - y_j|$")
    ax2.set_ylabel("Mean Shared Reliability ($\\bar{r}^c$)")
    ax2.set_title("(b) Unimodal Disagreement vs. $\\bar{r}^c$", loc="left", fontweight="bold", pad=4)
    ax2.set_ylim(0.24, 0.54)
    ax2.yaxis.set_major_locator(MaxNLocator(5))
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2.grid(True, zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.2, handletextpad=0.4)

    # Save
    save_fig(fig, save_stem)
    plt.close(fig)


def main():
    pt_path = PROJECT_ROOT / "result" / "experiments" / "exp04" / "exp04_sims_mr0.0.pt"
    data_path = PROJECT_ROOT / "data" / "SIMS" / "unaligned_39.pkl"
    out_dir = PROJECT_ROOT / "result" / "experiments" / "exp04" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_stem = out_dir / "exp04_sims_rc_unimodal_disagreement"

    data = load_sims_data_and_scores(pt_path, data_path)
    make_plot(data, save_stem)
    print(f"Figure saved to {save_stem}.pdf and {save_stem}.png")


if __name__ == "__main__":
    main()
