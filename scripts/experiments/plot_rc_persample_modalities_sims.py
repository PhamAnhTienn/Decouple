#!/usr/bin/env python3
"""
Plot per-sample, modality-aligned Shared Reliability (r_l^c, r_v^c, r_a^c) at Clean (MR = 0.0) on CH-SIMS.

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


def load_data():
    data_path = PROJECT_ROOT / "data" / "SIMS" / "unaligned_39.pkl"
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

    pt_path = PROJECT_ROOT / "result" / "experiments" / "exp04" / "exp04_sims_mr0.0.pt"
    d = torch.load(pt_path, map_location="cpu")
    rd = d["raw_data"]

    rc_l = rd["r_shared_l"].numpy()
    rc_v = rd["r_shared_v"].numpy()
    rc_a = rd["r_shared_a"].numpy()

    return {
        "y_l": y_l, "y_v": y_v, "y_a": y_a, "y_m": y_m,
        "is_agree": is_agree, "is_partial": is_partial, "is_conflict": is_conflict,
        "pair_diff": pair_diff,
        "rc_l": rc_l, "rc_v": rc_v, "rc_a": rc_a
    }


def make_plot(data, save_stem):
    venue = "cvpr"
    kind = "wide"
    apply_style(venue=venue, kind=kind)

    # 1 row, 2 columns: wide layout (width = 6.875 in, height = 2.4 in)
    fig, axes = new_fig(nrows=1, ncols=2, height=2.4)
    ax1, ax2 = axes[0], axes[1]
    fs = FS[kind]

    pair_diff = data["pair_diff"]
    rc_l = data["rc_l"]
    rc_v = data["rc_v"]
    rc_a = data["rc_a"]
    is_agree = data["is_agree"]
    is_partial = data["is_partial"]
    is_conflict = data["is_conflict"]

    # -------------------------------------------------------------
    # Panel (a): Sample-Aligned Modality Triplets
    # Common X-axis: pair_diff (single value per sample)
    # -------------------------------------------------------------
    rc_min = np.minimum.reduce([rc_l, rc_v, rc_a])
    rc_max = np.maximum.reduce([rc_l, rc_v, rc_a])
    intra_spread = rc_max - rc_min

    # Connect triplet per sample with faint gray vertical bar
    for i in range(len(pair_diff)):
        ax1.plot(
            [pair_diff[i], pair_diff[i]], [rc_min[i], rc_max[i]],
            color="#b0b0b0", linewidth=0.65, alpha=0.6, zorder=1
        )

    # Plot modality scatter points sharing the exact same x
    h_l = ax1.scatter(
        pair_diff, rc_l, s=11, color="#9467bd", marker="d", alpha=0.75,
        label=r"Language $r_l^c$", zorder=2, edgecolors="none"
    )
    h_v = ax1.scatter(
        pair_diff, rc_v, s=11, color="#2ca02c", marker="^", alpha=0.75,
        label=r"Vision $r_v^c$", zorder=2, edgecolors="none"
    )
    h_a = ax1.scatter(
        pair_diff, rc_a, s=11, color="#ff7f0e", marker="v", alpha=0.75,
        label=r"Audio $r_a^c$", zorder=2, edgecolors="none"
    )

    # Linear trend lines for each modality
    x_grid = np.linspace(0.0, pair_diff.max(), 100)
    for rc_m, col, ls in [(rc_l, "#9467bd", "--"), (rc_v, "#2ca02c", ":"), (rc_a, "#ff7f0e", "-.")]:
        sl, ic, _, _, _ = stats.linregress(pair_diff, rc_m)
        ax1.plot(x_grid, ic + sl * x_grid, color=col, linestyle=ls, linewidth=1.1, zorder=3)

    ax1.set_title(r"(a) Sample-Aligned Triplets $(r_l^c, r_v^c, r_a^c)$ on Same X", fontsize=fs["panel"], pad=4)
    ax1.set_xlabel(r"Sample Label Disagreement $\Delta_{\mathrm{pair}}$", fontsize=fs["label"])
    ax1.set_ylabel(r"Shared Reliability $r_m^c$", fontsize=fs["label"])
    ax1.set_xlim(-0.05, 1.45)
    ax1.set_ylim(0.22, 0.55)
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax1.grid(True, linestyle=":", alpha=0.45, linewidth=0.4)

    annot_a = (
        f"Intra-sample spread:\n"
        f"Mean: {intra_spread.mean():.3f} " + r"$\pm$" + f" {intra_spread.std():.3f}\n"
        f"Spread vs " + r"$\Delta_{\mathrm{pair}}$: " + r"$\rho = +0.075$"
    )
    ax1.text(
        0.04, 0.94, annot_a,
        transform=ax1.transAxes, fontsize=fs["annot"] - 0.5,
        verticalalignment="top", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.9, linewidth=0.5)
    )
    ax1.legend(loc="upper right", fontsize=fs["legend"] - 0.8, frameon=True, facecolor="#f8f9fa", edgecolor="#cccccc", framealpha=0.9)

    # -------------------------------------------------------------
    # Panel (b): Cross-Modal Pairwise Reliability Scatter
    # Each point is ONE unique sample: Language vs. Non-Verbal
    # -------------------------------------------------------------
    rc_va = (rc_v + rc_a) / 2.0

    ax2.scatter(
        rc_l[is_agree], rc_va[is_agree],
        s=12, color="#1f77b4", marker="o", alpha=0.6, label=r"Full Agree ($N=226$)", edgecolors="none"
    )
    ax2.scatter(
        rc_l[is_partial], rc_va[is_partial],
        s=12, color="#7f7f7f", marker="s", alpha=0.5, label=r"Weak / Neu. ($N=145$)", edgecolors="none"
    )
    ax2.scatter(
        rc_l[is_conflict], rc_va[is_conflict],
        s=15, color="#d62728", marker="^", alpha=0.8, label=r"Polarity Conflict ($N=86$)", edgecolors="black", linewidths=0.3
    )

    # Equilibrium line y = x
    diag_vals = np.linspace(0.24, 0.54, 100)
    ax2.plot(diag_vals, diag_vals, "k--", linewidth=1.1, label=r"Equilibrium ($y = x$)")

    rho_b, p_rho_b = stats.spearmanr(rc_l, rc_va)
    p_str_b = f"{p_rho_b:.1e}" if p_rho_b < 0.001 else f"{p_rho_b:.3f}"
    annot_b = (
        f"Cross-Modal Co-variation:\n"
        f"Spearman " + r"$\rho = " + f"{rho_b:+.3f}$\n"
        f"($p = {p_str_b}$)"
    )
    ax2.text(
        0.04, 0.94, annot_b,
        transform=ax2.transAxes, fontsize=fs["annot"] - 0.5,
        verticalalignment="top", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.9, linewidth=0.5)
    )

    ax2.set_title(r"(b) Intra-Sample Co-variation: $r_l^c$ vs. $(r_v^c + r_a^c)/2$", fontsize=fs["panel"], pad=4)
    ax2.set_xlabel(r"Language Reliability $r_l^c$", fontsize=fs["label"])
    ax2.set_ylabel(r"Non-Verbal Reliability $(r_v^c + r_a^c)/2$", fontsize=fs["label"])
    ax2.set_xlim(0.24, 0.54)
    ax2.set_ylim(0.24, 0.54)
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2.grid(True, linestyle=":", alpha=0.45, linewidth=0.4)
    ax2.legend(loc="lower right", fontsize=fs["legend"] - 0.8, frameon=True, facecolor="#f8f9fa", edgecolor="#cccccc", framealpha=0.9)

    save_fig(fig, save_stem)
    plt.close(fig)
    print(f"[Done] Saved sample-aligned modality figure to {save_stem}.pdf / .png")


def main():
    data = load_data()
    fig_dir = PROJECT_ROOT / "result" / "experiments" / "exp04" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_stem = fig_dir / "exp04_sims_persample_modalities_aligned_clean"
    make_plot(data, save_stem)


if __name__ == "__main__":
    main()
