#!/usr/bin/env python3
"""
Plot 6-panel pairwise cross-modal Shared Reliability (r_m1^c vs. r_m2^c) at Clean (MR = 0.0) on CH-SIMS.

Follows CVPR style guidelines via scripts/experiments/paper_style.py:
- Two-column wide layout (width = 6.875 in).
- 2 rows x 3 columns grid.
- Times-like serif fonts, STIX math, PDF fonttype 42.
- Layout 'constrained', shared legend on top.
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


def load_data():
    data_path = PROJECT_ROOT / "data" / "SIMS" / "unaligned_39.pkl"
    with open(data_path, "rb") as f:
        raw = pickle.load(f)

    y_l = np.array(raw["test"]["regression_labels_T"])
    y_a = np.array(raw["test"]["regression_labels_A"])
    y_v = np.array(raw["test"]["regression_labels_V"])

    sign_l = np.sign(y_l)
    sign_v = np.sign(y_v)
    sign_a = np.sign(y_a)

    is_agree = (sign_l == sign_v) & (sign_v == sign_a) & (sign_l != 0)
    is_conflict = (sign_l * sign_v < 0) | (sign_l * sign_a < 0) | (sign_v * sign_a < 0)
    is_partial = (~is_agree) & (~is_conflict)

    pt_path = PROJECT_ROOT / "result" / "experiments" / "exp04" / "exp04_sims_mr0.0.pt"
    d = torch.load(pt_path, map_location="cpu")
    rd = d["raw_data"]

    rc_l = rd["r_shared_l"].numpy()
    rc_v = rd["r_shared_v"].numpy()
    rc_a = rd["r_shared_a"].numpy()

    return {
        "is_agree": is_agree, "is_partial": is_partial, "is_conflict": is_conflict,
        "rc_l": rc_l, "rc_v": rc_v, "rc_a": rc_a
    }


def make_plot(data, save_stem):
    venue = "cvpr"
    kind = "wide"
    apply_style(venue=venue, kind=kind)

    # 2 rows, 3 columns: 6 pairwise plots
    fig, axes = new_fig(nrows=2, ncols=3, height=4.3)
    fs = FS[kind]

    rc_l = data["rc_l"]
    rc_v = data["rc_v"]
    rc_a = data["rc_a"]
    is_agree = data["is_agree"]
    is_partial = data["is_partial"]
    is_conflict = data["is_conflict"]

    # 6 pairwise configurations: (panel_label, x_name, y_name, x_data, y_data, ax)
    pairs = [
        # Row 1
        ("(a) Language vs. Vision", r"Language $r_l^c$", r"Vision $r_v^c$", rc_l, rc_v, axes[0]),
        ("(b) Language vs. Audio", r"Language $r_l^c$", r"Audio $r_a^c$", rc_l, rc_a, axes[1]),
        ("(c) Vision vs. Audio", r"Vision $r_v^c$", r"Audio $r_a^c$", rc_v, rc_a, axes[2]),
        # Row 2 (transposed pairs)
        ("(d) Vision vs. Language", r"Vision $r_v^c$", r"Language $r_l^c$", rc_v, rc_l, axes[3]),
        ("(e) Audio vs. Language", r"Audio $r_a^c$", r"Language $r_l^c$", rc_a, rc_l, axes[4]),
        ("(f) Audio vs. Vision", r"Audio $r_a^c$", r"Vision $r_v^c$", rc_a, rc_v, axes[5]),
    ]

    lim_min, lim_max = 0.23, 0.54
    diag_vals = np.linspace(lim_min, lim_max, 100)

    h_agree = None
    h_partial = None
    h_conflict = None
    h_diag = None

    for title, xlabel, ylabel, x, y, ax in pairs:
        h_agree = ax.scatter(
            x[is_agree], y[is_agree],
            s=9, color="#1f77b4", marker="o", alpha=0.6, label="Full Agreement ($N=226$)", edgecolors="none"
        )
        h_partial = ax.scatter(
            x[is_partial], y[is_partial],
            s=9, color="#7f7f7f", marker="s", alpha=0.5, label="Weak / Neutral ($N=145$)", edgecolors="none"
        )
        h_conflict = ax.scatter(
            x[is_conflict], y[is_conflict],
            s=12, color="#d62728", marker="^", alpha=0.8, label="Polarity Conflict ($N=86$)",
            edgecolors="black", linewidths=0.3
        )

        h_diag, = ax.plot(
            diag_vals, diag_vals,
            color="black", linestyle="--", linewidth=1.0, label="Equilibrium ($y = x$)"
        )

        rho, p_val = stats.spearmanr(x, y)
        p_str = f"{p_val:.1e}" if p_val < 0.001 else f"{p_val:.3f}"
        annot_text = f"Spearman " + r"$\rho = " + f"{rho:+.3f}$\n" + f"($p = {p_str}$)"

        ax.text(
            0.05, 0.94, annot_text,
            transform=ax.transAxes, fontsize=fs["annot"] - 0.5,
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.9, linewidth=0.5)
        )

        ax.set_title(title, fontsize=fs["panel"] - 0.5, pad=3)
        ax.set_xlabel(xlabel, fontsize=fs["label"] - 0.5)
        ax.set_ylabel(ylabel, fontsize=fs["label"] - 0.5)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(True, linestyle=":", alpha=0.45, linewidth=0.4)

    # Place shared legend at the top
    handles = [h_agree, h_partial, h_conflict, h_diag]
    labels = ["Full Agreement ($N=226$)", "Weak / Neutral ($N=145$)", "Polarity Conflict ($N=86$)", "Equilibrium ($y = x$)"]
    shared_legend(fig, handles, labels, ncol=4)

    save_fig(fig, save_stem)
    plt.close(fig)
    print(f"[Done] Saved 6-panel pairwise modality plot to {save_stem}.pdf / .png")


def main():
    data = load_data()
    fig_dir = PROJECT_ROOT / "result" / "experiments" / "exp04" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_stem = fig_dir / "exp04_sims_pairwise_modalities_6panels_clean"
    make_plot(data, save_stem)


if __name__ == "__main__":
    main()

