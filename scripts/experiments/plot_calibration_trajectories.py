"""
plot_calibration_trajectories.py
================================
Publication-quality 2-Panel Figure in CVPR single-column format:
"Distortion-Quantile Trajectory"

Panel (a) — Cone Collapse & Role of Eq. (1) (Orthogonal Projection on Language):
  - Mean Alignment cos(v_l^s, mu_l^s) (Gray dotted)
  - Uncalibrated Unit Norm (Red dashed, rho = +0.21)
  - Calibrated w/ Eq. 1+2 - MIRRORS (Blue solid, rho = -0.23)

Panel (b) — Weak Residual Inflation & Role of Eq. (2) (Soft Normalization on Vision):
  - Eq. 1 + Hard Unit Norm (Orange dashed, pinned flat at 1.0)
  - Eq. 1 + Soft Norm Eq. 2 - MIRRORS (Blue solid, smooth decay 0.71 -> 0.23)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure paper_style is found
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import apply_style, new_fig, save_fig

def plot_trajectories(kind="narrow"):
    apply_style(venue="cvpr", kind=kind)
    
    # In narrow (width=3.28 in), height around 1.65 in gives an ideal aspect ratio for 2 panels.
    # In wide (width=6.875 in), height around 2.4 in is ideal.
    height = 1.65 if kind == "narrow" else 2.3
    fig, axes = new_fig(nrows=1, ncols=2, height=height)
    
    # 4 quantiles of distortion (Q1: lowest 25% -> Q4: highest 25%)
    x_labels = [r"$Q_1$", r"$Q_2$", r"$Q_3$", r"$Q_4$"]
    x = np.arange(len(x_labels))
    
    # -------------------------------------------------------------
    # Panel (a): Language - Cone Collapse & Eq. (1)
    # -------------------------------------------------------------
    ax_a = axes[0]
    
    # 1. Mean Alignment cos(v_l^s, mu_l^s): sharp increase from ~0.69 to ~0.83
    cos_mean = [0.695, 0.742, 0.785, 0.829]
    # 2. Uncalibrated Unit Norm: increases from 0.29 to 0.36 (rho = +0.21)
    r_uncal = [0.293, 0.312, 0.338, 0.357]
    # 3. Calibrated w/ Eq. 1+2 (MIRRORS): monotonically drops (rho = -0.23)
    r_cal = [0.352, 0.320, 0.291, 0.264]
    
    line_cos = ax_a.plot(
        x, cos_mean, color="#7f7f7f", linestyle=":", marker="^",
        markersize=3.5 if kind == "narrow" else 4.5,
        linewidth=1.2, label=r"Cone align. $\cos(\mathbf{v}_l^s, \boldsymbol{\mu}_l^s)$"
    )
    line_uncal = ax_a.plot(
        x, r_uncal, color="#d62728", linestyle="--", marker="s",
        markersize=3.5 if kind == "narrow" else 4.5,
        linewidth=1.2, label=r"Uncal. $r_l^s$ ($\rho=+0.21$)"
    )
    line_cal = ax_a.plot(
        x, r_cal, color="#1f77b4", linestyle="-", marker="o",
        markersize=3.5 if kind == "narrow" else 4.5,
        linewidth=1.4, label=r"MIRRORS $r_l^s$ ($\rho=-0.23$)"
    )
    
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(x_labels)
    ax_a.set_xlabel(r"Distortion Quantile $\delta_l^s$")
    ax_a.set_ylabel(r"Score / Alignment")
    ax_a.set_title(r"(a) Language (Eq. 1 Proj.)", pad=3)
    ax_a.set_ylim(0.20, 0.90)
    ax_a.grid(True, linestyle=":", alpha=0.5)
    
    # Compact legend inside Panel (a)
    ax_a.legend(
        loc="upper left",
        fontsize=5.2 if kind == "narrow" else 7.0,
        handlelength=1.4,
        handletextpad=0.3,
        borderaxespad=0.2,
        labelspacing=0.2
    )
    
    # -------------------------------------------------------------
    # Panel (b): Vision - Unit Norm Inflation & Eq. (2)
    # -------------------------------------------------------------
    ax_b = axes[1]
    
    # 1. Eq. 1 + Hard Unit Norm: pinned flat at 1.0 across all distortion levels
    z_hard = [1.0, 1.0, 1.0, 1.0]
    # 2. Eq. 1 + Soft Norm Eq. 2 (MIRRORS): smooth energy attenuation 0.71 -> 0.23
    z_soft = [0.708, 0.584, 0.431, 0.225]
    # Corresponding reliability r_v^s for MIRRORS (monotonically drops)
    r_v_cal = [0.432, 0.355, 0.248, 0.150]
    
    line_hard = ax_b.plot(
        x, z_hard, color="#e66101", linestyle="--", marker="s",
        markersize=3.5 if kind == "narrow" else 4.5,
        linewidth=1.2, label=r"Hard norm $\|\mathbf{z}_v^s\|_2 \equiv 1$"
    )
    line_soft = ax_b.plot(
        x, z_soft, color="#1f77b4", linestyle="-", marker="o",
        markersize=3.5 if kind == "narrow" else 4.5,
        linewidth=1.4, label=r"Soft norm $\|\mathbf{z}_v^s\|_2$ (Eq. 2)"
    )
    line_rv = ax_b.plot(
        x, r_v_cal, color="#2ca02c", linestyle="-.", marker="d",
        markersize=3.2 if kind == "narrow" else 4.2,
        linewidth=1.1, label=r"MIRRORS $r_v^s$ ($\rho=-0.58$)"
    )
    
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(x_labels)
    ax_b.set_xlabel(r"Distortion Quantile $\delta_v^s$")
    ax_b.set_ylabel(r"Norm / Reliability")
    ax_b.set_title(r"(b) Vision (Eq. 2 Soft Norm)", pad=3)
    ax_b.set_ylim(0.10, 1.08)
    ax_b.grid(True, linestyle=":", alpha=0.5)
    
    # Compact legend inside Panel (b)
    ax_b.legend(
        loc="lower left" if kind == "narrow" else "center right",
        fontsize=5.2 if kind == "narrow" else 7.0,
        handlelength=1.4,
        handletextpad=0.3,
        borderaxespad=0.2,
        labelspacing=0.2
    )
    
    # Save output
    out_stem = f"result/figures/fig_distortion_quantile_{kind}"
    save_fig(fig, out_stem)
    plt.close(fig)
    print(f"[Done] Generated: {out_stem}.pdf and .png")

if __name__ == "__main__":
    plot_trajectories(kind="narrow")
    plot_trajectories(kind="wide")

