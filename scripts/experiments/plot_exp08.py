#!/usr/bin/env python3
"""
plot_exp08.py
=============
Publication-ready heatmaps for Experiment 8 (Sender-Receiver beta behaviour)
following paper_style.py guidelines (CVPR / ACL / NeurIPS / ICML format).

Features:
- Preserves full 3-decimal precision (e.g., 0.443, 0.375, 0.131)
- Supports eye-friendly, publication-grade colormaps (Blues, YlGnBu, cividis, PuBu, GnBu)
- Automatically balances cell text contrast based on perceived luminance
- Generates 6 individual figures, 1 Main 2-panel figure (Audio->Vision and Vision->Audio side-by-side),
  and 1 Appendix 6-panel figure WITH cell numbers on all 6 heatmaps
- Vector PDF (embedded TrueType, fonttype 42) + 600 DPI preview PNG
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from paper_style import FS, VENUES, apply_style, new_fig, save_fig

MOD_NAMES = {"l": "Language", "v": "Vision", "a": "Audio"}
MOD_SYMS = {"l": r"\eta_l", "v": r"\eta_v", "a": r"\eta_a"}
MOD_UPPER = {"l": "L", "v": "V", "a": "A"}

ALL_PAIRS = [
    ("a_to_v", "a", "v", r"(a) Audio $\rightarrow$ Vision ($\beta_{V \leftarrow A}$)"),
    ("v_to_a", "v", "a", r"(b) Vision $\rightarrow$ Audio ($\beta_{A \leftarrow V}$)"),
    ("a_to_l", "a", "l", r"(c) Audio $\rightarrow$ Language ($\beta_{L \leftarrow A}$)"),
    ("v_to_l", "v", "l", r"(d) Vision $\rightarrow$ Language ($\beta_{L \leftarrow V}$)"),
    ("l_to_v", "l", "v", r"(e) Language $\rightarrow$ Vision ($\beta_{V \leftarrow L}$)"),
    ("l_to_a", "l", "a", r"(f) Language $\rightarrow$ Audio ($\beta_{A \leftarrow L}$)"),
]

SUPPORTED_CMAPS = [
    "Blues", "YlGnBu", "PuBu", "cividis", "GnBu", "viridis", "YlOrRd"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready CVPR heatmaps for Exp 08 from summary JSON."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT / "result" / "experiments" / "exp08" / "exp08_mosi_summary.json",
        help="Path to exp08_<dataset>_summary.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "result" / "experiments" / "exp08" / "figures",
        help="Output directory for generated figures",
    )
    parser.add_argument(
        "--venue",
        choices=sorted(VENUES),
        default="cvpr",
        help="Target conference template (default: cvpr)",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="Blues",
        choices=SUPPORTED_CMAPS,
        help="Colormap: Blues (gentle eye-friendly blue), YlGnBu (teal-blue), cividis (ergonomic), PuBu, etc.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=3,
        help="Number of decimal places to display in cells (default: 3)",
    )
    parser.add_argument(
        "--no-annot",
        action="store_true",
        help="Disable numerical cell text annotations in heatmaps",
    )
    return parser.parse_args()


def load_data(summary_path):
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary JSON not found: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def annotate_cells(ax, mat, im, fontsize=4.2, digits=3):
    """Add high-contrast numerical annotations inside each cell without clipping."""
    norm = im.norm
    cmap = im.get_cmap()
    nrows, ncols = mat.shape
    fmt = f"{{:.{digits}f}}"
    for i in range(nrows):
        for j in range(ncols):
            val = mat[i, j]
            r, g, b, _ = cmap(norm(val))
            # Perceived relative luminance
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            color = "black" if lum > 0.52 else "white"
            ax.text(
                j, i, fmt.format(val),
                ha="center", va="center",
                color=color, fontsize=fontsize,
                fontweight="normal",
            )

def add_cbar(fig, ax, im, digits=3):
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.9)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{digits}f"))
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    cbar.ax.tick_params(labelsize=5, length=2, pad=1)
    return cbar

def plot_single_pair(pair_key, send_mod, recv_mod, mat, rates, out_dir, venue="cvpr", cmap="Blues", digits=3, show_annot=True):
    """Plot an individual pair in CVPR narrow (single-column, 3.28 in) format."""
    apply_style(venue=venue, kind="narrow")
    fig, axes = new_fig(1, 1, height=2.95)
    ax = axes[0]

    im = ax.imshow(
        mat, cmap=cmap, origin="lower", aspect="auto",
        vmin=mat.min(), vmax=mat.max()
    )

    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels([f"{r:.1f}" for r in rates])
    ax.set_yticks(range(len(rates)))
    ax.set_yticklabels([f"{r:.1f}" for r in rates])

    s_name = MOD_NAMES[send_mod]
    r_name = MOD_NAMES[recv_mod]
    s_sym = MOD_SYMS[send_mod]
    r_sym = MOD_SYMS[recv_mod]
    s_up = MOD_UPPER[send_mod]
    r_up = MOD_UPPER[recv_mod]

    ax.set_xlabel(f"Sender ({s_name}) Missing Rate ${s_sym}$")
    ax.set_ylabel(f"Receiver ({r_name}) Missing Rate ${r_sym}$")
    ax.set_title(rf"$\beta_{{{r_up} \leftarrow {s_up}}}$", pad=3, fontweight="bold")

    if show_annot:
        annotate_cells(ax, mat, im, fontsize=4.2, digits=digits)

    cbar = add_cbar(fig, ax, im, digits)

    out_stem = out_dir / f"exp08_heatmap_{pair_key}_{venue}_{cmap.lower()}"
    save_fig(fig, out_stem)


def plot_a_to_v_and_v_to_a(data, out_dir, venue="cvpr", cmap="Blues", digits=3, show_annot=True):
    """
    Plot 2 figures: Audio -> Vision and Vision -> Audio standing side-by-side
    in CVPR double-column wide format (width = 6.875 in).
    """
    apply_style(venue=venue, kind="wide")
    fig, axes = new_fig(1, 2, height=3.10)

    rates = data["a_to_v"]["rates"]
    mat_a2v = np.array(data["a_to_v"]["beta_matrix"])
    mat_v2a = np.array(data["v_to_a"]["beta_matrix"])

    panels = [
        (axes[0], mat_a2v, r"(a) Audio $\rightarrow$ Vision ($\beta_{V \leftarrow A}$)",
         r"Sender (Audio) Missing Rate $\eta_a$", r"Receiver (Vision) Missing Rate $\eta_v$"),
        (axes[1], mat_v2a, r"(b) Vision $\rightarrow$ Audio ($\beta_{A \leftarrow V}$)",
         r"Sender (Vision) Missing Rate $\eta_v$", r"Receiver (Audio) Missing Rate $\eta_a$"),
    ]

    for ax, mat, title, xlabel, ylabel in panels:
        im = ax.imshow(
            mat, cmap=cmap, origin="lower", aspect="auto",
            vmin=mat.min(), vmax=mat.max()
        )
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([f"{r:.1f}" for r in rates])
        ax.set_yticks(range(len(rates)))
        ax.set_yticklabels([f"{r:.1f}" for r in rates])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="center", pad=3, fontweight="bold")

        if show_annot:
            annotate_cells(ax, mat, im, fontsize=5.0, digits=digits)

        cbar = add_cbar(fig, ax, im, digits)

    # Save as both exp08_a_to_v_and_v_to_a_... and exp08_main_2panel_...
    stem1 = out_dir / f"exp08_a_to_v_and_v_to_a_{venue}_{cmap.lower()}"
    save_fig(fig, stem1)

    stem2 = out_dir / f"exp08_main_2panel_{venue}_{cmap.lower()}"
    shutil.copyfile(stem1.with_suffix(".pdf"), stem2.with_suffix(".pdf"))
    shutil.copyfile(stem1.with_suffix(".png"), stem2.with_suffix(".png"))
    print(f"Also mirrored to: {stem2}.pdf and {stem2}.png")


def plot_appendix_all6(data, out_dir, venue="cvpr", cmap="Blues", digits=3, show_annot=True):
    """
    Plot all 6 pairs in a unified 2x3 grid for the Appendix (wide, 6.875 in).
    Displays numerical values with 3 decimal places on ALL 6 heatmaps.
    """
    apply_style(venue=venue, kind="wide")
    fig, axes = new_fig(2, 3, height=4.3)      # 5.60 -> 4.3

    fig.subplots_adjust(
        left=0.065, right=0.985,               # lề trái/phải nhỏ lại
        top=0.94,   bottom=0.085,              # lề trên/dưới nhỏ lại
        hspace=0.30,                           # 0.38 -> 0.30 (khoảng cách giữa 2 hàng)
        wspace=0.34,                           # 0.45 -> 0.34 (khoảng cách giữa các cột)
    )

    rates = data["a_to_v"]["rates"]
    tick_indices = list(range(0, len(rates), 2))
    tick_labels = [f"{rates[i]:.1f}" for i in tick_indices]

    for idx, (pkey, s_mod, r_mod, title) in enumerate(ALL_PAIRS):
        ax = axes[idx]
        mat = np.array(data[pkey]["beta_matrix"])
        im = ax.imshow(
            mat, cmap=cmap, origin="lower", aspect="auto",
            vmin=mat.min(), vmax=mat.max()
        )
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_indices)
        ax.set_yticklabels(tick_labels)

        s_name = MOD_NAMES[s_mod]
        r_name = MOD_NAMES[r_mod]
        s_sym = MOD_SYMS[s_mod]
        r_sym = MOD_SYMS[r_mod]

        ax.set_xlabel(f"Sender ({s_name})")
        ax.set_ylabel(f"Receiver ({r_name})")
        ax.set_title(title, loc="center", pad=2, fontweight="bold", fontsize=7.2)
        ax.tick_params(axis="both", length=2, pad=1)

        if show_annot:
            # Font size 3.7 pt fits 3 decimals (e.g. 0.443) cleanly in each of the 6 panels
            annotate_cells(ax, mat, im, fontsize=3.7, digits=digits)

        cbar = add_cbar(fig, ax, im, digits)

    save_fig(fig, out_dir / f"exp08_appendix_all6_{venue}_{cmap.lower()}")


def main():
    args = parse_args()
    summary_path = args.summary.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading summary from: {summary_path}")
    print(f"Colormap: {args.cmap} | Precision: {args.digits} decimals | Venue: {args.venue.upper()}")
    data = load_data(summary_path)

    rates = data["a_to_v"]["rates"]
    show_annot = not args.no_annot

    # 1. Plot all 6 individual heatmaps
    print(f"\n[1/3] Generating 6 individual {args.venue.upper()} narrow heatmaps ({args.cmap}, {args.digits} decimals)...")
    for pkey, s_mod, r_mod, _ in ALL_PAIRS:
        if pkey in data:
            mat = np.array(data[pkey]["beta_matrix"])
            plot_single_pair(
                pkey, s_mod, r_mod, mat, rates, out_dir,
                venue=args.venue, cmap=args.cmap, digits=args.digits, show_annot=show_annot
            )

    # 2. Plot 2 figures: Audio -> Vision and Vision -> Audio side by side
    print(f"\n[2/3] Generating 2 figures side-by-side: a_to_v and v_to_a ({args.venue.upper()} wide, {args.cmap})...")
    plot_a_to_v_and_v_to_a(
        data, out_dir, venue=args.venue, cmap=args.cmap, digits=args.digits, show_annot=show_annot
    )

    # 3. Plot Appendix 6-panel grid WITH numbers on all 6 heatmaps
    print(f"\n[3/3] Generating Appendix 6-panel {args.venue.upper()} wide figure WITH numbers on all 6 panels...")
    plot_appendix_all6(
        data, out_dir, venue=args.venue, cmap=args.cmap, digits=args.digits, show_annot=show_annot
    )

    print(f"\nAll publication figures successfully saved to: {out_dir}")


if __name__ == "__main__":
    main()

# python scripts/experiments/plot_exp08.py --cmap Blues --digits 3