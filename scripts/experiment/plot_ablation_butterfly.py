"""plot_ablation_butterfly.py - butterfly chart of ablation drops (MOSI | labels | MOSEI).

Bars   : drop in F1 w.r.t. the full model (MIRROR).
Diamond: drop in ACC-7 w.r.t. the full model.
Each dataset panel has its own x-scale. Style: CVPR, single column (narrow).

Usage (from scripts/experiment/):
    python plot_ablation_butterfly.py                 # -> figures/ablation_butterfly.{pdf,png}
    python plot_ablation_butterfly.py --out figures/fig_ablation
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import FS, MaxNLocator, apply_style, save_fig, shared_legend, width_in  # noqa: E402

VENUE, KIND = "cvpr", "narrow"

# --------------------------------------------------------------------------- data
# (label, group, ACC7, F1) -- group: "comp" = critical components, "loss" = loss regularization
MOSI_FULL = dict(acc7=32.79, f1=71.55)
MOSI = [
    (r"w/o $\mathcal{L}_{\mathrm{CPA}}$", "comp", 31.16, 70.78),
    ("w/o Calibration",                   "comp", 29.72, 67.82),
    ("w/o Shared",                        "comp", 30.77, 66.93),
    ("w/o Specific",                      "comp", 30.73, 65.27),
    (r"w/o $\mathcal{L}_{\mathrm{ort}}$", "loss", 29.88, 67.90),
]

# !!! MOCK numbers -- replace with the real MOSEI table when it is ready !!!
MOSEI_IS_MOCK = True
MOSEI_FULL = dict(acc7=54.20, f1=86.10)
MOSEI = [
    (r"w/o $\mathcal{L}_{\mathrm{CPA}}$", "comp", 53.60, 85.60),
    ("w/o Calibration",                        "comp", 52.70, 84.40),
    ("w/o Shared",                        "comp", 52.30, 83.70),
    ("w/o Specific",                      "comp", 51.80, 83.00),
    (r"w/o $\mathcal{L}_{\mathrm{ort}}$", "loss", 52.90, 84.60),
]

# False: all bars share one colour (legend = dF1 bar + dAcc7 diamond only).
# True : bars coloured by ablation group (components vs. loss) with a 3-entry legend.
GROUP_COLORS = False
COLORS = dict(comp="#3B6FB6", loss="#E08A1E" if GROUP_COLORS else "#3B6FB6")
DIAMOND = "#C0143C"


# --------------------------------------------------------------------------- helpers
def deltas(rows, full):
    """-> dict label -> (group, dF1, dAcc7); drops are negative numbers."""
    return {lab: (g, round(f1 - full["f1"], 2), round(acc - full["acc7"], 2)) for lab, g, acc, f1 in rows}


def neg_fmt(v, _pos=None):
    return "0" if abs(v) < 1e-9 else f"\u2212{abs(v):g}"


def draw_panel(ax, data, order, side, title, mock=False):
    """side='left': bars grow leftwards from the label column; 'right': rightwards."""
    sign = -1 if side == "left" else 1
    y = np.arange(len(order))
    f1 = np.array([abs(data[k][1]) for k in order])
    acc = np.array([abs(data[k][2]) for k in order])
    colors = [COLORS[data[k][0]] for k in order]

    plt.rcParams["hatch.linewidth"] = 0.5
    ax.barh(y, sign * f1, height=0.62, color=colors, zorder=2,
            edgecolor="white" if mock else "none", linewidth=0.0,
            hatch="///" if mock else None)  # hatched bars flag mock data
    ax.scatter(sign * acc, y, marker="D", s=11, c=DIAMOND, edgecolors="white",
               linewidths=0.5, zorder=4)

    lim = max(f1.max(), acc.max()) * 1.45
    pad = lim * 0.08
    for yi, a, b in zip(y, f1, acc):
        x = sign * (max(a, b) + pad)
        ax.text(x, yi, f"\u2212{a:.2f}", ha="right" if side == "left" else "left",
                va="center", fontsize=FS[KIND]["annot"], zorder=5)

    ax.set_xlim((-lim, 0) if side == "left" else (0, lim))
    ax.set_ylim(len(order) - 0.5, -0.5)  # first row on top
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MaxNLocator(3, prune=None))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(neg_fmt))
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "left" if side == "left" else "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, pad=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/ablation_butterfly")
    args = ap.parse_args()

    apply_style(VENUE, KIND)
    d_mosi, d_mosei = deltas(MOSI, MOSI_FULL), deltas(MOSEI, MOSEI_FULL)
    order = sorted(d_mosi, key=lambda k: d_mosi[k][1])  # largest F1 drop (most negative) on top

    width = width_in(VENUE, KIND)
    fig = plt.figure(figsize=(width, 2.05), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.0, hspace=0.0)
    ax_l, ax_c, ax_r = fig.subplots(1, 3, gridspec_kw=dict(width_ratios=[1.0, 0.82, 1.0]))

    draw_panel(ax_l, d_mosi, order, "left", "MOSI")
    draw_panel(ax_r, d_mosei, order, "right", "MOSEI (mock)" if MOSEI_IS_MOCK else "MOSEI",
               mock=MOSEI_IS_MOCK)

    # centre label column
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(len(order) - 0.5, -0.5)
    ax_c.axis("off")
    for i, k in enumerate(order):
        ax_c.text(0.5, i, k, ha="center", va="center", fontsize=FS[KIND]["tick"] + 0.5)

    if GROUP_COLORS:
        handles = [Patch(facecolor=COLORS["comp"], label="Components"),
                   Patch(facecolor=COLORS["loss"], label="Loss reg.")]
    else:
        handles = [Patch(facecolor=COLORS["comp"], label=r"$\Delta$F1")]
    handles += [
        Line2D([], [], marker="D", ls="", markersize=3.6, markerfacecolor=DIAMOND,
               markeredgecolor="white", markeredgewidth=0.5, label=r"$\Delta$Acc$_7$"),
    ]
    shared_legend(fig, handles, [h.get_label() for h in handles], ncol=len(handles))
    # fig.supxlabel("Drop w.r.t. full model (points)", fontsize=FS[KIND]["label"])

    save_fig(fig, args.out)
    if MOSEI_IS_MOCK:
        print("WARNING: MOSEI panel uses MOCK numbers.")


if __name__ == "__main__":
    main()