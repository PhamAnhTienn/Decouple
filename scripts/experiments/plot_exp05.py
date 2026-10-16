#!/usr/bin/env python3
"""Plot Exp05 (Selective / Asymmetric Corruption) from exp05_*_summary.json.

FIGURE RULES — follow FIGURE_GUIDE.md:
1. Create figures at the selected venue's final printed width (no later shrinking).
2. Use Times-like serif fonts, STIX math, and embedded TrueType fonts (PDF fonttype 42).
3. Use venue-specific font sizes; never go below 6 pt or above 9 pt.
4. Use layout='constrained'; do not use tight_layout(), subplots_adjust(), or bbox_inches='tight'.
5. Use one shared legend above the line-plot panels; keep axes/metrics comparable across panels.
6. Keep dimensions/style constants in paper_style.py; call apply_style(venue, kind).
7. Save vector PDF for the paper and PNG only as a preview; do not alter measured data.
8. Check final dimensions, embedded fonts, clipping, labels, and the paper's actual template.

The script only visualizes values saved by the experiment. It does not synthesize results.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from paper_style import FS, VENUES, apply_style, new_fig, save_fig, shared_legend

HERE = Path(__file__).resolve().parent
MODALITIES = ("l", "a", "v")  # Consistent L, A, V order across labels and heatmaps.
MODALITY_NAMES = {"l": "Language", "a": "Audio", "v": "Vision"}


REPO_ROOT = HERE.parents[1]
DEFAULT_SUMMARY = REPO_ROOT / "result" / "experiments" / "exp05" / "exp05_mosi_summary.json"
DEFAULT_OUT_DIR = REPO_ROOT / "result" / "experiments" / "exp05" / "figures"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create paper-ready Exp05 line plots and heatmaps from the experiment summary JSON."
    )
    parser.add_argument(
        "--summary", type=Path, default=DEFAULT_SUMMARY if DEFAULT_SUMMARY.exists() else HERE / "exp05_mosi_summary.json",
        help="Path to exp05_<dataset>_summary.json (default: result/experiments/exp05/exp05_mosi_summary.json).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="Output directory for PDF and PNG figures (default: result/experiments/exp05/figures).",
    )
    parser.add_argument(
        "--venue", choices=sorted(VENUES), default="acl",
        help="Paper template: acl, cvpr, iccv, iclr, neurips, or icml (default: acl).",
    )
    parser.add_argument(
        "--kind", choices=("narrow", "wide"), default="wide",
        help="Figure width. Use wide for two panels (default: wide).",
    )
    parser.add_argument(
        "--no-heatmap", action="store_true",
        help="Only create the line plot; by default both line plot and heatmap are saved.",
    )
    return parser.parse_args()


def load_summary(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Summary JSON not found: {path}\n"
            "Run the Exp05 experiment first, or pass --summary /path/to/exp05_<dataset>_summary.json."
        )
    with path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)

    entries = summary.get("mean_reliability")
    if not isinstance(entries, list) or not entries:
        raise ValueError("The summary JSON must contain a non-empty 'mean_reliability' list.")

    rate_rows, shared_rows, specific_rows, labels = [], [], [], []
    for i, entry in enumerate(entries):
        setting = entry.get("setting", {})
        shared = entry.get("r_shared", {})
        specific = entry.get("r_specific", {})
        if any(m not in setting for m in MODALITIES):
            raise ValueError(f"Setting {i} must contain missing rates for l, a and v.")
        if any(m not in shared or m not in specific for m in MODALITIES):
            raise ValueError(f"Entry {i} must contain r_shared and r_specific for l, a and v.")
        rates = [float(setting[m]) for m in MODALITIES]
        r_c = [float(shared[m]) for m in MODALITIES]
        r_s = [float(specific[m]) for m in MODALITIES]
        if not all(math.isfinite(x) for x in rates + r_c + r_s):
            raise ValueError(f"Non-finite value found in summary entry {i}.")
        rate_rows.append(rates)
        shared_rows.append(r_c)
        specific_rows.append(r_s)

        max_rate = max(rates)
        winners = [j for j, rate in enumerate(rates) if math.isclose(rate, max_rate, abs_tol=1e-8)]
        if len(winners) == 1:
            high_name = MODALITY_NAMES[MODALITIES[winners[0]]]
            labels.append(f"{high_name}\n({rates[0]:.1f}, {rates[1]:.1f}, {rates[2]:.1f})")
        else:
            labels.append(f"Setting {i + 1}\n({rates[0]:.1f}, {rates[1]:.1f}, {rates[2]:.1f})")

    return {
        "rates": np.asarray(rate_rows, dtype=float),
        "shared": np.asarray(shared_rows, dtype=float),
        "specific": np.asarray(specific_rows, dtype=float),
        "labels": labels,
    }


def get_common_ylim(arrays):
    """Use a data-adaptive y-range with margin so selective drops are visually salient."""
    values = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    low, high = float(np.min(values)), float(np.max(values))
    span = high - low
    if span <= 1e-12:
        span = max(abs(high), 1.0) * 0.1
    pad = span * 0.15
    y_min = max(0.0, low - pad)
    y_max = min(1.0, high + pad)
    return y_min, y_max


def common_heat_scale(arrays):
    """Return a tight color scale fitting the actual data range."""
    values = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    low, high = float(np.min(values)), float(np.max(values))
    if math.isclose(low, high, abs_tol=1e-12):
        pad = max(abs(low), 1.0) * 0.05
        return max(0.0, low - pad), min(1.0, high + pad)
    return low, high


def plot_line(data, out_dir, venue, kind):
    apply_style(venue, kind)
    fig, axes = new_fig(1, 2, height=2.75, sharey=True)
    fs = FS[kind]
    series = ((data["shared"], "Mean shared reliability"),
              (data["specific"], "Mean specific reliability"))
    markers = ("o", "s", "^")
    all_handles = []

    for panel_idx, (ax, (matrix, ylabel)) in enumerate(zip(axes, series)):
        for j, modality in enumerate(MODALITIES):
            (line,) = ax.plot(
                np.arange(len(data["labels"])), matrix[:, j],
                marker=markers[j], label=MODALITY_NAMES[modality],
                markeredgecolor="auto", zorder=3,
            )
            if panel_idx == 0:
                all_handles.append(line)
        ax.set_xticks(np.arange(len(data["labels"])))
        ax.set_xticklabels(data["labels"])
        ax.set_ylabel(ylabel)
        ax.set_xlabel("High-missing modality (rates ordered L, A, V)")
        title_text = "(a) Shared Reliability ($r^c$)" if panel_idx == 0 else "(b) Specific Reliability ($r^s$)"
        ax.set_title(title_text, loc="left", pad=3, fontweight="bold")
        ax.set_xlim(-0.18, len(data["labels"]) - 0.82 if len(data["labels"]) > 1 else 0.18)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylim(*get_common_ylim([data["shared"], data["specific"]]))
    shared_legend(fig, all_handles, [h.get_label() for h in all_handles], ncol=3)
    save_fig(fig, out_dir / "exp05_mean_reliability_line")


def plot_heatmap(data, out_dir, venue, kind):
    apply_style(venue, kind)
    fig, axes = new_fig(1, 2, height=2.75, sharey=True)
    fs = FS[kind]
    vmin, vmax = common_heat_scale([data["shared"], data["specific"]])
    matrices = ((data["shared"], "Mean shared reliability"),
                (data["specific"], "Mean specific reliability"))
    images = []

    for panel_idx, (ax, (matrix, ylabel)) in enumerate(zip(axes, matrices)):
        # Rows are reliability modalities (L, A, V); columns are asymmetric settings.
        image = ax.imshow(matrix.T, aspect="auto", interpolation="nearest", cmap="viridis",
                          vmin=vmin, vmax=vmax)
        images.append(image)
        ax.set_xticks(np.arange(len(data["labels"])))
        short_labels = []
        for idx, label in enumerate(data["labels"]):
            first = label.split("\n", maxsplit=1)[0]
            short_labels.append(first)
        ax.set_xticklabels(short_labels)
        ax.set_yticks(np.arange(len(MODALITIES)))
        ax.set_yticklabels([MODALITY_NAMES[m] for m in MODALITIES])
        ax.set_xlabel("Modality with highest missing rate")
        ax.set_ylabel("Reliability modality")
        title_text = "(a) Shared Reliability ($r^c$)" if panel_idx == 0 else "(b) Specific Reliability ($r^s$)"
        ax.set_title(title_text, loc="left", pad=3, fontweight="bold")
        ax.set_xticks(np.arange(-0.5, len(data["labels"]), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(MODALITIES), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)
        threshold = (vmin + vmax) / 2.0
        for row in range(matrix.shape[1]):
            for col in range(matrix.shape[0]):
                value = matrix[col, row]
                ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                        color="black" if value >= threshold else "white", fontsize=fs["annot"])

    # One shared colorbar across both panels, with the same scale for fair comparison.
    colorbar = fig.colorbar(
        images[0], ax=axes.tolist(), orientation="horizontal", location="top",
        shrink=0.72, pad=0.12, aspect=35,
    )
    colorbar.set_label("Mean reliability")
    colorbar.ax.xaxis.set_major_locator(MaxNLocator(4))
    colorbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    save_fig(fig, out_dir / "exp05_mean_reliability_heatmap")


def main():
    args = parse_args()
    data = load_summary(args.summary.expanduser().resolve())
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_line(data, out_dir, args.venue, args.kind)
    if not args.no_heatmap:
        plot_heatmap(data, out_dir, args.venue, args.kind)

    print(f"Input summary: {args.summary.expanduser().resolve()}")
    print(f"Output directory: {out_dir}")
    print("Note: if the summary contains only the three default settings, figures compare those settings; they do not imply a multi-rate sweep.")


if __name__ == "__main__":
    main()

# python scripts/experiments/plot_exp05.py   
#     --summary /home/tien/NCKH/DecAlign/result/experiments/exp05/exp05_mosi_summary.json  
#     --out-dir /home/tien/NCKH/DecAlign/result/experiments/exp05/figures   
#     --venue acl --kind wide