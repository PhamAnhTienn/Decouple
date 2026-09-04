#!/usr/bin/env python3
"""
Reconstruction Benefit vs. Conflict Score — Post-hoc Plotting Script.

This script loads saved .pt score files from different DMD training variants
(AllRec, NoRecon, NoRec-L, NoRec-V, NoRec-A) and generates publication-quality
figures showing the correlation between conflict scores and reconstruction benefit.

Definitions:
  Overall benefit:
    B_i^model = ℓ(y_i, ŷ_i^NoRecon) − ℓ(y_i, ŷ_i^AllRec)

  Per-modality benefit:
    B_{i,m}^branch = ℓ(y_i, ŷ_i^NoRec-m) − ℓ(y_i, ŷ_i^AllRec)

  Where ℓ = L1 loss (MAE), per sample.

Usage:
  python scripts/plot_reconstruction_benefit.py --dataset mosi --seed 1111

  # Custom paths
  python scripts/plot_reconstruction_benefit.py \\
      --score-dir result/dmd_scores \\
      --output-dir result/dmd_recon_analysis \\
      --dataset mosi --seed 1111 --n-bins 10
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

# ────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────
MODALITY_KEYS = ("l", "v", "a")
MODALITY_NAMES = {"l": "Text", "v": "Video", "a": "Audio"}
MODALITY_TAGS = {"l": "NoRecL", "v": "NoRecV", "a": "NoRecA"}


# ────────────────────────────────────────────────────────────────────
# Score Loading
# ────────────────────────────────────────────────────────────────────

def load_scores(score_dir, dataset, run_tag, seed):
    """Load a saved .pt score file."""
    path = Path(score_dir) / f"{dataset}_dmd_scores_{run_tag}_seed{seed}.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Score file not found: {path}\n"
            f"  → Did you train DMD with --run_tag {run_tag} --seeds {seed}?"
        )
    data = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  [✓] Loaded {path.name}  ({data['predictions'].shape[0]} samples)")
    return data


def compute_per_sample_loss(predictions, labels):
    """Per-sample L1 loss: ℓ(y_i, ŷ_i) = |y_i − ŷ_i|."""
    return (predictions.view(-1) - labels.view(-1)).abs()


# ────────────────────────────────────────────────────────────────────
# Plotting Utilities
# ────────────────────────────────────────────────────────────────────

def _setup_matplotlib():
    """Configure matplotlib for publication-quality output."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


# Color palette — accessible + modern
COLORS = {
    "overall": "#7C3AED",  # purple
    "l": "#2563EB",        # vivid blue
    "v": "#16A34A",        # vivid green
    "a": "#DC2626",        # vivid red
}
FILL_COLORS = {
    "overall": "#C4B5FD",
    "l": "#93C5FD",
    "v": "#86EFAC",
    "a": "#FCA5A5",
}


def _bin_and_average(C, B, n_bins=10):
    """
    Compute quantile-based bins of C and return bin centers,
    E[B|bin], SE[B|bin], and sample counts.
    """
    bin_edges = np.quantile(C, np.linspace(0, 1, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    actual_bins = len(bin_edges) - 1

    centers, means, ses, counts = [], [], [], []
    for i in range(actual_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < actual_bins - 1:
            mask = (C >= lo) & (C < hi)
        else:
            mask = (C >= lo) & (C <= hi)
        n = mask.sum()
        if n == 0:
            continue
        centers.append((lo + hi) / 2)
        means.append(B[mask].mean())
        ses.append(B[mask].std() / np.sqrt(n))
        counts.append(n)

    return (np.array(centers), np.array(means),
            np.array(ses), np.array(counts))


def _draw_panel(ax, C, B, color, fill_color, label, n_bins, show_ylabel=True):
    """Draw a single conflict-vs-benefit panel."""
    from matplotlib.ticker import MaxNLocator

    centers, means, ses, counts = _bin_and_average(C, B, n_bins)
    if len(centers) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#9CA3AF")
        return

    # ±SE shading
    ax.fill_between(centers, means - ses, means + ses,
                    alpha=0.25, color=fill_color, zorder=1)

    # Main line
    ax.plot(centers, means, color=color, linewidth=2.2,
            marker="o", markersize=6, markeredgecolor="white",
            markeredgewidth=1.2, zorder=3, label=label)

    # Zero reference
    ax.axhline(0, color="#6B7280", linewidth=0.8, linestyle="--", alpha=0.7, zorder=0)

    # Background shading: green for B>0 (recon helps), red for B<0 (recon hurts)
    y_margin = max(abs(means.min()), abs(means.max())) * 0.25
    y_margin = max(y_margin, 0.005)  # minimum margin
    ylims = [min(means.min() - y_margin, -0.005), max(means.max() + y_margin, 0.005)]
    ax.axhspan(0, ylims[1] * 1.5, color="#DCFCE7", alpha=0.22, zorder=0)
    ax.axhspan(ylims[0] * 1.5, 0, color="#FEE2E2", alpha=0.22, zorder=0)

    ax.set_xlabel("Conflict Score $C$")
    if show_ylabel:
        ax.set_ylabel(r"Reconstruction Benefit $\mathbb{E}[B]$")
    ax.legend(loc="best", framealpha=0.85, edgecolor="#D1D5DB")

    # Annotation: count per bin
    for cx, cnt in zip(centers, counts):
        ax.annotate(f"n={int(cnt)}", (cx, ylims[0] * 0.9),
                    fontsize=7, ha="center", color="#9CA3AF", style="italic")

    ax.set_ylim(ylims)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    # Pearson & Spearman correlation (on raw data)
    from scipy.stats import spearmanr
    r_pearson = np.corrcoef(C, B)[0, 1]
    rho, _ = spearmanr(C, B)
    ax.annotate(
        f"$r = {r_pearson:+.3f}$\n$\\rho = {rho:+.3f}$",
        xy=(0.97, 0.97), xycoords="axes fraction",
        fontsize=9, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D1D5DB", alpha=0.9)
    )


# ────────────────────────────────────────────────────────────────────
# Figure 1: Overall B^model vs Overall C
# ────────────────────────────────────────────────────────────────────

def plot_overall_benefit(allrec, norecon, output_path, n_bins=10, dataset="MOSI"):
    """
    Single-panel: E[B_i^model | C_i ∈ bin] vs C_i
    B_i^model = ℓ(y, ŷ^NoRecon) − ℓ(y, ŷ^AllRec)
    """
    plt = _setup_matplotlib()

    loss_allrec = compute_per_sample_loss(allrec['predictions'], allrec['labels'])
    loss_norecon = compute_per_sample_loss(norecon['predictions'], norecon['labels'])
    B = (loss_norecon - loss_allrec).numpy()
    C = allrec['conflict_overall'].numpy()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    _draw_panel(ax, C, B, COLORS["overall"], FILL_COLORS["overall"],
                r"$B^{model}$", n_bins, show_ylabel=True)
    ax.set_xlabel("Overall Conflict Score $C_i$")
    ax.set_ylabel(r"Overall Recon. Benefit $\mathbb{E}[B_i^{model}]$")
    ax.set_title(f"Overall Reconstruction Benefit — {dataset.upper()}",
                 fontweight="bold", pad=10)

    # Interpretation text
    fig.text(0.5, -0.04,
             "$B > 0$: reconstruction training improves prediction  ·  "
             "$B < 0$: reconstruction training hurts prediction",
             ha="center", fontsize=9, color="#4B5563", style="italic")

    _save_and_close(fig, output_path)

    # Print summary
    pos = (B > 0).sum()
    neg = (B < 0).sum()
    print(f"    Overall B: mean={B.mean():.4f}, "
          f"positive={pos}/{len(B)} ({100*pos/len(B):.1f}%), "
          f"negative={neg}/{len(B)} ({100*neg/len(B):.1f}%)")


# ────────────────────────────────────────────────────────────────────
# Figure 2: Per-modality B^branch vs C_m  (3 panels)
# ────────────────────────────────────────────────────────────────────

def plot_modality_benefit(allrec, norec_per_mod, output_path,
                          n_bins=10, dataset="MOSI"):
    """
    3-panel figure:
      Panel m: E[B_{i,m}^branch | C_{i,m} ∈ bin] vs C_{i,m}
      B_{i,m}^branch = ℓ(y, ŷ^NoRec-m) − ℓ(y, ŷ^AllRec)
    """
    plt = _setup_matplotlib()

    loss_allrec = compute_per_sample_loss(allrec['predictions'], allrec['labels'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for idx, m in enumerate(MODALITY_KEYS):
        ax = axes[idx]
        tag = MODALITY_TAGS[m]
        if tag not in norec_per_mod:
            ax.text(0.5, 0.5, f"No {tag} data\n(train with --run_tag {tag})",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="#9CA3AF")
            ax.set_title(f"{MODALITY_NAMES[m]}", fontweight="bold", pad=8)
            continue

        norec_data = norec_per_mod[tag]
        loss_norec = compute_per_sample_loss(norec_data['predictions'],
                                             norec_data['labels'])
        B = (loss_norec - loss_allrec).numpy()
        C = allrec[f'conflict_{m}'].numpy()

        _draw_panel(ax, C, B, COLORS[m], FILL_COLORS[m],
                    MODALITY_NAMES[m], n_bins, show_ylabel=(idx == 0))
        ax.set_xlabel(f"Conflict Score $C_{{i,{m}}}$")
        if idx == 0:
            ax.set_ylabel(r"Recon. Benefit $\mathbb{E}[B_{i,m}^{branch}]$")
        ax.set_title(f"{MODALITY_NAMES[m]}", fontweight="bold", pad=8)

        # Summary
        pos = (B > 0).sum()
        neg = (B < 0).sum()
        print(f"    B_{m} ({MODALITY_NAMES[m]}): mean={B.mean():.4f}, "
              f"positive={pos}/{len(B)} ({100*pos/len(B):.1f}%), "
              f"negative={neg}/{len(B)} ({100*neg/len(B):.1f}%)")

    fig.suptitle(
        f"Per-Modality Conflict Score vs. Reconstruction Benefit — "
        f"DMD on {dataset.upper()}",
        fontsize=15, fontweight="bold", y=1.03
    )

    fig.text(0.5, -0.04,
             "$B_{i,m}^{branch} > 0$: modality $m$ reconstruction helps  ·  "
             "$B_{i,m}^{branch} < 0$: modality $m$ reconstruction hurts",
             ha="center", fontsize=9, color="#4B5563", style="italic")

    _save_and_close(fig, output_path)


# ────────────────────────────────────────────────────────────────────
# Figure 3: Per-modality Scatter + Trend
# ────────────────────────────────────────────────────────────────────

def plot_modality_scatter(allrec, norec_per_mod, output_path,
                          n_bins=10, dataset="MOSI"):
    """
    Supplementary scatter plot with per-sample (C_{i,m}, B_{i,m}^branch)
    and polynomial trend line.
    """
    plt = _setup_matplotlib()

    loss_allrec = compute_per_sample_loss(allrec['predictions'], allrec['labels'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for idx, m in enumerate(MODALITY_KEYS):
        ax = axes[idx]
        tag = MODALITY_TAGS[m]
        if tag not in norec_per_mod:
            ax.text(0.5, 0.5, f"No {tag} data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="#9CA3AF")
            ax.set_title(f"{MODALITY_NAMES[m]}", fontweight="bold", pad=8)
            continue

        norec_data = norec_per_mod[tag]
        loss_norec = compute_per_sample_loss(norec_data['predictions'],
                                             norec_data['labels'])
        B = (loss_norec - loss_allrec).numpy()
        C = allrec[f'conflict_{m}'].numpy()

        # Scatter
        ax.scatter(C, B, s=14, alpha=0.35, color=COLORS[m],
                   edgecolors="none", zorder=2)
        ax.axhline(0, color="#6B7280", linewidth=0.8, linestyle="--",
                   alpha=0.7, zorder=0)

        # Polynomial trend (degree 3)
        if len(C) > 10:
            order = np.argsort(C)
            C_s, B_s = C[order], B[order]
            try:
                coeffs = np.polyfit(C_s, B_s, 3)
                poly = np.poly1d(coeffs)
                xs = np.linspace(C_s.min(), C_s.max(), 200)
                ax.plot(xs, poly(xs), color="black", linewidth=2,
                        linestyle="-", zorder=4, label="Poly-3 trend")
            except np.linalg.LinAlgError:
                pass

        ax.set_xlabel(f"Conflict Score $C_{{i,{m}}}$")
        if idx == 0:
            ax.set_ylabel(r"Recon. Benefit $B_{i,m}^{branch}$")
        ax.set_title(f"{MODALITY_NAMES[m]}", fontweight="bold", pad=8)

        # Pearson r annotation
        from scipy.stats import spearmanr
        r = np.corrcoef(C, B)[0, 1]
        rho, p_val = spearmanr(C, B)
        ax.annotate(
            f"$r = {r:+.3f}$\n$\\rho = {rho:+.3f}$\n$p = {p_val:.1e}$",
            xy=(0.97, 0.97), xycoords="axes fraction",
            fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D1D5DB", alpha=0.9)
        )
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        f"Per-Sample Conflict vs. Reconstruction Benefit — "
        f"DMD on {dataset.upper()}",
        fontsize=15, fontweight="bold", y=1.03
    )
    _save_and_close(fig, output_path)


# ────────────────────────────────────────────────────────────────────
# CSV Export
# ────────────────────────────────────────────────────────────────────

def export_csv(allrec, norecon, norec_per_mod, output_path):
    """Export all per-sample scores to a single CSV for downstream analysis."""
    loss_allrec = compute_per_sample_loss(allrec['predictions'], allrec['labels'])

    n = allrec['labels'].shape[0]
    rows = []

    # Overall benefit
    if norecon is not None:
        loss_norecon = compute_per_sample_loss(norecon['predictions'], norecon['labels'])
        B_overall = (loss_norecon - loss_allrec).numpy()
    else:
        B_overall = np.full(n, np.nan)

    # Per-modality benefit
    B_mod = {}
    for m in MODALITY_KEYS:
        tag = MODALITY_TAGS[m]
        if tag in norec_per_mod:
            loss_norec = compute_per_sample_loss(
                norec_per_mod[tag]['predictions'], norec_per_mod[tag]['labels']
            )
            B_mod[m] = (loss_norec - loss_allrec).numpy()
        else:
            B_mod[m] = np.full(n, np.nan)

    for i in range(n):
        rows.append({
            "sample_idx": i,
            "label": float(allrec['labels'][i]),
            "pred_AllRec": float(allrec['predictions'][i]),
            "loss_AllRec": float(loss_allrec[i]),
            "C_overall": float(allrec['conflict_overall'][i]),
            "C_l": float(allrec['conflict_l'][i]),
            "C_v": float(allrec['conflict_v'][i]),
            "C_a": float(allrec['conflict_a'][i]),
            "B_overall": float(B_overall[i]),
            "B_l": float(B_mod['l'][i]),
            "B_v": float(B_mod['v'][i]),
            "B_a": float(B_mod['a'][i]),
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [✓] CSV saved to {output_path}  ({n} samples)")


# ────────────────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────────────────

def _save_and_close(fig, path):
    """Save figure as both the requested format and always as PNG."""
    import matplotlib.pyplot as plt
    path = Path(path)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    print(f"  [✓] Saved {path}")
    # Also save PNG if the primary format is PDF
    if path.suffix.lower() == ".pdf":
        png_path = path.with_suffix(".png")
        fig.savefig(png_path, bbox_inches="tight", pad_inches=0.25)
        print(f"  [✓] Saved {png_path}")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--score-dir", type=Path, default=Path("result/dmd_scores"),
                   help="Directory containing saved .pt score files")
    p.add_argument("--output-dir", type=Path, default=Path("result/dmd_recon_analysis"),
                   help="Output directory for figures and CSV")
    p.add_argument("--dataset", default="mosi", choices=("mosi", "mosei", "sims"))
    p.add_argument("--seed", type=int, default=1111)
    p.add_argument("--n-bins", type=int, default=10,
                   help="Number of quantile bins for the binned plots")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Reconstruction Benefit Analysis — {args.dataset.upper()}, seed={args.seed}")
    print(f"{'='*65}")

    # ── Load AllRec (required) ──
    print("\n[1] Loading score files...")
    allrec = load_scores(args.score_dir, args.dataset, "AllRec", args.seed)

    # ── Load NoRecon (optional, for overall B^model) ──
    norecon = None
    try:
        norecon = load_scores(args.score_dir, args.dataset, "NoRecon", args.seed)
    except FileNotFoundError as e:
        print(f"  [!] {e}")
        print("      → Skipping overall benefit plot.")

    # ── Load NoRec-L, NoRec-V, NoRec-A (optional) ──
    norec_per_mod = {}
    for m in MODALITY_KEYS:
        tag = MODALITY_TAGS[m]
        try:
            norec_per_mod[tag] = load_scores(args.score_dir, args.dataset, tag, args.seed)
        except FileNotFoundError:
            print(f"  [!] {tag} not found — skipping modality {MODALITY_NAMES[m]}")

    # ── Verify sample counts match ──
    n_allrec = allrec['predictions'].shape[0]
    if norecon is not None:
        assert norecon['predictions'].shape[0] == n_allrec, \
            f"Sample count mismatch: AllRec={n_allrec} vs NoRecon={norecon['predictions'].shape[0]}"
    for tag, data in norec_per_mod.items():
        assert data['predictions'].shape[0] == n_allrec, \
            f"Sample count mismatch: AllRec={n_allrec} vs {tag}={data['predictions'].shape[0]}"

    # ── Figure 1: Overall B^model vs C ──
    if norecon is not None:
        print("\n[2] Plotting overall reconstruction benefit...")
        plot_overall_benefit(
            allrec, norecon,
            args.output_dir / f"{args.dataset}_overall_benefit.pdf",
            n_bins=args.n_bins, dataset=args.dataset
        )

    # ── Figure 2: Per-modality B^branch vs C_m ──
    if norec_per_mod:
        print("\n[3] Plotting per-modality reconstruction benefit...")
        plot_modality_benefit(
            allrec, norec_per_mod,
            args.output_dir / f"{args.dataset}_modality_benefit.pdf",
            n_bins=args.n_bins, dataset=args.dataset
        )

    # ── Figure 3: Per-modality scatter ──
    if norec_per_mod:
        print("\n[4] Plotting per-modality scatter...")
        plot_modality_scatter(
            allrec, norec_per_mod,
            args.output_dir / f"{args.dataset}_modality_scatter.pdf",
            n_bins=args.n_bins, dataset=args.dataset
        )

    # ── CSV export ──
    print("\n[5] Exporting CSV...")
    export_csv(
        allrec, norecon, norec_per_mod,
        args.output_dir / f"{args.dataset}_reconstruction_benefit_seed{args.seed}.csv"
    )

    # ── Summary ──
    print(f"\n{'='*65}")
    print(f"  [✓] All done! Output: {args.output_dir}/")
    print(f"{'='*65}")
    print()

    # Print available vs missing files
    print("  Score files used:")
    print(f"    AllRec  : ✓")
    print(f"    NoRecon : {'✓' if norecon else '✗ (train with --run_tag NoRecon)'}")
    for m in MODALITY_KEYS:
        tag = MODALITY_TAGS[m]
        status = "✓" if tag in norec_per_mod else f"✗ (train with --run_tag {tag})"
        print(f"    {tag:8s}: {status}")
    print()


if __name__ == "__main__":
    main()
