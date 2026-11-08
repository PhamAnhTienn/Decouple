"""
Experiment 12c – Sample-level disagreement between shared and specific reliability
==================================================================================
Do r_shared and r_specific, within the SAME modality, give genuinely different
per-sample quality assessments?

Key design choices vs exp12b:
  (1) Works on RAW reliability values (Δᵢ = r_specificᵢ − r_sharedᵢ), not ranks.
      Ranks inflate tiny differences when one score is nearly constant.
  (2) When per-sample corruption is available in the .pt file, scatter points are
      colored by true corruption. When corruption is absent (standard exp12 checkpoints),
      points are colored by the signed reliability gap Δ = r_specific − r_shared
      (diverging colormap), making the disagreement visually salient rather than
      collapsing to monochrome gray.

Output (2 conference-ready figures following paper_style.py):
  Figure 1 – Scatter: per-sample (r_shared, r_specific) colored by corruption or Δ,
             with diagonal y = x + agreement band |Δ| < ε.
  Figure 2 – Disagreement rate curve: P(|Δ| > ε) vs ε for each modality
             at selected missing rates, with a shuffled null baseline.

Usage:
  Real data:    python exp12c_raw_disagreement.py --dataset mosi
  Synthetic:    python exp12c_raw_disagreement.py --demo
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy import stats

# Import paper_style
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import FS, VENUES, apply_style, new_fig, save_fig, shared_legend, width_in

MODALITIES = ["l", "v", "a"]
MOD_NAMES = {"l": "Language", "v": "Vision", "a": "Audio"}
EPS_GRID = np.linspace(0.0, 0.5, 51)


# =====================================================================
# Loading helpers
# =====================================================================

def _flat(a):
    """Flatten multi-dim per-sample arrays to 1-D by averaging extra dims."""
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(a.shape[0], -1).mean(axis=1) if a.ndim > 1 else a


def load_raw(save_dir, dataset, mr):
    """Load a single exp12_<dataset>_mr<mr>.pt file.
    Returns dict {modality: {rc, rs, corr}} or None if missing."""
    path = Path(save_dir) / f"exp12_{dataset}_mr{mr}.pt"
    if not path.exists():
        return None
    import torch
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    out = {}
    for m in MODALITIES:
        corr = _flat(blob["corruption"][m]) if "corruption" in blob else None
        # u = 1 - r  →  r = 1 - u
        out[m] = dict(
            rc=1.0 - _flat(blob["u_shared"][m]),
            rs=1.0 - _flat(blob["u_specific"][m]),
            corr=corr,
        )
    return out


def make_demo(rates, n=1500, seed=0):
    """Synthetic data: corruption degrades r_specific more than r_shared."""
    rng = np.random.default_rng(seed)
    out = {}
    for mr in rates:
        out[mr] = {}
        for _ in MODALITIES:
            c = np.clip(rng.beta(2, 2, n) * 2 * mr, 0, 1) if mr > 0 else np.zeros(n)
            common = rng.normal(size=n)
            rs = np.clip(0.90 - 0.60 * c + 0.04 * common + 0.07 * rng.normal(size=n), 0, 1)
            rc = np.clip(0.85 - 0.20 * c + 0.04 * common + 0.10 * rng.normal(size=n), 0, 1)
            out[mr][_] = dict(rc=rc, rs=rs, corr=c)
    return out


# =====================================================================
# Analysis
# =====================================================================

def eps_curve(d):
    """Compute P(|Δ| > ε) for each ε in EPS_GRID."""
    a = np.abs(d)
    return np.array([(a > e).mean() for e in EPS_GRID])


def analyse(raw, n_null=20, seed=0):
    """Return processed data dict and summary dict."""
    rng = np.random.default_rng(seed)
    data, summary = {}, {}
    for mr, per in raw.items():
        data[mr], summary[mr] = {}, {}
        for m, d in per.items():
            rc, rs, corr = d["rc"], d["rs"], d["corr"]
            delta = rs - rc
            # Null: shuffle r_specific across samples
            null = np.mean(
                [eps_curve(rs[rng.permutation(len(rs))] - rc) for _ in range(n_null)],
                axis=0,
            )
            s = dict(
                n=int(len(rc)),
                mean_r_shared=float(rc.mean()),
                mean_r_specific=float(rs.mean()),
                std_r_shared=float(rc.std()),
                std_r_specific=float(rs.std()),
                mean_delta=float(delta.mean()),
                mean_abs_delta=float(np.abs(delta).mean()),
                spearman_rho=float(stats.spearmanr(rc, rs)[0]),
            )
            if corr is not None and np.ptp(corr) > 0:
                s["spearman_r_shared_vs_corr"] = float(stats.spearmanr(rc, corr)[0])
                s["spearman_r_specific_vs_corr"] = float(stats.spearmanr(rs, corr)[0])
            data[mr][m] = dict(rc=rc, rs=rs, corr=corr, delta=delta,
                               curve=eps_curve(delta), null_curve=null)
            summary[mr][m] = s
    return data, summary


# =====================================================================
# Figure 1 – Scatter (r_shared vs r_specific) with paper_style
# =====================================================================

def fig_scatter(data, show_rates, eps, out_stem, venue="acl", kind="wide"):
    """
    Grid: rows = modalities (L, V, A), cols = selected missing rates.
    Points colored by corruption (if available) or by signed reliability gap Δ = rs - rc.
    """
    apply_style(venue, kind)
    nr = len(show_rates)
    fs = FS[kind]

    # Create figure via paper_style
    fig, axes_flat = new_fig(nrows=3, ncols=nr, height=4.8, sharex=True, sharey=True)
    axes = axes_flat.reshape(3, nr)

    # Global axis limits
    allv = np.concatenate(
        [np.concatenate([data[r][m]["rc"], data[r][m]["rs"]])
         for r in show_rates for m in MODALITIES]
    )
    lo, hi = max(0.0, float(allv.min()) - 0.03), min(1.0, float(allv.max()) + 0.03)

    # Determine coloring mode
    has_corr = any(
        data[r][m]["corr"] is not None and np.ptp(data[r][m]["corr"]) > 0
        for r in show_rates for m in MODALITIES
    )

    if has_corr:
        cmap = "magma_r"
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
        cbar_label = "True corruption $c$"
    else:
        # Diverging colormap centered at 0 for Δ = rs - rc
        all_deltas = np.concatenate(
            [data[r][m]["delta"] for r in show_rates for m in MODALITIES]
        )
        vlim = float(max(0.35, min(0.6, np.percentile(np.abs(all_deltas), 98))))
        cmap = "RdBu_r"
        norm = matplotlib.colors.Normalize(vmin=-vlim, vmax=vlim)
        cbar_label = r"Reliability gap $\Delta = r_{\mathrm{specific}} - r_{\mathrm{shared}}$"

    sc = None
    for j, mr in enumerate(show_rates):
        for i, m in enumerate(MODALITIES):
            d, ax = data[mr][m], axes[i, j]
            color_vals = d["corr"] if has_corr else d["delta"]

            sc = ax.scatter(
                d["rc"], d["rs"],
                c=color_vals,
                cmap=cmap,
                norm=norm,
                s=3.0,
                alpha=0.55,
                rasterized=True,
                zorder=3,
            )

            # Diagonal y = x and agreement band |Δ| <= ε
            x_band = np.array([lo, hi])
            ax.plot(x_band, x_band, "k--", lw=0.75, zorder=2)
            ax.fill_between(
                x_band, x_band - eps, x_band + eps,
                color="gray", alpha=0.15, zorder=1,
            )

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")

            # Column title (top row only)
            if i == 0:
                ax.set_title(f"Missing rate = {mr:.0%}", fontsize=fs["panel"], pad=3)

            # Row label (leftmost column only)
            if j == 0:
                ax.set_ylabel(
                    f"{MOD_NAMES[m]}\n$r_{{\\mathrm{{specific}}}}$",
                    fontsize=fs["label"],
                )

            # Column label (bottom row only)
            if i == 2:
                ax.set_xlabel("$r_{\\mathrm{shared}}$", fontsize=fs["label"])

    # Shared colorbar
    cbar = fig.colorbar(sc, ax=axes, shrink=0.65, aspect=24, pad=0.02)
    cbar.ax.tick_params(labelsize=fs["tick"])
    cbar.set_label(cbar_label, fontsize=fs["label"])

    # Shared legend above subplots
    handles = [
        Line2D([], [], color="k", linestyle="--", lw=0.75),
        Patch(facecolor="gray", alpha=0.25, edgecolor="none"),
    ]
    labels = ["Agreement ($y = x$)", rf"Agreement band ($|\Delta| \leq {eps}$)"]
    shared_legend(fig, handles, labels, ncol=2)

    save_fig(fig, out_stem)


# =====================================================================
# Figure 2 – Disagreement rate curve P(|Δ| > ε) with paper_style
# =====================================================================

def fig_eps_curve(data, show_rates, eps, out_stem, venue="acl", kind="wide"):
    """
    One panel per modality. Solid = model curve, dotted = shuffled null.
    Vertical gray dashed line at chosen tolerance ε.
    """
    apply_style(venue, kind)
    fs = FS[kind]
    cmap = plt.get_cmap("viridis")

    fig, axes = new_fig(nrows=1, ncols=3, height=2.35, sharey=True)

    legend_handles = []
    legend_labels = []

    for i, m in enumerate(MODALITIES):
        ax = axes[i]
        panel_letter = chr(ord("a") + i)
        ax.set_title(f"({panel_letter}) {MOD_NAMES[m]}", loc="left", fontsize=fs["panel"], pad=3)

        for k, mr in enumerate(show_rates):
            col = cmap(k / max(1, len(show_rates) - 1))
            (line_model,) = ax.plot(
                EPS_GRID, data[mr][m]["curve"],
                color=col, lw=1.2,
                label=f"MR = {mr:.0%}",
            )
            ax.plot(
                EPS_GRID, data[mr][m]["null_curve"],
                color=col, ls=":", lw=1.1, alpha=0.75,
            )
            if i == 0:
                legend_handles.append(line_model)
                legend_labels.append(f"MR = {mr:.0%}")

        # Reference tolerance line
        ax.axvline(eps, color="gray", lw=0.8, ls="--")

        ax.set_xlabel(r"Tolerance $\varepsilon$", fontsize=fs["label"])
        ax.set_xlim(0.0, 0.5)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle=":", alpha=0.45, linewidth=0.4)

        if i == 0:
            ax.set_ylabel(
                r"$P(|r_{\mathrm{specific}} - r_{\mathrm{shared}}| > \varepsilon)$",
                fontsize=fs["label"],
            )

    # Proxy handles for dotted null and vertical threshold
    null_proxy = Line2D([], [], color="gray", linestyle=":", lw=1.1)
    eps_proxy = Line2D([], [], color="gray", linestyle="--", lw=0.8)
    legend_handles.extend([null_proxy, eps_proxy])
    legend_labels.extend(["Shuffled null (dotted)", rf"Tolerance $\varepsilon = {eps}$"])

    shared_legend(fig, legend_handles, legend_labels, ncol=len(legend_handles))

    save_fig(fig, out_stem)


# =====================================================================
# Console table
# =====================================================================

def print_table(summary, rates):
    print("\n" + "=" * 100)
    print("EXPERIMENT 12c – Sample-level raw-scale disagreement")
    print("=" * 100)
    print(f"{'MR':<6}{'Mod':<10}{'σ(r_c)':>8}{'σ(r_s)':>8}{'E|Δ|':>8}"
          f"{'ρ(c,s)':>8}{'ρ(rc,cor)':>10}{'ρ(rs,cor)':>10}")
    print("-" * 100)
    for r in rates:
        for m in MODALITIES:
            s = summary[r][m]
            ext = ""
            if "spearman_r_shared_vs_corr" in s:
                ext = f"{s['spearman_r_shared_vs_corr']:>10.3f}{s['spearman_r_specific_vs_corr']:>10.3f}"
            print(f"{r:<6.1f}{MOD_NAMES[m]:<10}{s['std_r_shared']:>8.4f}{s['std_r_specific']:>8.4f}"
                  f"{s['mean_abs_delta']:>8.4f}{s['spearman_rho']:>8.3f}{ext}")
    print("=" * 100)


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Exp12c: raw-scale sample-level disagreement (2 figures following paper_style.py)"
    )
    ap.add_argument("--dataset", default="mosi", choices=["mosi", "mosei", "sims"])
    ap.add_argument("--save_dir", default="./result/experiments/exp12")
    ap.add_argument("--missing_rates", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--show_rates", type=float, nargs="+",
                    default=[0.0, 0.3, 0.6, 0.9])
    ap.add_argument("--eps", type=float, default=0.1,
                    help="Threshold |Δ| to consider disagreement")
    ap.add_argument("--venue", choices=sorted(VENUES), default="acl",
                    help="Paper template: acl, cvpr, iccv, iclr, neurips, or icml")
    ap.add_argument("--kind", choices=("narrow", "wide"), default="wide",
                    help="Figure width layout (default: wide)")
    ap.add_argument("--demo", action="store_true",
                    help="Run with synthetic data (no .pt files needed)")
    args = ap.parse_args()

    out = os.path.join(args.save_dir, "sample_level_raw")
    os.makedirs(out, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if args.demo:
        raw = make_demo(args.missing_rates)
    else:
        raw = {}
        for mr in args.missing_rates:
            r = load_raw(args.save_dir, args.dataset, mr)
            if r is None:
                print(f"[WARN] Missing exp12_{args.dataset}_mr{mr}.pt – skipped")
            else:
                raw[mr] = r
    if not raw:
        raise SystemExit("No data available. Run exp12 first or use --demo.")

    rates = sorted(raw)
    show = [r for r in args.show_rates if r in raw] or rates[::max(1, len(rates) // 4)]

    # ------------------------------------------------------------------
    # Analyse
    # ------------------------------------------------------------------
    data, summary = analyse(raw)
    print_table(summary, rates)

    # ------------------------------------------------------------------
    # Two publication-ready figures (saved as both .pdf and .png)
    # ------------------------------------------------------------------
    stem1 = os.path.join(out, f"exp12c_scatter_{args.dataset}")
    stem2 = os.path.join(out, f"exp12c_eps_curve_{args.dataset}")

    fig_scatter(data, show, args.eps, stem1, venue=args.venue, kind=args.kind)
    fig_eps_curve(data, show, args.eps, stem2, venue=args.venue, kind=args.kind)

    # Save JSON summary
    json_path = os.path.join(out, f"exp12c_{args.dataset}_summary.json")
    with open(json_path, "w") as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=2)

    print(f"\nSummary JSON: {json_path}")


if __name__ == "__main__":
    main()
