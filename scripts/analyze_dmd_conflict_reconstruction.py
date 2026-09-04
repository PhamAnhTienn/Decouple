#!/usr/bin/env python3
"""
Conflict Score vs. Reconstruction Benefit Analysis for DMD.

This script implements the following experimental pipeline:
1. Load a pre-trained DMD checkpoint (frozen backbone).
2. Extract original projected features (proj_x_m) and reconstructed features (recon_m)
   as well as shared features (c_m) for each modality m ∈ {l, v, a}.
3. Compute per-sample conflict score:
       h_m^c = Norm(MeanPool_t(c_m))
       C_m   = 1 − (1/2) Σ_{n≠m} cos(h_m^c, h_n^c)
4. Train frozen linear probes g_m on the *train* split using mean-pooled
   original projected features → labels.
5. On the *test* split, compute reconstruction benefit:
       B_m = ℓ(y, ŷ_m^orig) − ℓ(y, ŷ_m^rec)
   where ℓ is L1 loss (MAE).
6. Plot E[B_m | C_m ∈ bin] vs conflict bin for each modality.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Allow running from repository root: `python scripts/analyze_dmd_...`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_config_regression
from data_loader import MMDataLoader
from models import build_model
from utils.functions import setup_seed


MODALITY_NAMES = {"l": "Text", "v": "Video", "a": "Audio"}
MODALITY_KEYS = ("l", "v", "a")


# ────────────────────────────────────────────────────────────────────
# Stage 1: Feature Extraction
# ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, dataloader, device):
    """Run the frozen DMD model on an entire dataloader and collect features."""
    model.eval()

    all_orig = {m: [] for m in MODALITY_KEYS}       # proj_x_m  (B, d, T)
    all_recon = {m: [] for m in MODALITY_KEYS}       # recon_m   (B, d, T)
    all_shared = {m: [] for m in MODALITY_KEYS}      # c_m after encoder_c  (T, B, d)
    all_labels = []

    for batch in dataloader:
        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)
        labels = batch["labels"]["M"].to(device).view(-1)

        out = model(text, audio, vision, is_distill=True)

        # Original projected features: shape (B, d, T)
        for m in MODALITY_KEYS:
            all_orig[m].append(out[f"origin_{m}"].cpu())
            all_recon[m].append(out[f"recon_{m}"].cpu())
            # c_m in output is (T, B, d) after permute in forward
            all_shared[m].append(out[f"c_{m}"].permute(1, 2, 0).cpu())  # → (B, d, T)

        all_labels.append(labels.cpu())

    # Concatenate
    features = {}
    for m in MODALITY_KEYS:
        features[f"orig_{m}"] = torch.cat(all_orig[m], dim=0)      # (N, d, T)
        features[f"recon_{m}"] = torch.cat(all_recon[m], dim=0)     # (N, d, T)
        features[f"shared_{m}"] = torch.cat(all_shared[m], dim=0)   # (N, d, T)
    features["labels"] = torch.cat(all_labels, dim=0)               # (N,)
    return features


# ────────────────────────────────────────────────────────────────────
# Stage 2: Conflict Score
# ────────────────────────────────────────────────────────────────────

def compute_conflict_scores(features):
    """
    C_m = 1 − (1/2) Σ_{n≠m} cos(h_m^c, h_n^c)
    where h_m^c = Norm(MeanPool_t(c_m)).
    """
    pooled = {}
    for m in MODALITY_KEYS:
        # shared features: (N, d, T) → mean over T → (N, d)
        h = features[f"shared_{m}"].mean(dim=2)
        pooled[m] = F.normalize(h, dim=1, eps=1e-8)

    conflict = {}
    for m in MODALITY_KEYS:
        others = [n for n in MODALITY_KEYS if n != m]
        cos_sum = sum(
            F.cosine_similarity(pooled[m], pooled[n], dim=1)
            for n in others
        )
        conflict[m] = 1.0 - cos_sum / 2.0  # (N,)
    return conflict


# ────────────────────────────────────────────────────────────────────
# Stage 3: Frozen Linear Probes
# ────────────────────────────────────────────────────────────────────

def mean_pool_features(feat_3d):
    """(N, d, T) → (N, d) via mean over temporal dim."""
    return feat_3d.mean(dim=2)


def train_probe(X_train, y_train, hidden_dim=128, epochs=200, lr=1e-3, weight_decay=1e-4):
    """Train a small MLP probe: Linear → ReLU → Linear → scalar."""
    in_dim = X_train.shape[1]
    probe = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = TensorDataset(X_train, y_train.unsqueeze(1))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    probe.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            pred = probe(xb)
            loss = F.l1_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        if (epoch + 1) % 50 == 0:
            avg = total_loss / len(dataset)
            print(f"    Probe epoch {epoch+1}/{epochs}  MAE = {avg:.4f}")
    probe.eval()
    return probe


# ────────────────────────────────────────────────────────────────────
# Stage 4: Reconstruction Benefit
# ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_reconstruction_benefit(probe, X_orig, X_recon, labels):
    """
    B_m = ℓ(y, ŷ_orig) − ℓ(y, ŷ_rec)      (per sample)
    Positive B_m → reconstruction helps; Negative → harmful.
    """
    y = labels.unsqueeze(1)
    pred_orig = probe(X_orig)
    pred_recon = probe(X_recon)
    loss_orig = (pred_orig - y).abs().squeeze(1)     # per-sample L1
    loss_recon = (pred_recon - y).abs().squeeze(1)
    benefit = loss_orig - loss_recon                  # (N,)
    return benefit


# ────────────────────────────────────────────────────────────────────
# Stage 5: Plotting
# ────────────────────────────────────────────────────────────────────

def plot_conflict_vs_benefit(conflict_scores, benefits, output_path, n_bins=10, dataset_name="MOSI"):
    """
    Publication-quality figure: E[B_m | C_m ∈ bin] vs conflict bin.
    Three modalities as sub-panels in one row.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    # ── Style ──
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

    # Color palette (accessible + modern)
    COLORS = {
        "l": "#2563EB",   # vivid blue
        "v": "#16A34A",   # vivid green
        "a": "#DC2626",   # vivid red
    }
    FILL_COLORS = {
        "l": "#93C5FD",
        "v": "#86EFAC",
        "a": "#FCA5A5",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    all_benefits_flat = np.concatenate([benefits[m].numpy() for m in MODALITY_KEYS])
    y_margin = max(abs(all_benefits_flat.min()), abs(all_benefits_flat.max())) * 0.15

    for idx, m in enumerate(MODALITY_KEYS):
        ax = axes[idx]
        C = conflict_scores[m].numpy()
        B = benefits[m].numpy()

        # Quantile-based bins for equal-count bins
        bin_edges = np.quantile(C, np.linspace(0, 1, n_bins + 1))
        bin_edges = np.unique(bin_edges)
        actual_bins = len(bin_edges) - 1

        bin_centers, bin_means, bin_stds, bin_counts = [], [], [], []
        for i in range(actual_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i < actual_bins - 1:
                mask = (C >= lo) & (C < hi)
            else:
                mask = (C >= lo) & (C <= hi)
            if mask.sum() == 0:
                continue
            bin_centers.append((lo + hi) / 2)
            bin_means.append(B[mask].mean())
            bin_stds.append(B[mask].std() / np.sqrt(mask.sum()))  # SE
            bin_counts.append(mask.sum())

        bin_centers = np.array(bin_centers)
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)

        # Fill regions
        ax.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds,
                        alpha=0.25, color=FILL_COLORS[m], zorder=1)

        # Main line
        ax.plot(bin_centers, bin_means, color=COLORS[m], linewidth=2.2,
                marker="o", markersize=6, markeredgecolor="white", markeredgewidth=1.2,
                zorder=3, label=f"{MODALITY_NAMES[m]}")

        # Zero reference line
        ax.axhline(0, color="#6B7280", linewidth=0.8, linestyle="--", alpha=0.7, zorder=0)

        # Shade background: green for B>0, red for B<0
        ylims = [min(bin_means.min() - y_margin, -0.01), max(bin_means.max() + y_margin, 0.01)]
        ax.axhspan(0, ylims[1] * 1.5, color="#DCFCE7", alpha=0.25, zorder=0)
        ax.axhspan(ylims[0] * 1.5, 0, color="#FEE2E2", alpha=0.25, zorder=0)

        # Annotations
        ax.set_xlabel("Conflict Score $C_m$")
        if idx == 0:
            ax.set_ylabel("Reconstruction Benefit $\\mathbb{E}[B_m]$")
        ax.set_title(f"{MODALITY_NAMES[m]}", fontweight="bold", pad=8)
        ax.legend(loc="best", framealpha=0.85, edgecolor="#D1D5DB")

        # Annotation: count per bin
        for cx, cnt in zip(bin_centers, bin_counts):
            ax.annotate(f"n={cnt}", (cx, ylims[0] * 0.85),
                       fontsize=7, ha="center", color="#9CA3AF", style="italic")

        ax.set_ylim(ylims)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    fig.suptitle(
        f"Conflict Score vs. Reconstruction Benefit — DMD on {dataset_name.upper()}",
        fontsize=15, fontweight="bold", y=1.03
    )

    # Add interpretation text
    fig.text(
        0.5, -0.06,
        "$B_m > 0$: reconstruction improves task prediction  ·  "
        "$B_m < 0$: reconstruction hurts task prediction",
        ha="center", fontsize=10, color="#4B5563", style="italic"
    )

    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.25)
    print(f"[✓] Figure saved to {output_path}")
    plt.close()


def plot_scatter_overlay(conflict_scores, benefits, output_path, dataset_name="MOSI"):
    """
    Supplementary scatter plot with per-sample (C_m, B_m) and a LOWESS trend.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    COLORS = {"l": "#2563EB", "v": "#16A34A", "a": "#DC2626"}

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for idx, m in enumerate(MODALITY_KEYS):
        ax = axes[idx]
        C = conflict_scores[m].numpy()
        B = benefits[m].numpy()

        ax.scatter(C, B, s=12, alpha=0.35, color=COLORS[m], edgecolors="none", zorder=2)
        ax.axhline(0, color="#6B7280", linewidth=0.8, linestyle="--", alpha=0.7, zorder=0)

        # Polynomial trend (degree 3)
        if len(C) > 10:
            order = np.argsort(C)
            C_sorted, B_sorted = C[order], B[order]
            try:
                coeffs = np.polyfit(C_sorted, B_sorted, 3)
                poly = np.poly1d(coeffs)
                xs = np.linspace(C_sorted.min(), C_sorted.max(), 200)
                ax.plot(xs, poly(xs), color="black", linewidth=2, linestyle="-", zorder=4,
                        label="Poly-3 trend")
            except np.linalg.LinAlgError:
                pass

        ax.set_xlabel("Conflict Score $C_m$")
        if idx == 0:
            ax.set_ylabel("Reconstruction Benefit $B_m$")
        ax.set_title(f"{MODALITY_NAMES[m]}", fontweight="bold", pad=8)
        ax.legend(loc="best", fontsize=9)

        # Pearson r
        r = np.corrcoef(C, B)[0, 1]
        ax.annotate(f"$r = {r:.3f}$", xy=(0.97, 0.97), xycoords="axes fraction",
                   fontsize=10, ha="right", va="top",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D1D5DB", alpha=0.9))

    fig.suptitle(
        f"Per-Sample Conflict vs. Reconstruction Benefit — DMD on {dataset_name.upper()}",
        fontsize=15, fontweight="bold", y=1.03
    )
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.25)
    print(f"[✓] Scatter plot saved to {output_path}")
    plt.close()


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=Path("pt/dmd-mosi-missing0.0.pth"),
                   help="Path to DMD checkpoint (.pth)")
    p.add_argument("--dataset", default="mosi", choices=("mosi", "mosei", "sims"))
    p.add_argument("--data-dir", default="data")
    p.add_argument("--config-file", default="config/dmd_mosi_config.json")
    p.add_argument("--seed", type=int, default=1111)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--probe-epochs", type=int, default=200,
                   help="Training epochs for the frozen linear probe")
    p.add_argument("--probe-hidden", type=int, default=128)
    p.add_argument("--n-bins", type=int, default=10,
                   help="Number of quantile bins for the bar plot")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=Path, default=Path("result/dmd_conflict_recon"))
    return p.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    setup_seed(args.seed)
    device = torch.device(args.device)

    # ── Load config & data ──
    cfg = get_config_regression("dmd", args.dataset, args.config_file, args.data_dir)
    cfg.batch_size = args.batch_size
    cfg.data_missing = False
    cfg.missing_rate_eval_test = 0.0
    cfg.cur_seed = args.seed
    cfg.device = device
    dataloaders = MMDataLoader(cfg, num_workers=0)

    # ── Load model ──
    model = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("[✓] Model loaded and frozen.")

    # ── Extract features ──
    print("\n[Stage 1] Extracting features on train split...")
    train_feats = extract_features(model, dataloaders["train"], device)
    print(f"    Train samples: {train_feats['labels'].shape[0]}")

    print("[Stage 1] Extracting features on test split...")
    test_feats = extract_features(model, dataloaders["test"], device)
    print(f"    Test samples: {test_feats['labels'].shape[0]}")

    # ── Compute conflict scores (test set) ──
    print("\n[Stage 2] Computing conflict scores on test set...")
    conflict_scores = compute_conflict_scores(test_feats)
    for m in MODALITY_KEYS:
        print(f"    C_{m}: mean={conflict_scores[m].mean():.4f}, "
              f"std={conflict_scores[m].std():.4f}, "
              f"range=[{conflict_scores[m].min():.4f}, {conflict_scores[m].max():.4f}]")

    # ── Train probes on train split ──
    print("\n[Stage 3] Training frozen probes on train split...")
    probes = {}
    for m in MODALITY_KEYS:
        print(f"  ─ Modality: {MODALITY_NAMES[m]}")
        X_train = mean_pool_features(train_feats[f"orig_{m}"])
        y_train = train_feats["labels"]
        probes[m] = train_probe(X_train, y_train,
                                hidden_dim=args.probe_hidden,
                                epochs=args.probe_epochs)

    # ── Compute reconstruction benefit on test split ──
    print("\n[Stage 4] Computing reconstruction benefit on test set...")
    benefits = {}
    for m in MODALITY_KEYS:
        X_orig_test = mean_pool_features(test_feats[f"orig_{m}"])
        X_recon_test = mean_pool_features(test_feats[f"recon_{m}"])
        y_test = test_feats["labels"]
        B = compute_reconstruction_benefit(probes[m], X_orig_test, X_recon_test, y_test)
        benefits[m] = B
        pos_frac = (B > 0).float().mean().item() * 100
        neg_frac = (B < 0).float().mean().item() * 100
        print(f"    B_{m}: mean={B.mean():.4f}, "
              f"positive={pos_frac:.1f}%, negative={neg_frac:.1f}%")

    # ── Save numerical results ──
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import csv
    csv_path = args.output_dir / f"{args.dataset}_per_sample.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_idx", "label", "modality",
            "conflict_score", "reconstruction_benefit"
        ])
        writer.writeheader()
        n = test_feats["labels"].shape[0]
        for m in MODALITY_KEYS:
            for i in range(n):
                writer.writerow({
                    "sample_idx": i,
                    "label": float(test_feats["labels"][i]),
                    "modality": m,
                    "conflict_score": float(conflict_scores[m][i]),
                    "reconstruction_benefit": float(benefits[m][i]),
                })
    print(f"\n[✓] Per-sample CSV saved to {csv_path}")

    # ── Correlation summary ──
    print("\n[Stage 5] Correlation summary:")
    for m in MODALITY_KEYS:
        C_np = conflict_scores[m].numpy()
        B_np = benefits[m].numpy()
        r_pearson = np.corrcoef(C_np, B_np)[0, 1]
        # Spearman
        from scipy.stats import spearmanr
        rho, p_val = spearmanr(C_np, B_np)
        print(f"    {MODALITY_NAMES[m]:6s}  Pearson r = {r_pearson:+.4f}   "
              f"Spearman ρ = {rho:+.4f}  (p = {p_val:.2e})")

    # ── Plot ──
    print("\n[Stage 6] Generating figures...")
    main_fig_path = args.output_dir / f"{args.dataset}_conflict_vs_benefit.pdf"
    plot_conflict_vs_benefit(conflict_scores, benefits, main_fig_path,
                            n_bins=args.n_bins, dataset_name=args.dataset)

    scatter_fig_path = args.output_dir / f"{args.dataset}_scatter_conflict_benefit.pdf"
    plot_scatter_overlay(conflict_scores, benefits, scatter_fig_path,
                        dataset_name=args.dataset)

    # Also save as PNG for quick viewing
    main_png_path = args.output_dir / f"{args.dataset}_conflict_vs_benefit.png"
    plot_conflict_vs_benefit(conflict_scores, benefits, main_png_path,
                            n_bins=args.n_bins, dataset_name=args.dataset)

    scatter_png_path = args.output_dir / f"{args.dataset}_scatter_conflict_benefit.png"
    plot_scatter_overlay(conflict_scores, benefits, scatter_png_path,
                        dataset_name=args.dataset)

    print(f"\n{'='*60}")
    print(f"[✓] All done. Output directory: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


# python scripts/analyze_dmd_conflict_reconstruction.py --checkpoint pt/dmd-mosi-missing0.0.pth --dataset mosi --data-dir data --config-file config/dmd_mosi_config.json --device cuda --seed 1111 --probe-epochs 200 --n-bins 10
