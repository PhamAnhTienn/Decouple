"""
Cluster-aware Visualization of DP Shared Feature Space
=======================================================
Visualizes the effect of OT-based cluster loss and prototype separation loss
on the shared (modality-invariant) space. Produces:

  Figure 1: t-SNE with prototypes & sentiment markers (publication style)
  Figure 2: Per-modality cluster assignment heatmap
  Figure 3: Radar chart of quantitative metrics

Usage:
    python scripts/visualize_cluster_shared_space_dp.py \
        --checkpoint pt/dp-mosi-missing0.0.pth \
        --config_file config/dp_mosi_config.json \
        --dataset mosi --split test --missing_data --missing_rate 0.0
"""

import argparse, os, sys, warnings
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config_regression
from data_loader import MMDataLoader
from models import build_model
from utils import setup_seed

warnings.filterwarnings("ignore")

# ──────────────────────── sentiment binning ────────────────────────

SENT_NAMES = {
    0: "Highly Neg",
    1: "Negative",
    2: "Weakly Neg",
    3: "Neutral",
    4: "Weakly Pos",
    5: "Positive",
    6: "Highly Pos",
}

SENT_MARKERS = {
    0: "s",   # square
    1: "^",   # triangle up
    2: "p",   # pentagon
    3: "o",   # circle
    4: "*",   # star
    5: "D",   # diamond
    6: "P",   # plus (filled)
}

MOD_COLORS = {"L": "#3b4cc0", "V": "#8B4513", "A": "#b40426"}
MOD_NAMES  = {"L": "Language", "V": "Vision", "A": "Audio"}


def bin_labels_7class(labels):
    """Map continuous MOSI scores to 7-class bins like the reference figure."""
    rounded = np.clip(np.round(labels).astype(int), -3, 3)
    # -3→0, -2→1, -1→2, 0→3, 1→4, 2→5, 3→6
    return rounded + 3


# ──────────────────────── feature extraction ────────────────────────

def extract_features(model, dataloader, device):
    """Extract h_c_{l,v,a}, h_shared, prototypes, and labels."""
    model.eval()
    h_l_all, h_v_all, h_a_all, h_shared_all, labels_all = [], [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            text   = batch["text"].to(device)
            audio  = batch["audio"].to(device)
            vision = batch["vision"].to(device)

            out = model(text, audio, vision, is_distill=False)

            h_l_all.append(out["repr_l_homo"].cpu().numpy())
            h_v_all.append(out["repr_v_homo"].cpu().numpy())
            h_a_all.append(out["repr_a_homo"].cpu().numpy())
            # h_l_all.append(out["h_c_l"].cpu().numpy())
            # h_v_all.append(out["h_c_v"].cpu().numpy())
            # h_a_all.append(out["h_c_a"].cpu().numpy())
            h_shared_all.append(out["h_shared"].cpu().numpy())
            labels_all.append(batch["labels"]["M"].numpy().reshape(-1))

    # Prototypes (K, D) – already L2-normalized at save time
    protos = F.normalize(model.prototypes.weight.data, dim=1, p=2).cpu().numpy()

    return (
        np.concatenate(h_l_all),
        np.concatenate(h_v_all),
        np.concatenate(h_a_all),
        np.concatenate(h_shared_all),
        np.concatenate(labels_all),
        protos,
    )


# ──────────────────────── Figure 1: t-SNE ────────────────────────

def plot_tsne_with_prototypes(h_l, h_v, h_a, protos, labels, save_path,
                              perplexity=30, seed=42):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    N = h_l.shape[0]
    K = protos.shape[0]
    bins = bin_labels_7class(labels)

    # Stack: [L ; V ; A ; Prototypes]
    all_feat = np.concatenate([h_l, h_v, h_a, protos], axis=0)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                max_iter=2000)
    emb = tsne.fit_transform(all_feat)
    print(f"  t-SNE KL divergence: {tsne.kl_divergence_:.4f}")

    emb_l = emb[:N]
    emb_v = emb[N:2*N]
    emb_a = emb[2*N:3*N]
    emb_p = emb[3*N:]

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    # Plot each modality
    for mod_key, emb_m, label_tag in [("L", emb_l, bins),
                                       ("V", emb_v, bins),
                                       ("A", emb_a, bins)]:
        for sid in sorted(SENT_NAMES.keys()):
            mask = label_tag == sid
            if mask.sum() == 0:
                continue
            ax.scatter(
                emb_m[mask, 0], emb_m[mask, 1],
                c=MOD_COLORS[mod_key],
                marker=SENT_MARKERS[sid],
                s=36, alpha=0.6,
                edgecolors="white", linewidths=0.3,
                zorder=3,
            )

    # Plot prototypes as large black stars
    ax.scatter(
        emb_p[:, 0], emb_p[:, 1],
        c="black", marker="X", s=220,
        edgecolors="gold", linewidths=1.5,
        zorder=5, label="Prototypes",
    )
    for k in range(K):
        ax.annotate(f"C{k}", (emb_p[k, 0], emb_p[k, 1]),
                     fontsize=7, fontweight="bold", color="black",
                     ha="center", va="bottom",
                     xytext=(0, 6), textcoords="offset points")

    # ── Legends ──
    # Sentiment legend (markers)
    sent_handles = [
        Line2D([0], [0], marker=SENT_MARKERS[sid], color="w",
               markerfacecolor="gray", markeredgecolor="gray",
               markersize=8, linewidth=0, label=SENT_NAMES[sid])
        for sid in sorted(SENT_NAMES.keys())
    ]
    leg1 = ax.legend(handles=sent_handles, title="Sentiment",
                     loc="lower left", fontsize=7.5, title_fontsize=8.5,
                     framealpha=0.9, edgecolor="#ccc", ncol=2)
    ax.add_artist(leg1)

    # Modality legend (colors)
    mod_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=MOD_COLORS[m], markersize=9,
               linewidth=0, label=MOD_NAMES[m])
        for m in ["L", "V", "A"]
    ]
    # Prototype legend
    mod_handles.append(
        Line2D([0], [0], marker="X", color="w", markerfacecolor="black",
               markeredgecolor="gold", markersize=10, linewidth=0,
               label="Prototype")
    )
    ax.legend(handles=mod_handles, title="Modality",
              loc="upper right", fontsize=8.5, title_fontsize=9,
              framealpha=0.9, edgecolor="#ccc")
    ax.add_artist(leg1)

    ax.set_title("t-SNE of Shared Space with Prototypes\n(CMU-MOSI Test Set)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("t-SNE Dim 1", fontsize=10)
    ax.set_ylabel("t-SNE Dim 2", fontsize=10)
    ax.grid(True, alpha=0.15, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {save_path}")
    plt.close()


# ──────────────────────── Figure 2: Cluster assignment heatmap ────────────────────────

def plot_cluster_heatmap(model, h_l, h_v, h_a, h_shared, labels, save_path):
    """Show how each sentiment bin distributes across prototypes for each modality."""
    import matplotlib.pyplot as plt

    bins = bin_labels_7class(labels)
    C = F.normalize(model.prototypes.weight.data, dim=1, p=2)  # (K, D)
    K = C.shape[0]

    # OT assignment from fused representation
    h_shared_t = torch.from_numpy(h_shared).to(C.device)
    S_bar = h_shared_t @ C.t()
    Q_star = model.sinkhorn_knopp(S_bar, epsilon=0.05, sinkhorn_iterations=3).cpu().numpy()

    # Per-modality softmax predictions
    tau = 0.1
    preds = {}
    for key, h_np in [("L", h_l), ("V", h_v), ("A", h_a)]:
        h_t = torch.from_numpy(h_np).to(C.device)
        s = (h_t @ C.t()) / tau
        preds[key] = F.softmax(s, dim=1).cpu().numpy()

    unique_bins = sorted(np.unique(bins))
    n_bins = len(unique_bins)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.patch.set_facecolor("white")

    datasets = [("Q* (OT Target)", Q_star)] + [(f"{MOD_NAMES[m]} pred", preds[m]) for m in ["L", "V", "A"]]

    for ax, (title, assign) in zip(axes, datasets):
        # Average assignment per sentiment bin → (n_bins, K)
        mat = np.zeros((n_bins, K))
        for i, b in enumerate(unique_bins):
            mask = bins == b
            if mask.sum() > 0:
                mat[i] = assign[mask].mean(axis=0)

        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0)
        ax.set_xticks(range(K))
        ax.set_xticklabels([f"C{k}" for k in range(K)], fontsize=8)
        ax.set_yticks(range(n_bins))
        ax.set_yticklabels([SENT_NAMES[b] for b in unique_bins], fontsize=8)
        ax.set_xlabel("Prototype", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")

        # Annotate values
        for i in range(n_bins):
            for j in range(K):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=6.5, color="black" if mat[i,j] < 0.3 else "white")

    fig.colorbar(im, ax=axes, shrink=0.8, label="Mean Assignment Probability")
    fig.suptitle("Cluster Assignment: OT Target vs Per-Modality Predictions",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {save_path}")
    plt.close()


# ──────────────────── Figure 3: Quantitative metrics bar chart ────────────────────

def plot_metrics(h_l, h_v, h_a, protos, labels, save_path):
    """
    Compute and visualize:
      - Silhouette score (cluster quality)
      - Inter/Intra class distance ratio
      - Cross-modal alignment (mean pairwise distance between modalities for same sample)
      - Prototype separation (mean pairwise distance between prototypes)
    """
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt

    bins = bin_labels_7class(labels)
    N = h_l.shape[0]

    # 1. Silhouette on fused features
    h_fused = (h_l + h_v + h_a) / 3.0
    h_fused_norm = h_fused / (np.linalg.norm(h_fused, axis=1, keepdims=True) + 1e-8)

    # Cluster assignment = argmax of cosine sim to prototypes
    sims = h_fused_norm @ protos.T  # (N, K)
    cluster_ids = sims.argmax(axis=1)

    n_unique = len(np.unique(cluster_ids))
    sil = silhouette_score(h_fused_norm, cluster_ids) if n_unique >= 2 else 0.0

    # 2. Inter/Intra sentiment distance ratio
    unique_bins = np.unique(bins)
    intra, inter = [], []
    all_feat = np.concatenate([h_l, h_v, h_a])
    all_bins = np.concatenate([bins, bins, bins])
    rng = np.random.RandomState(42)
    for b in unique_bins:
        in_c = all_feat[all_bins == b]
        out_c = all_feat[all_bins != b]
        if len(in_c) < 2 or len(out_c) < 2:
            continue
        n_s = min(200, len(in_c))
        idx_in = rng.choice(len(in_c), n_s, replace=len(in_c) < n_s)
        idx_out = rng.choice(len(out_c), n_s, replace=True)
        for i in range(0, n_s - 1, 2):
            intra.append(np.linalg.norm(in_c[idx_in[i]] - in_c[idx_in[i+1]]))
        for i in range(n_s):
            inter.append(np.linalg.norm(in_c[idx_in[i]] - out_c[idx_out[i]]))
    ratio = np.mean(inter) / (np.mean(intra) + 1e-8)

    # 3. Cross-modal alignment
    d_lv = np.mean(np.linalg.norm(h_l - h_v, axis=1))
    d_la = np.mean(np.linalg.norm(h_l - h_a, axis=1))
    d_va = np.mean(np.linalg.norm(h_v - h_a, axis=1))
    alignment = (d_lv + d_la + d_va) / 3.0

    # 4. Prototype separation
    K = protos.shape[0]
    proto_dists = []
    for i in range(K):
        for j in range(i+1, K):
            proto_dists.append(np.linalg.norm(protos[i] - protos[j]))
    proto_sep = np.mean(proto_dists)

    # ── Print metrics ──
    print("\n" + "=" * 60)
    print("QUANTITATIVE METRICS")
    print("=" * 60)
    print(f"  Silhouette Score (cluster)      : {sil:.4f}")
    print(f"  Inter/Intra Sentiment Ratio     : {ratio:.4f}")
    print(f"  Cross-Modal Alignment (↓ better): {alignment:.4f}")
    print(f"  Prototype Separation  (↑ better): {proto_sep:.4f}")
    print(f"  Clusters used: {n_unique}/{K}")
    print("=" * 60)

    # ── Bar chart ──
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.patch.set_facecolor("white")

    metrics = [
        ("Silhouette\nScore", sil, "#2ecc71", "[-1, 1]"),
        ("Inter/Intra\nRatio", ratio, "#3498db", "> 1 good"),
        ("Cross-Modal\nAlignment", alignment, "#e74c3c", "↓ better"),
        ("Prototype\nSeparation", proto_sep, "#9b59b6", "↑ better"),
    ]

    for ax, (name, val, color, note) in zip(axes, metrics):
        bar = ax.bar([name], [val], color=color, width=0.5, edgecolor="white", linewidth=1.5)
        ax.bar_label(bar, fmt="%.4f", fontsize=11, fontweight="bold", padding=4)
        ax.set_title(name, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel("")
        ax.text(0.5, 0.02, note, transform=ax.transAxes, ha="center",
                fontsize=8, color="gray", style="italic")
        ax.tick_params(labelbottom=False, labelsize=9)
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.set_axisbelow(True)

    fig.suptitle("Shared Space Quality Metrics", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {save_path}")
    plt.close()


# ──────────────────── Figure 4: Per-modality t-SNE panels ────────────────────

def plot_per_modality_panels(h_l, h_v, h_a, protos, labels, save_path,
                              perplexity=30, seed=42):
    """
    3-panel t-SNE like the reference image: one panel per modality,
    each showing how well samples cluster by sentiment around prototypes.
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.spatial import ConvexHull

    N = h_l.shape[0]
    K = protos.shape[0]
    bins = bin_labels_7class(labels)

    # Joint t-SNE so all panels share the same embedding space
    all_feat = np.concatenate([h_l, h_v, h_a, protos], axis=0)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, max_iter=2000)
    emb = tsne.fit_transform(all_feat)

    emb_dict = {"L": emb[:N], "V": emb[N:2*N], "A": emb[2*N:3*N]}
    emb_p = emb[3*N:]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("white")

    for ax, mod_key in zip(axes, ["L", "V", "A"]):
        ax.set_facecolor("#fafafa")
        emb_m = emb_dict[mod_key]

        # Draw convex hulls around each cluster assignment
        sims = np.concatenate([h_l, h_v, h_a], axis=0) if mod_key == "L" else None
        h_mod = {"L": h_l, "V": h_v, "A": h_a}[mod_key]
        csim = h_mod @ protos.T
        cids = csim.argmax(axis=1)

        hull_colors = plt.cm.Set2(np.linspace(0, 1, K))
        for k in range(K):
            mask_k = cids == k
            if mask_k.sum() >= 3:
                pts = emb_m[mask_k]
                try:
                    hull = ConvexHull(pts)
                    verts = np.append(hull.vertices, hull.vertices[0])
                    ax.fill(pts[verts, 0], pts[verts, 1],
                            alpha=0.08, color=hull_colors[k])
                    ax.plot(pts[verts, 0], pts[verts, 1],
                            linewidth=1.0, linestyle="--", color=hull_colors[k], alpha=0.4)
                except Exception:
                    pass

        # Plot samples
        for sid in sorted(SENT_NAMES.keys()):
            mask = bins == sid
            if mask.sum() == 0:
                continue
            ax.scatter(emb_m[mask, 0], emb_m[mask, 1],
                       c=MOD_COLORS[mod_key], marker=SENT_MARKERS[sid],
                       s=40, alpha=0.65, edgecolors="white", linewidths=0.3, zorder=3)

        # Prototypes
        ax.scatter(emb_p[:, 0], emb_p[:, 1], c="black", marker="X", s=200,
                   edgecolors="gold", linewidths=1.5, zorder=5)
        for k in range(K):
            ax.annotate(f"C{k}", (emb_p[k, 0], emb_p[k, 1]),
                         fontsize=7, fontweight="bold", ha="center", va="bottom",
                         xytext=(0, 6), textcoords="offset points")

        ax.set_title(f"{MOD_NAMES[mod_key]}", fontsize=13, fontweight="bold",
                     color=MOD_COLORS[mod_key])
        ax.set_xlabel("t-SNE Dim 1", fontsize=9)
        ax.set_ylabel("t-SNE Dim 2", fontsize=9)
        ax.grid(True, alpha=0.15, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)

    # Shared legend at bottom
    sent_handles = [
        Line2D([0], [0], marker=SENT_MARKERS[sid], color="w",
               markerfacecolor="gray", markeredgecolor="gray",
               markersize=8, linewidth=0, label=SENT_NAMES[sid])
        for sid in sorted(SENT_NAMES.keys())
    ]
    sent_handles.append(
        Line2D([0], [0], marker="X", color="w", markerfacecolor="black",
               markeredgecolor="gold", markersize=10, linewidth=0, label="Prototype")
    )
    fig.legend(handles=sent_handles, loc="lower center", ncol=8,
               fontsize=8.5, framealpha=0.9, edgecolor="#ccc",
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Per-Modality Shared Space with Cluster Boundaries",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {save_path}")
    plt.close()


# ──────────────────────── main ────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cluster-aware visualization of DP shared space")
    parser.add_argument("--checkpoint", type=str, default="pt/dp-mosi-missing0.0.pth")
    parser.add_argument("--dataset", type=str, default="mosi")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--config_file", type=str, default="config/dp_mosi_config.json")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--missing_data", action="store_true")
    parser.add_argument("--missing_rate", type=float, default=0.0)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config & model
    model_args = get_config_regression("dp", args.dataset, args.config_file, data_dir=args.data_dir)
    model_args.device = device
    model_args.model_save_path = args.checkpoint
    model_args.data_missing = args.missing_data
    model_args.missing_rate_eval_test = args.missing_rate
    model_args.cur_seed = 1111

    dataloader = MMDataLoader(model_args, num_workers=0)
    model = build_model(model_args)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Extract features
    print("Extracting features...")
    h_l, h_v, h_a, h_shared, labels, protos = extract_features(model, dataloader[args.split], device)
    print(f"  N={h_l.shape[0]}, D={h_l.shape[1]}, K={protos.shape[0]}")
    print(f"  Labels: [{labels.min():.2f}, {labels.max():.2f}]")

    # Output directory
    out_dir = args.output_dir or str(PROJECT_ROOT / "result" / "figures")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{args.dataset}_{args.split}_{args.missing_rate:.1f}"

    # ── Generate all figures ──
    print("\n[1/4] t-SNE with prototypes...")
    plot_tsne_with_prototypes(
        h_l, h_v, h_a, protos, labels,
        save_path=os.path.join(out_dir, f"tsne_cluster_{tag}.png"),
        perplexity=args.perplexity, seed=args.seed,
    )

    print("\n[2/4] Cluster assignment heatmap...")
    plot_cluster_heatmap(
        model, h_l, h_v, h_a, h_shared, labels,
        save_path=os.path.join(out_dir, f"cluster_heatmap_{tag}.png"),
    )

    print("\n[3/4] Quantitative metrics...")
    plot_metrics(
        h_l, h_v, h_a, protos, labels,
        save_path=os.path.join(out_dir, f"metrics_{tag}.png"),
    )

    print("\n[4/4] Per-modality panels...")
    plot_per_modality_panels(
        h_l, h_v, h_a, protos, labels,
        save_path=os.path.join(out_dir, f"per_modality_{tag}.png"),
        perplexity=args.perplexity, seed=args.seed,
    )

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
