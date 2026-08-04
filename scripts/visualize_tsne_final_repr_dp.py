"""
t-SNE Visualization of DP Final Representation (Pre-Prediction)
================================================================
Visualises the *final fused representation* of the DP model — the
vector formed by concatenating the gated heterogeneous cross-modal
branches (last_h_l, last_h_v, last_h_a) with the shared-encoder
fusion (c_fusion), i.e. ``last_hs_proj``.

This is the representation immediately before the final linear
prediction head, so it captures **everything the model knows** about
a sample.  A t-SNE plot of this space reveals whether the model has
learned semantically discriminative clusters (i.e. samples with the
same sentiment score are grouped together).

Unlike ``visualize_tsne_shared_space_dp.py``, which plots only the
modality-invariant shared features to evaluate cross-modal alignment,
this script focuses on **class-level separability** in the final
representation — the kind of plot typically shown in papers to
demonstrate that a model produces a well-structured embedding space.

Usage:
    python scripts/visualize_tsne_final_repr_dp.py \\
        --checkpoint /home/tien/NCKH/DecAlign/pt/dp-mosi.pth \\
        --dataset mosi \\
        --data_dir ./data \\
        --config_file config/dp_mosi_config.json \\
        --gpu_id 0
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config_regression
from data_loader import MMDataLoader
from models import build_model
from utils import setup_seed


def extract_final_representation(model, dataloader, device):
    """
    Forward pass through the DP model, extracting the final fused
    representation ``last_hs_proj`` (right before the output layer).

    This vector is the concatenation of gated cross-modal branches
    and the shared-encoder fusion, followed by a residual MLP:
        last_hs = [σ(W_l·h_l), σ(W_v·h_v), σ(W_a·h_a), σ(W_c·c_fusion)]
        last_hs_proj = MLP(last_hs) + last_hs   (residual connection)

    Returns:
        representations: numpy array [N, combined_dim]
        labels: numpy array [N] (regression scores)
    """
    model.eval()

    all_reprs = []
    all_labels = []

    with torch.no_grad():
        for batch_data in dataloader:
            text = batch_data['text'].to(device)
            audio = batch_data['audio'].to(device)
            vision = batch_data['vision'].to(device)
            labels = batch_data['labels']['M']

            # DP forward
            output = model(text, audio, vision, is_distill=False)

            # --- Replicate the gating + concatenation logic from DP.forward ---
            # The model computes last_hs_proj internally but does not expose it
            # in the output dict, so we reconstruct it from the exposed tensors.
            last_h_l = output['last_h_l']   # (B, 2*d)
            last_h_v = output['last_h_v']   # (B, 2*d)
            last_h_a = output['last_h_a']   # (B, 2*d)

            # c_fusion = cat(c_l_att[-1], c_v_att[-1], c_a_att[-1])  → (B, 3*d)
            # The model already exposes h_c_l etc., but we need the raw
            # self-attention outputs for c_fusion. Instead, we reconstruct
            # last_hs_proj by hooking into the model's own layers.
            #
            # Actually, it's easier to just re-use the gating weights and
            # projection layers directly:
            last_h_l_gated = torch.sigmoid(model.weight_l(last_h_l))
            last_h_v_gated = torch.sigmoid(model.weight_v(last_h_v))
            last_h_a_gated = torch.sigmoid(model.weight_a(last_h_a))

            # For c_fusion, we need the self-attention pooled shared features.
            # We can get them from the model's forward output keys.
            c_l = output['c_l']  # (T, B, D)
            c_v = output['c_v']
            c_a = output['c_a']

            c_l_att = model.self_attentions_c_l(c_l)
            if type(c_l_att) == tuple:
                c_l_att = c_l_att[0]
            c_l_att = c_l_att[-1]

            c_v_att = model.self_attentions_c_v(c_v)
            if type(c_v_att) == tuple:
                c_v_att = c_v_att[0]
            c_v_att = c_v_att[-1]

            c_a_att = model.self_attentions_c_a(c_a)
            if type(c_a_att) == tuple:
                c_a_att = c_a_att[0]
            c_a_att = c_a_att[-1]

            c_fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)
            c_fusion_gated = torch.sigmoid(model.weight_c(c_fusion))

            last_hs = torch.cat([last_h_l_gated, last_h_v_gated,
                                 last_h_a_gated, c_fusion_gated], dim=1)
            last_hs_proj = model.proj2(
                F.dropout(F.relu(model.proj1(last_hs), inplace=False),
                          p=model.output_dropout, training=False))
            last_hs_proj = last_hs_proj + last_hs

            all_reprs.append(last_hs_proj.cpu().numpy())
            all_labels.append(labels.numpy().reshape(-1))

    representations = np.concatenate(all_reprs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return representations, labels


def create_sentiment_bins(labels):
    """
    Bin continuous sentiment scores into 7 discrete categories for visualization,
    matching the Acc_7 metric in metricsTop.py.

    MOSI scores are clipped to [-3, 3] and rounded to the nearest integer,
    yielding 7 bins: {-3, -2, -1, 0, +1, +2, +3}.

    Returns:
        bin_ids: array of integer bin indices per sample (0–6)
    """
    rounded = np.round(labels).astype(int)
    rounded = np.clip(rounded, -3, 3)

    bin_map = {-3: 0, -2: 1, -1: 2, 0: 3, 1: 4, 2: 5, 3: 6}
    bin_ids = np.array([bin_map[r] for r in rounded])

    return bin_ids


def plot_tsne(representations, labels, save_path, perplexity=30, seed=42):
    """
    Create publication-quality t-SNE visualization of the final representation.

    Unlike the shared-space plot which uses different markers per modality,
    here each point is a *single sample* (the fused representation already
    combines all modalities).  Color = 7-class sentiment bin.
    """
    from sklearn.manifold import TSNE
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # ── Publication rcParams ──────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'mathtext.fontset': 'dejavusans',
    })

    print("Running t-SNE dimensionality reduction...")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        max_iter=2000,
    )
    embeddings = tsne.fit_transform(representations)
    print(f"t-SNE completed. KL divergence: {tsne.kl_divergence_:.4f}")

    # Create sentiment bins (7-class)
    bin_ids = create_sentiment_bins(labels)

    # =============================================
    # Publication-quality plot (7 bins)
    # =============================================

    # Diverging 7-color palette: cool → neutral → warm
    sentiment_colors = {
        0: '#08306b',   # -3  Very Strong Neg — darkest navy
        1: '#2171b5',   # -2  Strong Neg — medium blue
        2: '#6baed6',   # -1  Negative — light blue
        3: '#969696',   #  0  Neutral — mid gray
        4: '#fb6a4a',   # +1  Positive — salmon red
        5: '#cb181d',   # +2  Strong Pos — crimson
        6: '#67000d',   # +3  Very Strong Pos — darkest red
    }

    sentiment_names = {
        0: '−3  (Very Strong Neg)',
        1: '−2  (Strong Neg)',
        2: '−1  (Negative)',
        3: ' 0  (Neutral)',
        4: '+1  (Positive)',
        5: '+2  (Strong Pos)',
        6: '+3  (Very Strong Pos)',
    }

    # ── Figure & axes ─────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')

    # ── Spines: keep only left & bottom, thicken ─────────────────────
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.2)
        ax.spines[spine].set_color('#333333')

    # ── Plot each sentiment bin ───────────────────────────────────────
    for sent_id in sorted(sentiment_colors.keys()):
        mask = bin_ids == sent_id
        if mask.sum() == 0:
            continue
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=sentiment_colors[sent_id],
            marker='o',
            s=26,
            alpha=0.72,
            edgecolors='white',
            linewidths=0.25,
            zorder=3,
            rasterized=True,
        )

    # ── Legend ─────────────────────────────────────────────────────────
    color_handles = [
        mpatches.Patch(
            facecolor=sentiment_colors[sid],
            edgecolor='#888888',
            linewidth=0.6,
            label=sentiment_names[sid],
        )
        for sid in sorted(sentiment_colors.keys())
    ]
    legend = ax.legend(
        handles=color_handles,
        title='Sentiment Bin',
        loc='upper left',
        fontsize=7.5,
        title_fontsize=8.5,
        framealpha=0.92,
        edgecolor='#bbbbbb',
        borderpad=0.6,
        handlelength=1.0,
        handleheight=0.8,
        labelspacing=0.35,
    )
    legend.get_title().set_fontweight('bold')

    # ── Title & axis labels ───────────────────────────────────────────
    ax.set_title(
        't-SNE of DP Final Representation\n'
        '(Pre-Prediction Fused Embedding, CMU-MOSI Test Set)',
        fontsize=12,
        fontweight='bold',
        pad=12,
    )
    ax.set_xlabel('t-SNE Dimension 1', fontsize=10.5, labelpad=6)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=10.5, labelpad=6)
    ax.tick_params(axis='both', labelsize=9, width=1.0, length=4,
                   direction='out', color='#555555')

    # Light grid
    ax.grid(True, alpha=0.18, linestyle='--', linewidth=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved t-SNE plot to: {save_path}")
    plt.close()

    # =============================================
    # Quantitative analysis: class separability
    # =============================================
    _bin_names = {
        0: '-3 (VStrNeg)', 1: '-2 (StrNeg)', 2: '-1 (Neg)',
        3: ' 0 (Neutral)', 4: '+1 (Pos)',    5: '+2 (StrPos)',
        6: '+3 (VStrPos)',
    }

    print("\n" + "="*70)
    print("QUANTITATIVE ANALYSIS: Class Separability in Final Representation")
    print("="*70)

    sample_bin_ids = create_sentiment_bins(labels)
    unique_bins = np.unique(sample_bin_ids)

    # ── Per-class centroid and spread ─────────────────────────────────
    print(f"\n{'Sentiment Bin':<20} {'N':>5} {'Centroid Norm':>14} {'Intra-class σ':>14}")
    print("-" * 58)

    centroids = {}
    for bid in unique_bins:
        mask = sample_bin_ids == bid
        n_samples = mask.sum()
        if n_samples < 2:
            continue
        cluster_pts = embeddings[mask]
        centroid = cluster_pts.mean(axis=0)
        centroids[bid] = centroid
        centroid_norm = np.linalg.norm(centroid)
        intra_std = np.mean(np.linalg.norm(cluster_pts - centroid, axis=1))
        print(f"{_bin_names[bid]:<20} {n_samples:>5} {centroid_norm:>14.2f} {intra_std:>14.2f}")

    # ── Inter-class centroid distance matrix ──────────────────────────
    bid_list = sorted(centroids.keys())
    print(f"\n{'Inter-Class Centroid Distance Matrix':}")
    print("-" * 60)

    header = f"{'':>20}" + "".join(f"{_bin_names[b][:10]:>12}" for b in bid_list)
    print(header)
    for bi in bid_list:
        row = f"{_bin_names[bi][:14]:>20}"
        for bj in bid_list:
            dist = np.linalg.norm(centroids[bi] - centroids[bj])
            row += f"{dist:>12.2f}"
        print(row)

    # ── Raw feature-space intra/inter ratio ───────────────────────────
    print("\n" + "="*70)
    print("RAW FEATURE SPACE: Intra-class vs Inter-class Distance")
    print("="*70)

    intra_dists = []
    inter_dists = []

    for bid in unique_bins:
        in_class = representations[sample_bin_ids == bid]
        out_class = representations[sample_bin_ids != bid]

        if len(in_class) < 2 or len(out_class) < 2:
            continue

        n_sample = min(500, len(in_class))
        idx_in = np.random.choice(len(in_class), n_sample, replace=False)
        idx_out = np.random.choice(len(out_class), n_sample, replace=True)

        for i in range(0, n_sample - 1, 2):
            intra_dists.append(np.linalg.norm(in_class[idx_in[i]] - in_class[idx_in[i+1]]))

        for i in range(n_sample):
            inter_dists.append(np.linalg.norm(in_class[idx_in[i]] - out_class[idx_out[i]]))

    mean_intra = np.mean(intra_dists)
    mean_inter = np.mean(inter_dists)
    ratio = mean_inter / (mean_intra + 1e-8)

    print(f"\nMean Intra-class Distance : {mean_intra:.4f}")
    print(f"Mean Inter-class Distance : {mean_inter:.4f}")
    print(f"Inter / Intra Ratio       : {ratio:.4f}")
    print(f"\n→ Ratio >> 1.0 indicates good class separation in the final representation")
    print(f"  (model successfully learns semantically discriminative embeddings)")


def main():
    parser = argparse.ArgumentParser(
        description='t-SNE visualization of DP final representation (pre-prediction)')
    parser.add_argument('--checkpoint', type=str, default='pt/dp-mosi.pth',
                        help='Path to trained DP model checkpoint')
    parser.add_argument('--dataset', type=str, default='mosi', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--config_file', type=str, default='', help='Config JSON path')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'],
                        help='Data split to visualize')
    parser.add_argument('--missing_data', action='store_true', help='Missing data')
    parser.add_argument('--missing_rate', type=float, default=0.0, help='Missing rate for test')
    parser.add_argument('--perplexity', type=float, default=30, help='t-SNE perplexity')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='', help='Output image path')
    parser.add_argument('--config_override', action='append', default=[],
                        help='Override config values, e.g. hidden_size=128')
    args = parser.parse_args()

    setup_seed(args.seed)

    # Set device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config — use 'dp' model name
    config_file = args.config_file if args.config_file else ''
    model_args = get_config_regression('dp', args.dataset, config_file, data_dir=args.data_dir)
    model_args.device = device
    model_args.model_save_path = args.checkpoint

    model_args.data_missing = args.missing_data
    model_args.missing_rate_eval_test = args.missing_rate
    model_args.cur_seed = 1111

    # Apply config overrides
    for override in args.config_override:
        if '=' in override:
            key, value = override.split('=', 1)
            try:
                import ast
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
            setattr(model_args, key, value)

    # Load data
    print(f"Loading {args.dataset} dataset from {args.data_dir}...")
    dataloader = MMDataLoader(model_args, num_workers=0)

    # Build model and load checkpoint
    print(f"Building DP model and loading checkpoint from {args.checkpoint}...")
    model = build_model(model_args)

    state_dict = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Model loaded. Extracting final representation from {args.split} set...")

    # Extract features
    representations, labels = extract_final_representation(
        model, dataloader[args.split], device
    )

    print(f"Extracted final representation: shape={representations.shape}")
    print(f"Labels range: [{labels.min():.2f}, {labels.max():.2f}], N={len(labels)}")

    # Output path
    if args.output:
        save_path = args.output
    else:
        os.makedirs(str(PROJECT_ROOT / 'result' / 'figures'), exist_ok=True)
        save_path = str(PROJECT_ROOT / 'result' / 'figures' /
                        f'tsne_dp_final_repr_{args.dataset}_{args.split}_{model_args.missing_rate_eval_test:.1f}.png')

    # Generate t-SNE plot
    plot_tsne(
        representations, labels,
        save_path=save_path,
        perplexity=args.perplexity,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()

# python scripts/visualize_tsne_final_repr_dp.py \
#     --checkpoint /home/tien/NCKH/DecAlign/pt/dp-mosi-missing0.0.pth \
#     --config_file config/dp_mosi_config.json \
#     --missing_data \
#     --missing_rate 0.0 \
#     --dataset mosi \
#     --split test
