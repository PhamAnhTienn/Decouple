"""
t-SNE Visualization of DP Shared Feature Space (Modality-Invariant Encoder)
============================================================================
Demonstrates the alignment quality of the shared (modality-invariant)
encoder in the DP model.  The shared encoder `encoder_c` maps all three
modalities through a single Conv1d, producing c_l, c_v, c_a.

Usage:
    python scripts/visualize_tsne_shared_space_dp.py \
        --checkpoint /home/tien/NCKH/DecAlign/pt/dp-mosi.pth \
        --dataset mosi \
        --data_dir ./data \
        --config_file config/dp_mosi_config.json \
        --gpu_id 0
"""

import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config_regression
from data_loader import MMDataLoader
from models import build_model
from utils import setup_seed


def extract_shared_features(model, dataloader, device):
    """
    Forward pass through the DP model, extracting shared (modality-invariant)
    representations c_l, c_v, c_a from the shared encoder `encoder_c`.

    The DP model returns c_l, c_v, c_a with shape (seq_len, batch, d).
    We take the last time-step (matching the self-attention pooling used
    for prediction in the model) to get a fixed-size vector per sample.

    Returns:
        shared_t, shared_v, shared_a: numpy arrays [N, hidden_size]
        labels: numpy array [N] (regression scores)
    """
    model.eval()

    all_shared_t = []
    all_shared_v = []
    all_shared_a = []
    all_labels = []

    with torch.no_grad():
        for batch_data in dataloader:
            text = batch_data['text'].to(device)
            audio = batch_data['audio'].to(device)
            vision = batch_data['vision'].to(device)
            text_m = batch_data['text_m'].to(device)
            audio_m = batch_data['audio_m'].to(device)
            vision_m = batch_data['vision_m'].to(device)
            labels = batch_data['labels']['M']

            # DP forward: (text, audio, video, is_distill)
            # Use missing-rate inputs to match evaluation setting
            output = model(text, audio, vision, is_distill=False)

            # c_l, c_v, c_a have shape (seq_len, batch, d) after permute in DP.forward
            # Take the last time-step as the utterance-level representation
            c_l = output['repr_l_homo']  # (batch, d)
            c_v = output['repr_v_homo']  # (batch, d)
            c_a = output['repr_a_homo']  # (batch, d)

            all_shared_t.append(c_l.cpu().numpy())
            all_shared_v.append(c_v.cpu().numpy())
            all_shared_a.append(c_a.cpu().numpy())
            all_labels.append(labels.numpy().reshape(-1))

    shared_t = np.concatenate(all_shared_t, axis=0)
    shared_v = np.concatenate(all_shared_v, axis=0)
    shared_a = np.concatenate(all_shared_a, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return shared_t, shared_v, shared_a, labels


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
    
    # Each integer from -3 to +3 maps to a unique bin index (0–6)
    # This mirrors metricsTop.__multiclass_acc used for Acc_7
    bin_map = {-3: 0, -2: 1, -1: 2, 0: 3, 1: 4, 2: 5, 3: 6}
    
    bin_ids = np.array([bin_map[r] for r in rounded])
    
    return bin_ids


def plot_tsne(shared_t, shared_v, shared_a, labels, save_path, perplexity=30, seed=42):
    """
    Create publication-quality t-SNE visualization of the shared feature space.
    
    Layout: Single plot showing all three modalities with:
      - Color = sentiment intensity bin (7-class, matching Acc_7)
      - Marker shape = modality (Text / Audio / Video)
    """
    from sklearn.manifold import TSNE
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    
    # ── Publication rcParams ──────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'mathtext.fontset': 'dejavusans',
    })
    
    print("Running t-SNE dimensionality reduction...")
    
    N = shared_t.shape[0]
    
    # Stack all modality features together: [3N, hidden_size]
    all_features = np.concatenate([shared_t, shared_v, shared_a], axis=0)
    modality_ids = np.array([0]*N + [1]*N + [2]*N)  # 0=Text, 1=Video, 2=Audio
    all_labels = np.concatenate([labels, labels, labels], axis=0)
    
    # Run t-SNE on the combined features
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        max_iter=2000,
    )
    embeddings = tsne.fit_transform(all_features)
    print(f"t-SNE completed. KL divergence: {tsne.kl_divergence_:.4f}")
    
    # Create sentiment bins (7-class)
    bin_ids = create_sentiment_bins(all_labels)
    
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
    
    modality_markers = {
        0: 'o',    # Text — circle
        1: 's',    # Video — square
        2: '^',    # Audio — triangle
    }
    modality_names = {0: 'Text', 1: 'Video', 2: 'Audio'}
    
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
    
    # ── Plot each modality × sentiment bin combination ────────────────
    for mod_id in [0, 1, 2]:
        for sent_id in sorted(sentiment_colors.keys()):
            mask = (modality_ids == mod_id) & (bin_ids == sent_id)
            if mask.sum() == 0:
                continue
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=sentiment_colors[sent_id],
                marker=modality_markers[mod_id],
                s=22,
                alpha=0.70,
                edgecolors='white',
                linewidths=0.25,
                zorder=3,
                rasterized=True,
            )
    
    # ── Legends ───────────────────────────────────────────────────────
    # Sentiment legend (colors) — 7 entries
    color_handles = [
        mpatches.Patch(
            facecolor=sentiment_colors[sid],
            edgecolor='#888888',
            linewidth=0.6,
            label=sentiment_names[sid],
        )
        for sid in sorted(sentiment_colors.keys())
    ]
    legend1 = ax.legend(
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
    legend1.get_title().set_fontweight('bold')
    ax.add_artist(legend1)
    
    # Modality legend (markers)
    marker_handles = [
        Line2D(
            [0], [0],
            marker=modality_markers[mid],
            color='w',
            markerfacecolor='#555555',
            markeredgecolor='#888888',
            markersize=7,
            label=modality_names[mid],
            linewidth=0,
        )
        for mid in sorted(modality_markers.keys())
    ]
    legend2 = ax.legend(
        handles=marker_handles,
        title='Modality',
        loc='upper right',
        fontsize=7.5,
        title_fontsize=8.5,
        framealpha=0.92,
        edgecolor='#bbbbbb',
        borderpad=0.6,
        handlelength=1.0,
        labelspacing=0.35,
    )
    legend2.get_title().set_fontweight('bold')
    ax.add_artist(legend1)  # re-add legend1 after overwrite
    
    # ── Title & axis labels ───────────────────────────────────────────
    ax.set_title(
        't-SNE of DP Shared Feature Space (Modality-Invariant Encoder)\n'
        'Shared Encoder (encoder_c) on CMU-MOSI Test Set',
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
    # Quantitative analysis: cross-modality misalignment
    # =============================================
    # Local bin-name lookup for the 7-class scheme
    _bin_names = {
        0: '-3 (VStrNeg)', 1: '-2 (StrNeg)', 2: '-1 (Neg)',
        3: ' 0 (Neutral)', 4: '+1 (Pos)',    5: '+2 (StrPos)',
        6: '+3 (VStrPos)',
    }
    
    print("\n" + "="*70)
    print("QUANTITATIVE ANALYSIS: Sentiment Conditional Alignment Quality")
    print("="*70)
    
    emb_t = embeddings[:N]
    emb_v = embeddings[N:2*N]
    emb_a = embeddings[2*N:]
    sample_bin_ids = create_sentiment_bins(labels)
    
    unique_bins = np.unique(sample_bin_ids)
    
    print(f"\n{'Sentiment Bin':<20} {'N':>5} {'Text↔Video':>12} {'Text↔Audio':>12} {'Video↔Audio':>12}")
    print("-" * 65)
    
    for bid in unique_bins:
        mask = sample_bin_ids == bid
        n_samples = mask.sum()
        if n_samples < 2:
            continue
        
        # Mean distance between paired modality samples within same bin
        # (same utterance, same label)
        d_tv = np.mean(np.linalg.norm(emb_t[mask] - emb_v[mask], axis=1))
        d_ta = np.mean(np.linalg.norm(emb_t[mask] - emb_a[mask], axis=1))
        d_va = np.mean(np.linalg.norm(emb_v[mask] - emb_a[mask], axis=1))
        
        print(f"{_bin_names[bid]:<20} {n_samples:>5} {d_tv:>12.2f} {d_ta:>12.2f} {d_va:>12.2f}")
    
    # Cross-bin analysis: how close are DIFFERENT sentiment samples across modalities?
    print(f"\n{'Cross-Sentiment Centroid Distance Matrix (Text↔Video pairs)':}")
    print("-" * 60)
    
    centroids_t = {}
    centroids_v = {}
    for bid in unique_bins:
        mask = sample_bin_ids == bid
        if mask.sum() >= 2:
            centroids_t[bid] = emb_t[mask].mean(axis=0)
            centroids_v[bid] = emb_v[mask].mean(axis=0)
    
    bid_list = sorted(centroids_t.keys())
    header = f"{'':>20}" + "".join(f"{'V-'+_bin_names[b][:8]:>14}" for b in bid_list)
    print(header)
    for bt in bid_list:
        row = f"{'T-'+_bin_names[bt][:12]:>20}"
        for bv in bid_list:
            dist = np.linalg.norm(centroids_t[bt] - centroids_v[bv])
            marker = " ◄" if bt != bv and dist < np.linalg.norm(centroids_t[bt] - centroids_v[bt]) else ""
            row += f"{dist:>11.2f}{marker:>3}"
        print(row)
    
    print("\n◄ = Cross-sentiment centroid distance SMALLER than same-sentiment distance")
    print("    (indicates negative transfer: wrong-label features pulled closer)")
    
    # Also report the raw feature space distances (not t-SNE) for robustness
    print("\n" + "="*70)
    print("RAW FEATURE SPACE: Intra-class vs Inter-class Distance (Shared Features)")
    print("="*70)
    
    all_raw = np.concatenate([shared_t, shared_v, shared_a], axis=0)
    all_bin_ids = np.concatenate([sample_bin_ids, sample_bin_ids, sample_bin_ids])
    
    # Compute within-class and between-class average distances
    unique_bins_raw = np.unique(all_bin_ids)
    intra_dists = []
    inter_dists = []
    
    for bid in unique_bins_raw:
        in_class = all_raw[all_bin_ids == bid]
        out_class = all_raw[all_bin_ids != bid]
        
        if len(in_class) < 2 or len(out_class) < 2:
            continue
        
        # Sample for efficiency
        n_sample = min(500, len(in_class))
        idx_in = np.random.choice(len(in_class), n_sample, replace=False)
        idx_out = np.random.choice(len(out_class), n_sample, replace=True)
        
        # Intra-class distance (within same sentiment)
        for i in range(0, n_sample - 1, 2):
            intra_dists.append(np.linalg.norm(in_class[idx_in[i]] - in_class[idx_in[i+1]]))
        
        # Inter-class distance (different sentiment)
        for i in range(n_sample):
            inter_dists.append(np.linalg.norm(in_class[idx_in[i]] - out_class[idx_out[i]]))
    
    mean_intra = np.mean(intra_dists)
    mean_inter = np.mean(inter_dists)
    ratio = mean_inter / (mean_intra + 1e-8)
    
    print(f"\nMean Intra-class Distance : {mean_intra:.4f}")
    print(f"Mean Inter-class Distance : {mean_inter:.4f}")
    print(f"Inter / Intra Ratio       : {ratio:.4f}")
    print(f"\n→ Ratio close to 1.0 indicates poor class separation in shared space")
    print(f"  (shared encoder aligns distributions but may NOT preserve class structure)")


def main():
    parser = argparse.ArgumentParser(description='t-SNE visualization of DP shared space')
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
    
    # Load config — use 'dp' model name instead of 'misa'
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
            # Try to parse as Python literal
            try:
                import ast
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
            setattr(model_args, key, value)
    
    # Load data
    print(f"Loading {args.dataset} dataset from {args.data_dir}...")
    dataloader = MMDataLoader(model_args, num_workers=0)  # num_workers=0 for debugging
    
    # Build model and load checkpoint
    print(f"Building DP model and loading checkpoint from {args.checkpoint}...")
    model = build_model(model_args)
    
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded. Extracting shared features from {args.split} set...")
    
    # Extract features
    shared_t, shared_v, shared_a, labels = extract_shared_features(
        model, dataloader[args.split], device
    )
    
    print(f"Extracted features: T={shared_t.shape}, V={shared_v.shape}, A={shared_a.shape}")
    print(f"Labels range: [{labels.min():.2f}, {labels.max():.2f}], N={len(labels)}")
    
    # Output path
    if args.output:
        save_path = args.output
    else:
        os.makedirs(str(PROJECT_ROOT / 'result' / 'figures'), exist_ok=True)
        save_path = str(PROJECT_ROOT / 'result' / 'figures' / f'tsne_dp_shared_{args.dataset}_{args.split}_{model_args.missing_rate_eval_test:.1f}.png')
    
    # Generate t-SNE plot
    plot_tsne(
        shared_t, shared_v, shared_a, labels,
        save_path=save_path,
        perplexity=args.perplexity,
        seed=args.seed,
    )
    
    # # Save raw features for further analysis
    # np_save_path = save_path.replace('.png', '_features.npz')
    # np.savez(
    #     np_save_path,
    #     shared_t=shared_t,
    #     shared_v=shared_v,
    #     shared_a=shared_a,
    #     labels=labels,
    # )
    # print(f"Raw features saved to: {np_save_path}")


if __name__ == '__main__':
    main()

# python scripts/visualize_tsne_shared_space_dp.py \
#     --checkpoint /home/tien/NCKH/DecAlign/pt/dp-mosi-missing0.0.pth \
#     --config_file config/dp_mosi_config.json \
#     --missing_data \
#     --missing_rate 0.0 \
#     --dataset mosi \
#     --split test