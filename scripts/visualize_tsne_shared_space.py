"""
t-SNE Visualization of MISA Shared Feature Space
==================================================
Demonstrates the "Global Marginal Matching Gap":
  CMD only matches global marginal distributions across modalities,
  ignoring per-class (sentiment-conditional) structure. This can cause
  features with DIFFERENT sentiment labels from DIFFERENT modalities
  to be pulled together — a form of negative transfer.

Usage:
    python scripts/visualize_tsne_shared_space.py \
        --checkpoint /home/tien/NCKH/DecAlign/pt/misa-mosi-missing0.5.pth \
        --dataset mosi \
        --data_dir ./data \
        --config_file config/misa_mosi_config.json \
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
    Forward pass through the MISA model, extracting shared representations
    for each modality (utt_shared_t, utt_shared_v, utt_shared_a).
    
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
            
            # Forward pass — this populates model.utt_shared_*
            _ = model(text, audio, vision, text_m, audio_m, vision_m)
            
            all_shared_t.append(model.utt_shared_t.cpu().numpy())
            all_shared_v.append(model.utt_shared_v.cpu().numpy())
            all_shared_a.append(model.utt_shared_a.cpu().numpy())
            all_labels.append(labels.numpy().reshape(-1))
    
    shared_t = np.concatenate(all_shared_t, axis=0)
    shared_v = np.concatenate(all_shared_v, axis=0)
    shared_a = np.concatenate(all_shared_a, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    return shared_t, shared_v, shared_a, labels


def create_sentiment_bins(labels):
    """
    Bin continuous sentiment scores into discrete categories for visualization.
    MOSI scores range from -3 to 3. We use rounded integer bins as in acc computation.
    Then group into broader categories for clearer visualization.
    
    Returns:
        bin_labels: array of string labels per sample
        bin_ids: array of integer bin indices per sample
    """
    rounded = np.round(labels).astype(int)
    rounded = np.clip(rounded, -3, 3)
    
    # Map to broader sentiment categories for visual clarity
    # Strong Negative: [-3, -2], Negative: [-1], Neutral: [0], 
    # Positive: [1], Strong Positive: [2, 3]
    bin_labels = []
    bin_ids = []
    for r in rounded:
        if r <= -2:
            bin_labels.append("Strong Neg (-3,-2)")
            bin_ids.append(0)
        elif r == -1:
            bin_labels.append("Negative (-1)")
            bin_ids.append(1)
        elif r == 0:
            bin_labels.append("Neutral (0)")
            bin_ids.append(2)
        elif r == 1:
            bin_labels.append("Positive (1)")
            bin_ids.append(3)
        else:  # r >= 2
            bin_labels.append("Strong Pos (2,3)")
            bin_ids.append(4)
    
    return np.array(bin_labels), np.array(bin_ids)


def plot_tsne(shared_t, shared_v, shared_a, labels, save_path, perplexity=30, seed=42):
    """
    Create publication-quality t-SNE visualization of the shared feature space.
    
    Layout: Single plot showing all three modalities with:
      - Color = sentiment intensity bin  
      - Marker shape = modality (Text / Audio / Video)
    
    This reveals the Global Marginal Matching Gap: CMD aligns marginal
    distributions but features with different sentiments get entangled.
    """
    from sklearn.manifold import TSNE
    import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    
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
        learning_rate='auto',
        init='pca',
    )
    embeddings = tsne.fit_transform(all_features)
    print(f"t-SNE completed. KL divergence: {tsne.kl_divergence_:.4f}")
    
    # Create sentiment bins
    bin_labels, bin_ids = create_sentiment_bins(all_labels)
    
    # =============================================
    # Publication-quality plot
    # =============================================
    
    # Color palette — sequential diverging to match sentiment polarity
    # Strong Neg → Neg → Neutral → Pos → Strong Pos
    sentiment_colors = {
        0: '#1b4f72',   # Strong Negative — deep navy blue
        1: '#5dade2',   # Negative — sky blue
        2: '#7d8c8e',   # Neutral — muted gray
        3: '#e74c3c',   # Positive — warm red
        4: '#922b21',   # Strong Positive — dark crimson
    }
    
    sentiment_names = {
        0: 'Strong Neg (≤-2)',
        1: 'Negative (-1)',
        2: 'Neutral (0)',
        3: 'Positive (+1)',
        4: 'Strong Pos (≥+2)',
    }
    
    modality_markers = {
        0: 'o',    # Text — circle
        1: 's',    # Video — square
        2: '^',    # Audio — triangle
    }
    modality_names = {0: 'Text', 1: 'Video', 2: 'Audio'}
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')
    
    # Plot each modality × sentiment bin combination
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
                s=28,
                alpha=0.65,
                edgecolors='white',
                linewidths=0.3,
                zorder=3,
            )
    
    # ---- Legends ----
    # Sentiment legend (colors)
    color_handles = [
        mpatches.Patch(
            facecolor=sentiment_colors[sid],
            edgecolor='gray',
            linewidth=0.5,
            label=sentiment_names[sid]
        )
        for sid in sorted(sentiment_colors.keys())
    ]
    legend1 = ax.legend(
        handles=color_handles,
        title='Sentiment Intensity',
        loc='upper left',
        fontsize=9,
        title_fontsize=10,
        framealpha=0.9,
        edgecolor='#cccccc',
    )
    ax.add_artist(legend1)
    
    # Modality legend (markers)
    marker_handles = [
        Line2D(
            [0], [0],
            marker=modality_markers[mid],
            color='w',
            markerfacecolor='#555555',
            markeredgecolor='gray',
            markersize=8,
            label=modality_names[mid],
            linewidth=0,
        )
        for mid in sorted(modality_markers.keys())
    ]
    ax.legend(
        handles=marker_handles,
        title='Modality',
        loc='upper right',
        fontsize=9,
        title_fontsize=10,
        framealpha=0.9,
        edgecolor='#cccccc',
    )
    ax.add_artist(legend1)  # re-add legend1 after overwrite
    
    ax.set_title(
        't-SNE of MISA Shared Feature Space (CMD — Unconditional Alignment)\n'
        'Modality-Invariant Shared Encoder on CMU-MOSI Test Set',
        fontsize=13,
        fontweight='bold',
        pad=14,
    )
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.tick_params(labelsize=9)
    
    # Light grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    # plt.show()
    print(f"Saved t-SNE plot to: {save_path}")
    plt.close()
    
    # =============================================
    # Quantitative analysis: cross-modality misalignment
    # =============================================
    print("\n" + "="*70)
    print("QUANTITATIVE ANALYSIS: Sentiment Conditional Alignment Quality")
    print("="*70)
    
    emb_t = embeddings[:N]
    emb_v = embeddings[N:2*N]
    emb_a = embeddings[2*N:]
    _, sample_bin_ids = create_sentiment_bins(labels)
    
    unique_bins = np.unique(sample_bin_ids)
    
    print(f"\n{'Sentiment Bin':<25} {'N':>5} {'Text↔Video':>12} {'Text↔Audio':>12} {'Video↔Audio':>12}")
    print("-" * 70)
    
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
        
        print(f"{sentiment_names[bid]:<25} {n_samples:>5} {d_tv:>12.2f} {d_ta:>12.2f} {d_va:>12.2f}")
    
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
    header = f"{'':>25}" + "".join(f"{'Video-'+sentiment_names[b][:8]:>16}" for b in bid_list)
    print(header)
    for bt in bid_list:
        row = f"{'Text-'+sentiment_names[bt][:12]:>25}"
        for bv in bid_list:
            dist = np.linalg.norm(centroids_t[bt] - centroids_v[bv])
            marker = " ◄" if bt != bv and dist < np.linalg.norm(centroids_t[bt] - centroids_v[bt]) else ""
            row += f"{dist:>13.2f}{marker:>3}"
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
    print(f"  (CMD aligns distributions globally but does NOT preserve class structure)")


def main():
    parser = argparse.ArgumentParser(description='t-SNE visualization of MISA shared space')
    parser.add_argument('--checkpoint', type=str, default='pt/misa-mosi.pth',
                        help='Path to trained MISA model checkpoint')
    parser.add_argument('--dataset', type=str, default='mosi', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--config_file', type=str, default='', help='Config JSON path')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'],
                        help='Data split to visualize')
    parser.add_argument('--perplexity', type=float, default=30, help='t-SNE perplexity')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='', help='Output image path')
    parser.add_argument('--config_override', action='append', default=[],
                        help='Override config values, e.g. hidden_size=128')
    parser.add_argument('--missing_data', action='store_true', help='Use missing data')
    parser.add_argument('--missing_rate', type=float, default=0.7, help='Missing rate')
    args = parser.parse_args()
    
    setup_seed(args.seed)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    config_file = args.config_file if args.config_file else ''
    model_args = get_config_regression('misa', args.dataset, config_file, data_dir=args.data_dir)
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
    print(f"Building MISA model and loading checkpoint from {args.checkpoint}...")
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
        save_path = str(PROJECT_ROOT / 'result' / 'figures' / f'tsne_misa_shared_{args.dataset}_{args.split}_{model_args.missing_rate_eval_test:.1f}.png')
    
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

# python scripts/visualize_tsne_shared_space.py \
#     --checkpoint /home/tien/NCKH/DecAlign/pt/misa-mosi-missing0.0.pth \
#     --config_file config/misa_mosi_config.json \
#     --dataset mosi \
#     --missing_data \
#     --missing_rate 0.0 \
#     --split test