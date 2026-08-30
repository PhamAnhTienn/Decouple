"""
t-SNE Visualization of DP Shared Feature Space WITH Temporal Prototypes
========================================================================
Visualizes the modality-invariant features (c_l, c_v, c_a) together with
the learned temporal prototypes (proto_l, proto_v, proto_a) from the DP model.

Prototypes are displayed as star (☆) markers colored by their sentiment class,
matching the style of Figure 3 in the reference paper.

Usage:
    python scripts/visualize_tsne_shared_space_dp_proto.py \
        --checkpoint /home/tien/NCKH/DecAlign/pt/dp-mosi-missing0.0.pth \
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
            labels = batch_data['labels']['M']

            output = model(text, audio, vision, is_distill=False)

            # repr_l/v/a_homo: (batch, d) — flattened common features
            c_l = output['repr_l_homo']
            c_v = output['repr_v_homo']
            c_a = output['repr_a_homo']

            all_shared_t.append(c_l.cpu().numpy())
            all_shared_v.append(c_v.cpu().numpy())
            all_shared_a.append(c_a.cpu().numpy())
            all_labels.append(labels.numpy().reshape(-1))

    shared_t = np.concatenate(all_shared_t, axis=0)
    shared_v = np.concatenate(all_shared_v, axis=0)
    shared_a = np.concatenate(all_shared_a, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return shared_t, shared_v, shared_a, labels


def extract_prototypes(model):
    """
    Extract temporal prototypes from the DP model and compute their
    temporal mean to get a single vector per class per modality.

    proto_l/v/a shape: (7, T_m, D) → temporal mean → (7, D)

    Returns:
        proto_l, proto_v, proto_a: numpy arrays [7, D]
        valid_mask: boolean array [7] — True if prototype is initialized
    """
    proto_l = model.proto_l.detach().cpu()  # (7, T_l, D)
    proto_v = model.proto_v.detach().cpu()  # (7, T_v, D)
    proto_a = model.proto_a.detach().cpu()  # (7, T_a, D)

    # Temporal mean: (7, T_m, D) → (7, D)
    mu_l = proto_l.mean(dim=1).numpy()
    mu_v = proto_v.mean(dim=1).numpy()
    mu_a = proto_a.mean(dim=1).numpy()

    # Valid prototypes = non-zero
    valid_mask = np.array([
        (np.abs(mu_l[c]).sum() > 0) and
        (np.abs(mu_v[c]).sum() > 0) and
        (np.abs(mu_a[c]).sum() > 0)
        for c in range(7)
    ])

    return mu_l, mu_v, mu_a, valid_mask


def create_sentiment_bins(labels):
    """
    Bin continuous sentiment scores into 7 discrete categories,
    matching the Acc_7 metric: clip to [-3,3], round to int → bins 0–6.
    """
    rounded = np.round(labels).astype(int)
    rounded = np.clip(rounded, -3, 3)
    bin_map = {-3: 0, -2: 1, -1: 2, 0: 3, 1: 4, 2: 5, 3: 6}
    return np.array([bin_map[r] for r in rounded])


def plot_tsne_with_prototypes(shared_t, shared_v, shared_a, labels,
                               proto_l, proto_v, proto_a, valid_mask,
                               save_path, perplexity=30, seed=42):
    """
    Publication-quality t-SNE visualization with 3 per-modality subplots.

    Layout: 1 row × 3 columns (Text | Video | Audio).
    Each subplot shows:
      - That modality's shared features, colored by 7-class sentiment
      - That modality's temporal prototypes as star (☆) markers
      - A shared bottom legend: Features, Class 1–7, Prototypes

    t-SNE is computed jointly across all modalities + prototypes so the
    spatial layout is consistent across panels.
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'mathtext.fontset': 'dejavusans',
    })

    print("Running t-SNE dimensionality reduction...")

    N = shared_t.shape[0]
    valid_classes = np.where(valid_mask)[0]
    K = len(valid_classes)

    # Build prototype arrays for valid classes only
    protos_l_valid = proto_l[valid_classes]  # (K, D)
    protos_v_valid = proto_v[valid_classes]
    protos_a_valid = proto_a[valid_classes]

    # Stack: [N_text, N_video, N_audio, K_proto_l, K_proto_v, K_proto_a]
    all_features = np.concatenate([
        shared_t, shared_v, shared_a,
        protos_l_valid, protos_v_valid, protos_a_valid,
    ], axis=0)

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(all_features) - 1),
        random_state=seed,
        max_iter=2000,
    )
    embeddings_all = tsne.fit_transform(all_features)
    print(f"t-SNE completed. KL divergence: {tsne.kl_divergence_:.4f}")

    # Split embeddings: features per modality + prototypes per modality
    emb_t = embeddings_all[:N]
    emb_v = embeddings_all[N:2*N]
    emb_a = embeddings_all[2*N:3*N]
    emb_proto_l = embeddings_all[3*N:3*N+K]
    emb_proto_v = embeddings_all[3*N+K:3*N+2*K]
    emb_proto_a = embeddings_all[3*N+2*K:]

    bin_ids = create_sentiment_bins(labels)

    # ── Color palette (7 classes) — vibrant, paper-style ──
    sentiment_colors = {
        0: '#7b2d8e',   # Class 1 — purple
        1: '#2171b5',   # Class 2 — blue
        2: '#17becf',   # Class 3 — cyan
        3: '#78c9a0',   # Class 4 — teal/mint
        4: '#f0e442',   # Class 5 — yellow
        5: '#f5a623',   # Class 6 — orange
        6: '#e45756',   # Class 7 — red
    }

    class_names = {
        0: 'Class 1', 1: 'Class 2', 2: 'Class 3', 3: 'Class 4',
        4: 'Class 5', 5: 'Class 6', 6: 'Class 7',
    }

    # Per-modality data: (features_emb, protos_emb, title, suffix)
    modality_data = [
        (emb_t, emb_proto_l, 'Text',  'text'),
        (emb_v, emb_proto_v, 'Video', 'video'),
        (emb_a, emb_proto_a, 'Audio', 'audio'),
    ]

    # Shared axis limits for consistent spatial layout across images
    all_emb = embeddings_all[:, :2]
    x_min, x_max = all_emb[:, 0].min(), all_emb[:, 0].max()
    y_min, y_max = all_emb[:, 1].min(), all_emb[:, 1].max()
    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.08

    # Build legend handles (shared across all figures)
    legend_handles = []
    legend_handles.append(
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#aaaaaa', markeredgecolor='#aaaaaa',
               markersize=7, linewidth=0, label='Features')
    )
    for sid in sorted(sentiment_colors.keys()):
        legend_handles.append(
            mpatches.Patch(
                facecolor=sentiment_colors[sid],
                edgecolor='none',
                label=class_names[sid],
            )
        )
    legend_handles.append(
        Line2D([0], [0], marker='*', color='w',
               markerfacecolor='#888888', markeredgecolor='black',
               markersize=12, linewidth=0, label='Prototypes')
    )

    # Derive base path for per-modality filenames
    base, ext = os.path.splitext(save_path)

    # ── Save one image per modality ──
    for emb_feat, emb_proto, mod_title, mod_suffix in modality_data:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # Plot features colored by sentiment class
        for sent_id in sorted(sentiment_colors.keys()):
            mask = bin_ids == sent_id
            if mask.sum() == 0:
                continue
            ax.scatter(
                emb_feat[mask, 0], emb_feat[mask, 1],
                c=sentiment_colors[sent_id],
                marker='o',
                s=18,
                alpha=0.65,
                edgecolors='none',
                zorder=2,
                rasterized=True,
            )

        # Plot prototypes as stars
        for idx, c in enumerate(valid_classes):
            ax.scatter(
                emb_proto[idx, 0], emb_proto[idx, 1],
                marker='*', s=450,
                c=sentiment_colors[c],
                edgecolors='black', linewidths=1.2,
                zorder=10,
            )

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        ax.set_title(
            f't-SNE — {mod_title}-modality Features \u0026 Prototypes',
            fontsize=13, fontweight='bold', pad=12,
        )

        ax.legend(
            handles=legend_handles,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.08),
            ncol=len(legend_handles),
            fontsize=8,
            frameon=False,
            handletextpad=0.3,
            columnspacing=0.7,
        )

        out_path = f"{base}_{mod_suffix}{ext}"
        plt.tight_layout(pad=1.0)
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved {mod_title} plot to: {out_path}")
        plt.close()

    # ── Quantitative analysis ──
    _bin_names = {
        0: '-3 (VStrNeg)', 1: '-2 (StrNeg)', 2: '-1 (Neg)',
        3: ' 0 (Neutral)', 4: '+1 (Pos)',    5: '+2 (StrPos)',
        6: '+3 (VStrPos)',
    }

    print("\n" + "="*70)
    print("QUANTITATIVE ANALYSIS: Sentiment Conditional Alignment Quality")
    print("="*70)

    sample_bin_ids = create_sentiment_bins(labels)

    unique_bins = np.unique(sample_bin_ids)

    print(f"\n{'Sentiment Bin':<20} {'N':>5} {'Text↔Video':>12} {'Text↔Audio':>12} {'Video↔Audio':>12}")
    print("-" * 65)

    for bid in unique_bins:
        mask = sample_bin_ids == bid
        n_samples = mask.sum()
        if n_samples < 2:
            continue
        d_tv = np.mean(np.linalg.norm(emb_t[mask] - emb_v[mask], axis=1))
        d_ta = np.mean(np.linalg.norm(emb_t[mask] - emb_a[mask], axis=1))
        d_va = np.mean(np.linalg.norm(emb_v[mask] - emb_a[mask], axis=1))
        print(f"{_bin_names[bid]:<20} {n_samples:>5} {d_tv:>12.2f} {d_ta:>12.2f} {d_va:>12.2f}")

    # Prototype-to-feature distance analysis
    print("\n" + "="*70)
    print("PROTOTYPE ANALYSIS: Mean Distance from Features to Class Prototype")
    print("="*70)

    for mod_name, emb_mod, emb_proto_mod in [
        ('Text',  emb_t, emb_proto_l),
        ('Video', emb_v, emb_proto_v),
        ('Audio', emb_a, emb_proto_a),
    ]:
        print(f"\n  {mod_name}:")
        for idx, c in enumerate(valid_classes):
            mask = sample_bin_ids == c
            if mask.sum() < 2:
                continue
            dist = np.mean(np.linalg.norm(emb_mod[mask] - emb_proto_mod[idx], axis=1))
            print(f"    Class {c} ({_bin_names[c]:<15}): mean dist = {dist:.2f}  (N={mask.sum()})")

    # Raw feature space stats
    print("\n" + "="*70)
    print("RAW FEATURE SPACE: Intra-class vs Inter-class Distance")
    print("="*70)

    all_raw = np.concatenate([shared_t, shared_v, shared_a], axis=0)
    all_bin_ids = np.concatenate([sample_bin_ids, sample_bin_ids, sample_bin_ids])
    unique_bins_raw = np.unique(all_bin_ids)
    intra_dists = []
    inter_dists = []

    for bid in unique_bins_raw:
        in_class = all_raw[all_bin_ids == bid]
        out_class = all_raw[all_bin_ids != bid]
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


def main():
    parser = argparse.ArgumentParser(
        description='t-SNE visualization of DP shared space with temporal prototypes')
    parser.add_argument('--checkpoint', type=str, default='pt/dp-mosi.pth')
    parser.add_argument('--dataset', type=str, default='mosi')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--config_file', type=str, default='')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'valid', 'test'])
    parser.add_argument('--missing_data', action='store_true')
    parser.add_argument('--missing_rate', type=float, default=0.0)
    parser.add_argument('--perplexity', type=float, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--config_override', action='append', default=[],
                        help='Override config values, e.g. hidden_size=128')
    args = parser.parse_args()

    setup_seed(args.seed)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config
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

    # Check if prototypes are initialized
    if hasattr(model, 'proto_initialized') and model.proto_initialized.item():
        print("✓ Temporal prototypes are initialized in this checkpoint.")
    else:
        print("⚠ WARNING: Prototypes may not be initialized in this checkpoint!")
        print("  The prototype stars may appear at origin (0,0). Ensure the checkpoint")
        print("  was trained for ≥2 epochs with prototype caching enabled.")

    # Extract features
    print(f"Extracting shared features from {args.split} set...")
    shared_t, shared_v, shared_a, labels = extract_shared_features(
        model, dataloader[args.split], device
    )
    print(f"Extracted features: T={shared_t.shape}, V={shared_v.shape}, A={shared_a.shape}")
    print(f"Labels range: [{labels.min():.2f}, {labels.max():.2f}], N={len(labels)}")

    # Extract prototypes
    print("Extracting temporal prototypes...")
    proto_l, proto_v, proto_a, valid_mask = extract_prototypes(model)
    print(f"Valid prototypes: {valid_mask.sum()}/7 classes — {np.where(valid_mask)[0].tolist()}")

    # Output path
    if args.output:
        save_path = args.output
    else:
        os.makedirs(str(PROJECT_ROOT / 'result' / 'figures'), exist_ok=True)
        save_path = str(PROJECT_ROOT / 'result' / 'figures' /
                        f'tsne_dp_shared_proto_{args.dataset}_{args.split}_{model_args.missing_rate_eval_test:.1f}.png')

    # Generate t-SNE plot
    plot_tsne_with_prototypes(
        shared_t, shared_v, shared_a, labels,
        proto_l, proto_v, proto_a, valid_mask,
        save_path=save_path,
        perplexity=args.perplexity,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()

# python scripts/visualize_tsne_shared_space_dp_proto.py \
#     --checkpoint /home/tien/NCKH/DecAlign/pt/dp-mosi-missing0.0.pth \
#     --config_file config/dp_mosi_config.json \
#     --missing_data \
#     --missing_rate 0.0 \
#     --dataset mosi \
#     --split test
