"""
t-SNE Across Datasets (CVPR Paper Style)
=========================================
Plot t-SNE of the final representation for **one model** across multiple
datasets (MOSI, MOSEI, SIMS) in a grid layout.

Layout mirrors the CVPR reference figure 2: 2×2 or Nx1 grid of subplots,
each subplot = one dataset.  Colour = continuous sentiment intensity
(Negative → Positive diverging colourbar) with per-dataset value range.

Usage examples:
    # Plot DP model across 3 datasets at missing_rate=0.5
    python scripts/tsne_across_datasets.py \\
        --model_name dp --split test \\
        --missing_rate 0.5 \\
        --datasets \\
            "MOSI | pt/dp-mosi-missing0.5.pth | config/dp_mosi_config.json | mosi | ./data" \\
            "MOSEI | pt/dp-mosei-missing0.5.pth | config/dp_mosei_config.json | mosei | ./data" \\
            "CH-SIMS | pt/dp-sims-missing0.5.pth | config/dp_sims_config.json | sims | ./data"

    # Plot DecAlign across datasets (no missing)
    python scripts/tsne_across_datasets.py \\
        --model_name decalign --split test \\
        --datasets \\
            "MOSI | pt/decalign-mosi-missing0.0.pth | config/dec_mosi_config.json | mosi | ./data" \\
            "MOSEI | pt/decalign-mosei-missing0.0.pth | config/dec_mosei_config.json | mosei | ./data" \\
            "SIMS | pt/decalign-sims-missing0.0.pth | config/dec_sims_config.json | sims | ./data"

Each --datasets value is  "title | checkpoint | config_file | dataset_name | data_dir"
"""

import argparse
import math
import os
import sys
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

# ── Paper style ──────────────────────────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "experiments"))
from paper_style import apply_style, save_fig  # noqa: E402

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


# =====================================================================
# Feature extraction helpers (same as tsne_compare_models.py)
# =====================================================================

def _extract_dp(model, dataloader, device, use_missing):
    """Extract last_hs_proj from DP model (6-input forward)."""
    model.eval()
    all_reprs, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            vision = batch['vision'].to(device)
            labels = batch['labels']['M']

            if use_missing and 'text_m' in batch:
                text_m = batch['text_m'].to(device)
                audio_m = batch['audio_m'].to(device)
                vision_m = batch['vision_m'].to(device)
                out = model(text_m, audio_m, vision_m, text, audio, vision)
            else:
                out = model(text, audio, vision, text, audio, vision)

            all_reprs.append(out['last_hs_proj'].cpu().numpy())
            all_labels.append(labels.numpy().reshape(-1))

    return np.concatenate(all_reprs), np.concatenate(all_labels)


def _extract_dmd(model, dataloader, device, use_missing):
    """Extract final fused repr from DMD model (3-input forward)."""
    model.eval()
    all_reprs, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch['labels']['M']
            if use_missing and 'text_m' in batch:
                text = batch['text_m'].to(device)
                audio = batch['audio_m'].to(device)
                vision = batch['vision_m'].to(device)
            else:
                text = batch['text'].to(device)
                audio = batch['audio'].to(device)
                vision = batch['vision'].to(device)

            out = model(text, audio, vision, is_distill=False)

            last_h_l = out['last_h_l']
            last_h_v = out['last_h_v']
            last_h_a = out['last_h_a']

            c_l = out['c_l']; c_v = out['c_v']; c_a = out['c_a']
            c_l_att = model.self_attentions_c_l(c_l)
            c_l_att = c_l_att[0][-1] if isinstance(c_l_att, tuple) else c_l_att[-1]
            c_v_att = model.self_attentions_c_v(c_v)
            c_v_att = c_v_att[0][-1] if isinstance(c_v_att, tuple) else c_v_att[-1]
            c_a_att = model.self_attentions_c_a(c_a)
            c_a_att = c_a_att[0][-1] if isinstance(c_a_att, tuple) else c_a_att[-1]
            c_fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)

            last_h_l = torch.sigmoid(model.weight_l(last_h_l))
            last_h_v = torch.sigmoid(model.weight_v(last_h_v))
            last_h_a = torch.sigmoid(model.weight_a(last_h_a))
            c_fusion = torch.sigmoid(model.weight_c(c_fusion))
            last_hs = torch.cat([last_h_l, last_h_v, last_h_a, c_fusion], dim=1)
            last_hs_proj = model.proj2(
                F.dropout(F.relu(model.proj1(last_hs), inplace=False),
                          p=model.output_dropout, training=False))
            last_hs_proj = last_hs_proj + last_hs

            all_reprs.append(last_hs_proj.cpu().numpy())
            all_labels.append(labels.numpy().reshape(-1))

    return np.concatenate(all_reprs), np.concatenate(all_labels)


def _extract_decalign(model, dataloader, device, use_missing):
    """Extract final_rep from DecAlign model (3-input forward)."""
    model.eval()
    all_reprs, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch['labels']['M']
            if use_missing and 'text_m' in batch:
                text = batch['text_m'].to(device)
                audio = batch['audio_m'].to(device)
                vision = batch['vision_m'].to(device)
            else:
                text = batch['text'].to(device)
                audio = batch['audio'].to(device)
                vision = batch['vision'].to(device)

            out = model(text, audio, vision, is_distill=False)
            all_reprs.append(out['final_rep'].cpu().numpy())
            all_labels.append(labels.numpy().reshape(-1))

    return np.concatenate(all_reprs), np.concatenate(all_labels)


def _extract_misa(model, dataloader, device, use_missing):
    """Extract fused representation from MISA model."""
    model.eval()
    all_reprs, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch['labels']['M']
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            vision = batch['vision'].to(device)

            text_m = batch.get('text_m')
            audio_m = batch.get('audio_m')
            vision_m = batch.get('vision_m')

            if use_missing and text_m is not None:
                text_m = text_m.to(device)
                audio_m = audio_m.to(device)
                vision_m = vision_m.to(device)
                model.alignment(text, audio, vision, text_m, audio_m, vision_m)
            else:
                model.alignment(text, audio, vision)

            h = torch.stack((
                model.utt_private_t, model.utt_private_v, model.utt_private_a,
                model.utt_shared_t, model.utt_shared_v, model.utt_shared_a,
            ), dim=0)
            h = model.transformer_encoder(h)
            h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)

            all_reprs.append(h.cpu().numpy())
            all_labels.append(labels.numpy().reshape(-1))

    return np.concatenate(all_reprs), np.concatenate(all_labels)


EXTRACTOR_MAP = {
    'dp':       _extract_dp,
    'dmd':      _extract_dmd,
    'decalign': _extract_decalign,
    'misa':     _extract_misa,
}


# =====================================================================
# Plotting
# =====================================================================

def _get_label_range(dataset_name):
    """Return (vmin, vmax) for the sentiment colorbar."""
    ranges = {
        'mosi':  (-3, 3),
        'mosei': (-3, 3),
        'sims':  (-1, 1),
    }
    return ranges.get(dataset_name, (-3, 3))


def _make_cmap():
    """Diverging blue-white-red colourmap for sentiment."""
    return mcolors.LinearSegmentedColormap.from_list(
        "sentiment",
        ["#08306b", "#2171b5", "#6baed6", "#bdd7e7",
         "#f7f7f7",
         "#fcbba1", "#fb6a4a", "#cb181d", "#67000d"],
        N=256,
    )


def plot_tsne_across_datasets(panel_data, save_stem, perplexity=30, seed=42):
    """
    Create a CVPR-style grid t-SNE figure showing one model on multiple datasets.

    Parameters
    ----------
    panel_data : list[dict]
        Each dict has keys: 'title', 'dataset_name', 'features' (N,D),
        'labels' (N,)
    save_stem : str | Path
    """
    from sklearn.manifold import TSNE

    n = len(panel_data)
    ncols = min(n, 2) if n > 1 else 1
    nrows = math.ceil(n / ncols)

    # ── Apply paper style ─────────────────────────────────────────────
    apply_style("cvpr", "wide")
    cmap = _make_cmap()

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6.875, 6.875 / ncols * 0.95 * nrows + 0.5),
        layout="constrained",
        squeeze=False,
    )
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.06, hspace=0.12)

    for idx, panel in enumerate(panel_data):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        feats = panel['features']
        labels = panel['labels']
        ds_name = panel['dataset_name']
        title = panel['title']
        vmin, vmax = _get_label_range(ds_name)

        print(f"  [{idx+1}/{n}] t-SNE for '{title}' "
              f"({feats.shape[0]} samples, dim={feats.shape[1]})...")

        tsne = TSNE(n_components=2, perplexity=perplexity,
                    random_state=seed, max_iter=2000)
        emb = tsne.fit_transform(feats)
        print(f"    KL divergence: {tsne.kl_divergence_:.4f}")

        sc = ax.scatter(
            emb[:, 0], emb[:, 1],
            c=labels, cmap=cmap, vmin=vmin, vmax=vmax,
            s=8, alpha=0.75,
            edgecolors='white', linewidths=0.15,
            rasterized=True,
        )

        ax.set_xlabel('t-SNE Dimension 1', fontsize=7)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=7)
        ax.tick_params(labelsize=6)

        # ── Per-panel colourbar ───────────────────────────────────────
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.04)
        cb = fig.colorbar(sc, cax=cax)
        cb.ax.tick_params(labelsize=5.5)

        # Positive ↑ / Negative ↓ labels on the colourbar
        cb.ax.text(0.5, 1.02, r'Positive $\uparrow$', transform=cb.ax.transAxes,
                   ha='center', va='bottom', fontsize=5.5, color='#cb181d',
                   fontweight='bold')
        cb.ax.text(0.5, -0.02, r'Negative $\downarrow$', transform=cb.ax.transAxes,
                   ha='center', va='top', fontsize=5.5, color='#08306b',
                   fontweight='bold')
        cb.set_label('Sentiment\nIntensity', fontsize=6, labelpad=2)

        # Title as subplot label: (a) MOSI, (b) MOSEI, etc.
        panel_letter = chr(ord('a') + idx)
        ax.set_title(f't-SNE Visualization for {title} Dataset',
                     fontsize=7.5, fontweight='bold', pad=6)
        # Also add panel letter below
        ax.text(0.5, -0.15, f'({panel_letter}) {title}',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=7.5, fontweight='bold')

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # ── Save ──────────────────────────────────────────────────────────
    save_stem = Path(save_stem)
    save_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_stem.with_suffix(".pdf"), dpi=600)
    fig.savefig(save_stem.with_suffix(".png"), dpi=600)
    print(f"Saved → {save_stem}.pdf / .png")
    plt.close(fig)


# =====================================================================
# CLI
# =====================================================================

def parse_dataset_entry(s):
    """Parse 'title | checkpoint | config_file | dataset_name | data_dir' string."""
    parts = [p.strip() for p in s.split('|')]
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            f"Expected 'title | checkpoint | config | dataset_name | data_dir', got: {s!r}"
        )
    return {
        'title':        parts[0],
        'checkpoint':   parts[1],
        'config':       parts[2],
        'dataset_name': parts[3].lower(),
        'data_dir':     parts[4],
    }


def main():
    parser = argparse.ArgumentParser(
        description='t-SNE across datasets for one model (CVPR paper style)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name (dp / dmd / decalign / misa)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'valid', 'test'])
    parser.add_argument('--missing_rate', type=float, default=0.0,
                        help='Missing rate for test (0.0 = no missing)')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--perplexity', type=float, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='',
                        help='Output path stem (without extension)')
    parser.add_argument('--datasets', nargs='+', type=parse_dataset_entry,
                        required=True,
                        help=('Each: "title | checkpoint | config | dataset_name | data_dir". '
                              'Example: "MOSI | pt/dp-mosi.pth | config/dp_mosi_config.json | mosi | ./data"'))
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model_name = args.model_name.lower()
    extractor = EXTRACTOR_MAP.get(model_name)
    if extractor is None:
        raise ValueError(f"No extractor for model '{model_name}'. "
                         f"Supported: {sorted(EXTRACTOR_MAP)}")

    use_missing = args.missing_rate > 0
    panel_data = []

    for entry in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {entry['title']}  ({entry['dataset_name']})")
        print(f"  checkpoint: {entry['checkpoint']}")
        print(f"  config:     {entry['config']}")
        print(f"{'='*60}")

        cfg = get_config_regression(
            model_name, entry['dataset_name'],
            entry['config'], data_dir=entry['data_dir'],
        )
        cfg.device = device
        cfg.model_save_path = entry['checkpoint']
        cfg.data_missing = use_missing
        cfg.missing_rate_eval_test = args.missing_rate
        cfg.cur_seed = 1111

        dataloader = MMDataLoader(cfg, num_workers=0)
        model = build_model(cfg)
        state = torch.load(entry['checkpoint'], map_location='cpu')
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()

        reprs, labels = extractor(model, dataloader[args.split], device, use_missing)
        print(f"  Extracted: shape={reprs.shape}, "
              f"labels=[{labels.min():.2f}, {labels.max():.2f}]")

        panel_data.append({
            'title':        entry['title'],
            'dataset_name': entry['dataset_name'],
            'features':     reprs,
            'labels':       labels,
        })

        del model, dataloader
        torch.cuda.empty_cache()

    # ── Plot ──────────────────────────────────────────────────────────
    if args.output:
        save_stem = args.output
    else:
        ds_names = "_".join(e['dataset_name'] for e in args.datasets)
        save_stem = str(PROJECT_ROOT / 'result' / 'figures' /
                        f'tsne_{model_name}_datasets_{ds_names}_mr{args.missing_rate}')

    plot_tsne_across_datasets(
        panel_data,
        save_stem=save_stem,
        perplexity=args.perplexity,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()

# # Ví dụ: DP model trên 3 datasets, missing_rate=0.0
# conda activate baseline
# python scripts/tsne_across_datasets.py \
#     --model_name dp --split test \
#     --missing_rate 0.0 \
#     --datasets \
#         "MOSI | pt/dp-mosi-missing0.0.pth | config/dp_mosi_config.json | mosi | ./data" \
#         "MOSEI | pt/dp-mosei-missing0.0.pth | config/dp_mosei_config.json | mosei | ./data" \
#         "CH-SIMS | pt/dp-sims-missing0.0.pth | config/dp_sims_config.json | sims | ./data"
