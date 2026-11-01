"""
t-SNE Compare Models / Components (CVPR Paper Style)
=====================================================
Plot t-SNE of the final representation (pre-classifier fused embedding)
for multiple models or ablation variants **side-by-side on the same dataset**.

Layout mirrors the CVPR reference figure: one row of subplots, each subplot
is a different model/ablation.  Colour = continuous sentiment intensity
(Negative → Positive diverging colourbar).

Usage examples:
    # Compare 3 models on MOSI at missing_rate=0.5
    python scripts/tsne_compare_models.py \\
        --dataset mosi --data_dir ./data --split test \\
        --missing_rate 0.5 \\
        --entries \\
            "w/o MSD | dp | pt/dp-mosi-missing0.5.pth | config/dp_mosi_config.json" \\
            "w/o CCE | dmd | pt/dmd-mosi-missing0.5.pth | config/dmd_mosi_config.json" \\
            "DecAlign | dp | pt/dp-mosi-missing0.5.pth | config/dp_mosi_config.json"

    # Compare ablation variants
    python scripts/tsne_compare_models.py \\
        --dataset mosi --data_dir ./data --split test \\
        --missing_rate 0.5 \\
        --entries \\
            "w/o Reliability | dp | pt/dp-mosi-missing0.5-rel_uniform.pth | config/dp_mosi_config.json" \\
            "Shared Only | dp | pt/dp-mosi-missing0.5-rel_shared_only.pth | config/dp_mosi_config.json" \\
            "Full (Ours) | dp | pt/dp-mosi-missing0.5.pth | config/dp_mosi_config.json"

Each --entries value is  "title | model_name | checkpoint_path | config_file"
"""

import argparse
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


# =====================================================================
# Feature extraction helpers (one per model family)
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

            # Reconstruct last_hs_proj (same gating logic as DMD.forward)
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

            # Fused representation = concat of all private + shared
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


def plot_tsne_compare(all_embeddings, all_labels, titles, dataset_name,
                      save_stem, perplexity=30, seed=42):
    """
    Create a CVPR-style multi-panel t-SNE comparison figure.

    Parameters
    ----------
    all_embeddings : list[np.ndarray]  — each (N, D) raw features
    all_labels     : list[np.ndarray]  — each (N,) sentiment scores
    titles         : list[str]         — subplot titles
    dataset_name   : str
    save_stem      : str | Path        — output path (without extension)
    """
    from sklearn.manifold import TSNE

    n_panels = len(all_embeddings)

    # ── Apply paper style ─────────────────────────────────────────────
    apply_style("cvpr", "wide")

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(6.875, 6.875 / n_panels * 0.95 + 0.35),
        layout="constrained",
        squeeze=False,
    )
    axes = axes.ravel()
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.04)

    vmin, vmax = _get_label_range(dataset_name)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "sentiment",
        ["#08306b", "#2171b5", "#6baed6", "#bdd7e7",   # blues
         "#f7f7f7",                                       # neutral white
         "#fcbba1", "#fb6a4a", "#cb181d", "#67000d"],     # reds
        N=256,
    )

    for idx, (feats, labels, title) in enumerate(
        zip(all_embeddings, all_labels, titles)
    ):
        ax = axes[idx]

        print(f"  [{idx+1}/{n_panels}] Running t-SNE for '{title}' "
              f"({feats.shape[0]} samples, dim={feats.shape[1]})...")

        tsne = TSNE(n_components=2, perplexity=perplexity,
                    random_state=seed, max_iter=2000)
        emb_2d = tsne.fit_transform(feats)
        print(f"    KL divergence: {tsne.kl_divergence_:.4f}")

        sc = ax.scatter(
            emb_2d[:, 0], emb_2d[:, 1],
            c=labels, cmap=cmap, vmin=vmin, vmax=vmax,
            s=8, alpha=0.75,
            edgecolors='white', linewidths=0.15,
            rasterized=True,
        )

        ax.set_title(title, fontweight='bold', pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ── Shared colourbar below all panels ─────────────────────────────
    cbar = fig.colorbar(
        sc, ax=axes.tolist(), orientation='horizontal',
        fraction=0.035, pad=0.04, aspect=40, shrink=0.6,
    )
    cbar.set_label('')
    neg_label = "Negative"
    pos_label = "Positive"
    cbar.ax.text(-0.02, 0.5, neg_label, transform=cbar.ax.transAxes,
                 ha='right', va='center', fontsize=7)
    cbar.ax.text(1.02, 0.5, pos_label, transform=cbar.ax.transAxes,
                 ha='left', va='center', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

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

def parse_entry(s):
    """Parse 'title | model_name | checkpoint | config_file' string."""
    parts = [p.strip() for p in s.split('|')]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"Expected 'title | model | checkpoint | config', got: {s!r}"
        )
    return {
        'title':      parts[0],
        'model_name': parts[1].lower(),
        'checkpoint': parts[2],
        'config':     parts[3],
    }


def main():
    parser = argparse.ArgumentParser(
        description='t-SNE compare models/ablations (CVPR paper style)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dataset', type=str, default='mosi',
                        help='Dataset name (mosi / mosei / sims)')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'valid', 'test'])
    parser.add_argument('--missing_rate', type=float, default=0.0,
                        help='Missing rate for test set (0.0 = no missing)')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--perplexity', type=float, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='',
                        help='Output path stem (without extension)')
    parser.add_argument('--entries', nargs='+', type=parse_entry, required=True,
                        help=('Each entry: "title | model_name | checkpoint | config". '
                              'Example: "DecAlign | dp | pt/dp-mosi.pth | config/dp_mosi_config.json"'))
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    use_missing = args.missing_rate > 0

    all_embeddings = []
    all_labels = []
    titles = []

    for entry in args.entries:
        print(f"\n{'='*60}")
        print(f"Loading: {entry['title']}  ({entry['model_name']})")
        print(f"  checkpoint: {entry['checkpoint']}")
        print(f"  config:     {entry['config']}")
        print(f"{'='*60}")

        cfg = get_config_regression(
            entry['model_name'], args.dataset,
            entry['config'], data_dir=args.data_dir,
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

        extractor = EXTRACTOR_MAP.get(entry['model_name'])
        if extractor is None:
            raise ValueError(f"No extractor for model '{entry['model_name']}'. "
                             f"Supported: {sorted(EXTRACTOR_MAP)}")

        reprs, labels = extractor(model, dataloader[args.split], device, use_missing)
        print(f"  Extracted: shape={reprs.shape}, "
              f"labels=[{labels.min():.2f}, {labels.max():.2f}]")

        all_embeddings.append(reprs)
        all_labels.append(labels)
        titles.append(entry['title'])

        # Free GPU memory before loading next model
        del model, dataloader
        torch.cuda.empty_cache()

    # ── Plot ──────────────────────────────────────────────────────────
    if args.output:
        save_stem = args.output
    else:
        names = "_".join(e['model_name'] for e in args.entries)
        save_stem = str(PROJECT_ROOT / 'result' / 'figures' /
                        f'tsne_compare_{args.dataset}_mr{args.missing_rate}_{names}')

    plot_tsne_compare(
        all_embeddings, all_labels, titles,
        dataset_name=args.dataset,
        save_stem=save_stem,
        perplexity=args.perplexity,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()

# Ví dụ: so sánh 3 ablation variants trên MOSI, missing_rate=0.5
# conda activate baseline
# python scripts/tsne_compare_models.py \
#     --dataset mosi --data_dir ./data --split test \
#     --missing_rate 0.5 \
#     --entries \
#         "w/o Reliability | dp | pt/dp-mosi-missing0.5-rel_uniform.pth | config/dp_mosi_config.json" \
#         "Shared Only | dp | pt/dp-mosi-missing0.5-rel_shared_only.pth | config/dp_mosi_config.json" \
#         "Full (Ours) | dp | pt/dp-mosi-missing0.5.pth | config/dp_mosi_config.json"
