#!/usr/bin/env python3
"""Summarize the train-bank local-coherence ablation on the test split.

Expected experiment layout (created by run_specific_neighbor_ablation.sh):
  result/specific_neighbor_ablation/<variant>/mr<rate>/
    uncertainty/uncertainty_mosi_mr<rate>_seed<seed>.pt
    normal/mosi.csv
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import pandas as pd
import torch

from paper_style import apply_style, new_fig, save_fig, shared_legend


VARIANTS = ('complement', 'self', 'random')
VARIANT_LABELS = {
    'complement': 'Complement',
    'self': 'Self-conditioned',
    'random': 'Random',
}
VARIANT_TICKS = {'complement': 'Comp.', 'self': 'Self', 'random': 'Rand.'}
VARIANT_COLORS = {
    'complement': '#4E79A7',
    'self': '#59A14F',
    'random': '#E15759',
}
MODALITIES = ('l', 'v', 'a')


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location='cpu')


def uncertainty_path(root: Path, variant: str, rate: float, dataset: str, seed: int):
    return (
        root / variant / f'mr{rate:.1f}' / 'uncertainty' /
        f'uncertainty_{dataset}_mr{rate:.1f}_seed{seed}.pt'
    )


def collect_coherence(root: Path, dataset: str, seed: int, rates: list[float]):
    rows = []
    for variant in VARIANTS:
        for rate in rates:
            path = uncertainty_path(root, variant, rate, dataset, seed)
            if not path.is_file():
                raise FileNotFoundError(f'Missing uncertainty file: {path}')
            result = load_checkpoint(path)
            metadata = result.get('metadata', {})
            source = metadata.get('neighbor_source')
            if source != 'train_bank':
                raise ValueError(
                    f'{path} used {source!r}; expected train_bank. Re-run the experiment '
                    'after updating main.py before interpreting this ablation.'
                )
            if metadata.get('neighbor_mode') != variant:
                raise ValueError(
                    f'{path} reports neighbor_mode={metadata.get("neighbor_mode")!r}, '
                    f'but is stored under {variant!r}.'
                )
            for modality in MODALITIES:
                key = f'specific_local_coherence_{modality}'
                if key not in result:
                    raise KeyError(f'{path} lacks {key}. Re-run --mode uncertainty.')
                values = result[key].detach().cpu().reshape(-1).tolist()
                rows.extend({
                    'variant': variant,
                    'missing_rate': rate,
                    'seed': seed,
                    'modality': modality,
                    'specific_cosine': value,
                } for value in values)
    return pd.DataFrame(rows)


def collect_performance(root: Path, dataset: str, rates: list[float]):
    rows = []
    metric_columns = ('F1_score Mean', 'MAE Mean', 'Corr Mean')
    for variant in VARIANTS:
        for rate in rates:
            path = root / variant / f'mr{rate:.1f}' / 'normal' / f'{dataset}.csv'
            if not path.is_file():
                raise FileNotFoundError(f'Missing downstream result: {path}')
            frame = pd.read_csv(path)
            if 'Missing Rate' not in frame: