"""Single source of truth for final-size, ACL-style Matplotlib figures."""

from pathlib import Path
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


# ACL: A4 paper, 2.5 cm margins, 0.6 cm column separation.
VENUES = {
    'acl': dict(layout='two-column', narrow=7.7 / 2.54, wide=16.0 / 2.54),
}

FS = {
    'narrow': dict(base=7.0, label=8.0, tick=6.5, legend=7.0, panel=8.0),
    'wide': dict(base=7.5, label=8.5, tick=7.0, legend=7.5, panel=8.5),
}
SERIF_CANDIDATES = [
    'Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'TeX Gyre Termes',
]
_CURRENT = {'venue': None, 'kind': None}


def apply_style(venue='acl', kind='wide'):
    """Apply final printed sizes; figures are never subsequently scaled."""
    if venue not in VENUES:
        raise KeyError(f'Unknown venue: {venue}')
    if kind not in FS:
        raise ValueError("kind must be 'narrow' or 'wide'")

    available = {font.name for font in fm.fontManager.ttflist}
    serif = [font for font in SERIF_CANDIDATES if font in available]
    if not serif:
        warnings.warn('No Times-like font found; Matplotlib will fall back to DejaVu Serif.')

    fs = FS[kind]
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': serif + SERIF_CANDIDATES + ['DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': fs['base'],
        'axes.labelsize': fs['label'],
        'axes.titlesize': fs['panel'],
        'xtick.labelsize': fs['tick'],
        'ytick.labelsize': fs['tick'],
        'legend.fontsize': fs['legend'],
        'axes.linewidth': 0.8 if kind == 'wide' else 0.7,
        'axes.labelpad': 2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3.0 if kind == 'wide' else 2.5,
        'ytick.major.size': 3.0 if kind == 'wide' else 2.5,
        'xtick.major.width': 0.7 if kind == 'wide' else 0.6,
        'ytick.major.width': 0.7 if kind == 'wide' else 0.6,
        'xtick.major.pad': 2,
        'ytick.major.pad': 2,
        'lines.linewidth': 1.5 if kind == 'wide' else 1.3,
        'lines.markersize': 4.5 if kind == 'wide' else 4.0,
        'lines.markeredgewidth': 0.5,
        'grid.linewidth': 0.4,
        'grid.alpha': 0.5,
        'grid.linestyle': ':',
        'legend.frameon': False,
        'figure.dpi': 150,
        'savefig.dpi': 600,
    })
    _CURRENT.update(venue=venue, kind=kind)


def new_fig(nrows=1, ncols=1, height=None, panel_aspect=None, **kwargs):
    """Return a constrained-layout figure at its final printed dimensions."""
    if _CURRENT['venue'] is None:
        raise RuntimeError('Call apply_style() before new_fig().')
    venue, kind = _CURRENT['venue'], _CURRENT['kind']
    width = VENUES[venue][kind]
    if height is None:
        aspect = panel_aspect or (0.65 if kind == 'wide' else 0.8)
        height = (width / ncols) * aspect * nrows + 0.5
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width, height), layout='constrained',
        squeeze=False, **kwargs,
    )
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.06, hspace=0.08)
    return fig, axes.ravel()


def shared_legend(fig, handles, labels, ncol, **kwargs):
    return fig.legend(
        handles, labels, loc='outside upper center', ncol=ncol, frameon=False,
        handlelength=1.6, columnspacing=1.2, handletextpad=0.5, **kwargs,
    )


def save_fig(fig, stem):
    """Save exact-size PDF and PNG without a tight bounding box."""
    venue, kind = _CURRENT['venue'], _CURRENT['kind']
    target_width = VENUES[venue][kind]
    width, height = fig.get_size_inches()
    if abs(width - target_width) > 1e-3:
        raise AssertionError(f'Figure width {width:.3f} != target {target_width:.3f} in')
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.png'))
    plt.close(fig)
    print(f'Saved {stem}.pdf/.png ({width:.2f} x {height:.2f} in; ACL/{kind}).')