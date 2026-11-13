"""paper_style.py - single source of truth for conference-style Matplotlib figures.

Adapted from the supplied FIGURE_GUIDE.md. Plotting scripts should call
apply_style(venue, kind), use new_fig(), and save through save_fig().
"""
from pathlib import Path
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # re-exported helpers

# Widths in inches. Only set verified=True after checking the actual venue template.
VENUES = {
    "acl":     dict(layout="two-column", narrow=7.7 / 2.54, wide=16.0 / 2.54, verified=True),
    "cvpr":    dict(layout="two-column", narrow=(6.875 - 0.3125) / 2, wide=6.875, verified=True),
    "iclr":    dict(layout="single-column", narrow=0.48 * 5.5, wide=5.5, verified=True),
    "neurips": dict(layout="single-column", narrow=0.48 * 5.5, wide=5.5, verified=False),
    "icml":    dict(layout="two-column", narrow=3.25, wide=6.75, verified=False),
}
VENUES["iccv"] = VENUES["cvpr"]

# Final printed font sizes in pt, from FIGURE_GUIDE.md.
FS = {
    "narrow": dict(base=7.0, label=8.0, tick=6.5, legend=7.0, panel=8.0, annot=6.5),
    "wide":   dict(base=7.5, label=8.5, tick=7.0, legend=7.5, panel=8.5, annot=7.0),
}
LW = dict(narrow=1.3, wide=1.5)
MS = dict(narrow=4.0, wide=4.5)
MEW = 0.5
SPINE = dict(narrow=0.7, wide=0.8)
TICK_LEN = dict(narrow=2.5, wide=3.0)
TICK_W = dict(narrow=0.6, wide=0.7)
SERIF_CANDIDATES = [
    "Times New Roman", "Liberation Serif", "Nimbus Roman", "TeX Gyre Termes", "FreeSerif"
]
_CUR = dict(venue=None, kind=None)


def width_in(venue, kind):
    return VENUES[venue][kind]


def apply_style(venue="acl", kind="narrow"):
    """Apply the shared conference style; call once before creating each figure."""
    if venue not in VENUES:
        raise KeyError(f"unknown venue {venue!r}; choose one of {sorted(VENUES)}")
    if kind not in ("narrow", "wide"):
        raise ValueError("kind must be 'narrow' or 'wide'")
    if not VENUES[venue]["verified"]:
        warnings.warn(
            f"{venue}: widths are NOT verified; measure the actual template and update VENUES."
        )
    _CUR.update(venue=venue, kind=kind)

    available = {font.name for font in fm.fontManager.ttflist}
    found = [name for name in SERIF_CANDIDATES if name in available]
    if not found:
        warnings.warn(
            "No Times-like font found; Matplotlib will fall back to DejaVu Serif. "
            "Install Liberation Serif / Nimbus Roman for closer Times compatibility."
        )
    fs = FS[kind]
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": found + SERIF_CANDIDATES + ["DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": fs["base"],
        "axes.labelsize": fs["label"],
        "axes.titlesize": fs["panel"],
        "xtick.labelsize": fs["tick"],
        "ytick.labelsize": fs["tick"],
        "legend.fontsize": fs["legend"],
        "axes.linewidth": SPINE[kind],
        "axes.labelpad": 2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": TICK_LEN[kind],
        "ytick.major.size": TICK_LEN[kind],
        "xtick.major.width": TICK_W[kind],
        "ytick.major.width": TICK_W[kind],
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "lines.linewidth": LW[kind],
        "lines.markersize": MS[kind],
        "lines.markeredgewidth": MEW,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.45,
        "grid.linestyle": ":",
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
    })


def _current():
    if _CUR["venue"] is None:
        raise RuntimeError("Call apply_style(venue, kind) before new_fig().")
    return _CUR["venue"], _CUR["kind"]


def new_fig(nrows=1, ncols=1, height=None, panel_aspect=None, **kwargs):
    """Create a figure at the target final printed width (nominal scale factor 1.0)."""
    venue, kind = _current()
    width = width_in(venue, kind)
    if height is None:
        aspect = panel_aspect or (0.8 if kind == "narrow" else 0.65)
        height = (width / ncols) * aspect * nrows + 0.5
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width, height), layout="constrained", squeeze=False, **kwargs
    )
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.06, hspace=0.08)
    return fig, axes.ravel()


def shared_legend(fig, handles, labels, ncol, **kwargs):
    """Place one legend above the entire figure (Matplotlib >= 3.7)."""
    return fig.legend(
        handles, labels, loc="outside upper center", ncol=ncol, frameon=False,
        handlelength=1.6, columnspacing=1.2, handletextpad=0.5, **kwargs
    )


def latex_snippet():
    venue, kind = _current()
    two_columns = VENUES[venue]["layout"] == "two-column"
    if two_columns and kind == "narrow":
        return r"\begin{figure}[t] ... \includegraphics[width=\columnwidth]{...}"
    if two_columns:
        return r"\begin{figure*}[t] ... \includegraphics[width=\textwidth]{...}"
    if kind == "wide":
        return r"\begin{figure}[t] ... \includegraphics[width=\linewidth]{...}"
    return r"two figures side-by-side: \includegraphics[width=0.48\linewidth]{...}"


def save_fig(fig, stem):
    """Save vector PDF and preview PNG without bbox_inches='tight' or resizing."""
    venue, kind = _current()
    width, height = fig.get_size_inches()
    target_width = width_in(venue, kind)
    if abs(width - target_width) >= 1e-3:
        raise AssertionError(f"Figure width {width:.3f} in != target {target_width:.3f} in")
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"))
    print(
        f"Saved {stem}.pdf and {stem}.png [{venue}/{kind}], "
        f"{width:.2f} x {height:.2f} in; PDF width target={target_width * 72:.1f} pt. "
        f"LaTeX: {latex_snippet()}"
    )
    plt.close(fig)


def effective_pt(font_pt, fig_width_in, target_width_in):
    """Printed font size after scaling an existing figure to the target width."""
    return font_pt * target_width_in / fig_width_in