"""Create the MOSI missing-modality motivation figure.

The figure compares one MOSI test example per model under MR=0.0 and MR=0.5
for SARM, DecAlign, and DMD. It uses only two scores available for every model:

  u_con      = 1 - mean cosine similarity to the other shared features
  u_specific = 1 - mean cosine similarity to K nearest specific features,
               where neighbours are retrieved from the other shared spaces.

Examples:
  python scripts/analyze_case_study.py
  python scripts/analyze_case_study.py --reuse_cache

Outputs: result/figures/motivation/ and result/case_study_scores/.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
# Allow ``python scripts/analyze_case_study.py`` from the repository root.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CHECKPOINT_DIR = ROOT / "pt"
CACHE_DIR = ROOT / "result" / "case_study_scores"
FIG_DIR = ROOT / "result" / "figures" / "motivation"

MODALITIES = ("l", "v", "a")
MODEL_SPECS = {
    "sarm": {
        "model_name": "dp",
        "display_name": "SARM",
        "config": ROOT / "config" / "dp_mosi_config.json",
    },
    "decalign": {
        "model_name": "decalign",
        "display_name": "DecAlign",
        "config": ROOT / "config" / "dec_mosi_config.json",
    },
    "dmd": {
        "model_name": "dmd",
        "display_name": "DMD",
        "config": ROOT / "config" / "dmd_mosi_config.json",
    },
}
MODEL_ORDER = ("sarm", "decalign", "dmd")

# Hardcoded Acc-7 values for the figure (from official evaluation)
ACC7_VALUES = {
    0.0: {"sarm": 44.23, "decalign": 37.73, "dmd": 44.0},
    0.5: {"sarm": 33.67, "decalign": 27.0,  "dmd": 30.67},
}

ACL_COL_W = 3.03  # in, \columnwidth
SHOW_VALUES = False  # số trên thanh ở 5 pt rất nhỏ; bật nếu thật cần

_DISPLAY = {"sarm": "SARM", "decalign": "DecAlign", "dmd": "DMD"} 

COLOR_CON, COLOR_SPEC = "#F28E2B", "#4E79A7"
EDGE_CON, EDGE_SPEC = "#D07520", "#3A6186"
MISS_FACE = "#FBE4E4"      # band behind the missing modality
ACCENT = "#C0392B"         # red: missing tick / delta label
 
SUMMARY_TITLE = r"Text missing: Change on $u^{\mathrm{specific}}$"
 
 
def _np(x) -> np.ndarray:
    """torch.Tensor / ndarray -> ndarray (keeps this module torch-free)."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
 
 
def _minus(s: str) -> str:
    return s.replace("-", "\u2212")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer MOSI checkpoints and plot shared/specific uncertainty."
    )
    parser.add_argument("--checkpoint_dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--cache_dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--fig_dir", type=Path, default=FIG_DIR)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override inference batch size (useful on a small GPU/CPU).")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--reuse_cache", action="store_true",
                        help="Do not run inference; require the six cached score files.")
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER,
                        default=list(MODEL_ORDER),
                        help="Models to infer. The comparative figure needs all three.")
    parser.add_argument("--summary", choices=("dataset", "case"),
                        default="dataset")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def set_seed(seed: int) -> None:
    """Keep modality masking and the selected case reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_path(checkpoint_dir: Path, model_key: str, missing_rate: float) -> Path:
    filename_key = "dp" if model_key == "sarm" else model_key
    path = checkpoint_dir / f"{filename_key}-mosi-missing{missing_rate:.1f}.pth"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def cache_path(cache_dir: Path, model_key: str, missing_rate: float, seed: int) -> Path:
    return cache_dir / f"{model_key}_mosi_mr{missing_rate:.1f}_seed{seed}.pt"


def build_args(model_key: str, missing_rate: float, seed: int, device: torch.device,
               batch_size: int | None) -> SimpleNamespace:
    """Build the exact architecture configuration for each checkpoint."""
    from config import get_config_regression

    spec = MODEL_SPECS[model_key]
    args = get_config_regression(
        spec["model_name"], "mosi", config_file=spec["config"], data_dir=ROOT / "data"
    )
    args.data_missing = True
    args.missing_rate_eval_test = missing_rate
    args.cur_seed = seed
    args.device = device
    if batch_size is not None:
        args.batch_size = batch_size
    return args


def _mean_pool(feature: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Pool either (T, B, D) or (B, D, T) features into (B, D)."""
    if feature.ndim != 3:
        raise ValueError(f"Expected a 3-D representation, got {tuple(feature.shape)}")
    if feature.shape[1] == batch_size:  # (T, B, D), used by SARM
        return feature.mean(dim=0)
    if feature.shape[0] == batch_size:  # (B, D, T), used by DecAlign
        return feature.mean(dim=2)
    raise ValueError(f"Cannot locate batch dimension in representation {tuple(feature.shape)}")

def _bar(ax, x, val, w, fc, ec, ref=None, z=3.0):
    """Solid bar. If `ref` (value at MR=0.0) is given, draw the change on top:
    gain -> hatched segment above the old level, loss -> dashed outline."""
    if ref is None:
        ax.bar(x, val, w, color=fc, edgecolor=ec, linewidth=0.5, zorder=z)
        return
    ax.bar(x, min(val, ref), w, color=fc, edgecolor=ec, linewidth=0.5, zorder=z)
    d = val - ref
    if d >= 0:
        ax.bar(x, d, w, bottom=ref, facecolor=fc, edgecolor="white", hatch="//////",
               linewidth=0, zorder=z)
        ax.bar(x, d, w, bottom=ref, facecolor="none", edgecolor=ec, linewidth=0.8,
               zorder=z + 0.1)
    else:
        ax.bar(x, -d, w, bottom=val, facecolor="none", edgecolor=ec, linewidth=0.7,
               linestyle=(0, (2, 1.5)), zorder=z)
 
 
def _draw_summary_panel(ax, results, sample_indices, mode: str) -> None:
    """Bottom panel: change of the *text* score (MR 0.5 - MR 0.0) for every model."""
    off, w = 0.205, 0.37
    tops, bots = [0.0], [0.0]
    entries = []
    for gi, mk in enumerate(MODEL_ORDER):
        for score, fc, ec, dx in (("u_con", COLOR_CON, EDGE_CON, -off),
                                  ("u_specific", COLOR_SPEC, EDGE_SPEC, +off)):
            a = _np(results[0.0][mk][score]["l"])
            b = _np(results[0.5][mk][score]["l"])
            if mode == "case":
                si = sample_indices[mk]
                delta, err = float(b[si] - a[si]), 0.0
            else:
                diff = b - a
                delta = float(diff.mean())
                err = float(diff.std(ddof=1) / math.sqrt(len(diff)))
            entries.append((gi + dx, delta, err, fc, ec, score))
            tops.append(delta + err)
            bots.append(delta - err)
 
    lo, hi = min(bots), max(tops)
    span = max(hi - lo, 1e-6)
    ax.set_ylim(lo - (0.28 * span if lo < 0 else 0.0), hi + 0.34 * span)
 
    for x, delta, err, fc, ec, score in entries:
        ax.bar(x, delta, w, color=fc, edgecolor=ec, linewidth=0.5, zorder=3)
        if err > 0:
            ax.errorbar(x, delta, yerr=err, color="#333333", linewidth=0.5,
                        capsize=1.2, capthick=0.5, zorder=4)
        y = delta + err + 0.03 * span if delta >= 0 else delta - err - 0.03 * span
        spec = score == "u_specific"
        ax.text(x, y, _minus(f"{delta:+.2f}"), ha="center",
                va="bottom" if delta >= 0 else "top",
                fontsize=6.2 if spec else 5.6, fontweight="bold" if spec else "normal",
                color=ACCENT if spec else "#444444", zorder=5)
 
    ax.axhline(0, color="#555555", linewidth=0.5, zorder=4)
    ax.set_xlim(-0.55, len(MODEL_ORDER) - 0.45)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([_DISPLAY[k] for k in MODEL_ORDER], fontweight="bold")
    ax.tick_params(axis="x", length=0, pad=2)
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylabel(r"$\Delta$ on text", fontsize=7, labelpad=2)
    ax.set_title(SUMMARY_TITLE, fontweight="bold", loc="left", pad=2)
    # ax.set_xlabel("mean $\\pm$ s.e. over the MOSI test set" if mode == "dataset"
    #               else "same samples as above", fontsize=5.8, color="#555555", labelpad=1)

def forward_and_representations(
    model_key: str, model: torch.nn.Module, batch: dict, device: torch.device
) -> Tuple[torch.Tensor, dict, dict]:
    """Return predictions plus normalized shared and specific features."""
    text = batch["text"].to(device)
    audio = batch["audio"].to(device)
    vision = batch["vision"].to(device)
    text_m = batch["text_m"].to(device)
    audio_m = batch["audio_m"].to(device)
    vision_m = batch["vision_m"].to(device)
    batch_size = text.shape[0]

    if model_key == "sarm":
        output = model(text_m, audio, vision, text, audio, vision)
        prediction = output["output_logit"]
        shared = {m: _mean_pool(output[f"c_{m}"], batch_size) for m in MODALITIES}
        specific = {m: _mean_pool(output[f"s_{m}"], batch_size) for m in MODALITIES}
    elif model_key == "decalign":
        # DecAlign has one input per modality: pass the corrupted test inputs.
        output = model(text_m, audio, vision)
        prediction = output["output_logit"]
        shared = {m: _mean_pool(output[f"c_{m}"], batch_size) for m in MODALITIES}
        specific = {m: _mean_pool(output[f"s_{m}"], batch_size) for m in MODALITIES}
    elif model_key == "dmd":
        # DMD takes (text, audio, video) — same 3 inputs.
        output = model(text_m, audio, vision)
        prediction = output["output_logit"]
        shared = {m: _mean_pool(output[f"c_{m}"], batch_size) for m in MODALITIES}
        specific = {m: _mean_pool(output[f"s_{m}"], batch_size) for m in MODALITIES}
    else:  # pragma: no cover - protected by argparse / MODEL_SPECS
        raise KeyError(model_key)

    shared = {m: F.normalize(value.detach(), dim=1).cpu() for m, value in shared.items()}
    specific = {m: F.normalize(value.detach(), dim=1).cpu() for m, value in specific.items()}
    return prediction.detach().cpu().view(-1), shared, specific


def compute_scores(shared: Dict[str, torch.Tensor], specific: Dict[str, torch.Tensor],
                   neighbors: int) -> Tuple[dict, dict]:
    """Calculate u_con and u_specific for every test sample and modality."""
    n_samples = shared["l"].shape[0]
    if n_samples < 2:
        raise ValueError("At least two examples are required for KNN uncertainty.")
    k = min(neighbors, n_samples - 1)
    u_con, u_specific = {}, {}
    for modality in MODALITIES:
        others = [other for other in MODALITIES if other != modality]
        consensus = sum(
            F.cosine_similarity(shared[modality], shared[other], dim=1)
            for other in others
        ) / len(others)
        u_con[modality] = 1.0 - consensus

        # Retrieve neighbours without modality m, then assess m's private
        # representation against those neighbours.
        leave_one_out = F.normalize(sum(shared[other] for other in others), dim=1)
        similarities = leave_one_out @ leave_one_out.T
        similarities.fill_diagonal_(-math.inf)
        neighbour_index = similarities.topk(k, dim=1).indices
        neighbour_specific = specific[modality][neighbour_index]
        neighbour_similarity = F.cosine_similarity(
            specific[modality].unsqueeze(1), neighbour_specific, dim=2
        )
        u_specific[modality] = 1.0 - neighbour_similarity.mean(dim=1)
    return u_con, u_specific


def compute_performance(predictions: torch.Tensor, labels: torch.Tensor) -> dict:
    """Compute MOSI metrics on the same test pass used for the scores."""
    from utils.metricsTop import MetricsTop

    return MetricsTop("regression").getMetrics("mosi")(
        predictions.view(-1, 1), labels.view(-1, 1)
    )


def infer_checkpoint(model_key: str, missing_rate: float, args: argparse.Namespace,
                     device: torch.device) -> dict:
    """Run one checkpoint on MOSI test and return portable CPU tensors."""
    from data_loader import MMDataLoader
    from models import build_model

    set_seed(args.seed)
    model_args = build_args(model_key, missing_rate, args.seed, device, args.batch_size)
    loaders = MMDataLoader(model_args, num_workers=args.num_workers)
    checkpoint = checkpoint_path(args.checkpoint_dir, model_key, missing_rate)
    model = build_model(model_args).to(device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    pred_list, label_list, id_list = [], [], []
    shared_parts = {m: [] for m in MODALITIES}
    specific_parts = {m: [] for m in MODALITIES}
    with torch.no_grad():
        for batch in tqdm(
            loaders["test"],
            desc=f"{MODEL_SPECS[model_key]['display_name']} MR={missing_rate:.1f}",
        ):
            prediction, shared, specific = forward_and_representations(model_key, model, batch, device)
            pred_list.append(prediction)
            label_list.append(batch["labels"]["M"].view(-1).cpu())
            id_list.extend(str(item) for item in batch["id"])
            for modality in MODALITIES:
                shared_parts[modality].append(shared[modality])
                specific_parts[modality].append(specific[modality])

    predictions = torch.cat(pred_list)
    labels = torch.cat(label_list)
    shared = {m: torch.cat(parts) for m, parts in shared_parts.items()}
    specific = {m: torch.cat(parts) for m, parts in specific_parts.items()}
    u_con, u_specific = compute_scores(shared, specific, args.neighbors)
    metrics = compute_performance(predictions, labels)

    del model, loaders
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "ids": id_list,
        "labels": labels,
        "predictions": predictions,
        "u_con": u_con,
        "u_specific": u_specific,
        "metrics": metrics,
        "metadata": {
            "model": model_key,
            "display_name": MODEL_SPECS[model_key]["display_name"],
            "missing_rate": missing_rate,
            "seed": args.seed,
            "neighbors": args.neighbors,
            "checkpoint": str(checkpoint),
        },
    }


def load_or_infer_all(args: argparse.Namespace, device: torch.device) -> dict:
    if set(args.models) != set(MODEL_ORDER):
        raise ValueError("The comparative figure requires --models sarm decalign misa.")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for missing_rate in (0.0, 0.5):
        results[missing_rate] = {}
        for model_key in MODEL_ORDER:
            path = cache_path(args.cache_dir, model_key, missing_rate, args.seed)
            if args.reuse_cache:
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing cached inference: {path}. Run without --reuse_cache first."
                    )
                data = torch.load(path, map_location="cpu", weights_only=False)
                if data["metadata"].get("neighbors") != args.neighbors:
                    raise ValueError(f"{path} was created with a different --neighbors value.")
                print(f"Loaded {path.relative_to(ROOT)}")
            else:
                print(f"\nInferring {MODEL_SPECS[model_key]['display_name']} at MR={missing_rate:.1f} ...")
                data = infer_checkpoint(model_key, missing_rate, args, device)
                torch.save(data, path)
                print(f"Saved {path.relative_to(ROOT)}")
            results[missing_rate][model_key] = data
    return results


def average_score(data: dict, score_name: str, sample_index: int) -> float:
    return float(np.mean([data[score_name][m][sample_index].item() for m in MODALITIES]))


def display_sample_id(sample_id: object) -> str:
    """Make MOSI IDs literal text instead of Matplotlib mathtext."""
    return str(sample_id).replace("$", "").replace("_", "-")


def select_case(results: dict) -> dict:
    reference_ids = results[0.0]["sarm"]["ids"]
    for model_key in MODEL_ORDER:
        for missing_rate in (0.0, 0.5):
            if results[missing_rate][model_key]["ids"] != reference_ids:
                raise RuntimeError("Model outputs are not aligned to the same MOSI test order.")
 
    selected: Dict[str, int] = {}
    for model_key in MODEL_ORDER:
        d0, d5 = results[0.0][model_key], results[0.5][model_key]
        n = len(d0["ids"])
 
        candidates = []
        for i in range(n):
            avg_con_0 = np.mean([d0["u_con"][m][i].item() for m in MODALITIES])
            avg_spec_0 = np.mean([d0["u_specific"][m][i].item() for m in MODALITIES])
            if avg_con_0 <= avg_spec_0:                 # shared dominates at MR=0
                continue
 
            t_con_0, t_con_5 = d0["u_con"]["l"][i].item(), d5["u_con"]["l"][i].item()
            t_spec_0, t_spec_5 = d0["u_specific"]["l"][i].item(), d5["u_specific"]["l"][i].item()
            if t_spec_5 <= t_con_5:                     # specific dominates at MR=0.5
                continue
 
            jump = t_spec_5 - t_spec_0
            if jump <= 0:
                continue
            d_con = t_con_5 - t_con_0
            if d_con > 0.5 * jump:                      # shared also reacts -> weak story
                continue
 
            gap = t_spec_5 - t_con_5
            candidates.append((i, (jump - max(d_con, 0.0)) * gap))
 
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            selected[model_key] = candidates[0][0]
        else:                                           # relaxed fallback
            selected[model_key] = int(_np(d5["u_specific"]["l"]).argmax())
    return selected


def write_summary(results: dict, sample_indices: dict, output_path: Path) -> None:
    fieldnames = [
        "missing_rate", "model", "sample_index", "sample_id",
        "u_con_T", "u_spec_T", "u_con_V", "u_spec_V", "u_con_A", "u_spec_A",
        "Acc_2", "MAE", "Corr",
    ]
    rows = []
    for missing_rate in (0.0, 0.5):
        for model_key in MODEL_ORDER:
            si = sample_indices[model_key]
            data = results[missing_rate][model_key]
            metrics = data["metrics"]
            row = {
                "missing_rate": missing_rate,
                "model": MODEL_SPECS[model_key]["display_name"],
                "sample_index": si,
                "sample_id": data["ids"][si],
                "Acc_2": metrics["Acc_2"],
                "MAE": metrics["MAE"],
                "Corr": metrics["Corr"],
            }
            for m, tag in zip(MODALITIES, ("T", "V", "A")):
                row[f"u_con_{tag}"] = data["u_con"][m][si].item()
                row[f"u_spec_{tag}"] = data["u_specific"][m][si].item()
            rows.append(row)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_motivation_figure_1col(results: dict, sample_indices: dict, output_dir: Path,
                                fig_w: float = ACL_COL_W, fig_h: float = 4.3,
                                summary: str = "dataset",
                                acc7_values: dict | None = None,
                                display_names: dict | None = None) -> None:
    """summary: "dataset" (mean +- s.e. over the whole test set) or "case" (same sample)."""
    global _DISPLAY
    if display_names:
        _DISPLAY = display_names
    if acc7_values is None:
        acc7_values = ACC7_VALUES                                   # noqa: F821  (from the original file)
 
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes",
                       "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.6,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "font.size": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.titlesize": 8, "legend.fontsize": 7,
        "hatch.linewidth": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
 
    MOD_LABELS_MAP = {"l": "T", "v": "V", "a": "A"}
    MRS = (0.0, 0.5)
 
    # -- x layout: unchanged (2 MR blocks x 3 modality groups x 2 bars) --
    bar_w, group_gap, block_gap = 0.38, 0.30, 0.90
    pos, group_centre, block_centre, block_edges = {}, {}, {}, {}
    x = 0.0
    for mr in MRS:
        left = x - bar_w / 2
        for m in MODALITIES:
            pos[(mr, m)] = (x, x + bar_w)
            group_centre[(mr, m)] = x + bar_w / 2
            x += 2 * bar_w + group_gap
        right = x - group_gap - bar_w / 2
        block_edges[mr] = (left, right)
        block_centre[mr] = (left + right) / 2
        x += block_gap - group_gap
    x_lo, x_hi = block_edges[0.0][0] - 0.15, block_edges[0.5][1] + 0.15
    divider_x = (block_edges[0.0][1] + block_edges[0.5][0]) / 2
 
    # -- figure: 3 model rows + 1 summary row --
    fig = plt.figure(figsize=(fig_w, fig_h), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, hspace=0.07)
    gs = fig.add_gridspec(len(MODEL_ORDER) + 1, 1, height_ratios=[1, 1, 1, 0.82])
    axes = []
    for r in range(len(MODEL_ORDER)):
        axes.append(fig.add_subplot(gs[r], sharey=axes[0] if axes else None))
    ax_sum = fig.add_subplot(gs[len(MODEL_ORDER)])
 
    # y range: much less headroom than before, because MR/Acc-7 boxes are gone
    global_ymax = 0.0
    for mk in MODEL_ORDER:
        si = sample_indices[mk]
        for mr in MRS:
            for m in MODALITIES:
                global_ymax = max(global_ymax,
                                  results[mr][mk]["u_con"][m][si].item(),
                                  results[mr][mk]["u_specific"][m][si].item())
    y_top = max(0.7, math.ceil(global_ymax / 0.78 * 10) / 10)
 
    for row, (ax, model_key) in enumerate(zip(axes, MODEL_ORDER)):
        si = sample_indices[model_key]
        t_ref_con = results[0.0][model_key]["u_con"]["l"][si].item()
        t_ref_spec = results[0.0][model_key]["u_specific"]["l"][si].item()
 
        for mr in MRS:
            data = results[mr][model_key]
            for m in MODALITIES:
                u_con_val = data["u_con"][m][si].item()
                u_spec_val = data["u_specific"][m][si].item()
                x_con, x_spec = pos[(mr, m)]
                is_missing = (m == "l" and mr == 0.5)
 
                if is_missing:
                    ax.axvspan(x_con - bar_w / 2 - 0.08, x_spec + bar_w / 2 + 0.08,
                               facecolor=MISS_FACE, edgecolor="none", zorder=1)
                    _bar(ax, x_con, u_con_val, bar_w, COLOR_CON, EDGE_CON, ref=t_ref_con)
                    _bar(ax, x_spec, u_spec_val, bar_w, COLOR_SPEC, EDGE_SPEC, ref=t_ref_spec)
                    d = u_spec_val - t_ref_spec
                    ax.text(x_spec, max(u_spec_val, t_ref_spec) + 0.02 * y_top,
                            _minus(f"{d:+.2f}"), ha="center", va="bottom", fontsize=6.5,
                            fontweight="bold", color=ACCENT, zorder=6)
                else:
                    _bar(ax, x_con, u_con_val, bar_w, COLOR_CON, EDGE_CON)
                    _bar(ax, x_spec, u_spec_val, bar_w, COLOR_SPEC, EDGE_SPEC)
       
            # Acc-7: highlight SARM in bold black
            is_sarm = model_key == "sarm"

            ax.text(
                block_edges[mr][1], 0.985,
                rf"$\star$ Acc-7 {acc7_values[mr][model_key]:.1f}%",
                transform=ax.get_xaxis_transform(),
                ha="right",
                va="top",
                fontsize=6,
                color="black" if is_sarm else "#444444",
                fontweight="bold" if is_sarm else "normal",
                zorder=5,
            )

        # MR headers only once (top row) instead of 6 boxes
        if row == 0:
            for mr in MRS:
                ax.text(block_centre[mr], 1.0, f"MR = {mr:.1f}",
                        transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=7, fontweight="bold", color="#222222")
 
        ax.axvline(divider_x, color="#BBBBBB", linewidth=0.5, linestyle=(0, (3, 2)), zorder=2)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0, y_top)
        ax.set_xticks([group_centre[(mr, m)] for mr in MRS for m in MODALITIES])
        ax.set_xticklabels([MOD_LABELS_MAP[m] for _ in MRS for m in MODALITIES],
                           fontweight="bold")
        # the missing modality: red tick label
        for lab, (mr, m) in zip(ax.get_xticklabels(), [(a, b) for a in MRS for b in MODALITIES]):
            if m == "l" and mr == 0.5:
                lab.set_color(ACCENT)
        ax.tick_params(axis="x", length=0, pad=2)
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.3, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(_DISPLAY[model_key], fontweight="bold", loc="left", pad=2)
 
    # "missing" caption once, under the last model row
    ax_last = axes[-1]
    ax_last.text(group_centre[(0.5, "l")], -0.17, "missing", transform=ax_last.get_xaxis_transform(),
                 ha="center", va="top", fontsize=5.8, color=ACCENT, style="italic")
    axes[1].set_ylabel("Uncertainty score", fontsize=7.5, labelpad=2)
 
    _draw_summary_panel(ax_sum, results, sample_indices, summary)
 
    legend_handles = [
        Patch(facecolor=COLOR_CON, edgecolor=EDGE_CON, linewidth=0.5, label=r"$u^{\mathrm{con}}$"),
        Patch(facecolor=COLOR_SPEC, edgecolor=EDGE_SPEC, linewidth=0.5,
              label=r"$u^{\mathrm{specific}}$"),
        Patch(facecolor=MISS_FACE, edgecolor="#E3B5B5", linewidth=0.5, label="Missing text"),
        Patch(facecolor="#BDBDBD", edgecolor="white", hatch="//////", linewidth=0,
              label=r"$\Delta$ vs. MR = 0.0"),
    ]
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=4, frameon=False,
               fontsize=6.5, handlelength=1.2, columnspacing=1.0, handletextpad=0.4)
 
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"mosi_shared_specific_case_study_1col.{suffix}",
                    dpi=400, facecolor="white")
    plt.close(fig)


def print_summary(results: dict, sample_indices: dict) -> None:
    print("\nSelected MOSI test examples (per-model)")
    for model_key in MODEL_ORDER:
        si = sample_indices[model_key]
        sid = results[0.0][model_key]["ids"][si]
        print(f"  {MODEL_SPECS[model_key]['display_name']:>10}: index={si}, id={sid}")

    print("\n  MR   Model       sample  u_con_T  u_spec_T  u_con_V  u_spec_V  u_con_A  u_spec_A   Acc-2(%)")
    print("  " + "-" * 100)
    for missing_rate in (0.0, 0.5):
        for model_key in MODEL_ORDER:
            si = sample_indices[model_key]
            data = results[missing_rate][model_key]
            vals = []
            for m in MODALITIES:
                vals.append(data["u_con"][m][si].item())
                vals.append(data["u_specific"][m][si].item())
            acc = 100 * data["metrics"]["Acc_2"]
            print(
                f"  {missing_rate:.1f}  {MODEL_SPECS[model_key]['display_name']:<10} "
                f" {si:>5}   "
                + "  ".join(f"{v:7.4f}" for v in vals)
                + f"   {acc:6.2f}"
            )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Running on {device}. Checkpoints: {args.checkpoint_dir}")
    results = load_or_infer_all(args, device)
    sample_indices = select_case(results)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    plot_motivation_figure_1col(results, sample_indices, args.fig_dir,
                                summary=args.summary,
                                display_names={k: v["display_name"] for k, v in MODEL_SPECS.items()})
    summary_path = args.fig_dir / "mosi_shared_specific_case_study.csv"
    write_summary(results, sample_indices, summary_path)
    print_summary(results, sample_indices)
    print(f"\nFigure: {args.fig_dir / 'mosi_shared_specific_case_study.pdf'}")
    print(f"Data:   {summary_path}")


if __name__ == "__main__":
    main()