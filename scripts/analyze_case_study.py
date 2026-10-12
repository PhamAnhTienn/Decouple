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
    """Choose a per-model example where text u_specific jumps at MR=0.5.

    For each model, find a sample satisfying:
      1. MR=0.0: mean u_con > mean u_specific  (shared uncertainty dominates)
      2. MR=0.5: text u_specific jumps high and exceeds text u_con

    Returns ``{model_key: sample_index}``.
    """
    reference_ids = results[0.0]["sarm"]["ids"]
    for model_key in MODEL_ORDER:
        for missing_rate in (0.0, 0.5):
            if results[missing_rate][model_key]["ids"] != reference_ids:
                raise RuntimeError("Model outputs are not aligned to the same MOSI test order.")

    selected: Dict[str, int] = {}
    for model_key in MODEL_ORDER:
        d0 = results[0.0][model_key]
        d5 = results[0.5][model_key]
        n = len(d0["ids"])

        candidates = []
        for i in range(n):
            # Criterion 1: at MR=0.0, mean u_con > mean u_specific
            avg_con_0 = np.mean([d0["u_con"][m][i].item() for m in MODALITIES])
            avg_spec_0 = np.mean([d0["u_specific"][m][i].item() for m in MODALITIES])
            if avg_con_0 <= avg_spec_0:
                continue

            # Criterion 2: at MR=0.5, text u_specific > text u_con
            t_con_5 = d5["u_con"]["l"][i].item()
            t_spec_5 = d5["u_specific"]["l"][i].item()
            if t_spec_5 <= t_con_5:
                continue

            # Criterion 3: text u_specific actually jumped from MR=0.0
            t_spec_0 = d0["u_specific"]["l"][i].item()
            jump = t_spec_5 - t_spec_0
            if jump <= 0:
                continue

            # Score: jump magnitude × gap at MR=0.5
            gap = t_spec_5 - t_con_5
            candidates.append((i, jump * gap))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            selected[model_key] = candidates[0][0]
        else:
            # Relaxed fallback: biggest text u_specific at MR=0.5
            text_spec_5 = d5["u_specific"]["l"]
            selected[model_key] = int(text_spec_5.argmax().item())

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


def plot_motivation_figure(results: dict, sample_indices: dict, output_dir: Path) -> None:
    """2×3 grid — rows = MR conditions, columns = models.

    Each model column uses its own best-case sample.  Row labels
    (MR = 0.0 / 0.5) appear once on the left.  No suptitle.
    Bars are grouped by modality (T, V, A) with u_con and u_specific
    side-by-side within each group.  A soft highlight rectangle marks the
    Text group when text u_specific spikes at MR=0.5.  Acc-7 is shown as
    a compact corner label (bold black for SARM, gold for others).
    """
    from matplotlib.patches import FancyBboxPatch
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "font.size": 9.5,
        "axes.labelsize": 11.0,
        "axes.titlesize": 12.0,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 11.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # -- Colour & hatch definitions --
    COLOR_CON = "#F28E2B"       # orange  (u_con)
    COLOR_SPEC = "#4E79A7"      # steel-blue  (u_specific)
    EDGE_CON = "#D07520"
    EDGE_SPEC = "#3A6186"
    HIGHLIGHT_COLOR = "#FFEFD5"  # soft peach for text-spike highlight
    HIGHLIGHT_EDGE = "#FFDAB9"

    MOD_LABELS_MAP = {"l": "T", "v": "V", "a": "A"}
    MOD_HATCHES = {"l": "///", "v": "\\\\", "a": "xx"}

    # -- Grouped bar positions --
    # 3 modality groups, each with 2 bars (u_con, u_specific).
    # Within a group the two bars sit side-by-side; between groups a gap.
    bar_w = 0.35
    group_gap = 0.45  # extra space between modality groups
    group_centres = []  # centre x of each modality group
    bar_positions = []  # (x_con, x_spec) per modality
    x_cursor = 0.0
    for _i, _m in enumerate(MODALITIES):
        x_con = x_cursor
        x_spec = x_cursor + bar_w
        bar_positions.append((x_con, x_spec))
        group_centres.append((x_con + x_spec) / 2)
        x_cursor = x_spec + bar_w + group_gap

    fig, axes_grid = plt.subplots(
        2, len(MODEL_ORDER), figsize=(12.0, 5.8), sharey=True,
    )

    # Collect global y-max for consistent scale (first pass)
    global_ymax = 0.0

    MR_LABELS = {0.0: "MR = 0.0", 0.5: "MR = 0.5"}

    # Pre-compute text u_spec at MR=0.0 for highlight detection
    text_spec_mr0 = {}
    for model_key in MODEL_ORDER:
        si = sample_indices[model_key]
        text_spec_mr0[model_key] = results[0.0][model_key]["u_specific"]["l"][si].item()

    for col, model_key in enumerate(MODEL_ORDER):
        si = sample_indices[model_key]

        for row, missing_rate in enumerate((0.0, 0.5)):
            ax = axes_grid[row, col]
            data = results[missing_rate][model_key]

            all_vals = []
            for mi, m in enumerate(MODALITIES):
                tag = MOD_LABELS_MAP[m]
                hatch = MOD_HATCHES[m]
                u_con_val = data["u_con"][m][si].item()
                u_spec_val = data["u_specific"][m][si].item()
                x_con, x_spec = bar_positions[mi]

                bar_con = ax.bar(x_con, u_con_val, bar_w, color=COLOR_CON,
                                 edgecolor=EDGE_CON, linewidth=0.7, zorder=3,
                                 hatch=hatch)
                bar_spec = ax.bar(x_spec, u_spec_val, bar_w, color=COLOR_SPEC,
                                  edgecolor=EDGE_SPEC, linewidth=0.7, zorder=3,
                                  hatch=hatch)

                # Value annotations
                for bar_obj, val, edge_c in [(bar_con[0], u_con_val, EDGE_CON),
                                              (bar_spec[0], u_spec_val, EDGE_SPEC)]:
                    ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                            val + 0.015, f"{val:.2f}",
                            ha="center", va="bottom", fontsize=7.5,
                            color=edge_c, fontweight="bold")

                all_vals.extend([u_con_val, u_spec_val])

                # -- Highlight rectangle for Text modality spike --
                if m == "l" and missing_rate == 0.5:
                    t_spec_now = u_spec_val
                    t_spec_before = text_spec_mr0[model_key]
                    jump = t_spec_now - t_spec_before
                    if jump > 0.05:  # meaningful spike threshold
                        rect_x = x_con - bar_w * 0.35
                        rect_w = (x_spec + bar_w) - x_con + bar_w * 0.7 - bar_w
                        rect_h = max(u_con_val, u_spec_val) + 0.08
                        highlight = FancyBboxPatch(
                            (rect_x, 0), rect_w, rect_h,
                            boxstyle="round,pad=0.04",
                            facecolor=HIGHLIGHT_COLOR, edgecolor=HIGHLIGHT_EDGE,
                            linewidth=1.0, alpha=0.55, zorder=1,
                        )
                        ax.add_patch(highlight)

            # -- Acc-7 label (compact, upper-right corner) --
            acc7 = ACC7_VALUES[missing_rate][model_key]
            if model_key == "sarm":
                acc_color = "#000000"
                acc_bg = "#F5F5DC"
            else:
                acc_color = "#6B4C00"
                acc_bg = "#FFFDE7"
            ax.text(
                0.97, 0.95,
                f"★ Acc7 = {acc7:.1f}%",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9.0, fontweight="bold", color=acc_color,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=acc_bg,
                          edgecolor=acc_color, alpha=0.85, linewidth=0.8),
            )

            # -- X-axis --
            ax.set_xticks(group_centres)
            ax.set_xticklabels(
                [MOD_LABELS_MAP[m] for m in MODALITIES], fontsize=9.5,
                fontweight="bold",
            )
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, zorder=0)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            global_ymax = max(global_ymax, max(all_vals))

            # Column title: model name (only on top row)
            if row == 0:
                ax.set_title(
                    MODEL_SPECS[model_key]["display_name"],
                    fontweight="bold", fontsize=13,
                )

            # Row label on leftmost column
            if col == 0:
                ax.set_ylabel(
                    f"{MR_LABELS[missing_rate]}\nUncertainty score",
                    fontsize=11, fontweight="bold",
                )

    # -- Global y-limits --
    y_top = max(1.0, math.ceil((global_ymax + 0.15) * 4) / 4)
    for ax_row in axes_grid:
        for ax in ax_row:
            ax.set_ylim(0, y_top)

    # -- Legend --
    legend_handles = [
        Patch(facecolor=COLOR_CON, edgecolor=EDGE_CON, label=r"$u^{con}$"),
        Patch(facecolor=COLOR_SPEC, edgecolor=EDGE_SPEC, label=r"$u^{specific}$"),
        Patch(facecolor="#E0E0E0", edgecolor="#999999", hatch="///", label="Text (T)"),
        Patch(facecolor="#E0E0E0", edgecolor="#999999", hatch="\\\\", label="Video (V)"),
        Patch(facecolor="#E0E0E0", edgecolor="#999999", hatch="xx", label="Audio (A)"),
        Patch(facecolor=HIGHLIGHT_COLOR, edgecolor=HIGHLIGHT_EDGE,
              linewidth=1.0, alpha=0.55, label="Text spike"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=6, frameon=True,
        bbox_to_anchor=(0.5, -0.005), handlelength=1.6, columnspacing=1.0,
        framealpha=0.92, edgecolor="#CCCCCC", fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1.0])
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"mosi_shared_specific_case_study.{suffix}",
                    dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.03)
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
    plot_motivation_figure(results, sample_indices, args.fig_dir)
    summary_path = args.fig_dir / "mosi_shared_specific_case_study.csv"
    write_summary(results, sample_indices, summary_path)
    print_summary(results, sample_indices)
    print(f"\nFigure: {args.fig_dir / 'mosi_shared_specific_case_study.pdf'}")
    print(f"Data:   {summary_path}")


if __name__ == "__main__":
    main()

