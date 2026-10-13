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


def plot_motivation_figure_1col(results: dict, sample_indices: dict, output_dir: Path,
                                fig_w: float = ACL_COL_W, fig_h: float = 4.3) -> None:
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
        "font.size": 6.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # -- Colours (unchanged) --
    COLOR_CON, COLOR_SPEC = "#F28E2B", "#4E79A7"
    EDGE_CON, EDGE_SPEC = "#D07520", "#3A6186"
    HIGHLIGHT_COLOR, HIGHLIGHT_EDGE = "#FFEFD5", "#FFDAB9"
    MOD_LABELS_MAP = {"l": "T", "v": "V", "a": "A"}
    MRS = (0.0, 0.5)

    # -- x layout: 2 MR blocks, each with 3 modality groups of 2 bars --
    bar_w, group_gap, block_gap = 0.38, 0.30, 0.90
    pos = {}            # (mr, modality) -> (x_con, x_spec)
    group_centre = {}   # (mr, modality) -> x
    block_centre = {}   # mr -> x
    block_edges = {}    # mr -> (left, right)
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

    fig, axes = plt.subplots(len(MODEL_ORDER), 1, figsize=(fig_w, fig_h),
                             sharey=True, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, hspace=0.06)

    global_ymax = 0.0
    for ax, model_key in zip(axes, MODEL_ORDER):
        si = sample_indices[model_key]
        text_spec_mr0 = results[0.0][model_key]["u_specific"]["l"][si].item()

        for mr in MRS:
            data = results[mr][model_key]
            for m in MODALITIES:
                u_con_val = data["u_con"][m][si].item()
                u_spec_val = data["u_specific"][m][si].item()
                x_con, x_spec = pos[(mr, m)]

                if m == "l" and mr == 0.5 and u_spec_val - text_spec_mr0 > 0.05:
                    ax.axvspan(x_con - bar_w / 2 - 0.08, x_spec + bar_w / 2 + 0.08,
                               facecolor=HIGHLIGHT_COLOR, edgecolor=HIGHLIGHT_EDGE,
                               linewidth=0.5, alpha=0.55, zorder=1)

                for xx, val, fc, ec in ((x_con, u_con_val, COLOR_CON, EDGE_CON),
                                        (x_spec, u_spec_val, COLOR_SPEC, EDGE_SPEC)):
                    ax.bar(xx, val, bar_w, color=fc, edgecolor=ec, linewidth=0.5, zorder=3)
                    if SHOW_VALUES:
                        ax.text(xx, val + 0.015, f"{val:.2f}".lstrip("0"),
                                ha="center", va="bottom", fontsize=5, color="#333333")
                global_ymax = max(global_ymax, u_con_val, u_spec_val)

            # block label + Acc-7 (above each block)
            acc7 = ACC7_VALUES[mr][model_key]
            acc_color, acc_bg = (("#000000", "#F5F5DC") if model_key == "sarm"
                                 else ("#6B4C00", "#FFFDE7"))
            ax.text(block_centre[mr], 0.97,
                    rf"MR = {mr:.1f}" "\n" rf"$\star$ Acc-7 = {acc7:.1f}%",
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=6.5, fontweight="bold", color=acc_color, linespacing=1.25,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=acc_bg,
                              edgecolor=acc_color, alpha=0.85, linewidth=0.5),
                    zorder=5)

        ax.axvline(divider_x, color="#BBBBBB", linewidth=0.5, linestyle=(0, (3, 2)), zorder=2)
        ax.set_xlim(x_lo, x_hi)
        ax.set_xticks([group_centre[(mr, m)] for mr in MRS for m in MODALITIES])
        ax.set_xticklabels([MOD_LABELS_MAP[m] for _ in MRS for m in MODALITIES],
                           fontweight="bold")
        ax.tick_params(axis="x", length=0, pad=2)
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(MODEL_SPECS[model_key]["display_name"], fontweight="bold",
                     loc="left", pad=2)

    # headroom cho hộp MR/Acc-7 (chiếm ~35% chiều cao trục)
    y_top = max(1.0, math.ceil(global_ymax / 0.62 * 10) / 10)
    for ax in axes:
        ax.set_ylim(0, y_top)

    fig.supylabel("Uncertainty score", fontsize=7.5)

    legend_handles = [
        Patch(facecolor=COLOR_CON, edgecolor=EDGE_CON, linewidth=0.5,
              label=r"$u^{\mathrm{con}}$"),
        Patch(facecolor=COLOR_SPEC, edgecolor=EDGE_SPEC, linewidth=0.5,
              label=r"$u^{\mathrm{specific}}$"),
        Patch(facecolor=HIGHLIGHT_COLOR, edgecolor=HIGHLIGHT_EDGE, linewidth=0.5,
              alpha=0.55, label="Text spike"),
    ]
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=3,
               frameon=False, handlelength=1.3, columnspacing=1.4, handletextpad=0.5)

    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"mosi_shared_specific_case_study_1col.{suffix}",
                    dpi=400, facecolor="white")  # không dùng bbox_inches="tight"
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
    plot_motivation_figure_1col(results, sample_indices, args.fig_dir)
    summary_path = args.fig_dir / "mosi_shared_specific_case_study.csv"
    write_summary(results, sample_indices, summary_path)
    print_summary(results, sample_indices)
    print(f"\nFigure: {args.fig_dir / 'mosi_shared_specific_case_study.pdf'}")
    print(f"Data:   {summary_path}")


if __name__ == "__main__":
    main()

