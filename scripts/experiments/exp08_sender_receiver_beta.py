"""
Experiment 8 – Sender–Receiver Behaviour (β heatmap)
=====================================================
Verify that the reliability-gated complementary exchange coefficient β
behaves as designed:

    β_{m←n} = γ_{m,n}^s · r_n^s / (r_m^s + r_n^s)
    where  γ_{m,n}^s = 1 − (1 − r_m^s)(1 − r_n^s)

Expected behaviour for a given sender→receiver pair (e.g. Audio → Vision):
  - Sender good, receiver weak   → β high  (receiver accepts help)
  - Sender weak, receiver good   → β low   (receiver ignores noisy sender)
  - Both weak                    → β low   (joint gate γ suppresses)

This script sweeps sender and receiver missing rates independently,
computing mean β for a chosen pair, and outputs a 2D heatmap.

Output
------
* 2D heatmap: x = sender missing rate, y = receiver missing rate, color = mean β
* Saved to result/experiments/exp08/
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import get_config_regression
from data_loader import MMDataLoader
from models import build_model
from utils import assign_gpu, setup_seed

logger = logging.getLogger("DecAlign")
MODALITIES = ["l", "v", "a"]
MOD_NAMES = {"l": "Language", "v": "Vision", "a": "Audio"}


# =====================================================================
# Create asymmetric dataloader (per-modality missing rates)
# =====================================================================

def create_dataloader_with_rates(args, mr_dict, num_workers=4, mode="test"):
    """
    Create test DataLoader with custom per-modality missing rates.
    mr_dict: {'l': float, 'a': float, 'v': float}
    """
    with open(args.featurePath, "rb") as f:
        data = pickle.load(f)

    split = data[mode]
    N_samples = len(split["regression_labels" if args.train_mode == "regression" else "classification_labels"])

    if getattr(args, "use_bert", False):
        text_bert = split["text_bert"]
        if text_bert.ndim == 3 and text_bert.shape[1] == 3 and np.issubdtype(text_bert.dtype, np.integer):
            text = text_bert
        else:
            text = text_bert.astype(np.float32)
    else:
        text = split["text"].astype(np.float32)

    audio = split["audio"].astype(np.float32)
    vision = split["vision"].astype(np.float32)
    audio[audio == -np.inf] = 0
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    vision = np.nan_to_num(vision, nan=0.0, posinf=0.0, neginf=0.0)

    labels_M = np.array(split["regression_labels" if args.train_mode == "regression" else "classification_labels"]).astype(np.float32)
    raw_text = split["raw_text"]
    ids = split["id"]

    if getattr(args, "need_data_aligned", False):
        audio_lengths = np.sum(text[:, 1, :], axis=1, dtype=np.int32)
        vision_lengths = audio_lengths.copy()
    else:
        audio_lengths = split.get("audio_lengths", [audio.shape[1]] * N_samples)
        vision_lengths = split.get("vision_lengths", [vision.shape[1]] * N_samples)

    seed = getattr(args, "cur_seed", 1111)
    np.random.seed(seed)

    # Text missing
    input_mask = text[:, 1, :]
    input_len = np.argmin(input_mask, axis=1)
    mr_l_arr = mr_dict["l"] * np.ones((N_samples, 1))
    text_miss_mask = (np.random.uniform(size=input_mask.shape) > mr_l_arr.repeat(input_mask.shape[1], 1)) * input_mask
    for i in range(N_samples):
        if input_len[i] > 0:
            text_miss_mask[i, 0] = 1
            text_miss_mask[i, input_len[i] - 1] = 1
    text_m_ids = text_miss_mask * text[:, 0, :] + 100 * np.ones_like(text[:, 0, :]) * (input_mask - text_miss_mask)
    text_m_full = np.concatenate([
        np.expand_dims(text_m_ids, 1),
        np.expand_dims(text_miss_mask, 1),
        np.expand_dims(text[:, 2, :], 1),
    ], axis=1)

    # Audio missing
    audio_mask = np.array([[1]*int(l) + [0]*(audio.shape[1]-int(l)) for l in audio_lengths])
    mr_a_arr = mr_dict["a"] * np.ones((N_samples, 1))
    audio_miss_mask = (np.random.uniform(size=audio_mask.shape) > mr_a_arr.repeat(audio_mask.shape[1], 1)) * audio_mask
    audio_m = audio_miss_mask.reshape(audio.shape[0], audio.shape[1], 1) * audio

    # Vision missing
    vision_mask = np.array([[1]*int(l) + [0]*(vision.shape[1]-int(l)) for l in vision_lengths])
    mr_v_arr = mr_dict["v"] * np.ones((N_samples, 1))
    vision_miss_mask = (np.random.uniform(size=vision_mask.shape) > mr_v_arr.repeat(vision_mask.shape[1], 1)) * vision_mask
    vision_m = vision_miss_mask.reshape(vision.shape[0], vision.shape[1], 1) * vision

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return N_samples
        def __getitem__(self, idx):
            if np.issubdtype(text.dtype, np.integer):
                tt = torch.LongTensor(text[idx])
                tm = torch.LongTensor(text_m_full[idx])
            else:
                tt = torch.Tensor(text[idx])
                tm = torch.Tensor(text_m_full[idx])
            return {
                "text": tt, "audio": torch.Tensor(audio[idx]), "vision": torch.Tensor(vision[idx]),
                "text_m": tm, "audio_m": torch.Tensor(audio_m[idx]), "vision_m": torch.Tensor(vision_m[idx]),
                "labels": {"M": torch.Tensor(labels_M[idx].reshape(-1))},
                "index": idx, "id": ids[idx], "raw_text": raw_text[idx],
            }

    return torch.utils.data.DataLoader(_DS(), batch_size=args.batch_size, shuffle=False, num_workers=num_workers)


# =====================================================================
# Compute β for a sender → receiver pair
# =====================================================================

def compute_beta_values(model, dataloader, device, m_recv, m_send):
    """
    Compute mean β_{recv←send} across all test samples.

    β_{m←n} = γ_{m,n} · r_n / (r_m + r_n)
    γ_{m,n} = 1 − (1 − r_m)(1 − r_n)

    Returns:
        mean_beta: float
        all_betas: (N,) numpy array
    """
    model.eval()
    betas = []

    with torch.no_grad():
        for batch in dataloader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            text_m = batch["text_m"].to(device)
            audio_m = batch["audio_m"].to(device)
            vision_m = batch["vision_m"].to(device)

            output = model(text_m, audio_m, vision_m, text, audio, vision)

            r_m = output["r_specific"][m_recv]  # (B,)
            r_n = output["r_specific"][m_send]  # (B,)

            gamma_mn = 1.0 - (1.0 - r_m) * (1.0 - r_n)
            denom = (r_m + r_n).clamp(min=1e-8)
            beta = gamma_mn * r_n / denom  # (B,)

            betas.append(beta.cpu().numpy())

    all_betas = np.concatenate(betas)
    return float(np.mean(all_betas)), all_betas


# =====================================================================
# Sweep sender & receiver missing rates
# =====================================================================

def sweep_beta(model, model_args, m_recv, m_send, rates, num_workers, third_mod_rate=0.0):
    """
    Sweep sender and receiver missing rates, compute mean β for each combination.

    Args:
        m_recv, m_send: modality keys ('l', 'a', 'v')
        rates: list of missing rates to sweep
        third_mod_rate: fixed rate for the third modality

    Returns:
        beta_matrix: (len(rates), len(rates)) — [recv_rate_idx, sender_rate_idx]
    """
    third_mod = [m for m in MODALITIES if m not in [m_recv, m_send]][0]
    n = len(rates)
    beta_matrix = np.zeros((n, n))

    for i, recv_rate in enumerate(rates):
        for j, send_rate in enumerate(rates):
            mr_dict = {
                m_recv: recv_rate,
                m_send: send_rate,
                third_mod: third_mod_rate,
            }
            dl = create_dataloader_with_rates(model_args, mr_dict, num_workers)
            mean_beta, _ = compute_beta_values(model, dl, model_args.device, m_recv, m_send)
            beta_matrix[i, j] = mean_beta
            print(f"  recv_mr={recv_rate:.1f}, send_mr={send_rate:.1f} → β={mean_beta:.4f}")

    return beta_matrix


# =====================================================================
# Plot heatmap
# =====================================================================

def plot_heatmap(beta_matrix, rates, m_recv, m_send, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(beta_matrix, cmap="YlOrRd", aspect="auto", origin="lower",
                   vmin=beta_matrix.min(), vmax=beta_matrix.max())

    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels([f"{r:.1f}" for r in rates])
    ax.set_yticks(range(len(rates)))
    ax.set_yticklabels([f"{r:.1f}" for r in rates])

    ax.set_xlabel(f"Sender ({MOD_NAMES[m_send]}) Missing Rate", fontsize=12)
    ax.set_ylabel(f"Receiver ({MOD_NAMES[m_recv]}) Missing Rate", fontsize=12)
    ax.set_title(f"β_{{{MOD_NAMES[m_recv]}←{MOD_NAMES[m_send]}}}", fontsize=14)

    # Add readable annotations based on actual cell luminance
    norm = im.norm
    cmap = im.get_cmap()

    for i in range(len(rates)):
        for j in range(len(rates)):
            value = beta_matrix[i, j]

            # Actual RGBA color of this cell
            r, g, b, _ = cmap(norm(value))

            # Perceived brightness (relative luminance)
            luminance = (
                0.2126 * r
                + 0.7152 * g
                + 0.0722 * b
            )

            # Dark text on light cells, white on dark cells
            text_color = "black" if luminance > 0.5 else "white"

            ax.text(
                j, i, f"{value:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="medium",
                path_effects=[
                    pe.withStroke(linewidth=0.4, foreground="white")
                    if text_color == "black"
                    else pe.withStroke(linewidth=0.4, foreground="black")
                ],
            )

    plt.colorbar(im, ax=ax, label="Mean β")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to {save_path}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Exp08: Sender–Receiver β Behaviour")
    parser.add_argument("--dataset", type=str, default="mosi", choices=["mosi", "mosei", "sims"])
    parser.add_argument("--config_file", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Checkpoint path (default: checkpoint/dp-{dataset}-missing0.0.pth)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint")
    parser.add_argument("--save_dir", type=str, default="./result/experiments/exp08")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1111)
    # Pair selection
    parser.add_argument("--sender", type=str, default="a", choices=["l", "a", "v"],
                        help="Sender modality")
    parser.add_argument("--receiver", type=str, default="v", choices=["l", "a", "v"],
                        help="Receiver modality")
    # Sweep config
    parser.add_argument("--rates", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        help="Missing rates to sweep")
    parser.add_argument("--third_rate", type=float, default=0.0,
                        help="Fixed missing rate for the third modality")
    parser.add_argument("--all_pairs", action="store_true",
                        help="Run all 6 sender→receiver pairs")
    args = parser.parse_args()

    if args.sender == args.receiver:
        raise ValueError("Sender and receiver must be different modalities")

    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    if args.config_file == "":
        config_file = str(Path(__file__).resolve().parents[2] / "config" / f"dp_{args.dataset}_config.json")
    else:
        config_file = args.config_file

    model_args = get_config_regression("dp", args.dataset, config_file, data_dir=args.data_dir)
    model_args.device = assign_gpu(args.gpu_ids)
    model_args.cur_seed = args.seed
    model_args.data_missing = True
    model_args.mode = "test"
    model_args.missing_rate_eval_test = 0.0

    if args.checkpoint:
        ckpt = args.checkpoint
    else:
        ckpt = os.path.join(args.checkpoint_dir, f"dp-{args.dataset}-missing0.0.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    setup_seed(args.seed)

    # Init data to set feature dims
    dl_init = MMDataLoader(model_args, args.num_workers)
    model = build_model(model_args).to(model_args.device)
    model.load_state_dict(torch.load(ckpt, map_location=model_args.device))

    # Build train bank
    logger.info("Building train-set KNN bank...")
    model.update_train_bank(model, dl_init["train"], model_args.device)

    # Determine pairs to run
    if args.all_pairs:
        pairs = [(r, s) for r in MODALITIES for s in MODALITIES if r != s]
    else:
        pairs = [(args.receiver, args.sender)]

    all_results = {}

    for m_recv, m_send in pairs:
        print(f"\n{'='*60}")
        print(f"  Pair: {MOD_NAMES[m_send]} → {MOD_NAMES[m_recv]}")
        print(f"{'='*60}")

        beta_matrix = sweep_beta(
            model, model_args, m_recv, m_send,
            args.rates, args.num_workers, args.third_rate,
        )

        plot_heatmap(
            beta_matrix, args.rates, m_recv, m_send,
            os.path.join(args.save_dir, f"beta_heatmap_{m_send}_to_{m_recv}_{args.dataset}.png"),
        )

        all_results[f"{m_send}_to_{m_recv}"] = {
            "beta_matrix": beta_matrix.tolist(),
            "rates": args.rates,
        }

    # Save
    torch.save(all_results, os.path.join(args.save_dir, f"exp08_{args.dataset}.pt"))
    with open(os.path.join(args.save_dir, f"exp08_{args.dataset}_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {args.save_dir}")


if __name__ == "__main__":
    main()

# python scripts/experiments/exp08.py \
#     --dataset mosi \
#     --all_pairs \
#     --save_dir result/experiments/exp08 \
#     --rates 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9