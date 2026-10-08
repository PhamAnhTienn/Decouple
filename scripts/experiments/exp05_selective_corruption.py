"""
Experiment 5 – Selective / Asymmetric Corruption
==================================================
Prove that reliability scores are **modality-selective**: when a specific
modality is heavily corrupted, its reliability drops while others stay high.

Asymmetric corruption settings (e.g. for 3 modalities L, A, V):
    (r_L, r_A, r_V) = (0.8, 0.2, 0.2)  →  Language is corrupted
    (r_L, r_A, r_V) = (0.2, 0.8, 0.2)  →  Audio   is corrupted
    (r_L, r_A, r_V) = (0.2, 0.2, 0.8)  →  Vision  is corrupted

Metrics:
  - **Top-1 identification accuracy**: predict which modality is most corrupted
    as m̂ = argmin_m r(m,i)  and check if it matches the ground-truth corrupted
    modality.
  - **Macro-F1** over 3 classes.
  - **Mean reliability per modality** under each asymmetric setting.

Output
------
* Heatmap / line plot data: mean r_L, r_A, r_V per setting
* Top-1 accuracy & Macro-F1
* Saved to result/experiments/exp05/
"""

import argparse
import json
import logging
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import get_config_regression
from data_loader import MMDataLoader, MMDataset
from models import build_model
from utils import assign_gpu, setup_seed

logger = logging.getLogger("DecAlign")
MODALITIES = ["l", "v", "a"]
MOD_NAMES = {"l": "Language", "v": "Vision", "a": "Audio"}


# =====================================================================
# Custom asymmetric missing-rate data loader
# =====================================================================

def create_asymmetric_dataloader(args, missing_rates_per_modality, num_workers=4, mode="test"):
    """
    Create a DataLoader where each modality has a DIFFERENT missing rate.

    Args:
        args: config namespace
        missing_rates_per_modality: dict {'l': 0.8, 'a': 0.2, 'v': 0.2}
        num_workers: number of DataLoader workers
        mode: 'test' or 'valid'

    Returns:
        DataLoader for the requested split
    """
    # We need to create datasets with custom per-modality missing rates.
    # The simplest approach: load the raw dataset, then manually apply
    # missing masks with different rates per modality.
    with open(args.featurePath, "rb") as f:
        data = pickle.load(f)

    split_data = data[mode]

    # Text
    if getattr(args, "use_bert", False):
        text_bert = split_data["text_bert"]
        if text_bert.ndim == 3 and text_bert.shape[1] == 3 and np.issubdtype(text_bert.dtype, np.integer):
            text = text_bert
        else:
            text = text_bert.astype(np.float32)
    else:
        text = split_data["text"].astype(np.float32)

    audio = split_data["audio"].astype(np.float32)
    vision = split_data["vision"].astype(np.float32)

    audio[audio == -np.inf] = 0
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    vision = np.nan_to_num(vision, nan=0.0, posinf=0.0, neginf=0.0)

    labels_M = np.array(split_data["regression_labels" if args.train_mode == "regression" else "classification_labels"]).astype(np.float32)
    raw_text = split_data["raw_text"]
    ids = split_data["id"]

    if getattr(args, "need_data_aligned", False):
        audio_lengths = np.sum(text[:, 1, :], axis=1, dtype=np.int32) if text.ndim == 3 and text.shape[1] == 3 else [audio.shape[1]] * len(audio)
        vision_lengths = audio_lengths.copy() if isinstance(audio_lengths, list) else audio_lengths.copy()
    else:
        audio_lengths = split_data.get("audio_lengths", [audio.shape[1]] * len(audio))
        vision_lengths = split_data.get("vision_lengths", [vision.shape[1]] * len(vision))

    N = len(labels_M)
    seed = getattr(args, "cur_seed", 1111)
    np.random.seed(seed)

    # Missing rates per modality
    mr_l = missing_rates_per_modality["l"]
    mr_a = missing_rates_per_modality["a"]
    mr_v = missing_rates_per_modality["v"]

    # Generate missing masks for text
    input_mask = text[:, 1, :] if text.ndim == 3 and text.shape[1] == 3 else np.ones_like(text[:, :, 0])
    input_len_text = np.argmin(input_mask, axis=1)
    mr_l_arr = mr_l * np.ones((N, 1))
    text_missing_mask = (np.random.uniform(size=input_mask.shape) > mr_l_arr.repeat(input_mask.shape[1], 1)) * input_mask
    for i in range(N):
        if input_len_text[i] > 0:
            text_missing_mask[i, 0] = 1
            text_missing_mask[i, input_len_text[i] - 1] = 1
    text_m = text_missing_mask * text[:, 0, :] + (100 * np.ones_like(text[:, 0, :])) * (input_mask - text_missing_mask)
    Input_ids_m = np.expand_dims(text_m, 1)
    Input_mask = np.expand_dims(text_missing_mask, 1)
    Segment_ids = np.expand_dims(text[:, 2, :], 1)
    text_m_full = np.concatenate((Input_ids_m, Input_mask, Segment_ids), axis=1)

    # Audio missing
    audio_mask = np.array([np.array([1] * int(l) + [0] * (audio.shape[1] - int(l))) for l in audio_lengths])
    mr_a_arr = mr_a * np.ones((N, 1))
    audio_missing_mask = (np.random.uniform(size=audio_mask.shape) > mr_a_arr.repeat(audio_mask.shape[1], 1)) * audio_mask
    audio_m = audio_missing_mask.reshape(audio.shape[0], audio.shape[1], 1) * audio

    # Vision missing
    vision_mask = np.array([np.array([1] * int(l) + [0] * (vision.shape[1] - int(l))) for l in vision_lengths])
    mr_v_arr = mr_v * np.ones((N, 1))
    vision_missing_mask = (np.random.uniform(size=vision_mask.shape) > mr_v_arr.repeat(vision_mask.shape[1], 1)) * vision_mask
    vision_m = vision_missing_mask.reshape(vision.shape[0], vision.shape[1], 1) * vision

    # Build a simple dataset class
    class AsymmetricDataset(torch.utils.data.Dataset):
        def __init__(self):
            pass

        def __len__(self):
            return N

        def __getitem__(self, idx):
            if np.issubdtype(text.dtype, np.integer):
                text_tensor = torch.LongTensor(text[idx])
                text_m_tensor = torch.LongTensor(text_m_full[idx])
            else:
                text_tensor = torch.Tensor(text[idx])
                text_m_tensor = torch.Tensor(text_m_full[idx])

            return {
                "raw_text": raw_text[idx],
                "text": text_tensor,
                "audio": torch.Tensor(audio[idx]),
                "vision": torch.Tensor(vision[idx]),
                "text_m": text_m_tensor,
                "audio_m": torch.Tensor(audio_m[idx]),
                "vision_m": torch.Tensor(vision_m[idx]),
                "index": idx,
                "id": ids[idx],
                "labels": {"M": torch.Tensor(labels_M[idx].reshape(-1))},
            }

    ds = AsymmetricDataset()
    return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)


# =====================================================================
# Collect reliability scores under a given asymmetric setting
# =====================================================================

def collect_reliability(model, dataloader, device):
    """
    Run inference and return per-sample reliability for each modality.

    Returns:
        r_shared:   dict m → (N,) numpy array
        r_specific: dict m → (N,) numpy array
    """
    model.eval()
    r_shared_acc = {m: [] for m in MODALITIES}
    r_specific_acc = {m: [] for m in MODALITIES}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Exp05] Collecting reliability"):
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            text_m = batch["text_m"].to(device)
            audio_m = batch["audio_m"].to(device)
            vision_m = batch["vision_m"].to(device)

            output = model(text_m, audio_m, vision_m, text, audio, vision)

            for m in MODALITIES:
                r_shared_acc[m].append(output["r_shared"][m].cpu())
                r_specific_acc[m].append(output["r_specific"][m].cpu())

    r_shared = {m: torch.cat(r_shared_acc[m]).numpy() for m in MODALITIES}
    r_specific = {m: torch.cat(r_specific_acc[m]).numpy() for m in MODALITIES}
    return r_shared, r_specific


# =====================================================================
# Identification: predict which modality is most corrupted
# =====================================================================

def compute_identification_metrics(all_r_shared, all_r_specific, settings):
    """
    For each sample, predict the corrupted modality as argmin_m r(m).
    Then compute Top-1 accuracy and Macro-F1.

    Args:
        all_r_shared:   list of dicts (one per setting)
        all_r_specific: list of dicts (one per setting)
        settings:       list of dicts with per-modality missing rates

    Returns:
        dict with accuracy and F1 for both shared and specific.
    """
    # Ground-truth: which modality has the highest missing rate in each setting?
    gt_labels = []
    pred_shared = []
    pred_specific = []

    for i, setting in enumerate(settings):
        mr_values = [setting[m] for m in MODALITIES]
        gt_mod = np.argmax(mr_values)  # modality index with highest missing rate
        N = len(all_r_shared[i][MODALITIES[0]])

        for j in range(N):
            gt_labels.append(gt_mod)

            # Predict corrupted modality = one with lowest reliability
            r_shared_vals = [all_r_shared[i][m][j] for m in MODALITIES]
            r_specific_vals = [all_r_specific[i][m][j] for m in MODALITIES]

            pred_shared.append(np.argmin(r_shared_vals))
            pred_specific.append(np.argmin(r_specific_vals))

    gt_labels = np.array(gt_labels)
    pred_shared = np.array(pred_shared)
    pred_specific = np.array(pred_specific)

    return {
        "shared": {
            "accuracy": float(accuracy_score(gt_labels, pred_shared)),
            "macro_f1": float(f1_score(gt_labels, pred_shared, average="macro")),
        },
        "specific": {
            "accuracy": float(accuracy_score(gt_labels, pred_specific)),
            "macro_f1": float(f1_score(gt_labels, pred_specific, average="macro")),
        },
    }


# =====================================================================
# Pretty-print
# =====================================================================

def print_results(settings, all_r_shared, all_r_specific, ident_metrics):
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: Selective / Asymmetric Corruption")
    print("=" * 80)

    print("\n--- Mean Reliability per Modality under Asymmetric Settings ---")
    print(f"{'Setting (mr_L, mr_A, mr_V)':<30} {'rᶜ_L':<8} {'rᶜ_A':<8} {'rᶜ_V':<8} {'rˢ_L':<8} {'rˢ_A':<8} {'rˢ_V':<8}")
    print("-" * 80)

    for i, setting in enumerate(settings):
        label = f"({setting['l']:.1f}, {setting['a']:.1f}, {setting['v']:.1f})"
        rc = {m: np.mean(all_r_shared[i][m]) for m in MODALITIES}
        rs = {m: np.mean(all_r_specific[i][m]) for m in MODALITIES}
        print(
            f"{label:<30} {rc['l']:.4f}  {rc['a']:.4f}  {rc['v']:.4f}  "
            f"{rs['l']:.4f}  {rs['a']:.4f}  {rs['v']:.4f}"
        )

    print("\n--- Corrupted Modality Identification ---")
    print(f"  Shared reliability:   Acc={ident_metrics['shared']['accuracy']:.4f}  Macro-F1={ident_metrics['shared']['macro_f1']:.4f}")
    print(f"  Specific reliability: Acc={ident_metrics['specific']['accuracy']:.4f}  Macro-F1={ident_metrics['specific']['macro_f1']:.4f}")
    print("=" * 80)


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Exp05: Selective / Asymmetric Corruption")
    parser.add_argument("--dataset", type=str, default="mosi", choices=["mosi", "mosei", "sims"])
    parser.add_argument("--config_file", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to a single checkpoint (e.g. checkpoint/dp-mosi-missing0.0.pth)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint")
    parser.add_argument("--base_missing_rate", type=float, default=0.0,
                        help="Missing rate of the checkpoint to load (for naming)")
    parser.add_argument("--save_dir", type=str, default="./result/experiments/exp05")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1111)
    # Asymmetric settings
    parser.add_argument("--high_rate", type=float, default=0.8, help="High corruption rate")
    parser.add_argument("--low_rate", type=float, default=0.2, help="Low corruption rate")
    args = parser.parse_args()

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
    model_args.missing_rate_eval_test = args.base_missing_rate

    # Resolve checkpoint
    if args.checkpoint:
        ckpt = args.checkpoint
    else:
        ckpt = os.path.join(args.checkpoint_dir, f"dp-{args.dataset}-missing{args.base_missing_rate}.pth")

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    setup_seed(args.seed)

    # Load model
    # We need to initialize dataloader once to set feature_dims on model_args
    model_args.missing_rate_eval_test = 0.0
    dl_init = MMDataLoader(model_args, args.num_workers)
    model = build_model(model_args).to(model_args.device)
    model.load_state_dict(torch.load(ckpt, map_location=model_args.device))

    # Build train bank
    logger.info("Building train-set KNN bank...")
    model.update_train_bank(model, dl_init["train"], model_args.device)

    # Define asymmetric settings
    h, lo = args.high_rate, args.low_rate
    settings = [
        {"l": h, "a": lo, "v": lo},  # Language corrupted
        {"l": lo, "a": h, "v": lo},   # Audio corrupted
        {"l": lo, "a": lo, "v": h},   # Vision corrupted
    ]

    all_r_shared = []
    all_r_specific = []

    for setting in settings:
        print(f"\n--- Setting: (L={setting['l']}, A={setting['a']}, V={setting['v']}) ---")
        dl_asym = create_asymmetric_dataloader(model_args, setting, args.num_workers, mode="test")
        r_shared, r_specific = collect_reliability(model, dl_asym, model_args.device)
        all_r_shared.append(r_shared)
        all_r_specific.append(r_specific)

    # Identification metrics
    ident_metrics = compute_identification_metrics(all_r_shared, all_r_specific, settings)

    # Print results
    print_results(settings, all_r_shared, all_r_specific, ident_metrics)

    # Save
    save_data = {
        "settings": settings,
        "all_r_shared": all_r_shared,
        "all_r_specific": all_r_specific,
        "identification_metrics": ident_metrics,
    }
    torch.save(save_data, os.path.join(args.save_dir, f"exp05_{args.dataset}.pt"))

    summary = {
        "settings": settings,
        "mean_reliability": [
            {
                "setting": settings[i],
                "r_shared": {m: float(np.mean(all_r_shared[i][m])) for m in MODALITIES},
                "r_specific": {m: float(np.mean(all_r_specific[i][m])) for m in MODALITIES},
            }
            for i in range(len(settings))
        ],
        "identification": ident_metrics,
    }
    with open(os.path.join(args.save_dir, f"exp05_{args.dataset}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.save_dir}")


if __name__ == "__main__":
    main()
