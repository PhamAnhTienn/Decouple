"""
Experiment 14 – Efficiency & Reproducibility
=============================================
Report computational overhead of the full SARM model compared to the backbone.

Metrics reported:
  - Total trainable parameters
  - SARM-specific parameters (CPA prototypes + memory bank + reliability modules)
  - Train time per epoch (seconds)
  - Inference time per sample (milliseconds)
  - Peak GPU memory (MB)
  - Reference-memory bank size (MB)
  - (Optional) KNN retrieval latency per batch (ms)

Output
------
* Appendix table comparing Backbone vs SARM
* Saved to result/experiments/exp14/
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import get_config_regression
from data_loader import MMDataLoader
from models import build_model
from utils import assign_gpu, setup_seed, count_parameters

logger = logging.getLogger("DecAlign")


# =====================================================================
# Parameter counting
# =====================================================================

def count_param_groups(model):
    """
    Count parameters by component group.
    Returns dict of {group_name: num_params}.
    """
    groups = {
        "total": 0,
        "bert": 0,
        "projections": 0,
        "shared_encoder": 0,
        "specific_encoders": 0,
        "decoders_crm": 0,
        "fusion_layers": 0,
        "fsr": 0,
        "prototypes_cpa": 0,
        "cross_attention_specific": 0,
        "self_attention_shared": 0,
        "prediction_heads": 0,
        "other": 0,
    }

    for name, param in model.named_parameters():
        n = param.numel()
        groups["total"] += n

        if "text_model" in name or "bert" in name.lower():
            groups["bert"] += n
        elif "proj_l" in name or "proj_a" in name or "proj_v" in name:
            groups["projections"] += n
        elif "encoder_c" in name:
            groups["shared_encoder"] += n
        elif "encoder_s_" in name:
            groups["specific_encoders"] += n
        elif "decoder_" in name:
            groups["decoders_crm"] += n
        elif "fusion_" in name:
            groups["fusion_layers"] += n
        elif "fsr" in name:
            groups["fsr"] += n
        elif "prototype" in name:
            groups["prototypes_cpa"] += n
        elif "trans_l_with_" in name or "trans_a_with_" in name or "trans_v_with_" in name:
            groups["cross_attention_specific"] += n
        elif "self_attentions_c_" in name or "trans_shared" in name:
            groups["self_attention_shared"] += n
        elif "proj1" in name or "proj2" in name or "out_layer" in name or "weight_" in name:
            groups["prediction_heads"] += n
        elif "trans_l_mem" in name or "trans_a_mem" in name or "trans_v_mem" in name:
            groups["cross_attention_specific"] += n
        else:
            groups["other"] += n

    # SARM-specific = everything except BERT + basic projections
    groups["sarm_specific"] = (
        groups["total"] - groups["bert"] - groups["projections"]
    )
    # Backbone = BERT + projections + basic encoders + basic prediction
    groups["backbone_approx"] = (
        groups["bert"] + groups["projections"] + groups["shared_encoder"]
        + groups["specific_encoders"]
    )

    return groups


# =====================================================================
# Memory bank size
# =====================================================================

def compute_bank_size(model):
    """
    Compute the memory bank size in bytes and MB.
    """
    total_bytes = 0
    for key in ["l", "v", "a"]:
        if key in model._train_bank_z_c:
            total_bytes += model._train_bank_z_c[key].nelement() * model._train_bank_z_c[key].element_size()
        if key in model._train_bank_z_s:
            total_bytes += model._train_bank_z_s[key].nelement() * model._train_bank_z_s[key].element_size()

    return {
        "bytes": total_bytes,
        "MB": total_bytes / (1024 ** 2),
        "entries": model._train_bank_z_c["l"].shape[0] if "l" in model._train_bank_z_c else 0,
        "dim": model._train_bank_z_c["l"].shape[1] if "l" in model._train_bank_z_c else 0,
    }


# =====================================================================
# Timing: train epoch
# =====================================================================

def time_train_epoch(model, dataloader, device, max_batches=None):
    """
    Time a single training epoch (forward + backward, no optimizer step).
    Returns seconds per epoch.
    """
    model.train()
    criterion = torch.nn.L1Loss()

    torch.cuda.synchronize()
    start = time.time()

    n_batches = 0
    for batch in tqdm(dataloader, desc="[Exp14] Timing train epoch"):
        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)
        text_m = batch.get("text_m", batch["text"]).to(device)
        audio_m = batch.get("audio_m", batch["audio"]).to(device)
        vision_m = batch.get("vision_m", batch["vision"]).to(device)
        labels = batch["labels"]["M"].to(device).view(-1, 1)

        output = model(text_m, audio_m, vision_m, text, audio, vision)
        loss = criterion(output["output_logit"], labels)
        loss.backward()

        # Zero gradients (no optimizer step)
        model.zero_grad()

        n_batches += 1
        if max_batches and n_batches >= max_batches:
            break

    torch.cuda.synchronize()
    elapsed = time.time() - start

    return elapsed, n_batches


# =====================================================================
# Timing: inference
# =====================================================================

def time_inference(model, dataloader, device, max_batches=None):
    """
    Time inference (forward only).
    Returns total time, number of samples, per-sample latency.
    """
    model.eval()

    torch.cuda.synchronize()
    start = time.time()

    n_samples = 0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Exp14] Timing inference"):
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            text_m = batch.get("text_m", batch["text"]).to(device)
            audio_m = batch.get("audio_m", batch["audio"]).to(device)
            vision_m = batch.get("vision_m", batch["vision"]).to(device)

            _ = model(text_m, audio_m, vision_m, text, audio, vision)
            n_samples += text.shape[0]
            n_batches += 1
            if max_batches and n_batches >= max_batches:
                break

    torch.cuda.synchronize()
    elapsed = time.time() - start

    return elapsed, n_samples


# =====================================================================
# Timing: KNN retrieval
# =====================================================================

def time_knn_retrieval(model, dataloader, device, K=10, max_batches=5):
    """
    Time KNN retrieval specifically (bank lookup).
    """
    model.eval()
    times = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            text_m = batch.get("text_m", batch["text"]).to(device)
            audio_m = batch.get("audio_m", batch["audio"]).to(device)
            vision_m = batch.get("vision_m", batch["vision"]).to(device)

            # Run through encoders to get z_c, z_s
            output = model(text_m, audio_m, vision_m, text, audio, vision)

            # Extract z_c and z_s
            z_c = {m: F.normalize(output[f"c_{m}"].mean(dim=0), dim=1) for m in ["l", "v", "a"]}
            z_s = {m: F.normalize(output[f"s_{m}"].mean(dim=0), dim=1) for m in ["l", "v", "a"]}

            # Time just the bank lookup
            torch.cuda.synchronize()
            t0 = time.time()
            _ = model._neighbor_uncertainty_bank(z_c, z_s, K)
            torch.cuda.synchronize()
            t1 = time.time()

            times.append((t1 - t0) * 1000)  # ms

    return np.mean(times) if times else 0.0


# =====================================================================
# Peak GPU memory
# =====================================================================

def measure_peak_memory(model, dataloader, device, max_batches=3):
    """
    Measure peak GPU memory during a few forward + backward passes.
    """
    model.train()
    criterion = torch.nn.L1Loss()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    gc.collect()

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)
        text_m = batch.get("text_m", batch["text"]).to(device)
        audio_m = batch.get("audio_m", batch["audio"]).to(device)
        vision_m = batch.get("vision_m", batch["vision"]).to(device)
        labels = batch["labels"]["M"].to(device).view(-1, 1)

        output = model(text_m, audio_m, vision_m, text, audio, vision)
        loss = criterion(output["output_logit"], labels)
        loss.backward()
        model.zero_grad()

    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes / (1024 ** 2)  # MB


# =====================================================================
# Pretty-print
# =====================================================================

def print_table(results):
    print("\n" + "=" * 70)
    print("EXPERIMENT 14: Efficiency & Reproducibility")
    print("=" * 70)

    print(f"\n{'Metric':<40} {'Value':<30}")
    print("-" * 70)

    pg = results["param_groups"]
    print(f"{'Total Parameters':<40} {pg['total']:>15,}")
    print(f"{'  BERT':<40} {pg['bert']:>15,}")
    print(f"{'  Projections':<40} {pg['projections']:>15,}")
    print(f"{'  Shared Encoder':<40} {pg['shared_encoder']:>15,}")
    print(f"{'  Specific Encoders':<40} {pg['specific_encoders']:>15,}")
    print(f"{'  CRM Decoders':<40} {pg['decoders_crm']:>15,}")
    print(f"{'  Fusion Layers':<40} {pg['fusion_layers']:>15,}")
    print(f"{'  FSR':<40} {pg['fsr']:>15,}")
    print(f"{'  CPA Prototypes':<40} {pg['prototypes_cpa']:>15,}")
    print(f"{'  Cross-Attention (Specific)':<40} {pg['cross_attention_specific']:>15,}")
    print(f"{'  Self-Attention (Shared)':<40} {pg['self_attention_shared']:>15,}")
    print(f"{'  Prediction Heads':<40} {pg['prediction_heads']:>15,}")
    print(f"{'  Other':<40} {pg['other']:>15,}")
    print(f"{'  SARM-Specific (non-backbone)':<40} {pg['sarm_specific']:>15,}")
    print(f"{'  Backbone Approx':<40} {pg['backbone_approx']:>15,}")

    print()
    t = results["timing"]
    print(f"{'Train Time / Epoch (s)':<40} {t['train_time_per_epoch']:.2f}")
    print(f"{'Inference Time / Sample (ms)':<40} {t['inference_ms_per_sample']:.3f}")
    print(f"{'Peak GPU Memory (MB)':<40} {t['peak_gpu_memory_mb']:.1f}")

    if "bank" in results:
        b = results["bank"]
        print(f"{'Memory Bank Size (MB)':<40} {b['MB']:.2f}")
        print(f"{'Memory Bank Entries':<40} {b['entries']}")
        print(f"{'Memory Bank Dim':<40} {b['dim']}")

    if "knn_latency_ms" in results:
        print(f"{'KNN Retrieval Latency / Batch (ms)':<40} {results['knn_latency_ms']:.2f}")

    print("=" * 70)


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Exp14: Efficiency & Reproducibility")
    parser.add_argument("--dataset", type=str, default="mosi", choices=["mosi", "mosei", "sims"])
    parser.add_argument("--config_file", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint")
    parser.add_argument("--missing_rate", type=float, default=0.0)
    parser.add_argument("--save_dir", type=str, default="./result/experiments/exp14")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--max_train_batches", type=int, default=None,
                        help="Limit train batches for timing (None = full epoch)")
    parser.add_argument("--max_infer_batches", type=int, default=None,
                        help="Limit inference batches for timing")
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
    model_args.missing_rate_eval_test = args.missing_rate

    ckpt_path = os.path.join(args.checkpoint_dir, f"dp-{args.dataset}-missing{args.missing_rate}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    setup_seed(args.seed)

    dataloader = MMDataLoader(model_args, args.num_workers)
    model = build_model(model_args).to(model_args.device)
    model.load_state_dict(torch.load(ckpt_path, map_location=model_args.device))

    results = {}

    # 1. Parameter groups
    print("\n[1/5] Counting parameters...")
    results["param_groups"] = count_param_groups(model)

    # 2. Build train bank + measure bank size
    print("[2/5] Building train bank & measuring size...")
    model.update_train_bank(model, dataloader["train"], model_args.device)
    results["bank"] = compute_bank_size(model)

    # 3. KNN retrieval latency
    print("[3/5] Measuring KNN retrieval latency...")
    results["knn_latency_ms"] = time_knn_retrieval(
        model, dataloader["test"], model_args.device,
        K=getattr(model_args, "K_neighbors", 10),
    )

    # 4. Train epoch timing
    print("[4/5] Timing train epoch...")
    train_time, n_train_batches = time_train_epoch(
        model, dataloader["train"], model_args.device,
        max_batches=args.max_train_batches,
    )
    total_train_batches = len(dataloader["train"])
    if args.max_train_batches and n_train_batches < total_train_batches:
        train_time_full_epoch = train_time * (total_train_batches / n_train_batches)
    else:
        train_time_full_epoch = train_time

    # 5. Inference timing
    print("[5/5] Timing inference...")
    infer_time, n_infer_samples = time_inference(
        model, dataloader["test"], model_args.device,
        max_batches=args.max_infer_batches,
    )

    # 6. Peak memory
    print("[Bonus] Measuring peak GPU memory...")
    peak_mem = measure_peak_memory(model, dataloader["train"], model_args.device)

    results["timing"] = {
        "train_time_per_epoch": train_time_full_epoch,
        "train_batches_measured": n_train_batches,
        "train_batches_total": total_train_batches,
        "inference_total_time": infer_time,
        "inference_samples": n_infer_samples,
        "inference_ms_per_sample": (infer_time / max(n_infer_samples, 1)) * 1000,
        "peak_gpu_memory_mb": peak_mem,
    }

    # Print
    print_table(results)

    # Save
    save_path = os.path.join(args.save_dir, f"exp14_{args.dataset}_summary.json")
    # Convert to JSON-serializable
    json_results = json.loads(json.dumps(results, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
    with open(save_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
