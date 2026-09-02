#!/usr/bin/env python3
"""Probe whether modality conflict predicts MISA reconstruction behaviour.

The probe is deliberately observational: it never updates the checkpoint.  It
exports one row per (sample, target modality), so later analyses can replace
the conflict score without rerunning the backbone.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

# Allow `python scripts/analyze_misa_reconstruction.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_config_regression
from data_loader import MMDataLoader
from models import build_model
from utils.functions import setup_seed


MODALITIES = ("t", "v", "a")


def conflict(features):
    """CACR conflict: one minus mean cosine similarity to the other modalities."""
    normalized = {m: torch.nn.functional.normalize(features[m], dim=1, eps=1e-8)
                  for m in MODALITIES}
    values = {}
    for m in MODALITIES:
        others = [n for n in MODALITIES if n != m]
        similarity = sum((normalized[m] * normalized[n]).sum(dim=1) for n in others) / 2
        values[m] = 1.0 - similarity
    return values


def mean_cross_cosine(features):
    normalized = {m: torch.nn.functional.normalize(features[m], dim=1, eps=1e-8)
                  for m in MODALITIES}
    return {
        m: sum((normalized[m] * normalized[n]).sum(dim=1)
               for n in MODALITIES if n != m) / 2
        for m in MODALITIES
    }


def rank(values):
    """Ordinal ranks are sufficient here: these floating-point signals have no ties."""
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    result[order] = np.arange(len(values), dtype=float)
    return result


def correlations(rows):
    summary = {}
    for modality in MODALITIES:
        subset = [r for r in rows if r["modality"] == modality]
        x = np.asarray([r["conflict_shared"] for r in subset])
        targets = ("recon_mse", "latent_grad_norm", "alignment_change", "recon_delta_norm")
        summary[modality] = {}
        for target in targets:
            y = np.asarray([r[target] for r in subset])
            summary[modality][target] = {
                "pearson_r": float(np.corrcoef(x, y)[0, 1]),
                "spearman_rho": float(np.corrcoef(rank(x), rank(y))[0, 1]),
                "n": len(subset),
            }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("pt/misa-mosi-missing0.0.pth"))
    parser.add_argument("--dataset", default="mosi", choices=("mosi", "mosei", "sims", "iemocap"))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--config-file", default="config/misa_mosi_config.json")
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("result/reconstruction_probe"))
    return parser.parse_args()


def main():
    args_cli = parse_args()
    if not args_cli.checkpoint.is_file():
        raise FileNotFoundError(args_cli.checkpoint)
    if args_cli.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Pass --device cpu.")

    setup_seed(args_cli.seed)
    cfg = get_config_regression("misa", args_cli.dataset, args_cli.config_file, args_cli.data_dir)
    # The probe measures ordinary reconstruction, not missing-modality corruption.
    cfg.batch_size = args_cli.batch_size
    cfg.data_missing = False
    cfg.missing_rate_eval_test = 0.0
    cfg.cur_seed = args_cli.seed
    cfg.device = torch.device(args_cli.device)
    dataloaders = MMDataLoader(cfg, num_workers=0)
    model = build_model(cfg).to(cfg.device)
    state = torch.load(args_cli.checkpoint, map_location=cfg.device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    rows = []
    for batch in dataloaders[args_cli.split]:
        text = batch["text"].to(cfg.device)
        audio = batch["audio"].to(cfg.device)
        vision = batch["vision"].to(cfg.device)
        # Gradients are only used to measure sensitivity, never accumulated or stepped.
        model.zero_grad(set_to_none=True)
        model(text, audio, vision)
        original = {m: getattr(model, f"utt_{m}_orig") for m in MODALITIES}
        shared = {m: getattr(model, f"utt_shared_{m}") for m in MODALITIES}
        reconstructed = {m: getattr(model, f"utt_{m}_recon") for m in MODALITIES}
        latent = {m: getattr(model, f"utt_{m}") for m in MODALITIES}
        shared_conflict = conflict(shared)
        orig_alignment = mean_cross_cosine(original)
        recon_alignment = mean_cross_cosine(reconstructed)
        recon_conflict = conflict(reconstructed)

        batch_size = text.shape[0]
        per_sample_losses = {
            m: (reconstructed[m] - original[m]).square().mean(dim=1) for m in MODALITIES
        }
        # d ell_i / d(private_i + shared_i), retaining the graph until all modalities are read.
        latent_grads = {
            m: torch.autograd.grad(per_sample_losses[m].sum(), latent[m], retain_graph=m != "a")[0]
            .norm(dim=1)
            for m in MODALITIES
        }
        ids = batch["id"]
        labels = batch["labels"]["M"].view(batch_size).cpu().numpy()
        for m in MODALITIES:
            displacement = (reconstructed[m] - original[m]).norm(dim=1)
            for i in range(batch_size):
                rows.append({
                    "id": str(ids[i]), "label": float(labels[i]), "modality": m,
                    "conflict_shared": float(shared_conflict[m][i].detach().cpu()),
                    "conflict_recon": float(recon_conflict[m][i].detach().cpu()),
                    "recon_mse": float(per_sample_losses[m][i].detach().cpu()),
                    "latent_grad_norm": float(latent_grads[m][i].detach().cpu()),
                    "recon_delta_norm": float(displacement[i].detach().cpu()),
                    "orig_mean_cosine": float(orig_alignment[m][i].detach().cpu()),
                    "recon_mean_cosine": float(recon_alignment[m][i].detach().cpu()),
                    "alignment_change": float((recon_alignment[m][i] - orig_alignment[m][i]).detach().cpu()),
                })

    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / f"{args_cli.dataset}_{args_cli.split}_per_sample.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(args_cli.checkpoint), "split": args_cli.split,
        "score_definition": "1 - mean cosine(shared_m, shared_other)",
        "correlations": correlations(rows),
    }
    summary_path = args_cli.output_dir / f"{args_cli.dataset}_{args_cli.split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {csv_path} and {summary_path}")
    print(json.dumps(summary["correlations"], indent=2))


if __name__ == "__main__":
    main()
