"""
plot_appendix_calibration_distribution.py
=========================================
Plots the distribution of calibrated residual norms ||z_{m,i}^s||_2 after Eq. (2)
for MR = 0.0, 0.5, and 0.9.

Demonstrates that under Eq. (2), the calibrated vector norm smoothly transitions
from ~0.71 (1 / sqrt(2), intact representation) down towards 0 as missingness increases,
preserving energy degradation rather than being inflated to 1.0.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Project imports
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_config_regression
from data_loader import MMDataLoader
from models import build_model
from paper_style import apply_style, new_fig, save_fig

MOD_NAMES = {"l": "Language ($L$)", "v": "Vision ($V$)", "a": "Audio ($A$)"}
MR_COLORS = {"0.0": "#2ca02c", "0.5": "#1f77b4", "0.9": "#d62728"}

def extract_norms(missing_rates=[0.0, 0.5, 0.9]):
    config_file = str(ROOT_DIR / "config" / "dp_mosi_config.json")
    model_args = get_config_regression("dp", "mosi", config_file, data_dir=str(ROOT_DIR / "data"))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_args.device = device
    model_args.cur_seed = 1111
    model_args.data_missing = True
    model_args.mode = "test"
    
    extracted = {}
    
    for mr in missing_rates:
        mr_str = f"{mr:.1f}"
        model_args.missing_rate_eval_test = mr
        dl = MMDataLoader(model_args, num_workers=2)
        model = build_model(model_args).to(device)
        ckpt_path = str(ROOT_DIR / "checkpoint" / f"dp-mosi-missing{mr_str}.pth")
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint not found {ckpt_path}")
            continue
            
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        model.update_train_bank(model_forward_fn=model, train_dataloader=dl["train"], device=device)
        
        norms = {"l": [], "v": [], "a": []}
        with torch.no_grad():
            for batch in dl["test"]:
                t = batch.get("text_m", batch["text"]).to(device)
                a = batch.get("audio_m", batch["audio"]).to(device)
                v = batch.get("vision_m", batch["vision"]).to(device)
                out = model(t, a, v, t, a, v)
                for m in ["l", "v", "a"]:
                    s_bar = out[f"s_{m}"].mean(dim=0)  # (B, D)
                    mu = model._train_mu_dir_s[m].to(device)
                    sig = model._train_sigma_s[m].to(device)
                    p = s_bar - (s_bar * mu).sum(dim=1, keepdim=True) * mu
                    p_norm = p.norm(dim=1)
                    z_norm = p_norm / torch.sqrt(p_norm**2 + sig**2 + 1e-8)
                    norms[m].append(z_norm.cpu())
                    
        for m in ["l", "v", "a"]:
            norms[m] = torch.cat(norms[m]).numpy()
        extracted[mr_str] = norms
        print(f"Extracted MR {mr_str}: L={norms['l'].mean():.3f}, V={norms['v'].mean():.3f}, A={norms['a'].mean():.3f}")
        
    return extracted

def plot_distributions(data, kind="wide"):
    apply_style(venue="cvpr", kind=kind)
    fig, axes = new_fig(nrows=1, ncols=3, height=2.2 if kind == "wide" else 1.8)
    
    missing_rates = ["0.0", "0.5", "0.9"]
    modalities = ["l", "v", "a"]
    
    for idx, m in enumerate(modalities):
        ax = axes[idx]
        
        # Prepare data for violin / box
        dataset = [data[mr][m] for mr in missing_rates if mr in data]
        positions = [1, 2, 3]
        
        parts = ax.violinplot(
            dataset, positions=positions, showmeans=True, showextrema=True, widths=0.7
        )
        
        # Style violins
        for i, pc in enumerate(parts["bodies"]):
            mr = missing_rates[i]
            pc.set_facecolor(MR_COLORS[mr])
            pc.set_edgecolor("black")
            pc.set_alpha(0.65)
            pc.set_linewidth(0.8)
            
        parts["cmeans"].set_color("black")
        parts["cmeans"].set_linewidth(1.2)
        parts["cmins"].set_color("#444444")
        parts["cmaxes"].set_color("#444444")
        parts["cbars"].set_color("#444444")
        
        # Reference horizontal line at 1/sqrt(2) ≈ 0.707
        ax.axhline(1.0 / np.sqrt(2.0), color="#888888", linestyle=":", linewidth=1.0, alpha=0.8,
                   label=r"$1/\sqrt{2}\approx 0.71$ (Nominal)" if idx == 0 else "")
        # Reference line at 1.0 (Hard Unit Norm)
        ax.axhline(1.0, color="#e66101", linestyle="--", linewidth=1.0, alpha=0.7,
                   label=r"Hard Norm ($1.0$)" if idx == 0 else "")
        
        ax.set_xticks(positions)
        ax.set_xticklabels([r"$\text{MR}=0.0$", r"$\text{MR}=0.5$", r"$\text{MR}=0.9$"])
        ax.set_ylabel(r"Calibrated Norm $\|\mathbf{z}_{m,i}^s\|_2$" if idx == 0 else "")
        ax.set_title(MOD_NAMES[m], pad=3)
        ax.set_ylim(-0.02, 1.08)
        ax.grid(True, linestyle=":", alpha=0.4, axis="y")
        
        if idx == 0:
            ax.legend(loc="lower left", fontsize=6.2 if kind == "wide" else 5.2,
                      frameon=False, borderaxespad=0.2)
            
    out_stem = f"result/figures/fig_appendix_z_norm_distribution_{kind}"
    save_fig(fig, out_stem)
    plt.close(fig)
    print(f"[Done] Generated: {out_stem}.pdf and .png")

if __name__ == "__main__":
    data = extract_norms([0.0, 0.5, 0.9])
    plot_distributions(data, kind="wide")
    plot_distributions(data, kind="narrow")

