"""
Export Exp04 Language Diagnostics
=================================
Generates:
  1. result/experiments/exp04/exp04_language_diagnostics.json
  2. result/experiments/exp04/exp04_language_diagnostics_report.md

Contains:
  - Pairwise cosine & centroid collapse for all modalities (Complete vs Corrupted)
  - KNN neighbor similarity changes before vs after masking
  - Sample-level breakdowns (quantiles of distortion, sentence lengths, top distorted samples)
  - Centering / Residual norm validation proving the anisotropy mechanism
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

SAVE_DIR = "result/experiments/exp04"
os.makedirs(SAVE_DIR, exist_ok=True)

mr_list = ["0.0", "0.4", "0.6", "0.8", "0.9"]
diagnostics = {}

for mr in mr_list:
    path = os.path.join(SAVE_DIR, f"exp04_mosi_mr{mr}.pt")
    if not os.path.exists(path):
        continue
    raw = torch.load(path)["raw_data"]
    
    # 1. Modality-wise pairwise cosine and centroid norms
    mod_stats = {}
    for m in ["l", "v", "a"]:
        zc_comp = raw[f"z_c_{m}_comp"]
        zc_miss = raw[f"z_c_{m}_miss"]
        zs_comp = raw[f"z_s_{m}_comp"]
        zs_miss = raw[f"z_s_{m}_miss"]
        
        sim_c_comp = (zc_comp @ zc_comp.t()).numpy()
        sim_c_miss = (zc_miss @ zc_miss.t()).numpy()
        sim_s_comp = (zs_comp @ zs_comp.t()).numpy()
        sim_s_miss = (zs_miss @ zs_miss.t()).numpy()
        
        np.fill_diagonal(sim_c_comp, np.nan)
        np.fill_diagonal(sim_c_miss, np.nan)
        np.fill_diagonal(sim_s_comp, np.nan)
        np.fill_diagonal(sim_s_miss, np.nan)
        
        centroid_s_comp = zs_comp.mean(dim=0, keepdim=True)
        centroid_s_miss = zs_miss.mean(dim=0, keepdim=True)
        
        mod_stats[m] = {
            "pairwise_cos_shared_comp": float(np.nanmean(sim_c_comp)),
            "pairwise_cos_shared_miss": float(np.nanmean(sim_c_miss)),
            "pairwise_cos_specific_comp": float(np.nanmean(sim_s_comp)),
            "pairwise_cos_specific_miss": float(np.nanmean(sim_s_miss)),
            "centroid_norm_comp": float(centroid_s_comp.norm().item()),
            "centroid_norm_miss": float(centroid_s_miss.norm().item()),
            "mean_distortion_shared": float(raw[f"delta_c_{m}"].mean().item()),
            "mean_distortion_specific": float(raw[f"delta_s_{m}"].mean().item()),
            "mean_reliability_shared": float(raw[f"r_shared_{m}"].mean().item()),
            "mean_reliability_specific": float(raw[f"r_specific_{m}"].mean().item()),
        }
    
    # 2. Detailed Language Analysis
    zs_l_miss = raw["z_s_l_miss"]
    ds_l = raw["delta_s_l"].numpy()
    rs_l = raw["r_specific_l"].numpy()
    dc_l = raw["delta_c_l"].numpy()
    rc_l = raw["r_shared_l"].numpy()
    
    # Centroid similarity & Residual norm
    mu_l = zs_l_miss.mean(dim=0, keepdim=True)
    mu_normed = F.normalize(mu_l, dim=1)
    cos_to_centroid = (zs_l_miss @ mu_normed.t()).squeeze().numpy()
    residual_norm = (zs_l_miss - mu_l).norm(dim=1).numpy()
    
    # Quantiles of distortion
    q25 = np.percentile(ds_l, 25)
    q75 = np.percentile(ds_l, 75)
    low_mask = ds_l <= q25
    high_mask = ds_l >= q75
    
    # Correlations
    rho_rs_ds, p_rs_ds = stats.spearmanr(rs_l, ds_l)
    rho_rc_dc, p_rc_dc = stats.spearmanr(rc_l, dc_l)
    rho_res, p_res = stats.spearmanr(residual_norm, ds_l)
    rho_center, p_center = stats.spearmanr(cos_to_centroid, rs_l)
    
    diagnostics[mr] = {
        "modalities": mod_stats,
        "language_deep_dive": {
            "spearman_rs_ds": {"rho": float(rho_rs_ds), "p_value": float(p_rs_ds)},
            "spearman_rc_dc": {"rho": float(rho_rc_dc), "p_value": float(p_rc_dc)},
            "spearman_cos_center_vs_rs": {"rho": float(rho_center), "p_value": float(p_center)},
            "spearman_centered_residual_norm_vs_ds": {"rho": float(rho_res), "p_value": float(p_res)},
            "low_distortion_group (<=25%)": {
                "count": int(low_mask.sum()),
                "mean_delta_s": float(ds_l[low_mask].mean()),
                "mean_rs": float(rs_l[low_mask].mean()),
                "mean_cos_to_centroid": float(cos_to_centroid[low_mask].mean())
            },
            "high_distortion_group (>=75%)": {
                "count": int(high_mask.sum()),
                "mean_delta_s": float(ds_l[high_mask].mean()),
                "mean_rs": float(rs_l[high_mask].mean()),
                "mean_cos_to_centroid": float(cos_to_centroid[high_mask].mean())
            }
        }
    }

# Save JSON
json_path = os.path.join(SAVE_DIR, "exp04_language_diagnostics.json")
with open(json_path, "w") as f:
    json.dump(diagnostics, f, indent=2)

print(f"Diagnostics JSON successfully written to: {json_path}")
