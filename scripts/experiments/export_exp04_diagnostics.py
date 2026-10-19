"""
Export Exp04 Language Diagnostics
=================================
Generates:
  1. result/experiments/exp04/exp04_language_diagnostics.json
  2. result/experiments/exp04/exp04_language_diagnostics_report.md

Contains:
  - SARM KNN References direct cosine measurement (Query z_m^s vs Bank KNN references) across MRs
  - Modality comparison (Language vs Vision vs Audio)
  - Embedding concentration (Pairwise cosine & Centroid norm)
  - Sample-level breakdowns (quantiles of distortion & centroid proximity)
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

# SARM KNN direct measurements from task-321
SARM_KNN = {
    "0.0": {
        "l": {"knn_cos_comp": 0.5321, "knn_cos_miss": 0.5335, "diff_knn_cos": +0.0014, "r_s_comp": 0.4257, "r_s_miss": 0.4272, "diff_r_s": +0.0015, "query_bank_sim_comp": 0.8935, "query_bank_sim_miss": 0.8935},
        "v": {"knn_cos_comp": 0.5173, "knn_cos_miss": 0.5153, "diff_knn_cos": -0.0020, "r_s_comp": 0.3893, "r_s_miss": 0.3878, "diff_r_s": -0.0016, "query_bank_sim_comp": 0.9725, "query_bank_sim_miss": 0.9725},
        "a": {"knn_cos_comp": 0.5277, "knn_cos_miss": 0.5296, "diff_knn_cos": +0.0019, "r_s_comp": 0.4208, "r_s_miss": 0.4227, "diff_r_s": +0.0020, "query_bank_sim_comp": 0.9340, "query_bank_sim_miss": 0.9339},
    },
    "0.4": {
        "l": {"knn_cos_comp": 0.5221, "knn_cos_miss": 0.5369, "diff_knn_cos": +0.0149, "r_s_comp": 0.4195, "r_s_miss": 0.4272, "diff_r_s": +0.0077, "query_bank_sim_comp": 0.9410, "query_bank_sim_miss": 0.9368},
        "v": {"knn_cos_comp": 0.5333, "knn_cos_miss": 0.5174, "diff_knn_cos": -0.0159, "r_s_comp": 0.4029, "r_s_miss": 0.3912, "diff_r_s": -0.0117, "query_bank_sim_comp": 0.9799, "query_bank_sim_miss": 0.9780},
        "a": {"knn_cos_comp": 0.5965, "knn_cos_miss": 0.5799, "diff_knn_cos": -0.0166, "r_s_comp": 0.4745, "r_s_miss": 0.4643, "diff_r_s": -0.0101, "query_bank_sim_comp": 0.9284, "query_bank_sim_miss": 0.9267},
    },
    "0.6": {
        "l": {"knn_cos_comp": 0.4693, "knn_cos_miss": 0.5085, "diff_knn_cos": +0.0392, "r_s_comp": 0.3819, "r_s_miss": 0.4068, "diff_r_s": +0.0248, "query_bank_sim_comp": 0.9014, "query_bank_sim_miss": 0.8719},
        "v": {"knn_cos_comp": 0.4887, "knn_cos_miss": 0.4601, "diff_knn_cos": -0.0286, "r_s_comp": 0.3698, "r_s_miss": 0.3535, "diff_r_s": -0.0163, "query_bank_sim_comp": 0.9759, "query_bank_sim_miss": 0.9700},
        "a": {"knn_cos_comp": 0.5461, "knn_cos_miss": 0.4780, "diff_knn_cos": -0.0681, "r_s_comp": 0.4493, "r_s_miss": 0.4065, "diff_r_s": -0.0428, "query_bank_sim_comp": 0.9111, "query_bank_sim_miss": 0.9034},
    },
    "0.8": {
        "l": {"knn_cos_comp": 0.2970, "knn_cos_miss": 0.3900, "diff_knn_cos": +0.0929, "r_s_comp": 0.2833, "r_s_miss": 0.3226, "diff_r_s": +0.0393, "query_bank_sim_comp": 0.8898, "query_bank_sim_miss": 0.7550},
        "v": {"knn_cos_comp": 0.4650, "knn_cos_miss": 0.3744, "diff_knn_cos": -0.0906, "r_s_comp": 0.3526, "r_s_miss": 0.3068, "diff_r_s": -0.0457, "query_bank_sim_comp": 0.9667, "query_bank_sim_miss": 0.9471},
        "a": {"knn_cos_comp": 0.5288, "knn_cos_miss": 0.4052, "diff_knn_cos": -0.1236, "r_s_comp": 0.4399, "r_s_miss": 0.3710, "diff_r_s": -0.0690, "query_bank_sim_comp": 0.8947, "query_bank_sim_miss": 0.8846},
    },
    "0.9": {
        "l": {"knn_cos_comp": 0.2584, "knn_cos_miss": 0.4267, "diff_knn_cos": +0.1682, "r_s_comp": 0.2618, "r_s_miss": 0.3454, "diff_r_s": +0.0836, "query_bank_sim_comp": 0.9232, "query_bank_sim_miss": 0.5855},
        "v": {"knn_cos_comp": 0.4683, "knn_cos_miss": 0.2695, "diff_knn_cos": -0.1988, "r_s_comp": 0.3534, "r_s_miss": 0.2554, "diff_r_s": -0.0980, "query_bank_sim_comp": 0.9713, "query_bank_sim_miss": 0.9626},
        "a": {"knn_cos_comp": 0.5899, "knn_cos_miss": 0.3210, "diff_knn_cos": -0.2689, "r_s_comp": 0.4736, "r_s_miss": 0.3116, "diff_r_s": -0.1620, "query_bank_sim_comp": 0.9047, "query_bank_sim_miss": 0.9227},
    },
}

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
            "sarm_knn_refs": SARM_KNN[mr][m]
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
