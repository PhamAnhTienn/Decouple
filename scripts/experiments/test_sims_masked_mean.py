import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import get_config_regression
from data_loader import MMDataLoader
from models import build_model
from utils import setup_seed

def get_lengths_and_masks(batch, device):
    # Text length: sum of bert input_mask
    text = batch['text']
    if text.dim() == 3 and text.shape[1] == 3:
        # text is (B, 3, T): input_ids, input_mask, segment_ids
        l_lens = text[:, 1, :].sum(dim=1).long().to(device)
    else:
        l_lens = torch.full((text.shape[0],), text.shape[1], dtype=torch.long, device=device)
    
    if 'audio_lengths' in batch:
        a_lens = batch['audio_lengths'].long().to(device)
    else:
        a_lens = torch.full((batch['audio'].shape[0],), batch['audio'].shape[1], dtype=torch.long, device=device)
        
    if 'vision_lengths' in batch:
        v_lens = batch['vision_lengths'].long().to(device)
    else:
        v_lens = torch.full((batch['vision'].shape[0],), batch['vision'].shape[1], dtype=torch.long, device=device)
        
    return {'l': l_lens, 'v': v_lens, 'a': a_lens}

def masked_mean_tbd(x, lengths):
    """
    x: (T, B, D)
    lengths: (B,)
    returns: (B, D)
    """
    T, B, D = x.shape
    device = x.device
    # mask: (T, B)
    mask = torch.arange(T, device=device).unsqueeze(1) < lengths.to(device).unsqueeze(0)
    mask = mask.unsqueeze(2).float() # (T, B, 1)
    masked_sum = (x * mask).sum(dim=0) # (B, D)
    clamped_lens = lengths.to(device).float().clamp(min=1).unsqueeze(1) # (B, 1)
    return masked_sum / clamped_lens

def build_bank_custom(model, train_loader, device, use_masked=False, eps=1e-8):
    model.eval()
    bank_c = {'l': [], 'v': [], 'a': []}
    bank_s = {'l': [], 'v': [], 'a': []}
    
    with torch.no_grad():
        for batch in train_loader:
            text     = batch['text'].to(device)
            text_m   = batch.get('text_m', text).to(device)
            audio    = batch['audio'].to(device)
            audio_m  = batch.get('audio_m', audio).to(device)
            vision   = batch['vision'].to(device)
            vision_m = batch.get('vision_m', vision).to(device)
            
            output = model(text_m, audio_m, vision_m, text, audio, vision)
            lens = get_lengths_and_masks(batch, device)
            
            for m in ['l', 'v', 'a']:
                c_feat = output[f'c_{m}'] # (T, B, D)
                s_feat = output[f's_{m}'] # (T, B, D)
                if use_masked:
                    c_bar_b = masked_mean_tbd(c_feat, lens[m]).cpu()
                    s_bar_b = masked_mean_tbd(s_feat, lens[m]).cpu()
                else:
                    c_bar_b = c_feat.mean(dim=0).cpu()
                    s_bar_b = s_feat.mean(dim=0).cpu()
                bank_c[m].append(c_bar_b)
                bank_s[m].append(s_bar_b)
                
    mu_dir_c, sigma_c = {}, {}
    mu_dir_s, sigma_s = {}, {}
    bank_z_s = {}
    
    for m in ['l', 'v', 'a']:
        all_c = torch.cat(bank_c[m], dim=0)
        z_c_unit = F.normalize(all_c, dim=1, eps=eps)
        mu_c = F.normalize(z_c_unit.mean(dim=0, keepdim=True), dim=1, eps=eps)
        p_c = all_c - (all_c * mu_c).sum(dim=1, keepdim=True) * mu_c
        mu_dir_c[m] = mu_c
        sigma_c[m] = p_c.norm(p=2, dim=1).mean().clamp(min=eps)
        
        all_s = torch.cat(bank_s[m], dim=0)
        z_s_unit = F.normalize(all_s, dim=1, eps=eps)
        mu_s = F.normalize(z_s_unit.mean(dim=0, keepdim=True), dim=1, eps=eps)
        p_s = all_s - (all_s * mu_s).sum(dim=1, keepdim=True) * mu_s
        mu_dir_s[m] = mu_s
        sigma_s[m] = p_s.norm(p=2, dim=1).mean().clamp(min=eps)
        bank_z_s[m] = F.normalize(p_s, dim=1, eps=eps)
        
    return mu_dir_c, sigma_c, mu_dir_s, sigma_s, bank_z_s

def compute_uncertainty_and_reliability(c_bar, s_bar, mu_dir_c, sigma_c, mu_dir_s, sigma_s, bank_z_s, tau_c=1.0, tau_s=1.0, K=10, eps=1e-8):
    # Soft normalize
    z_ring_c = {}
    for m, v_bar in c_bar.items():
        mu = mu_dir_c[m].to(v_bar.device)
        sig = sigma_c[m].to(v_bar.device)
        proj_coeff = (v_bar * mu).sum(dim=1, keepdim=True)
        p_m = v_bar - proj_coeff * mu
        denom = torch.sqrt(p_m.pow(2).sum(dim=1, keepdim=True) + sig.pow(2) + eps)
        z_ring_c[m] = p_m / denom

    z_ring_s = {}
    for m, v_bar in s_bar.items():
        mu = mu_dir_s[m].to(v_bar.device)
        sig = sigma_s[m].to(v_bar.device)
        proj_coeff = (v_bar * mu).sum(dim=1, keepdim=True)
        p_m = v_bar - proj_coeff * mu
        denom = torch.sqrt(p_m.pow(2).sum(dim=1, keepdim=True) + sig.pow(2) + eps)
        z_ring_s[m] = p_m / denom

    # Consensus uncertainty
    u_c = {}
    for m in ['l', 'v', 'a']:
        others_sum = sum(z_ring_c[n] for n in ['l', 'v', 'a'] if n != m)
        con_dir = F.normalize(others_sum, dim=1, eps=eps)
        u_c[m] = 1.0 - (z_ring_c[m] * con_dir).sum(dim=1)

    # Specific uncertainty
    u_s = {}
    for m in ['l', 'v', 'a']:
        z_q = z_ring_s[m]
        b_z = bank_z_s[m].to(z_q.device)
        sim = z_q @ b_z.t()
        k_act = min(K, b_z.shape[0])
        topk, _ = sim.topk(k_act, dim=1)
        u_s[m] = 1.0 - topk.mean(dim=1)

    # Reliability
    r_c = {m: torch.clamp(torch.exp(-u_c[m] / tau_c), min=1e-4, max=1.0) for m in ['l', 'v', 'a']}
    r_s = {m: torch.clamp(torch.exp(-u_s[m] / tau_s), min=1e-4, max=1.0) for m in ['l', 'v', 'a']}

    gamma_c = 1.0 - (1.0 - r_c['l']) * (1.0 - r_c['v']) * (1.0 - r_c['a'])

    # Norms
    norm_c_bar = {m: c_bar[m].norm(p=2, dim=1) for m in ['l', 'v', 'a']}
    norm_s_bar = {m: s_bar[m].norm(p=2, dim=1) for m in ['l', 'v', 'a']}
    norm_z_ring_c = {m: z_ring_c[m].norm(p=2, dim=1) for m in ['l', 'v', 'a']}
    norm_z_ring_s = {m: z_ring_s[m].norm(p=2, dim=1) for m in ['l', 'v', 'a']}

    return {
        'u_c': u_c, 'u_s': u_s,
        'r_c': r_c, 'r_s': r_s,
        'gamma_c': gamma_c,
        'norm_c_bar': norm_c_bar, 'norm_s_bar': norm_s_bar,
        'norm_z_ring_c': norm_z_ring_c, 'norm_z_ring_s': norm_z_ring_s
    }

def evaluate_pipeline(config_file='config/dp_sims_config.json', ckpt_path='checkpoint/dp-sims-missing0.0.pth'):
    setup_seed(1111)
    args = get_config_regression('dp', 'sims', config_file, data_dir='./data')
    args.dst_feature_dim_nheads = [48, 6]
    args.nmb_prototypes = 5
    args.need_data_aligned = False
    args.featurePath = './data/SIMS/unaligned_39.pkl'
    args.cur_seed = 1111
    args.mode = 'test'
    args.data_missing = True
    args.missing_rate_eval_test = 0.0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device

    print("Building model & dataloaders...")
    model = build_model(args).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    dataloader = MMDataLoader(args, 2)
    train_loader = dataloader['train']
    test_loader = dataloader['test']

    print("1. Building unmasked train bank...")
    u_mu_c, u_sig_c, u_mu_s, u_sig_s, u_bank_s = build_bank_custom(model, train_loader, device, use_masked=False)

    print("2. Building masked train bank...")
    m_mu_c, m_sig_c, m_mu_s, m_sig_s, m_bank_s = build_bank_custom(model, train_loader, device, use_masked=True)

    print(f"Sigmas (Unmasked bank): c_sig={ {m: u_sig_c[m].item() for m in u_sig_c} }, s_sig={ {m: u_sig_s[m].item() for m in u_sig_s} }")
    print(f"Sigmas (Masked bank):   c_sig={ {m: m_sig_c[m].item() for m in m_sig_c} }, s_sig={ {m: m_sig_s[m].item() for m in m_sig_s} }")

    # Evaluate on test set under 3 configurations:
    # A: Unmasked pooling (Baseline)
    # B: Masked pooling + Unmasked bank (naive masked test)
    # C: Masked pooling + Masked bank (consistent masked system)
    results = {'A': [], 'B': [], 'C': []}
    unimodal_labels = {'l': [], 'v': [], 'a': [], 'm': []}

    print("3. Evaluating on test set...")
    with torch.no_grad():
        for batch in test_loader:
            text     = batch['text'].to(device)
            text_m   = batch.get('text_m', text).to(device)
            audio    = batch['audio'].to(device)
            audio_m  = batch.get('audio_m', audio).to(device)
            vision   = batch['vision'].to(device)
            vision_m = batch.get('vision_m', vision).to(device)
            lens     = get_lengths_and_masks(batch, device)

            output = model(text_m, audio_m, vision_m, text, audio, vision)

            # Collect unimodal labels
            for m_key, k in [('l', 'T'), ('v', 'V'), ('a', 'A')]:
                if k in batch['labels']:
                    unimodal_labels[m_key].append(batch['labels'][k].cpu())
            unimodal_labels['m'].append(batch['labels']['M'].cpu())

            # Unmasked c_bar, s_bar
            c_bar_unmasked = {m: output[f'c_{m}'].mean(dim=0) for m in ['l', 'v', 'a']}
            s_bar_unmasked = {m: output[f's_{m}'].mean(dim=0) for m in ['l', 'v', 'a']}

            # Masked c_bar, s_bar
            c_bar_masked = {m: masked_mean_tbd(output[f'c_{m}'], lens[m]) for m in ['l', 'v', 'a']}
            s_bar_masked = {m: masked_mean_tbd(output[f's_{m}'], lens[m]) for m in ['l', 'v', 'a']}

            # Config A: unmasked + unmasked bank
            res_A = compute_uncertainty_and_reliability(
                c_bar_unmasked, s_bar_unmasked, u_mu_c, u_sig_c, u_mu_s, u_sig_s, u_bank_s, tau_c=model.tau_c, tau_s=model.tau_s
            )
            # Config B: masked + unmasked bank
            res_B = compute_uncertainty_and_reliability(
                c_bar_masked, s_bar_masked, u_mu_c, u_sig_c, u_mu_s, u_sig_s, u_bank_s, tau_c=model.tau_c, tau_s=model.tau_s
            )
            # Config C: masked + masked bank
            res_C = compute_uncertainty_and_reliability(
                c_bar_masked, s_bar_masked, m_mu_c, m_sig_c, m_mu_s, m_sig_s, m_bank_s, tau_c=model.tau_c, tau_s=model.tau_s
            )

            results['A'].append(res_A)
            results['B'].append(res_B)
            results['C'].append(res_C)

    # Process unimodal labels
    for k in unimodal_labels:
        unimodal_labels[k] = torch.cat(unimodal_labels[k], dim=0).squeeze().numpy()
    
    # Calculate disagreement tiers
    sign_l = np.sign(unimodal_labels['l'])
    sign_v = np.sign(unimodal_labels['v'])
    sign_a = np.sign(unimodal_labels['a'])
    
    tier_full = (sign_l == sign_v) & (sign_v == sign_a) & (sign_l != 0)
    tier_maj = ~tier_full & ((sign_l == sign_v) | (sign_l == sign_a) | (sign_v == sign_a)) & ((sign_l != 0) | (sign_v != 0) | (sign_a != 0))
    tier_dis = ~tier_full & ~tier_maj

    print(f"\nTiers count: Full={tier_full.sum()}, Majority={tier_maj.sum()}, Ambiguous/Disagreement={tier_dis.sum()}")

    # Aggregate results
    def aggregate(res_list):
        out = {}
        # flatten dict
        keys_1 = ['gamma_c']
        keys_2 = ['u_c', 'u_s', 'r_c', 'r_s', 'norm_c_bar', 'norm_s_bar', 'norm_z_ring_c', 'norm_z_ring_s']
        for k in keys_1:
            out[k] = torch.cat([r[k].cpu() for r in res_list], dim=0).numpy()
        for k in keys_2:
            out[k] = {}
            for m in ['l', 'v', 'a']:
                out[k][m] = torch.cat([r[k][m].cpu() for r in res_list], dim=0).numpy()
        return out

    agg_A = aggregate(results['A'])
    agg_B = aggregate(results['B'])
    agg_C = aggregate(results['C'])

    # Print Comparison Table
    print("\n" + "="*80)
    print("COMPARISON: UNMASKED (BASELINE) vs MASKED POOLING")
    print("="*80)

    for cfg_name, agg in [("Config A: Default (Unmasked)", agg_A),
                          ("Config B: Masked Pooling (Unmasked Bank)", agg_B),
                          ("Config C: Masked Pooling (Masked Bank - Full)", agg_C)]:
        print(f"\n--- {cfg_name} ---")
        print("Feature Norms ||v̄||:")
        print(f"  c_bar: Language={agg['norm_c_bar']['l'].mean():.4f}, Vision={agg['norm_c_bar']['v'].mean():.4f}, Audio={agg['norm_c_bar']['a'].mean():.4f}")
        print(f"  s_bar: Language={agg['norm_s_bar']['l'].mean():.4f}, Vision={agg['norm_s_bar']['v'].mean():.4f}, Audio={agg['norm_s_bar']['a'].mean():.4f}")
        print("Projected Norms ||z̊||:")
        print(f"  z̊^c:   Language={agg['norm_z_ring_c']['l'].mean():.4f}, Vision={agg['norm_z_ring_c']['v'].mean():.4f}, Audio={agg['norm_z_ring_c']['a'].mean():.4f}")
        print(f"  z̊^s:   Language={agg['norm_z_ring_s']['l'].mean():.4f}, Vision={agg['norm_z_ring_s']['v'].mean():.4f}, Audio={agg['norm_z_ring_s']['a'].mean():.4f}")
        print("Uncertainty u:")
        print(f"  u^c:   Language={agg['u_c']['l'].mean():.4f}, Vision={agg['u_c']['v'].mean():.4f}, Audio={agg['u_c']['a'].mean():.4f} | Overall={(agg['u_c']['l'].mean()+agg['u_c']['v'].mean()+agg['u_c']['a'].mean())/3:.4f}")
        print(f"  u^s:   Language={agg['u_s']['l'].mean():.4f}, Vision={agg['u_s']['v'].mean():.4f}, Audio={agg['u_s']['a'].mean():.4f} | Overall={(agg['u_s']['l'].mean()+agg['u_s']['v'].mean()+agg['u_s']['a'].mean())/3:.4f}")
        print("Reliability r & Consensus Gate γ^c:")
        mean_rc = (agg['r_c']['l'] + agg['r_c']['v'] + agg['r_c']['a']) / 3.0
        mean_rs = (agg['r_s']['l'] + agg['r_s']['v'] + agg['r_s']['a']) / 3.0
        print(f"  r^c:   Language={agg['r_c']['l'].mean():.4f}, Vision={agg['r_c']['v'].mean():.4f}, Audio={agg['r_c']['a'].mean():.4f} | Mean={mean_rc.mean():.4f}")
        print(f"  r^s:   Language={agg['r_s']['l'].mean():.4f}, Vision={agg['r_s']['v'].mean():.4f}, Audio={agg['r_s']['a'].mean():.4f} | Mean={mean_rs.mean():.4f}")
        print(f"  γ^c:   Mean={agg['gamma_c'].mean():.4f}, Median={np.median(agg['gamma_c']):.4f}, Min={agg['gamma_c'].min():.4f}, Max={agg['gamma_c'].max():.4f}")

        # Tiers breakdown
        print("  Breakdown by Tiers for r^c:")
        print(f"    Full Agreement (N={tier_full.sum()}):  r^c={mean_rc[tier_full].mean():.4f}, r^s={mean_rs[tier_full].mean():.4f}, γ^c={agg['gamma_c'][tier_full].mean():.4f}")
        print(f"    Majority Agree (N={tier_maj.sum()}):  r^c={mean_rc[tier_maj].mean():.4f}, r^s={mean_rs[tier_maj].mean():.4f}, γ^c={agg['gamma_c'][tier_maj].mean():.4f}")
        print(f"    Disagreement   (N={tier_dis.sum()}):  r^c={mean_rc[tier_dis].mean():.4f}, r^s={mean_rs[tier_dis].mean():.4f}, γ^c={agg['gamma_c'][tier_dis].mean():.4f}")

    # Save summary json
    summary_data = {
        'Config_A_Unmasked': {
            'mean_rc': float(((agg_A['r_c']['l'] + agg_A['r_c']['v'] + agg_A['r_c']['a'])/3).mean()),
            'mean_rs': float(((agg_A['r_s']['l'] + agg_A['r_s']['v'] + agg_A['r_s']['a'])/3).mean()),
            'gamma_c': float(agg_A['gamma_c'].mean()),
            'r_c_tiers': {
                'full': float(((agg_A['r_c']['l'] + agg_A['r_c']['v'] + agg_A['r_c']['a'])/3)[tier_full].mean()),
                'majority': float(((agg_A['r_c']['l'] + agg_A['r_c']['v'] + agg_A['r_c']['a'])/3)[tier_maj].mean()),
                'disagreement': float(((agg_A['r_c']['l'] + agg_A['r_c']['v'] + agg_A['r_c']['a'])/3)[tier_dis].mean()),
            }
        },
        'Config_B_Masked_UnmaskedBank': {
            'mean_rc': float(((agg_B['r_c']['l'] + agg_B['r_c']['v'] + agg_B['r_c']['a'])/3).mean()),
            'mean_rs': float(((agg_B['r_s']['l'] + agg_B['r_s']['v'] + agg_B['r_s']['a'])/3).mean()),
            'gamma_c': float(agg_B['gamma_c'].mean()),
            'r_c_tiers': {
                'full': float(((agg_B['r_c']['l'] + agg_B['r_c']['v'] + agg_B['r_c']['a'])/3)[tier_full].mean()),
                'majority': float(((agg_B['r_c']['l'] + agg_B['r_c']['v'] + agg_B['r_c']['a'])/3)[tier_maj].mean()),
                'disagreement': float(((agg_B['r_c']['l'] + agg_B['r_c']['v'] + agg_B['r_c']['a'])/3)[tier_dis].mean()),
            }
        },
        'Config_C_Masked_MaskedBank': {
            'mean_rc': float(((agg_C['r_c']['l'] + agg_C['r_c']['v'] + agg_C['r_c']['a'])/3).mean()),
            'mean_rs': float(((agg_C['r_s']['l'] + agg_C['r_s']['v'] + agg_C['r_s']['a'])/3).mean()),
            'gamma_c': float(agg_C['gamma_c'].mean()),
            'r_c_tiers': {
                'full': float(((agg_C['r_c']['l'] + agg_C['r_c']['v'] + agg_C['r_c']['a'])/3)[tier_full].mean()),
                'majority': float(((agg_C['r_c']['l'] + agg_C['r_c']['v'] + agg_C['r_c']['a'])/3)[tier_maj].mean()),
                'disagreement': float(((agg_C['r_c']['l'] + agg_C['r_c']['v'] + agg_C['r_c']['a'])/3)[tier_dis].mean()),
            }
        }
    }
    with open('result/experiments/exp04/sims_masked_mean_comparison.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    print("\nSaved summary to result/experiments/exp04/sims_masked_mean_comparison.json")

if __name__ == '__main__':
    evaluate_pipeline()
