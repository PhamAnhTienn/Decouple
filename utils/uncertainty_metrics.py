"""
Uncertainty Metrics for Multimodal Disentangled Representations.

Computes 4 types of uncertainty scores (per sample, per modality):
  - Shared Consensus:  u_m^{c,con}  — disagreement among shared representations
  - Shared Cycle:      u_m^{c,cy}   — reconstruction drift in shared space
  - Specific Neighbor: u_m^{s,nei}  — local inconsistency in specific space
  - Specific Cycle:    u_m^{s,cy}   — reconstruction drift in specific space
"""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger('DecAlign')

MODALITIES = ['l', 'v', 'a']


class UncertaintyMetrics:
    """Compute and persist uncertainty metrics on a dataset split."""

    def __init__(self, model, dataloader, args, K: int = 10):
        self.model = model
        self.dataloader = dataloader
        self.args = args
        self.K = K

    # ------------------------------------------------------------------
    # Pass 1: collect per-sample representations from the model
    # ------------------------------------------------------------------
    def _collect_representations(self):
        """
        Run the model on every batch and collect:
          z_c_m   = Norm(MeanPool(h_m^c))        — shared pooled repr
          z_s_m   = Norm(MeanPool(h_m^s))        — specific pooled repr
          zt_c_m  = Norm(MeanPool(Enc_c(recon_m)))  — cycle-shared repr
          zt_s_m  = Norm(MeanPool(Enc_s(recon_m)))  — cycle-specific repr
        All shapes: list of (B, D) tensors → will be cat'd to (N, D).
        """
        self.model.eval()
        device = self.args.device

        # Accumulators
        acc = {f'z_c_{m}': [] for m in MODALITIES}
        acc.update({f'z_s_{m}': [] for m in MODALITIES})
        acc.update({f'zt_c_{m}': [] for m in MODALITIES})
        acc.update({f'zt_s_{m}': [] for m in MODALITIES})
        uni_preds = {m: [] for m in MODALITIES}
        labels_list, preds_list, ids_list = [], [], []

        with torch.no_grad():
            for batch_data in tqdm(self.dataloader, desc='[Uncertainty] Collecting representations'):
                text = batch_data['text'].to(device)
                audio = batch_data['audio'].to(device)
                vision = batch_data['vision'].to(device)

                # When data_missing is disabled, _m keys don't exist;
                # use the clean modality as both missing and clean input.
                text_m = batch_data.get('text_m', batch_data['text']).to(device)
                audio_m = batch_data.get('audio_m', batch_data['audio']).to(device)
                vision_m = batch_data.get('vision_m', batch_data['vision']).to(device)

                labels = batch_data['labels']['M'].to(device)
                if self.args.train_mode == 'classification':
                    labels = labels.view(-1).long()
                else:
                    labels = labels.view(-1, 1)

                try:
                    output = self.model(text_m, audio_m, vision_m, text, audio, vision)
                except TypeError:
                    output = self.model(text_m, audio_m, vision_m)

                # --- Model predictions ---
                if 'output_logit' in output:
                    preds_list.append(output['output_logit'].cpu())
                for m in MODALITIES:
                    if f'logits_{m}_hetero' in output:
                        uni_preds[m].append(output[f'logits_{m}_hetero'].cpu())

                # --- Shared representations z_c_m ---
                # c_l, c_v, c_a are (T, B, D) in the output dict
                for m in MODALITIES:
                    c_m = output[f'c_{m}']  # (T, B, D)
                    z_c = F.normalize(c_m.mean(dim=0), dim=1)  # (B, D)
                    acc[f'z_c_{m}'].append(z_c.cpu())

                # --- Specific representations z_s_m ---
                for m in MODALITIES:
                    s_m = output[f's_{m}']  # (T, B, D)
                    z_s = F.normalize(s_m.mean(dim=0), dim=1)  # (B, D)
                    acc[f'z_s_{m}'].append(z_s.cpu())

                # --- Cycle representations ---
                # recon_m is (B, D, T) (Conv1d format)
                # Re-encode through shared encoder → (B, D, T) → mean over T → normalize
                encoder_s_map = {
                    'l': self.model.encoder_s_l,
                    'v': self.model.encoder_s_v,
                    'a': self.model.encoder_s_a,
                }
                for m in MODALITIES:
                    recon_m = output[f'recon_{m}']  # (B, D, T)

                    # Cycle shared: Enc_c(recon_m)
                    c_recon = self.model.encoder_c(recon_m)  # (B, D, T)
                    zt_c = F.normalize(c_recon.mean(dim=2), dim=1)  # (B, D)
                    acc[f'zt_c_{m}'].append(zt_c.cpu())

                    # Cycle specific: Enc_s_m(recon_m)
                    s_recon = encoder_s_map[m](recon_m)  # (B, D, T)
                    zt_s = F.normalize(s_recon.mean(dim=2), dim=1)  # (B, D)
                    acc[f'zt_s_{m}'].append(zt_s.cpu())

                labels_list.append(labels.cpu())
                if 'id' in batch_data:
                    ids_list.extend(batch_data['id'])

        # Concatenate everything to (N, D) or (N,)
        data = {k: torch.cat(v, dim=0) for k, v in acc.items()}
        data['labels'] = torch.cat(labels_list, dim=0).squeeze()
        if preds_list:
            data['predictions'] = torch.cat(preds_list, dim=0).squeeze()
        for m in MODALITIES:
            if uni_preds[m]:
                data[f'pred_{m}'] = torch.cat(uni_preds[m], dim=0).squeeze()
        data['ids'] = ids_list
        return data

    # ------------------------------------------------------------------
    # Metric computations (all vectorized, no loops over samples)
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_sim(a, b):
        """Row-wise cosine similarity between (N, D) tensors → (N,)."""
        return F.cosine_similarity(a, b, dim=1)

    @staticmethod
    def _shared_consensus(z_c: dict):
        """
        u_m^{c,con,i} = 1 - (1/(|M|-1)) * Σ_{n≠m} cos(z_m^c, z_n^c)
        Returns dict m → (N,) tensor.
        """
        scores = {}
        for m in MODALITIES:
            others = [n for n in MODALITIES if n != m]
            cos_sum = sum(
                F.cosine_similarity(z_c[m], z_c[n], dim=1) for n in others
            )
            scores[m] = 1.0 - cos_sum / len(others)
        return scores

    @staticmethod
    def _cycle_uncertainty(z, zt):
        """
        u_m^{*,cy,i} = 1 - cos(z_m, z̃_m)
        Works for both shared and specific.
        Returns dict m → (N,) tensor.
        """
        return {
            m: 1.0 - F.cosine_similarity(z[m], zt[m], dim=1)
            for m in MODALITIES
        }

    def _neighbor_uncertainty(self, z_c: dict, z_s: dict):
        """
        Leave-one-out neighbor search + specific consistency.

        z_{-m}^c = Norm(Σ_{n≠m} z_n^c)
        N_K^m(i) = TopK_{j≠i}[cos(z_{-m}^{c,i}, z_{-m}^{c,j})]
        u_m^{s,nei,i} = 1 - (1/K) Σ_{j∈N_K} cos(z_m^{s,i}, sg(z_m^{s,j}))

        Returns dict m → (N,) tensor.
        """
        K = self.K
        N = z_c[MODALITIES[0]].shape[0]
        scores = {}

        for m in MODALITIES:
            # Leave-one-out shared feature
            others = [n for n in MODALITIES if n != m]
            z_minus_m = F.normalize(
                sum(z_c[n] for n in others), dim=1
            )  # (N, D)

            # Pairwise cosine similarity matrix (N, N)
            sim_matrix = z_minus_m @ z_minus_m.t()  # already L2-normed

            # Mask self-similarity (set diagonal to -inf)
            sim_matrix.fill_diagonal_(-float('inf'))

            # TopK neighbors
            k_actual = min(K, N - 1)
            _, topk_indices = sim_matrix.topk(k_actual, dim=1)  # (N, K)

            # Gather neighbor specific representations
            z_s_m = z_s[m]  # (N, D)
            z_s_neighbors = z_s_m[topk_indices]  # (N, K, D)

            # Cosine similarity with neighbors (stop-gradient on neighbors)
            # z_s_m: (N, D) → (N, 1, D)
            cos_neighbors = F.cosine_similarity(
                z_s_m.unsqueeze(1), z_s_neighbors.detach(), dim=2
            )  # (N, K)

            scores[m] = 1.0 - cos_neighbors.mean(dim=1)  # (N,)

        return scores

    # ------------------------------------------------------------------
    # Fusion weight computation
    # ------------------------------------------------------------------
    @staticmethod
    def compute_fusion_weights(u_shared_consensus, u_shared_cycle,
                               u_specific_neighbor, u_specific_cycle,
                               tau_c: float = 1.0, tau_s: float = 1.0,
                               r_min: float = 0.1):
        """
        Compute per-modality fusion weights from the 4 uncertainty metrics.

        Aggregation (mean):
            u_m^c = 0.5 * (u_m^{c,con} + u_m^{c,cy})
            u_m^s = 0.5 * (u_m^{s,nei} + u_m^{s,cy})

        Weight (exponential decay + clamp, stop-gradient at train time):
            r_m^c = clamp(exp(-u_m^c / tau_c), r_min, 1)
            r_m^s = clamp(exp(-u_m^s / tau_s), r_min, 1)

        Args:
            u_shared_consensus:  dict m → (N,) tensor
            u_shared_cycle:      dict m → (N,) tensor
            u_specific_neighbor: dict m → (N,) tensor
            u_specific_cycle:    dict m → (N,) tensor
            tau_c: temperature for shared weight
            tau_s: temperature for specific weight
            r_min: minimum weight (clamp floor)

        Returns:
            u_shared:   dict m → (N,) aggregated shared uncertainty
            u_specific: dict m → (N,) aggregated specific uncertainty
            r_shared:   dict m → (N,) shared fusion weight
            r_specific: dict m → (N,) specific fusion weight
        """
        u_shared, u_specific = {}, {}
        r_shared, r_specific = {}, {}

        for m in MODALITIES:
            # Aggregate: mean of the two metrics in each group
            u_c = 0.5 * (u_shared_consensus[m] + u_shared_cycle[m])
            u_s = 0.5 * (u_specific_neighbor[m] + u_specific_cycle[m])

            # Exponential decay → clamp to [r_min, 1]
            r_c = torch.clamp(torch.exp(-u_c / tau_c), min=r_min, max=1.0)
            r_s = torch.clamp(torch.exp(-u_s / tau_s), min=r_min, max=1.0)

            u_shared[m] = u_c
            u_specific[m] = u_s
            r_shared[m] = r_c
            r_specific[m] = r_s

        return u_shared, u_specific, r_shared, r_specific

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def compute_and_save(self, save_path: str,
                         tau_c: float = 1.0, tau_s: float = 1.0,
                         r_min: float = 0.1):
        """
        Run full pipeline: collect → compute → save.

        Args:
            save_path: path to the output .pt file
            tau_c: temperature for shared fusion weight
            tau_s: temperature for specific fusion weight
            r_min: minimum fusion weight (clamp floor)
        """
        logger.info('[Uncertainty] Starting representation collection ...')
        data = self._collect_representations()

        # Organize representations into convenient dicts
        z_c = {m: data[f'z_c_{m}'] for m in MODALITIES}
        z_s = {m: data[f'z_s_{m}'] for m in MODALITIES}
        zt_c = {m: data[f'zt_c_{m}'] for m in MODALITIES}
        zt_s = {m: data[f'zt_s_{m}'] for m in MODALITIES}

        N = z_c['l'].shape[0]
        logger.info(f'[Uncertainty] Collected {N} samples. Computing metrics ...')

        # 1. Shared Consensus
        u_shared_consensus = self._shared_consensus(z_c)

        # 2. Shared Cycle
        u_shared_cycle = self._cycle_uncertainty(z_c, zt_c)

        # 3. Specific Neighbor
        u_specific_neighbor = self._neighbor_uncertainty(z_c, z_s)

        # 4. Specific Cycle
        u_specific_cycle = self._cycle_uncertainty(z_s, zt_s)

        # 5. Fusion Weights
        u_shared, u_specific, r_shared, r_specific = self.compute_fusion_weights(
            u_shared_consensus, u_shared_cycle,
            u_specific_neighbor, u_specific_cycle,
            tau_c=tau_c, tau_s=tau_s, r_min=r_min,
        )

        # --- Assemble output dict ---
        result = {
            'labels': data['labels'],
            'ids': data['ids'],
        }
        if 'predictions' in data:
            result['predictions'] = data['predictions']
        for m in MODALITIES:
            if f'pred_{m}' in data:
                result[f'pred_{m}'] = data[f'pred_{m}']

        # Raw uncertainty scores (N,) each
        for m in MODALITIES:
            result[f'u_{m}_shared_consensus'] = u_shared_consensus[m]
            result[f'u_{m}_shared_cycle'] = u_shared_cycle[m]
            result[f'u_{m}_specific_neighbor'] = u_specific_neighbor[m]
            result[f'u_{m}_specific_cycle'] = u_specific_cycle[m]

        # Aggregated uncertainties and fusion weights (N,) each
        for m in MODALITIES:
            result[f'u_{m}_shared'] = u_shared[m]
            result[f'u_{m}_specific'] = u_specific[m]
            result[f'r_{m}_shared'] = r_shared[m]
            result[f'r_{m}_specific'] = r_specific[m]

        # Store representations for future t-SNE / analysis
        for m in MODALITIES:
            result[f'z_c_{m}'] = z_c[m]
            result[f'z_s_{m}'] = z_s[m]

        # Metadata
        result['metadata'] = {
            'model': getattr(self.args, 'model_name', 'dp'),
            'dataset': self.args.dataset_name,
            'missing_rate': getattr(self.args, 'missing_rate_eval_test', 0.0),
            'K_neighbors': self.K,
            'seed': getattr(self.args, 'cur_seed', -1),
            'model_path': getattr(self.args, 'model_save_path', ''),
            'tau_c': tau_c,
            'tau_s': tau_s,
            'r_min': r_min,
        }

        # Save
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, save_path)
        logger.info(f'[Uncertainty] Saved results to {save_path}')

        # Print summary
        self._print_summary(result)

        return result

    @staticmethod
    def _print_summary(result):
        """Pretty-print summary statistics of all uncertainty scores and fusion weights."""
        print('\n' + '=' * 70)
        print('UNCERTAINTY METRICS SUMMARY')
        print('=' * 70)
        metrics = ['shared_consensus', 'shared_cycle', 'specific_neighbor', 'specific_cycle']
        for metric in metrics:
            print(f'\n--- {metric.replace("_", " ").title()} ---')
            for m in MODALITIES:
                key = f'u_{m}_{metric}'
                vals = result[key]
                print(
                    f'  [{m.upper()}]  '
                    f'mean={vals.mean():.4f}  '
                    f'std={vals.std():.4f}  '
                    f'min={vals.min():.4f}  '
                    f'max={vals.max():.4f}'
                )

        # Aggregated uncertainties
        if f'u_l_shared' in result:
            print(f'\n--- Aggregated Shared Uncertainty (u_m^c) ---')
            for m in MODALITIES:
                vals = result[f'u_{m}_shared']
                print(
                    f'  [{m.upper()}]  '
                    f'mean={vals.mean():.4f}  '
                    f'std={vals.std():.4f}  '
                    f'min={vals.min():.4f}  '
                    f'max={vals.max():.4f}'
                )
            print(f'\n--- Aggregated Specific Uncertainty (u_m^s) ---')
            for m in MODALITIES:
                vals = result[f'u_{m}_specific']
                print(
                    f'  [{m.upper()}]  '
                    f'mean={vals.mean():.4f}  '
                    f'std={vals.std():.4f}  '
                    f'min={vals.min():.4f}  '
                    f'max={vals.max():.4f}'
                )

        # Fusion weights
        if f'r_l_shared' in result:
            print(f'\n--- Shared Fusion Weight (r_m^c) ---')
            for m in MODALITIES:
                vals = result[f'r_{m}_shared']
                print(
                    f'  [{m.upper()}]  '
                    f'mean={vals.mean():.4f}  '
                    f'std={vals.std():.4f}  '
                    f'min={vals.min():.4f}  '
                    f'max={vals.max():.4f}'
                )
            print(f'\n--- Specific Fusion Weight (r_m^s) ---')
            for m in MODALITIES:
                vals = result[f'r_{m}_specific']
                print(
                    f'  [{m.upper()}]  '
                    f'mean={vals.mean():.4f}  '
                    f'std={vals.std():.4f}  '
                    f'min={vals.min():.4f}  '
                    f'max={vals.max():.4f}'
                )

        print('=' * 70 + '\n')
