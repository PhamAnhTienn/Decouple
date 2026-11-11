import torch
import torch.nn as nn
import torch.nn.functional as F
from .subNets2 import BertTextEncoder
from .subNets2.transformers_encoder.transformer import TransformerEncoder


class DP(nn.Module):
    def __init__(self, args):
        super(DP, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        if args.dataset_name == 'mosi':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 375
        elif args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500
        elif args.dataset_name == 'sims':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 39, 39, 39
            else:
                # SIMS unaligned: Text=39, Vision=55, Audio=400
                self.len_l, self.len_v, self.len_a = 39, 55, 400
        elif hasattr(args, 'seq_lens') and args.seq_lens:
            self.len_l, self.len_a, self.len_v = args.seq_lens[0], args.seq_lens[1], args.seq_lens[2]
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask

        # Uncertainty hyperparameters
        self.tau_c = getattr(args, 'tau_c', 1.0)    # temperature for shared reliability
        self.tau_s = getattr(args, 'tau_s', 0.5)    # temperature for specific reliability
        self.K_neighbors = getattr(args, 'K_neighbors', 10)  # neighbors for specific uncertainty

        # ---- Ablation flags (default False = full model) ----
        self.ablate_shared_cal = getattr(args, 'ablate_shared_calibration', False)
        self.ablate_specific_cal = getattr(args, 'ablate_specific_calibration', False)
        self.ablate_residual_cal = getattr(args, 'ablate_residual_calibration', False)

        # ── Role-conditioned reliability ablation ──
        # Controls how r_shared / r_specific scores are routed to the
        # shared-consensus and specific-exchange modules.
        # Modes:
        #   'full'          – default: shared→r^c, specific→r^s  (Full SARM)
        #   'uniform'       – r^c = r^s = 1 for both modules
        #   'shared_only'   – both modules use r^c
        #   'specific_only' – both modules use r^s
        #   'averaged'      – both modules use (r^c + r^s) / 2
        #   'swapped'       – shared→r^s, specific→r^c  (role-swapped)
        self.reliability_mode = getattr(args, 'reliability_mode', 'full')

        # Population statistics for ABTT (Axis-Based Tangent Transform) Subspace
        # Populated by update_train_bank() after each epoch; empty until first update.
        # Keys: 'l', 'v', 'a'
        # Shared subspace (q=c)
        self._train_mu_dir_c: dict = {}  # m -> (1, D) unit cone axis μ̂_m^c on S^{d-1}
        self._train_sigma_c: dict = {}   # m -> scalar mean projection length σ_m^c
        # Specific subspace (q=s)
        self._train_mu_dir_s: dict = {}  # m -> (1, D) unit cone axis μ̂_m^s on S^{d-1}
        self._train_sigma_s: dict = {}   # m -> scalar mean projection length σ_m^s
        # ABTT unit bank for specific neighbor lookup
        self._train_bank_z_s: dict = {}  # m -> (N_train, D) Norm(p_{m,j}^{s,bank})

        combined_dim_low = self.d_a
        combined_dim_high = 2 * self.d_a
        # --- OLD: shared concat (3*D) ---
        # combined_dim = 2 * (self.d_l + self.d_a + self.d_v) + self.d_l * 3
        # --- NEW: shared consensus sum (D) ---
        combined_dim = 2 * (self.d_l + self.d_a + self.d_v) + self.d_l
        output_dim = 1

        # 1. Temporal convolutional layers for initial feature
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # 2.1 Modality-specific encoder
        self.encoder_s_l = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.encoder_s_v = nn.Conv1d(self.d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.encoder_s_a = nn.Conv1d(self.d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        # 2.2 Modality-invariant encoder
        self.encoder_c = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)

        # 3. Decoder for reconstruct three modalities (OLD — kept for reference)
        self.decoder_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)
        self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)

        # =====================================================================
        # 3b. Cross-Reconstruction Module (CRM) (Section 3.3)
        #     6 cross-reconstruction decoders D_{nm}
        #     3 fusion layers F_m
        # =====================================================================
        # Decoders: D_{nm} reconstructs target m from shared H_n ⊕ specific P_m
        # For target L: D_VL (from V's shared), D_AL (from A's shared)
        self.decoder_vl = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)
        self.decoder_al = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)
        # For target V: D_LV (from L's shared), D_AV (from A's shared)
        self.decoder_lv = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_av = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        # For target A: D_LA (from L's shared), D_VA (from V's shared)
        self.decoder_la = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)
        self.decoder_va = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)

        # Fusion layers F_m to combine two cross-reconstructed features per target
        self.fusion_l = nn.Linear(self.d_l * 2, self.d_l)
        self.fusion_v = nn.Linear(self.d_v * 2, self.d_v)
        self.fusion_a = nn.Linear(self.d_a * 2, self.d_a)

        # init prototype
        self.nmb_prototypes = args.nmb_prototypes
        self.prototypes = nn.Linear(self.d_a, args.nmb_prototypes, bias=False)

        # for align c_l, c_v, c_a
        self.self_attentions_c_l = self.get_network(self_type='l')
        self.self_attentions_c_v = self.get_network(self_type='v')
        self.self_attentions_c_a = self.get_network(self_type='a')

        # --- NEW: shared consensus sum projection (D) ---
        self.proj1_c = nn.Linear(self.d_l, self.d_l)
        self.proj2_c = nn.Linear(self.d_l, self.d_l)
        self.out_layer_c = nn.Linear(self.d_l, output_dim)

        # 4.2 Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # Self-attention transformer for shared features H̃ (CRM shared branch)
        self.trans_shared = self.get_network(self_type='shared', layers=3)

        # Shared branch prediction head (for TH̃)
        shared_dim = self.d_l * 3  # concat of H̃_L, H̃_V, H̃_A last tokens
        self.proj1_shared = nn.Linear(shared_dim, shared_dim)
        self.proj2_shared = nn.Linear(shared_dim, shared_dim)
        self.out_layer_shared = nn.Linear(shared_dim, output_dim)

        # 4. fc layers for homogeneous graph distillation
        self.proj1_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.proj2_l_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1))
        self.out_layer_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), output_dim)
        self.proj1_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj2_v_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1))
        self.out_layer_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), output_dim)
        self.proj1_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)
        self.proj2_a_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1))
        self.out_layer_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), output_dim)

        # 5. fc layers for heterogeneous graph distillation
        self.proj1_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_l_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_v_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_a_high = nn.Linear(combined_dim_high, output_dim)

        # 6. Ensemble Projection layers
        # weight for each modality
        self.weight_l = nn.Linear(2 * self.d_l, 2 * self.d_l)
        self.weight_v = nn.Linear(2 * self.d_v, 2 * self.d_v)
        self.weight_a = nn.Linear(2 * self.d_a, 2 * self.d_a)
        # --- NEW: shared consensus sum weight (D) ---
        self.weight_c = nn.Linear(self.d_l, self.d_l)
        # final project
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        elif self_type == 'shared':
            embed_dim, attn_dropout = self.d_l * 3, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    # ==================================================================
    # Uncertainty computation (all online, within each batch)
    # ==================================================================

    def _pool_norm(self, h):
        """MeanPool over time dim + L2 normalize.
        Args:
            h: (B, D, T) — Conv1d format
        Returns:
            z: (B, D) — L2-normalized pooled representation
        """
        return F.normalize(h.mean(dim=2), dim=1)

    def _abtt_soft_normalize(self, v_bar_dict, mu_dir_bank, sigma_bank, eps=1e-8):
        """
        ABTT Orthogonal Projection + Soft Tikhonov Normalization.

        Step 1 (ABTT Orthogonal Projection):
            p_{m,i} = v̄_{m,i} - (v̄_{m,i}^T μ̂_m) μ̂_m
        Step 2 (Soft Unit-Ball Normalization):
            z̊_{m,i} = p_{m,i} / sqrt(||p_{m,i}||_2^2 + σ_m^2)

        Properties:
        - When v̄ = 0 (100% frame loss): p = 0 → z̊ = 0
        - When norm small (heavy masking / collapse to [UNK]): ||z̊|| → 0 smoothly
        - When norm very large (outlier): ||z̊|| bounded above by 1.0

        Falls back to in-batch statistics when training-set stats are unavailable.

        Args:
            v_bar_dict:  dict m → (B, D)  — unnormalized mean-pooled repr
            mu_dir_bank: dict m → (1, D)  — unit cone axis from training set
            sigma_bank:  dict m → scalar  — mean ABTT projection length from training set
            eps: numerical stability constant
        Returns:
            dict m → (B, D) ABTT soft-normalized vectors z̊, ||z̊|| ∈ [0, 1)
        """
        if self.ablate_residual_cal:
            # Ablation: standard unit normalization without residual projection & soft Tikhonov scaling
            return {m: F.normalize(v_bar, dim=1, eps=eps) for m, v_bar in v_bar_dict.items()}

        z_ring = {}
        has_stats = bool(mu_dir_bank) and all(m in mu_dir_bank for m in v_bar_dict)

        for m, v_bar in v_bar_dict.items():
            if has_stats:
                mu_dir = mu_dir_bank[m].to(v_bar.device)                # (1, D)
                sigma = sigma_bank[m].to(v_bar.device)                  # scalar
            else:
                # Fallback in Epoch 1 before update_train_bank() runs
                z_unit = F.normalize(v_bar.detach(), dim=1, eps=eps)
                mu_dir = F.normalize(z_unit.mean(dim=0, keepdim=True), dim=1, eps=eps)
                p_batch = v_bar.detach() - (v_bar.detach() * mu_dir).sum(dim=1, keepdim=True) * mu_dir
                sigma = p_batch.norm(p=2, dim=1).mean().clamp(min=eps)

            # ABTT orthogonal projection
            proj_coeff = (v_bar * mu_dir).sum(dim=1, keepdim=True)      # (B, 1)
            p_m = v_bar - proj_coeff * mu_dir                           # (B, D)

            # Soft Tikhonov normalization into unit ball
            p_norm_sq = p_m.pow(2).sum(dim=1, keepdim=True)             # (B, 1)
            denom = torch.sqrt(p_norm_sq + sigma.pow(2) + eps)          # (B, 1)
            z_ring[m] = p_m / denom                                     # (B, D), ||z̊|| ∈ [0, 1)

        return z_ring

    def _shared_consensus(self, c_bar_dict, eps=1e-8):
        """
        Shared Uncertainty via cross-modality consensus on ABTT-projected vectors.

            z̊_m^c = ABTT_soft_normalize(c̄_m)
            u_m^c  = 1 - (z̊_m^c)^T Norm(Σ_{n≠m} z̊_n^c)

        Args:
            c_bar_dict: dict m → (B, D) — unnormalized mean-pooled shared repr
        Returns:
            dict m → (B,) consensus uncertainty in [0, 2]
        """
        z_ring_c = self._abtt_soft_normalize(
            c_bar_dict, self._train_mu_dir_c, self._train_sigma_c, eps=eps
        )
        modalities = list(z_ring_c.keys())
        scores = {}
        for m in modalities:
            others_sum = sum(z_ring_c[n] for n in modalities if n != m)  # (B, D)
            consensus_dir = F.normalize(others_sum, dim=1, eps=eps)     # (B, D)
            scores[m] = 1.0 - (z_ring_c[m] * consensus_dir).sum(dim=1) # (B,)
        return scores

    def _specific_uncertainty(self, s_bar_dict, eps=1e-8):
        """
        Specific Uncertainty via neighbor support in ABTT-projected space.

            z̊_m^s = ABTT_soft_normalize(s̄_m)
            N_K^m(i) = TopK_j [ (z̊_{m,i}^s)^T ẑ_{m,j}^{s,bank} ]
            u_m^s = 1 - (1/K) Σ_{j∈N_K^m(i)} (z̊_{m,i}^s)^T ẑ_{m,j}^{s,bank}

        where ẑ_{m,j}^{s,bank} = Norm(p_{m,j}^{s,bank}) are unit-norm ABTT-projected
        neighbor vectors from the training-set memory bank.

        Fallback (before first update_train_bank): uses radial energy
            u_m^s = 1 / (1 + ||z̊_m^s||_2)  ∈ (0, 1]

        Args:
            s_bar_dict: dict m → (B, D) — unnormalized mean-pooled specific repr
        Returns:
            dict m → (B,) specific uncertainty
        """
        z_ring_s = self._abtt_soft_normalize(
            s_bar_dict, self._train_mu_dir_s, self._train_sigma_s, eps=eps
        )

        # If bank not yet populated, fall back to radial energy uncertainty
        if not self._train_bank_z_s:
            return {
                m: 1.0 / (1.0 + z_ring_s[m].norm(p=2, dim=1))
                for m in z_ring_s
            }

        modalities = list(z_ring_s.keys())
        device = z_ring_s[modalities[0]].device
        K = self.K_neighbors
        scores = {}

        for m in modalities:
            z_q = z_ring_s[m]                                           # (B, D)
            bank_z_hat = self._train_bank_z_s[m].to(device)             # (N_train, D)

            # Dot-product similarity: (z̊_{m,i}^s)^T ẑ_{m,j}^{s,bank}
            sim_matrix = z_q @ bank_z_hat.t()                           # (B, N_train)

            k_actual = min(K, bank_z_hat.shape[0])
            topk_vals, _ = sim_matrix.topk(k_actual, dim=1)             # (B, K)
            scores[m] = 1.0 - topk_vals.mean(dim=1)                     # (B,)

        return scores

    @torch.no_grad()
    def update_train_bank(self, model_forward_fn, train_dataloader, device,
                          eps=1e-8):
        """
        Compute ABTT population statistics for both subspaces (q ∈ {c, s}).

        Passes the entire training set through the encoder in eval mode
        (no gradients) and records per modality m, per subspace q:
            μ̂_m^q = Norm(E_j[Norm(v̄_{m,j}^q)])             — unit cone axis
            σ_m^q = E_j[||p_{m,j}^q||_2]                    — mean ABTT projection length

        Additionally for specific branch (q=s):
            bank_z_s[m] = Norm(p_{m,j}^s)                    — ABTT unit neighbor bank

        Should be called once after each training epoch (before validation).

        Args:
            model_forward_fn: callable that accepts (text_m, audio_m, vision_m,
                              text, audio, vision) and returns the model output dict.
                              Typically just `model` itself.
            train_dataloader: DataLoader for the training split.
            device: torch device.
            eps: small constant to avoid division by zero in normalization.
        """
        self.eval()
        bank_c = {'l': [], 'v': [], 'a': []}
        bank_s = {'l': [], 'v': [], 'a': []}

        for batch_data in train_dataloader:
            text     = batch_data['text'].to(device)
            text_m   = batch_data['text_m'].to(device)
            audio    = batch_data['audio'].to(device)
            audio_m  = batch_data['audio_m'].to(device)
            vision   = batch_data['vision'].to(device)
            vision_m = batch_data['vision_m'].to(device)

            output = model_forward_fn(text_m, audio_m, vision_m, text, audio, vision)

            for m in ['l', 'v', 'a']:
                bank_c[m].append(output[f'c_{m}'].mean(dim=0).cpu())
                bank_s[m].append(output[f's_{m}'].mean(dim=0).cpu())

        for m in ['l', 'v', 'a']:
            all_s = torch.cat(bank_s[m], dim=0)                        # (N, D)
            if self.ablate_residual_cal:
                # Ablation: unit-normalized specific feature bank without residual projection
                self._train_bank_z_s[m] = F.normalize(all_s, dim=1, eps=eps)  # (N, D)
                continue

            # ── Shared stats (q=c) ──
            all_c = torch.cat(bank_c[m], dim=0)                        # (N, D)
            z_c_unit = F.normalize(all_c, dim=1, eps=eps)              # (N, D)
            mu_dir_c = F.normalize(
                z_c_unit.mean(dim=0, keepdim=True), dim=1, eps=eps
            )                                                          # (1, D)
            p_c = all_c - (all_c * mu_dir_c).sum(dim=1, keepdim=True) * mu_dir_c  # (N, D)
            self._train_mu_dir_c[m] = mu_dir_c
            self._train_sigma_c[m] = p_c.norm(p=2, dim=1).mean().clamp(min=eps)

            # ── Specific stats (q=s) + ABTT unit bank ──
            z_s_unit = F.normalize(all_s, dim=1, eps=eps)              # (N, D)
            mu_dir_s = F.normalize(
                z_s_unit.mean(dim=0, keepdim=True), dim=1, eps=eps
            )                                                          # (1, D)
            p_s = all_s - (all_s * mu_dir_s).sum(dim=1, keepdim=True) * mu_dir_s  # (N, D)
            self._train_mu_dir_s[m] = mu_dir_s
            self._train_sigma_s[m] = p_s.norm(p=2, dim=1).mean().clamp(min=eps)
            self._train_bank_z_s[m] = F.normalize(p_s, dim=1, eps=eps) # (N, D)

        self.train()

    def _compute_reliability(self, u_consensus, u_cycle_c, u_specific, u_cycle_s):
        """
        Aggregate per-modality uncertainties into reliability scores.

        u_m^c = u_consensus (or 0.5 * (u_con + u_cy_c) when cycle is used)
        u_m^s = u_specific  (or 0.5 * (u_spec + u_cy_s) when cycle is used)
        r_m^c = clamp(exp(-u_m^c / tau_c), r_min, 1)
        r_m^s = clamp(exp(-u_m^s / tau_s), r_min, 1)

        All returned with stop-gradient.

        Returns:
            r_shared:   dict m → (B,)
            r_specific: dict m → (B,)
        """
        r_shared, r_specific = {}, {}
        for m in u_consensus:
            # u_c = 0.5 * (u_consensus[m] + u_cycle_c[m])
            # u_s = 0.5 * (u_specific[m] + u_cycle_s[m])
            u_c = u_consensus[m]
            u_s = u_specific[m]
            r_c = torch.clamp(torch.exp(-u_c / self.tau_c), min=1e-4, max=1.0)
            r_s = torch.clamp(torch.exp(-u_s / self.tau_s), min=1e-4, max=1.0)
            r_shared[m] = r_c.detach()      # stop-gradient
            r_specific[m] = r_s.detach()     # stop-gradient
        return r_shared, r_specific

    # ==================================================================
    # Reliability-Aware Information Flow (Section 3.X)
    # ==================================================================

    def _reliability_gated_consensus_sum(self, c_att_dict, r_shared):
        """
        Reliability-gated consensus for shared representations (SUM version).

        Closed-form solution of:
            h̄^{c,i} = argmin_z [ γ^{c,i} Σ_m w_m ||z - h_m||^2 + (1-γ^{c,i}) ||z||^2 ]
        which gives:
            h̄^{c,i} = γ^{c,i} Σ_m (r_m^{c,i} / R^{c,i}) h_m^{c,i}

        where:
            γ^{c,i} = 1 - Π_m (1 - r_m^{c,i})      (joint reliability gate)
            R^{c,i} = Σ_m r_m^{c,i}                   (total shared reliability)

        Output is a single (B, D) vector (weighted average, NOT concat).

        Args:
            c_att_dict: dict m → (B, D) — self-attention shared output
            r_shared:   dict m → (B,) — shared reliability weights (stop-grad)
        Returns:
            c_fusion: (B, D) — reliability-gated consensus representation
        """
        modalities = list(c_att_dict.keys())

        # Joint reliability gate: γ^c = 1 - Π_m (1 - r_m^c)
        gamma_c = torch.ones_like(r_shared[modalities[0]])  # (B,)
        for m in modalities:
            gamma_c = gamma_c * (1.0 - r_shared[m])         # accumulate product
        gamma_c = 1.0 - gamma_c                             # (B,)

        # Total shared reliability: R^c = Σ_m r_m^c
        r_stack = torch.stack([r_shared[m] for m in modalities], dim=1)  # (B, |M|)
        R_c = r_stack.sum(dim=1, keepdim=True).clamp(min=1e-8)           # (B, 1)

        gamma_c_expand = gamma_c.unsqueeze(1)  # (B, 1)

        # h̄ = γ^c * Σ_m (r_m / R^c) * h_m
        h_bar = torch.zeros_like(c_att_dict[modalities[0]])  # (B, D)
        for i, m in enumerate(modalities):
            r_m = r_stack[:, i].unsqueeze(1)           # (B, 1)
            w_m = r_m / R_c                             # (B, 1)
            h_bar = h_bar + w_m * c_att_dict[m]         # (B, D)

        h_bar = gamma_c_expand * h_bar                  # (B, D)

        return h_bar  # (B, D)

    def _reliability_gated_complementary_exchange(self, h_m, b_cross, r_specific, m_recv, m_send):
        """
        Reliability-gated complementary exchange for specific representations.

        Pairwise joint reliability gate:
            γ_{m,n}^{s,i} = 1 - (1 - r_m^{s,i})(1 - r_n^{s,i})

        Mixing coefficient:
            β_{m←n}^i = γ_{m,n}^{s,i} * r_n^{s,i} / (r_m^{s,i} + r_n^{s,i})

        Update rule:
            h̃_{m←n}^{s,i} = h_m^{s,i} + β_{m←n}^i * (b_{m←n}^i - h_m^{s,i})

        A reliable sender exerts stronger influence on the receiver, whereas a
        reliable receiver is less affected by potentially unreliable incoming
        information. When both are unreliable, γ suppresses the update.

        Args:
            h_m:         (T, B, D) — receiver's own specific representation
            b_cross:     (T, B, D) — cross-attention message b_{m←n}
            r_specific:  dict m → (B,) — specific reliability (stop-grad)
            m_recv:      str — receiver modality key (m)
            m_send:      str — sender modality key (n)
        Returns:
            h_updated:   (T, B, D)
        """
        r_m = r_specific[m_recv]  # (B,)
        r_n = r_specific[m_send]  # (B,)

        # γ_{m,n} = 1 - (1 - r_m)(1 - r_n)
        gamma_mn = 1.0 - (1.0 - r_m) * (1.0 - r_n)          # (B,)

        # β_{m←n} = γ_{m,n} * r_n / (r_m + r_n)
        denom = (r_m + r_n).clamp(min=1e-8)
        beta = gamma_mn * r_n / denom                         # (B,)

        # Reshape for broadcasting: (1, B, 1)
        beta = beta.unsqueeze(0).unsqueeze(2)

        # h̃ = h_m + β * (b_{m←n} - h_m)
        return h_m + beta * (b_cross - h_m)

    def _align_time_attention(self, h_src, p_tgt):
        """
        Dynamically aligns source shared feature h_src to target sequence length
        via non-parametric soft cross-attention (Query = p_tgt, Key/Value = h_src).
        Preserves 100% of source temporal information without interpolation artifacts.

        Args:
            h_src: (B, d, T_src) — source shared feature
            p_tgt: (B, d, T_tgt) — target specific feature
        Returns:
            h_aligned: (B, d, T_tgt) — aligned shared feature matching T_tgt
        """
        if h_src.shape[-1] == p_tgt.shape[-1]:
            return h_src

        # q: (B, T_tgt, d), k: (B, T_src, d), v: (B, T_src, d)
        q = p_tgt.transpose(1, 2)
        k = h_src.transpose(1, 2)
        v = k

        d_k = q.size(-1)
        # Scaled dot-product attention scores: (B, T_tgt, T_src)
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / (d_k ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted combination of ALL T_src frames: (B, T_tgt, d)
        h_aligned = torch.bmm(attn_weights, v)

        return h_aligned.transpose(1, 2)  # (B, d, T_tgt)

    def forward(self, text, audio, video, text_c, audio_c, video_c):
        if self.use_bert:
            text = self.text_model(text)
            text_c = self.text_model(text_c)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2) # (B, D, T)
        x_v = video.transpose(1, 2)

        x_l_c = F.dropout(text_c.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a_c = audio_c.transpose(1, 2)
        x_v_c = video_c.transpose(1, 2)

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        proj_x_l_c = x_l_c if self.orig_d_l == self.d_l else self.proj_l(x_l_c)
        proj_x_a_c = x_a_c if self.orig_d_a == self.d_a else self.proj_a(x_a_c)
        proj_x_v_c = x_v_c if self.orig_d_v == self.d_v else self.proj_v(x_v_c)

        s_l = self.encoder_s_l(proj_x_l)
        s_v = self.encoder_s_v(proj_x_v)
        s_a = self.encoder_s_a(proj_x_a)

        c_l = self.encoder_c(proj_x_l)
        c_v = self.encoder_c(proj_x_v)
        c_a = self.encoder_c(proj_x_a)
        c_list = [c_l, c_v, c_a]

        # # --- OLD reconstruction (commented out, replaced by CRM) ---
        # recon_l = self.decoder_l(torch.cat([s_l, c_list[0]], dim=1))
        # recon_v = self.decoder_v(torch.cat([s_v, c_list[1]], dim=1))
        # recon_a = self.decoder_a(torch.cat([s_a, c_list[2]], dim=1))
        
        # s_l_r = self.encoder_s_l(recon_l)
        # s_v_r = self.encoder_s_v(recon_v)
        # s_a_r = self.encoder_s_a(recon_a)

        # ==============================================================
        # Cross-Reconstruction Module (CRM) (Section 3.3)
        #   X̃_{nm} = D_{nm}([H_n ⊕ P_m])
        #   X̃_m = F_m(X̃_{n1,m}, X̃_{n2,m})
        # ==============================================================
        # For target L: use shared from V and A (aligned to T_l) + specific from L
        c_v_l = self._align_time_attention(c_v, s_l)
        c_a_l = self._align_time_attention(c_a, s_l)
        x_tilde_vl = self.decoder_vl(torch.cat([c_v_l, s_l], dim=1))  # (B, d, T_l)
        x_tilde_al = self.decoder_al(torch.cat([c_a_l, s_l], dim=1))  # (B, d, T_l)
        x_tilde_l = self.fusion_l(
            torch.cat([x_tilde_vl.permute(0, 2, 1), x_tilde_al.permute(0, 2, 1)], dim=-1)
        ).permute(0, 2, 1)  # (B, d, T_l)

        # For target V: use shared from L and A (aligned to T_v) + specific from V
        c_l_v = self._align_time_attention(c_l, s_v)
        c_a_v = self._align_time_attention(c_a, s_v)
        x_tilde_lv = self.decoder_lv(torch.cat([c_l_v, s_v], dim=1))  # (B, d, T_v)
        x_tilde_av = self.decoder_av(torch.cat([c_a_v, s_v], dim=1))  # (B, d, T_v)
        x_tilde_v = self.fusion_v(
            torch.cat([x_tilde_lv.permute(0, 2, 1), x_tilde_av.permute(0, 2, 1)], dim=-1)
        ).permute(0, 2, 1)  # (B, d, T_v)

        # For target A: use shared from L and V (aligned to T_a) + specific from A
        c_l_a = self._align_time_attention(c_l, s_a)
        c_v_a = self._align_time_attention(c_v, s_a)
        x_tilde_la = self.decoder_la(torch.cat([c_l_a, s_a], dim=1))  # (B, d, T_a)
        x_tilde_va = self.decoder_va(torch.cat([c_v_a, s_a], dim=1))  # (B, d, T_a)
        x_tilde_a = self.fusion_a(
            torch.cat([x_tilde_la.permute(0, 2, 1), x_tilde_va.permute(0, 2, 1)], dim=-1)
        ).permute(0, 2, 1)  # (B, d, T_a)

        # Re-disentangle reconstructed features (Eq. 9):
        #   H̃_m = E_H(X̃_m),  P̃_m = E_P^m(X̃_m)
        h_tilde_l = self.encoder_c(x_tilde_l)     # (B, d, T)
        h_tilde_v = self.encoder_c(x_tilde_v)
        h_tilde_a = self.encoder_c(x_tilde_a)

        p_tilde_l = self.encoder_s_l(x_tilde_l)   # (B, d, T)
        p_tilde_v = self.encoder_s_v(x_tilde_v)
        p_tilde_a = self.encoder_s_a(x_tilde_a)

        # ==============================================================
        # Online Uncertainty Computation
        # ==============================================================
        # Unnormalized mean-pooled features (Conv1d format: B,D,T → B,D)
        c_bar = {
            'l': c_l.mean(dim=2),  # (B, D) — NOT L2-normalized
            'v': c_v.mean(dim=2),
            'a': c_a.mean(dim=2),
        }
        s_bar = {
            'l': s_l.mean(dim=2),  # (B, D) — NOT L2-normalized
            'v': s_v.mean(dim=2),
            'a': s_a.mean(dim=2),
        }

        # Shared uncertainty: ABTT Consensus
        # u_m^c = 1 - (z̊_m^c)^T Norm(Σ_{n≠m} z̊_n^c)
        u_consensus = self._shared_consensus(c_bar)
        # Specific uncertainty: ABTT Neighbor Support
        # u_m^s = 1 - (1/K) Σ_{j∈N_K} (z̊_m^s)^T ẑ_{m,j}^{s,bank}
        u_specific = self._specific_uncertainty(s_bar)

        # Reliability scores (all stop-gradient): r = clamp(exp(-u/tau), r_min, 1)
        r_shared, r_specific = self._compute_reliability(
            u_consensus, None, u_specific, None
        )

        # ── Role-conditioned reliability routing ──
        # Re-assign r_shared / r_specific based on ablation mode.
        # After this block, r_shared is fed to shared-consensus,
        # r_specific is fed to specific-exchange.
        if self.reliability_mode == 'uniform':
            # r^c = r^s = 1 → no reliability gating at all
            ones = {m: torch.ones_like(r_shared[m]) for m in r_shared}
            r_shared, r_specific = ones, {m: v.clone() for m, v in ones.items()}
        elif self.reliability_mode == 'shared_only':
            # Both modules use r^c
            r_specific = {m: r_shared[m].clone() for m in r_shared}
        elif self.reliability_mode == 'specific_only':
            # Both modules use r^s
            r_shared = {m: r_specific[m].clone() for m in r_specific}
        elif self.reliability_mode == 'averaged':
            # Both modules use (r^c + r^s) / 2
            avg = {m: ((r_shared[m] + r_specific[m]) / 2.0).detach() for m in r_shared}
            r_shared = avg
            r_specific = {m: v.clone() for m, v in avg.items()}
        elif self.reliability_mode == 'swapped':
            # Role-swapped: shared module gets r^s, specific module gets r^c
            r_shared, r_specific = (
                {m: r_specific[m].clone() for m in r_specific},
                {m: r_shared[m].clone() for m in r_shared},
            )
        # elif self.reliability_mode == 'full':  → default, no change needed

        # ==============================================================
        # Permute to (T, B, D) for transformer layers
        # ==============================================================
        s_l = s_l.permute(2, 0, 1) # (T, B, D)
        s_v = s_v.permute(2, 0, 1)
        s_a = s_a.permute(2, 0, 1)

        c_l = c_l.permute(2, 0, 1)
        c_v = c_v.permute(2, 0, 1)
        c_a = c_a.permute(2, 0, 1)

        hs_l_low = c_l.transpose(0, 1).contiguous().view(x_l.size(0), -1)
        repr_l_low = self.proj1_l_low(hs_l_low)
        hs_proj_l_low = self.proj2_l_low(
            F.dropout(F.relu(repr_l_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_low += hs_l_low
        logits_l_low = self.out_layer_l_low(hs_proj_l_low)

        hs_v_low = c_v.transpose(0, 1).contiguous().view(x_v.size(0), -1)
        repr_v_low = self.proj1_v_low(hs_v_low)
        hs_proj_v_low = self.proj2_v_low(
            F.dropout(F.relu(repr_v_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_low += hs_v_low
        logits_v_low = self.out_layer_v_low(hs_proj_v_low)

        hs_a_low = c_a.transpose(0, 1).contiguous().view(x_a.size(0), -1)
        repr_a_low = self.proj1_a_low(hs_a_low)
        hs_proj_a_low = self.proj2_a_low(
            F.dropout(F.relu(repr_a_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_a_low += hs_a_low
        logits_a_low = self.out_layer_a_low(hs_proj_a_low)

        # ==============================================================
        # Reliability-Gated Consensus (Shared Fusion)
        # γ^{c,i} = 1 - Π_m (1 - r_m^{c,i})
        # h̄^{c,i} = γ^{c,i} Σ_m w_m h_m^{c,i}
        # ==============================================================
        # self-attention for shared feature
        c_l_att = self.self_attentions_c_l(c_l)
        if type(c_l_att) == tuple:
            c_l_att = c_l_att[0]
        c_l_att = c_l_att[-1] # B, D

        c_v_att = self.self_attentions_c_v(c_v)
        if type(c_v_att) == tuple:
            c_v_att = c_v_att[0]
        c_v_att = c_v_att[-1]

        c_a_att = self.self_attentions_c_a(c_a)
        if type(c_a_att) == tuple:
            c_a_att = c_a_att[0]
        c_a_att = c_a_att[-1]

        c_att_dict = {'l': c_l_att, 'v': c_v_att, 'a': c_a_att}
        if self.ablate_shared_cal:
            # Ablation: equal-weight mean (no reliability gating)
            c_fusion = (c_att_dict['l'] + c_att_dict['v'] + c_att_dict['a']) / 3.0  # (B, D)
        else:
            # Reliability-gated consensus (sum version, single vector)
            c_fusion = self._reliability_gated_consensus_sum(c_att_dict, r_shared)  # (B, D)

        c_proj = self.proj2_c(
            F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=self.output_dropout,
                      training=self.training))
        c_proj += c_fusion
        logits_c = self.out_layer_c(c_proj)

        # ==============================================================
        # Reliability-Gated Complementary Exchange (Specific Cross-attention)
        # γ_{m,n}^s = 1 - (1 - r_m^s)(1 - r_n^s)
        # β_{m←n} = γ_{m,n}^s * r_n^s / (r_m^s + r_n^s)
        # h̃_{m←n} = h_m + β * (b_{m←n} - h_m)
        # ==============================================================
        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(s_l, s_a, s_a)  # (T, B, D): b_{l←a}
        h_l_with_vs = self.trans_l_with_v(s_l, s_v, s_v)  # (T, B, D): b_{l←v}
        if not self.ablate_specific_cal:
            h_l_with_as = self._reliability_gated_complementary_exchange(s_l, h_l_with_as, r_specific, 'l', 'a')
            h_l_with_vs = self._reliability_gated_complementary_exchange(s_l, h_l_with_vs, r_specific, 'l', 'v')
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(s_a, s_l, s_l)  # (T, B, D): b_{a←l}
        h_a_with_vs = self.trans_a_with_v(s_a, s_v, s_v)  # (T, B, D): b_{a←v}
        if not self.ablate_specific_cal:
            h_a_with_ls = self._reliability_gated_complementary_exchange(s_a, h_a_with_ls, r_specific, 'a', 'l')
            h_a_with_vs = self._reliability_gated_complementary_exchange(s_a, h_a_with_vs, r_specific, 'a', 'v')
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]

        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(s_v, s_l, s_l)  # (T, B, D): b_{v←l}
        h_v_with_as = self.trans_v_with_a(s_v, s_a, s_a)  # (T, B, D): b_{v←a}
        if not self.ablate_specific_cal:
            h_v_with_ls = self._reliability_gated_complementary_exchange(s_v, h_v_with_ls, r_specific, 'v', 'l')
            h_v_with_as = self._reliability_gated_complementary_exchange(s_v, h_v_with_as, r_specific, 'v', 'a')
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[-1]

        hs_proj_l_high = self.proj2_l_high(
            F.dropout(F.relu(self.proj1_l_high(last_h_l), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_high += last_h_l
        logits_l_high = self.out_layer_l_high(hs_proj_l_high)

        hs_proj_v_high = self.proj2_v_high(
            F.dropout(F.relu(self.proj1_v_high(last_h_v), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_high += last_h_v
        logits_v_high = self.out_layer_v_high(hs_proj_v_high)

        hs_proj_a_high = self.proj2_a_high(
            F.dropout(F.relu(self.proj1_a_high(last_h_a), inplace=True), p=self.output_dropout,
                      training=self.training))
        hs_proj_a_high += last_h_a
        logits_a_high = self.out_layer_a_high(hs_proj_a_high)

        last_h_l = torch.sigmoid(self.weight_l(last_h_l))
        last_h_v = torch.sigmoid(self.weight_v(last_h_v))
        last_h_a = torch.sigmoid(self.weight_a(last_h_a))
        c_fusion = torch.sigmoid(self.weight_c(c_fusion))

        # Last concat of shared and specific
        last_hs = torch.cat([last_h_l, last_h_v, last_h_a, c_fusion], dim=1)
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)

        # --- Cluster & prototype features ---
        # c_l, c_v, c_a are (T, B, D) at this point
        # Raw mean-pool over timestep dimension (for CMD loss)
        utt_shared_l = c_l.mean(dim=0)  # (B, D) — raw, no L2-norm
        utt_shared_v = c_v.mean(dim=0)  # (B, D)
        utt_shared_a = c_a.mean(dim=0)  # (B, D)
        # L2-normalized versions (for cluster/prototype losses)
        h_c_l = F.normalize(utt_shared_l, dim=1, p=2)  # (B, D)
        h_c_v = F.normalize(utt_shared_v, dim=1, p=2)  # (B, D)
        h_c_a = F.normalize(utt_shared_a, dim=1, p=2)  # (B, D)

        # ==============================================================
        # Reconstruction reliability weights (for training loop)
        # w_m^i = sg(r_m^{c,i} × r_m^{s,i}) — suppresses recon under uncertainty
        # L_m = Σ_i w_m^i ℓ_m^i / Σ_i w_m^i
        # ==============================================================
        recon_weights = {
            m: (r_shared[m] * r_specific[m]).detach()  # (B,) — already stop-grad
            for m in ['l', 'v', 'a']
        }

        res = {
            'logits_l_homo': logits_l_low,
            'logits_v_homo': logits_v_low,
            'logits_a_homo': logits_a_low,
            'repr_l_homo': repr_l_low,
            'repr_v_homo': repr_v_low,
            'repr_a_homo': repr_a_low,
            'origin_l': proj_x_l_c,
            'origin_v': proj_x_v_c,
            'origin_a': proj_x_a_c,
            's_l': s_l,
            's_v': s_v,
            's_a': s_a,
            'c_l': c_l,
            'c_v': c_v, # T, B, D
            'c_a': c_a,
            # # --- OLD reconstruction outputs (commented out, replaced by CRM) ---
            # 's_l_r': s_l_r,
            # 's_v_r': s_v_r,
            # 's_a_r': s_a_r,
            # 'recon_l': recon_l,
            # 'recon_v': recon_v,
            # 'recon_a': recon_a,
            # --- CRM cross-reconstruction outputs ---
            'x_tilde_l': x_tilde_l,  # (B, d, T) cross-reconstructed features
            'x_tilde_v': x_tilde_v,
            'x_tilde_a': x_tilde_a,
            'h_tilde_l': h_tilde_l,  # (B, d, T) re-disentangled shared
            'h_tilde_v': h_tilde_v,
            'h_tilde_a': h_tilde_a,
            'p_tilde_l': p_tilde_l,  # (B, d, T) re-disentangled specific (replaces s_l_r)
            'p_tilde_v': p_tilde_v,
            'p_tilde_a': p_tilde_a,
            'logits_l_hetero': logits_l_high,
            'logits_v_hetero': logits_v_high,
            'logits_a_hetero': logits_a_high,
            'repr_l_hetero': hs_proj_l_high,
            'repr_v_hetero': hs_proj_v_high,
            'repr_a_hetero': hs_proj_a_high,
            'last_h_l': h_ls[-1],
            'last_h_v': h_vs[-1],
            'last_h_a': h_as[-1],
            'c_fusion': c_fusion,          # gated, (B, 3*d)
            'last_hs_proj': last_hs_proj,  # (B, combined_dim) — pre-classifier repr    
            'logits_c': logits_c,
            # 'logits_shared': logits_shared, # shared branch prediction from CRM — commented out
            'output_logit': output,
            'h_c_l': h_c_l,         # (B, D) per-modality shared embeddings (L2-normed)
            'h_c_v': h_c_v,
            'h_c_a': h_c_a,
            # --- Uncertainty outputs ---
            'r_shared': r_shared,       # dict m → (B,) shared reliability
            'r_specific': r_specific,   # dict m → (B,) specific reliability
            'recon_weights': recon_weights,  # dict m → (B,) reconstruction suppression weights
        }
        return res

    @torch.no_grad()
    def sinkhorn_knopp(self, S, epsilon=0.05, sinkhorn_iterations=3):
        """
        Sinkhorn-Knopp algorithm for entropic OT.
        Args:
            S: similarity matrix (B, K)
            epsilon: entropic regularization
            sinkhorn_iterations: number of normalization iterations
        Returns:
            Q: transport plan (B, K), rows sum to 1/B, cols sum to 1/K
        """
        Q = torch.exp(S / epsilon).t()  # (K, B)
        B = Q.shape[1]
        K = Q.shape[0]

        # Make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for _ in range(sinkhorn_iterations):
            # Normalize rows: total weight per prototype must be 1/K
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= K
            # Normalize columns: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # columns sum to 1
        return Q.t()  # (B, K)

    def compute_cpa_loss(self, output, tau=0.1, epsilon=0.05, sinkhorn_iterations=3):
        """
        Cross-modal Prototype Alignment (CPA) loss for 3 modalities.

        For each directed modality pair (m → n) where m ≠ n:
          - q_m = sinkhorn_knopp(h_m @ C^T)   — soft cluster assignment (pseudo-label)
          - p_n = softmax(h_n @ C^T / τ)       — softmax probability
          - ℓ(h_n, q_m) = - Σ_k q_m^(k) log p_n^(k)

        L_CPA = (1 / 6B) Σ over all 6 directed cross-modal pairs
        """
        h_c_l = output['h_c_l']  # (B, D)
        h_c_v = output['h_c_v']  # (B, D)
        h_c_a = output['h_c_a']  # (B, D)

        # Prototype weights (K, D), already L2-normalized in training loop
        C = self.prototypes.weight  # (K, D)

        # --- Soft cluster assignment codes via Sinkhorn-Knopp (detached pseudo-labels) ---
        q_l = self.sinkhorn_knopp(h_c_l @ C.t(), epsilon=epsilon,
                                  sinkhorn_iterations=sinkhorn_iterations)  # (B, K)
        q_v = self.sinkhorn_knopp(h_c_v @ C.t(), epsilon=epsilon,
                                  sinkhorn_iterations=sinkhorn_iterations)  # (B, K)
        q_a = self.sinkhorn_knopp(h_c_a @ C.t(), epsilon=epsilon,
                                  sinkhorn_iterations=sinkhorn_iterations)  # (B, K)

        # --- Softmax probabilities (trained) ---
        log_p_l = F.log_softmax(h_c_l @ C.t() / tau, dim=1)  # (B, K)
        log_p_v = F.log_softmax(h_c_v @ C.t() / tau, dim=1)  # (B, K)
        log_p_a = F.log_softmax(h_c_a @ C.t() / tau, dim=1)  # (B, K)

        # --- Cross-modal prediction (6 directed pairs, no self-prediction) ---
        loss_cpa = 0.0
        loss_cpa -= torch.mean(torch.sum(q_v * log_p_l, dim=1))  # q_v → train l
        loss_cpa -= torch.mean(torch.sum(q_a * log_p_l, dim=1))  # q_a → train l
        loss_cpa -= torch.mean(torch.sum(q_l * log_p_v, dim=1))  # q_l → train v
        loss_cpa -= torch.mean(torch.sum(q_a * log_p_v, dim=1))  # q_a → train v
        loss_cpa -= torch.mean(torch.sum(q_l * log_p_a, dim=1))  # q_l → train a
        loss_cpa -= torch.mean(torch.sum(q_v * log_p_a, dim=1))  # q_v → train a

        loss_cpa /= 6.0  # average over 6 cross-modal pairs
        return loss_cpa