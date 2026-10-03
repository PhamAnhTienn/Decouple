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
        if args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500
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
        self.tau_s = getattr(args, 'tau_s', 1.0)    # temperature for specific reliability
        self.tau_f = getattr(args, 'tau_f', 1.0)    # temperature for shared fusion softmax
        self.r_min = getattr(args, 'r_min', 0.1)    # minimum reliability floor
        self.K_neighbors = getattr(args, 'K_neighbors', 10)  # neighbors for specific uncertainty

        combined_dim_low = self.d_a
        combined_dim_high = 2 * self.d_a
        combined_dim = 2 * (self.d_l + self.d_a + self.d_v) + self.d_l * 3
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

        # 3. Decoder for reconstruct three modalities
        self.decoder_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)
        self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)

        # init prototype
        self.nmb_prototypes = args.nmb_prototypes
        self.prototypes = nn.Linear(self.d_a, args.nmb_prototypes, bias=False)

        # for align c_l, c_v, c_a
        self.self_attentions_c_l = self.get_network(self_type='l')
        self.self_attentions_c_v = self.get_network(self_type='v')
        self.self_attentions_c_a = self.get_network(self_type='a')

        self.proj1_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.proj2_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.out_layer_c = nn.Linear(self.d_l * 3, output_dim)

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
        self.weight_c = nn.Linear(3 * self.d_l, 3 * self.d_l)
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

    def _shared_consensus(self, z_c_dict):
        """
        u_m^{c,con,i} = 1 - (1/(|M|-1)) * Σ_{n≠m} cos(z_m^c, z_n^c)

        Args:
            z_c_dict: dict m → (B, D) — L2-normed shared pooled repr
        Returns:
            dict m → (B,) consensus uncertainty
        """
        modalities = list(z_c_dict.keys())
        scores = {}
        for m in modalities:
            others = [n for n in modalities if n != m]
            cos_sum = sum(
                F.cosine_similarity(z_c_dict[m], z_c_dict[n], dim=1)
                for n in others
            )
            scores[m] = 1.0 - cos_sum / len(others)
        return scores

    def _cycle_uncertainty(self, z_dict, zt_dict):
        """
        u_m^{*,cy,i} = 1 - cos(z_m, z̃_m)
        Works for both shared and specific.

        Args:
            z_dict:  dict m → (B, D) — original pooled repr
            zt_dict: dict m → (B, D) — cycle-reconstructed pooled repr
        Returns:
            dict m → (B,) cycle uncertainty
        """
        return {
            m: 1.0 - F.cosine_similarity(z_dict[m], zt_dict[m], dim=1)
            for m in z_dict
        }

    def _neighbor_uncertainty(self, z_c_dict, z_s_dict, K):
        """
        In-batch neighbor uncertainty for specific representations.

        z_{-m}^c = Norm(Σ_{n≠m} z_n^c)
        N_K^m(i) = TopK_{j≠i}[cos(z_{-m}^{c,i}, z_{-m}^{c,j})]
        u_m^{s,nei,i} = 1 - (1/K) Σ_{j∈N_K} cos(z_m^{s,i}, sg(z_m^{s,j}))

        Args:
            z_c_dict: dict m → (B, D) — shared pooled repr (normed)
            z_s_dict: dict m → (B, D) — specific pooled repr (normed)
            K: number of neighbors
        Returns:
            dict m → (B,) neighbor uncertainty
        """
        modalities = list(z_c_dict.keys())
        B = z_c_dict[modalities[0]].shape[0]
        scores = {}

        for m in modalities:
            # Leave-one-out shared feature
            others = [n for n in modalities if n != m]
            z_minus_m = F.normalize(
                sum(z_c_dict[n] for n in others), dim=1
            )  # (B, D)

            # Pairwise cosine similarity (B, B); already L2-normed
            sim_matrix = z_minus_m @ z_minus_m.t()
            sim_matrix.fill_diagonal_(-float('inf'))  # mask self

            k_actual = min(K, B - 1)
            if k_actual <= 0:
                scores[m] = torch.zeros(B, device=z_c_dict[m].device)
                continue

            _, topk_idx = sim_matrix.topk(k_actual, dim=1)  # (B, K)

            z_s_m = z_s_dict[m]                    # (B, D)
            z_s_neighbors = z_s_m[topk_idx]         # (B, K, D)

            cos_neighbors = F.cosine_similarity(
                z_s_m.unsqueeze(1),
                z_s_neighbors.detach(),  # stop-gradient on neighbors
                dim=2,
            )  # (B, K)
            scores[m] = 1.0 - cos_neighbors.mean(dim=1)  # (B,)

        return scores

    def _compute_reliability(self, u_consensus, u_cycle_c, u_neighbor, u_cycle_s):
        """
        Aggregate per-modality uncertainties into reliability scores.

        u_m^c = 0.5 * (u_con + u_cy_c)
        u_m^s = 0.5 * (u_nei + u_cy_s)
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
            # u_s = 0.5 * (u_neighbor[m] + u_cycle_s[m])
            u_c = u_consensus[m]
            u_s = u_neighbor[m]
            r_c = torch.clamp(torch.exp(-u_c / self.tau_c), min=self.r_min, max=1.0)
            r_s = torch.clamp(torch.exp(-u_s / self.tau_s), min=self.r_min, max=1.0)
            r_shared[m] = r_c.detach()      # stop-gradient
            r_specific[m] = r_s.detach()     # stop-gradient
        return r_shared, r_specific

    def _reliability_weighted_shared_fusion(self, c_att_dict, r_shared):
        """
        Uncertainty-aware shared fusion (closed-form weighted average).

        Formula:
            h̄^c = Σ_m [ r_m^c / (Σ_n r_n^c) * h_m^{c,att} ]

        For backbone compatibility the output is a concat of per-modality
        reliability-scaled features: each modality is multiplied by its
        normalized weight w_m = r_m^c / Σ_n r_n^c, then concatenated to
        keep the (B, 3*D) shape expected by the downstream projection layer.

        Args:
            c_att_dict: dict m → (B, D) — self-attention shared output
            r_shared:   dict m → (B,) — shared reliability weights (stop-grad)
        Returns:
            c_fusion: (B, 3*D) — reliability-scaled concat
        """
        modalities = list(c_att_dict.keys())

        # Normalized reliability weights w_m = r_m / Σ_n r_n
        r_stack = torch.stack([r_shared[m] for m in modalities], dim=1)  # (B, |M|)
        r_sum = r_stack.sum(dim=1, keepdim=True).clamp(min=1e-8)         # (B, 1)
        w = r_stack / r_sum                                               # (B, |M|)
        # w is already detached because r_shared is stop-gradient

        reweighted = []
        for i, m in enumerate(modalities):
            wi = w[:, i].unsqueeze(1)            # (B, 1)
            reweighted.append(wi * c_att_dict[m])  # (B, D)

        return torch.cat(reweighted, dim=1)  # (B, 3*D)

    def _calibrate_cross_attention(self, h_m, h_cross, r_specific, m_query, m_kv):
        """
        Reliability-aware complementary exchange for specific representations.

        Closed-form solution of:
            tilde_h_{m←n} = argmin_x [ r_m ||x - h_m||^2 + r_n ||x - b_{m←n}||^2 ]

        which gives:
            tilde_h_{m←n} = (r_m * h_m + r_n * b_{m←n}) / (r_m + r_n)

        A reliable sender (high r_n) exerts stronger influence; a reliable
        receiver (high r_m) is less affected by potentially noisy messages.

        Args:
            h_m:         (T, B, D) — receiver's own specific representation
            h_cross:     (T, B, D) — cross-attention message b_{m←n}
            r_specific:  dict m → (B,) — specific reliability (stop-grad)
            m_query:     str — receiver modality key (m)
            m_kv:        str — sender modality key (n)
        Returns:
            h_calibrated: (T, B, D)
        """
        r_m = r_specific[m_query].unsqueeze(0).unsqueeze(2)  # (1, B, 1)
        r_n = r_specific[m_kv].unsqueeze(0).unsqueeze(2)     # (1, B, 1)
        denom = (r_m + r_n).clamp(min=1e-8)
        return (r_m * h_m + r_n * h_cross) / denom

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

        recon_l = self.decoder_l(torch.cat([s_l, c_list[0]], dim=1))
        recon_v = self.decoder_v(torch.cat([s_v, c_list[1]], dim=1))
        recon_a = self.decoder_a(torch.cat([s_a, c_list[2]], dim=1))

        s_l_r = self.encoder_s_l(recon_l)
        s_v_r = self.encoder_s_v(recon_v)
        s_a_r = self.encoder_s_a(recon_a)

        # ==============================================================
        # Online Uncertainty Computation
        # ==============================================================
        # Pool + normalize shared and specific features (Conv1d format: B,D,T)
        z_c = {
            'l': self._pool_norm(c_l),  # (B, D)
            'v': self._pool_norm(c_v),
            'a': self._pool_norm(c_a),
        }
        z_s = {
            'l': self._pool_norm(s_l),  # (B, D)
            'v': self._pool_norm(s_v),
            'a': self._pool_norm(s_a),
        }

        # Shared consensus uncertainty: u_m^c = 1 - (1/(|M|-1)) Σ_{n≠m} cos(z_m^c, z_n^c)
        u_consensus = self._shared_consensus(z_c)
        # Specific neighbor uncertainty: u_m^s via leave-one-out KNN on shared space
        u_neighbor  = self._neighbor_uncertainty(z_c, z_s, self.K_neighbors)

        # Reliability scores (all stop-gradient): r = clamp(exp(-u/tau), r_min, 1)
        r_shared, r_specific = self._compute_reliability(
            u_consensus, None, u_neighbor, None
        )

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
        # Uncertainty-aware Shared Fusion
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

        # Reliability-weighted fusion (replaces plain torch.cat)
        c_att_dict = {'l': c_l_att, 'v': c_v_att, 'a': c_a_att}
        c_fusion = self._reliability_weighted_shared_fusion(c_att_dict, r_shared)  # (B, 3*D)
        # c_fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)

        c_proj = self.proj2_c(
            F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=self.output_dropout,
                      training=self.training))
        c_proj += c_fusion
        logits_c = self.out_layer_c(c_proj)

        # ==============================================================
        # Reliability-Aware Specific Cross-attention
        # tilde_h_{m←n} = (r_m * h_m + r_n * b_{m←n}) / (r_m + r_n)
        # where b_{m←n} = CrossAttn(h_m^s, h_n^s)
        # ==============================================================
        # (V,A) --> L
        b_l_from_a = self.trans_l_with_a(s_l, s_a, s_a)  # (T, B, D): b_{l←a}
        b_l_from_v = self.trans_l_with_v(s_l, s_v, s_v)  # (T, B, D): b_{l←v}
        h_l_with_as = self._calibrate_cross_attention(s_l, b_l_from_a, r_specific, 'l', 'a')
        h_l_with_vs = self._calibrate_cross_attention(s_l, b_l_from_v, r_specific, 'l', 'v')
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

        # (L,V) --> A
        b_a_from_l = self.trans_a_with_l(s_a, s_l, s_l)  # (T, B, D): b_{a←l}
        b_a_from_v = self.trans_a_with_v(s_a, s_v, s_v)  # (T, B, D): b_{a←v}
        h_a_with_ls = self._calibrate_cross_attention(s_a, b_a_from_l, r_specific, 'a', 'l')
        h_a_with_vs = self._calibrate_cross_attention(s_a, b_a_from_v, r_specific, 'a', 'v')
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]

        # (L,A) --> V
        b_v_from_l = self.trans_v_with_l(s_v, s_l, s_l)  # (T, B, D): b_{v←l}
        b_v_from_a = self.trans_v_with_a(s_v, s_a, s_a)  # (T, B, D): b_{v←a}
        h_v_with_ls = self._calibrate_cross_attention(s_v, b_v_from_l, r_specific, 'v', 'l')
        h_v_with_as = self._calibrate_cross_attention(s_v, b_v_from_a, r_specific, 'v', 'a')
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
            's_l_r': s_l_r,
            's_v_r': s_v_r,
            's_a_r': s_a_r,
            'recon_l': recon_l,
            'recon_v': recon_v,
            'recon_a': recon_a,
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

    def compute_push_loss(self, margin=2.0):
        """
        Compute L_push: margin-based prototype separation loss.
        L_push = (1 / K(K-1)) * sum_{i!=j} max(0, m - ||c_i - c_j||^2)
        Encourages prototypes to be at least sqrt(margin) apart in L2 distance.
        """
        C = F.normalize(self.prototypes.weight, dim=1, p=2)  # (K, D)
        K = C.shape[0]

        # Pairwise squared L2 distances
        # ||c_i - c_j||^2 = ||c_i||^2 + ||c_j||^2 - 2 * c_i^T c_j
        # Since C is L2-normalized: ||c_i||^2 = 1
        dots = C @ C.t()  # (K, K)
        sq_dists = 2.0 - 2.0 * dots  # (K, K)

        # Mask out diagonal
        mask = ~torch.eye(K, dtype=torch.bool, device=C.device)
        sq_dists_off = sq_dists[mask]  # K*(K-1)

        # L_push = (1 / K(K-1)) * sum_{i!=j} max(0, m - ||c_i - c_j||^2)
        loss_push = torch.clamp(margin - sq_dists_off, min=0.0).sum() / (K * (K - 1))
        return loss_push