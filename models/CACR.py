import torch
import torch.nn as nn
import torch.nn.functional as F
from .subNets2 import BertTextEncoder
from .subNets2.transformers_encoder.transformer import TransformerEncoder


class FSR(nn.Module):
    """
    Fine-grained Sentiment Refinement module (Section 3.4).
    
    Extracts fine-grained emotional cues from visual- and audio-specific
    features and uses text-guided cross-attention to supplement textual semantics.
    """

    def __init__(self, d_model, num_heads, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1):
        super(FSR, self).__init__()
        # Feature refinement layers for P_V and P_A
        self.refine_v = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.refine_a = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        # Text-guided cross-attention: P_L as query, [P'_V; P'_A] as key/value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=False,
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        # FFN after cross-attention
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(relu_dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(res_dropout),
        )

    def forward(self, p_l, p_v, p_a):
        """
        Args:
            p_l: (T_l, B, d) language specific features
            p_v: (T_v, B, d) visual specific features
            p_a: (T_a, B, d) audio specific features
        Returns:
            p_l_prime: (T_l, B, d) enhanced language features
            p_v_prime: (T_v, B, d) refined visual features
            p_a_prime: (T_a, B, d) refined audio features
        """
        # Refine visual and audio specific features
        p_v_prime = p_v + self.refine_v(p_v)  # (T_v, B, d)
        p_a_prime = p_a + self.refine_a(p_a)  # (T_a, B, d)

        # Concatenate refined V and A as key/value for cross-attention
        kv = torch.cat([p_v_prime, p_a_prime], dim=0)  # (T_v + T_a, B, d)

        # Text-guided cross-attention: P_L queries [P'_V; P'_A]
        residual = p_l
        p_l_norm = self.layer_norm1(p_l)
        attn_out, _ = self.cross_attn(query=p_l_norm, key=kv, value=kv)
        p_l_enhanced = residual + attn_out  # (T_l, B, d)

        # FFN
        residual = p_l_enhanced
        p_l_enhanced = self.layer_norm2(p_l_enhanced)
        p_l_prime = residual + self.ffn(p_l_enhanced)  # (T_l, B, d)

        return p_l_prime, p_v_prime, p_a_prime


class CACR(nn.Module):
    def __init__(self, args):
        super(CACR, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(
                use_finetune=args.use_finetune,
                transformers=args.transformers,
                pretrained=args.pretrained,
            )
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads

        # Sequence lengths per dataset
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
        self.d = dst_feature_dims  # unified dimension d
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

        # CAW hyperparameters
        self.tau = getattr(args, 'tau', 0.5)
        self.omega_min = getattr(args, 'omega_min', 0.1)

        output_dim = 1

        # =====================================================================
        # 1. Temporal convolutional layers for dimension unification
        # =====================================================================
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # =====================================================================
        # 2. Feature Disentanglement (Section 3.1)
        #    E_H: shared encoder (one for all modalities)
        #    E_P^m: modality-specific encoders (one per modality)
        # =====================================================================
        # Modality-specific encoders E_P^m
        self.encoder_p_l = nn.Conv1d(self.d, self.d, kernel_size=1, padding=0, bias=False)
        self.encoder_p_v = nn.Conv1d(self.d, self.d, kernel_size=1, padding=0, bias=False)
        self.encoder_p_a = nn.Conv1d(self.d, self.d, kernel_size=1, padding=0, bias=False)

        # Modality-shared encoder E_H
        self.encoder_h = nn.Conv1d(self.d, self.d, kernel_size=1, padding=0, bias=False)

        # =====================================================================
        # 3. Cross-Reconstruction Module (CRM) (Section 3.3)
        #    6 cross-reconstruction decoders D_{nm}
        #    3 fusion layers F_m
        # =====================================================================
        # Decoders: D_{nm} reconstructs target m from shared H_n ⊕ specific P_m
        # For target L: D_VL (from V's shared), D_AL (from A's shared)
        self.decoder_vl = nn.Conv1d(self.d * 2, self.d, kernel_size=1, padding=0, bias=False)
        self.decoder_al = nn.Conv1d(self.d * 2, self.d, kernel_size=1, padding=0, bias=False)
        # For target V: D_LV (from L's shared), D_AV (from A's shared)
        self.decoder_lv = nn.Conv1d(self.d * 2, self.d, kernel_size=1, padding=0, bias=False)
        self.decoder_av = nn.Conv1d(self.d * 2, self.d, kernel_size=1, padding=0, bias=False)
        # For target A: D_LA (from L's shared), D_VA (from V's shared)
        self.decoder_la = nn.Conv1d(self.d * 2, self.d, kernel_size=1, padding=0, bias=False)
        self.decoder_va = nn.Conv1d(self.d * 2, self.d, kernel_size=1, padding=0, bias=False)

        # Fusion layers F_m to combine two cross-reconstructed features per target
        self.fusion_l = nn.Linear(self.d * 2, self.d)
        self.fusion_v = nn.Linear(self.d * 2, self.d)
        self.fusion_a = nn.Linear(self.d * 2, self.d)

        # =====================================================================
        # 4. Fine-grained Sentiment Refinement (FSR) (Section 3.4)
        # =====================================================================
        self.fsr = FSR(
            d_model=self.d,
            num_heads=self.num_heads,
            attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
        )

        # =====================================================================
        # 5. Fusion and Prediction (Section 3.5)
        #    Multimodal Transformer for specific features (cross-attention)
        #    Self-attention Transformer for shared features
        # =====================================================================
        # Cross-modal attention transformers for specific features
        # (V,A) --> L
        self.trans_l_with_a = self._get_cross_network('la')
        self.trans_l_with_v = self._get_cross_network('lv')
        # (L,V) --> A
        self.trans_a_with_l = self._get_cross_network('al')
        self.trans_a_with_v = self._get_cross_network('av')
        # (L,A) --> V
        self.trans_v_with_l = self._get_cross_network('vl')
        self.trans_v_with_a = self._get_cross_network('va')

        # Memory transformers for specific features (concat 2 cross-attention outputs)
        self.trans_l_mem = self._get_cross_network('l_mem', layers=3)
        self.trans_a_mem = self._get_cross_network('a_mem', layers=3)
        self.trans_v_mem = self._get_cross_network('v_mem', layers=3)

        # Self-attention transformer for shared features H̃
        self.trans_shared = self._get_cross_network('shared', layers=3)

        # =====================================================================
        # 6. Hierarchical Prediction Heads
        # =====================================================================
        # Specific branch prediction heads (for each TP'_m)
        specific_dim = 2 * self.d  # after concat of 2 cross-attention outputs
        self.pred_tp_l = self._make_pred_head(specific_dim, output_dim)
        self.pred_tp_v = self._make_pred_head(specific_dim, output_dim)
        self.pred_tp_a = self._make_pred_head(specific_dim, output_dim)

        # Shared branch prediction head (for TH̃)
        shared_dim = self.d * 3  # concat of H̃_L, H̃_V, H̃_A last tokens
        self.pred_shared = self._make_pred_head(shared_dim, output_dim)

        # Final fusion prediction head
        # M = Cat(TP'_L, TP'_V, TP'_A, TH̃)
        fusion_dim = specific_dim * 3 + shared_dim
        self.weight_l = nn.Linear(specific_dim, specific_dim)
        self.weight_v = nn.Linear(specific_dim, specific_dim)
        self.weight_a = nn.Linear(specific_dim, specific_dim)
        self.weight_shared = nn.Linear(shared_dim, shared_dim)
        self.pred_fusion = self._make_pred_head(fusion_dim, output_dim)

    def _get_cross_network(self, net_type, layers=-1):
        """Create a TransformerEncoder for the given cross-attention type."""
        if net_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d, self.attn_dropout
        elif net_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d, self.attn_dropout_a
        elif net_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d, self.attn_dropout_v
        elif net_type in ['l_mem']:
            embed_dim, attn_dropout = 2 * self.d, self.attn_dropout
        elif net_type in ['a_mem']:
            embed_dim, attn_dropout = 2 * self.d, self.attn_dropout_a
        elif net_type in ['v_mem']:
            embed_dim, attn_dropout = 2 * self.d, self.attn_dropout_v
        elif net_type == 'shared':
            embed_dim, attn_dropout = self.d * 3, self.attn_dropout
        else:
            raise ValueError(f"Unknown network type: {net_type}")

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=max(self.layers, layers),
            attn_dropout=attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask,
        )

    def _make_pred_head(self, in_dim, out_dim):
        """Create a 2-layer prediction head with residual connection."""
        return nn.ModuleDict({
            'proj1': nn.Linear(in_dim, in_dim),
            'proj2': nn.Linear(in_dim, in_dim),
            'out': nn.Linear(in_dim, out_dim),
        })

    def _pred_forward(self, head, x):
        """Forward through a prediction head with residual."""
        h = F.relu(head['proj1'](x), inplace=True)
        h = F.dropout(h, p=self.output_dropout, training=self.training)
        h = head['proj2'](h)
        h = h + x  # residual
        logits = head['out'](h)
        return logits, h

    def compute_conflict_weights(self, h_l, h_v, h_a):
        """
        Conflict-Aware Weighting (CAW) strategy (Section 3.2).
        
        Computes per-sample conflict scores from shared features and maps
        them to cross-reconstruction weights via exponential decay.
        
        Args:
            h_l: (B, d) global shared feature for language (temporal-pooled)
            h_v: (B, d) global shared feature for visual
            h_a: (B, d) global shared feature for audio
        Returns:
            w_l, w_v, w_a: (B,) per-sample weights for each modality
        """
        # Normalize for cosine similarity
        h_l_norm = F.normalize(h_l, dim=-1)
        h_v_norm = F.normalize(h_v, dim=-1)
        h_a_norm = F.normalize(h_a, dim=-1)

        # Pairwise cosine similarities
        s_lv = (h_l_norm * h_v_norm).sum(dim=-1)  # (B,)
        s_la = (h_l_norm * h_a_norm).sum(dim=-1)
        s_va = (h_v_norm * h_a_norm).sum(dim=-1)

        # Conflict scores: d_{m,n} = 1 - s_{m,n}
        d_lv = 1.0 - s_lv
        d_la = 1.0 - s_la
        d_va = 1.0 - s_va

        # Overall conflict score per modality: C_m = mean of d_{m,n} for n != m
        c_l = (d_lv + d_la) / 2.0  # (B,)
        c_v = (d_lv + d_va) / 2.0
        c_a = (d_la + d_va) / 2.0

        # Exponential decay: ω̂_m = exp(-C_m / τ), ω_m = max(ω̂_m, ω_min)
        w_l = torch.exp(-c_l / self.tau).clamp(min=self.omega_min)
        w_v = torch.exp(-c_v / self.tau).clamp(min=self.omega_min)
        w_a = torch.exp(-c_a / self.tau).clamp(min=self.omega_min)

        return w_l, w_v, w_a

    def forward(self, text, audio, video, is_distill=False):
        # =====================================================================
        # Step 0: Feature extraction and dimension unification
        # =====================================================================
        if self.use_bert:
            text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)  # (B, D_l, T)
        x_a = audio.transpose(1, 2)  # (B, D_a, T)
        x_v = video.transpose(1, 2)  # (B, D_v, T)

        # Project to unified dimension d
        proj_x_l = x_l if self.orig_d_l == self.d else self.proj_l(x_l)  # (B, d, T_l)
        proj_x_a = x_a if self.orig_d_a == self.d else self.proj_a(x_a)  # (B, d, T_a)
        proj_x_v = x_v if self.orig_d_v == self.d else self.proj_v(x_v)  # (B, d, T_v)

        # =====================================================================
        # Step 1: Feature Disentanglement (Section 3.1)
        #   H_m = E_H(X_m),  P_m = E_P^m(X_m)
        # =====================================================================
        # Specific features P_m: (B, d, T_m)
        p_l = self.encoder_p_l(proj_x_l)
        p_v = self.encoder_p_v(proj_x_v)
        p_a = self.encoder_p_a(proj_x_a)

        # Shared features H_m: (B, d, T_m)
        h_l = self.encoder_h(proj_x_l)
        h_v = self.encoder_h(proj_x_v)
        h_a = self.encoder_h(proj_x_a)

        # =====================================================================
        # Step 2: Conflict-Aware Weighting (Section 3.2)
        #   Temporal pool H_m → global h_m, compute conflict weights
        # =====================================================================
        # Temporal mean pooling: (B, d, T) → (B, d)
        h_l_global = h_l.mean(dim=2)
        h_v_global = h_v.mean(dim=2)
        h_a_global = h_a.mean(dim=2)

        w_l, w_v, w_a = self.compute_conflict_weights(h_l_global, h_v_global, h_a_global)

        # =====================================================================
        # Step 3: Cross-Reconstruction Module (CRM) (Section 3.3)
        #   X̃_{nm} = D_{nm}([H_n ⊕ P_m])
        #   X̃_m = F_m(X̃_{n1,m}, X̃_{n2,m})
        # =====================================================================
        # For target L: use shared from V and A + specific from L
        # Note: H_n and P_m may have different temporal lengths when unaligned.
        # The paper uses aligned data (same T), so we can concatenate directly.
        x_tilde_vl = self.decoder_vl(torch.cat([h_v, p_l.expand_as(h_v)], dim=1))  # (B, d, T_v)
        x_tilde_al = self.decoder_al(torch.cat([h_a, p_l.expand_as(h_a)], dim=1))  # (B, d, T_a)
        # Fuse: pool to common length then fuse
        # Since data is aligned (T_l = T_v = T_a), we can directly use them
        x_tilde_l = self.fusion_l(
            torch.cat([x_tilde_vl.permute(0, 2, 1), x_tilde_al.permute(0, 2, 1)], dim=-1)
        ).permute(0, 2, 1)  # (B, d, T)

        # For target V: use shared from L and A + specific from V
        x_tilde_lv = self.decoder_lv(torch.cat([h_l, p_v.expand_as(h_l)], dim=1))  # (B, d, T_l)
        x_tilde_av = self.decoder_av(torch.cat([h_a, p_v.expand_as(h_a)], dim=1))  # (B, d, T_a)
        x_tilde_v = self.fusion_v(
            torch.cat([x_tilde_lv.permute(0, 2, 1), x_tilde_av.permute(0, 2, 1)], dim=-1)
        ).permute(0, 2, 1)  # (B, d, T)

        # For target A: use shared from L and V + specific from A
        x_tilde_la = self.decoder_la(torch.cat([h_l, p_a.expand_as(h_l)], dim=1))  # (B, d, T_l)
        x_tilde_va = self.decoder_va(torch.cat([h_v, p_a.expand_as(h_v)], dim=1))  # (B, d, T_v)
        x_tilde_a = self.fusion_a(
            torch.cat([x_tilde_la.permute(0, 2, 1), x_tilde_va.permute(0, 2, 1)], dim=-1)
        ).permute(0, 2, 1)  # (B, d, T)

        # Re-disentangle reconstructed features (Eq. 9):
        #   H̃_m = E_H(X̃_m),  P̃_m = E_P^m(X̃_m)
        h_tilde_l = self.encoder_h(x_tilde_l)   # (B, d, T)
        h_tilde_v = self.encoder_h(x_tilde_v)
        h_tilde_a = self.encoder_h(x_tilde_a)

        p_tilde_l = self.encoder_p_l(x_tilde_l)  # (B, d, T)
        p_tilde_v = self.encoder_p_v(x_tilde_v)
        p_tilde_a = self.encoder_p_a(x_tilde_a)

        # =====================================================================
        # Step 4: Fine-grained Sentiment Refinement (FSR) (Section 3.4)
        #   Using original P_m features
        # =====================================================================
        # Convert to (T, B, d) for transformer-style processing
        p_l_seq = p_l.permute(2, 0, 1)  # (T, B, d)
        p_v_seq = p_v.permute(2, 0, 1)
        p_a_seq = p_a.permute(2, 0, 1)

        p_l_prime, p_v_prime, p_a_prime = self.fsr(p_l_seq, p_v_seq, p_a_seq)
        # p_l_prime: (T, B, d), p_v_prime: (T, B, d), p_a_prime: (T, B, d)

        # =====================================================================
        # Step 5: Fusion (Section 3.5)
        #   Multimodal Transformer for specific features
        #   Self-attention Transformer for shared features
        # =====================================================================
        # --- Specific branch: cross-modal attention on refined P'_m ---
        # (V,A) --> L
        h_l_with_a = self.trans_l_with_a(p_l_prime, p_a_prime, p_a_prime)
        h_l_with_v = self.trans_l_with_v(p_l_prime, p_v_prime, p_v_prime)
        h_ls = torch.cat([h_l_with_a, h_l_with_v], dim=2)  # (T, B, 2d)
        h_ls = self.trans_l_mem(h_ls)
        if isinstance(h_ls, tuple):
            h_ls = h_ls[0]
        tp_l = h_ls[-1]  # (B, 2d)

        # (L,V) --> A
        h_a_with_l = self.trans_a_with_l(p_a_prime, p_l_prime, p_l_prime)
        h_a_with_v = self.trans_a_with_v(p_a_prime, p_v_prime, p_v_prime)
        h_as = torch.cat([h_a_with_l, h_a_with_v], dim=2)
        h_as = self.trans_a_mem(h_as)
        if isinstance(h_as, tuple):
            h_as = h_as[0]
        tp_a = h_as[-1]  # (B, 2d)

        # (L,A) --> V
        h_v_with_l = self.trans_v_with_l(p_v_prime, p_l_prime, p_l_prime)
        h_v_with_a = self.trans_v_with_a(p_v_prime, p_a_prime, p_a_prime)
        h_vs = torch.cat([h_v_with_l, h_v_with_a], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if isinstance(h_vs, tuple):
            h_vs = h_vs[0]
        tp_v = h_vs[-1]  # (B, 2d)

        # --- Shared branch: self-attention on H̃ ---
        # H̃ = [H̃_L ; H̃_V ; H̃_A] concatenated along feature dim
        h_tilde_l_seq = h_tilde_l.permute(2, 0, 1)  # (T, B, d)
        h_tilde_v_seq = h_tilde_v.permute(2, 0, 1)
        h_tilde_a_seq = h_tilde_a.permute(2, 0, 1)

        # Concat along feature dimension: (T, B, 3d)
        h_tilde_concat = torch.cat([h_tilde_l_seq, h_tilde_v_seq, h_tilde_a_seq], dim=2)
        t_h_tilde = self.trans_shared(h_tilde_concat)
        if isinstance(t_h_tilde, tuple):
            t_h_tilde = t_h_tilde[0]
        t_h_tilde_last = t_h_tilde[-1]  # (B, 3d)

        # =====================================================================
        # Step 6: Hierarchical Predictions
        # =====================================================================
        # Specific branch predictions
        logits_tp_l, repr_tp_l = self._pred_forward(self.pred_tp_l, tp_l)
        logits_tp_v, repr_tp_v = self._pred_forward(self.pred_tp_v, tp_v)
        logits_tp_a, repr_tp_a = self._pred_forward(self.pred_tp_a, tp_a)

        # Shared branch prediction
        logits_shared, repr_shared = self._pred_forward(self.pred_shared, t_h_tilde_last)

        # Final fusion: M = Cat(TP'_L, TP'_V, TP'_A, TH̃)
        fusion = torch.cat([tp_l, tp_v, tp_a, t_h_tilde_last], dim=1)
        output, _ = self._pred_forward(self.pred_fusion, fusion)

        # =====================================================================
        # Return all intermediate representations for loss computation
        # =====================================================================
        res = {
            # Final output
            'output_logit': output,
            # Original projected features X_m (for reconstruction target)
            'origin_l': proj_x_l,
            'origin_v': proj_x_v,
            'origin_a': proj_x_a,
            # Shared features H_m
            'h_l': h_l,
            'h_v': h_v,
            'h_a': h_a,
            # Specific features P_m
            'p_l': p_l,
            'p_v': p_v,
            'p_a': p_a,
            # Cross-reconstructed features X̃_m
            'x_tilde_l': x_tilde_l,
            'x_tilde_v': x_tilde_v,
            'x_tilde_a': x_tilde_a,
            # Re-disentangled shared H̃_m
            'h_tilde_l': h_tilde_l,
            'h_tilde_v': h_tilde_v,
            'h_tilde_a': h_tilde_a,
            # Re-disentangled specific P̃_m
            'p_tilde_l': p_tilde_l,
            'p_tilde_v': p_tilde_v,
            'p_tilde_a': p_tilde_a,
            # Conflict weights (per-sample)
            'w_l': w_l,
            'w_v': w_v,
            'w_a': w_a,
            # Hierarchical prediction logits
            'logits_tp_l': logits_tp_l,
            'logits_tp_v': logits_tp_v,
            'logits_tp_a': logits_tp_a,
            'logits_shared': logits_shared,
        }
        return res