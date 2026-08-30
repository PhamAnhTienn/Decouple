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

        # Temporal prototype infrastructure
        self.T_l = self.len_l - args.conv1d_kernel_size_l + 1
        self.T_v = self.len_v - args.conv1d_kernel_size_v + 1
        self.T_a = self.len_a - args.conv1d_kernel_size_a + 1
        self.num_classes = 7
        self.register_buffer('proto_l', torch.zeros(7, self.T_l, self.d_l))
        self.register_buffer('proto_v', torch.zeros(7, self.T_v, self.d_v))
        self.register_buffer('proto_a', torch.zeros(7, self.T_a, self.d_a))
        self.register_buffer('proto_initialized', torch.zeros(1, dtype=torch.bool))
        self.cache_l = [[] for _ in range(7)]
        self.cache_v = [[] for _ in range(7)]
        self.cache_a = [[] for _ in range(7)]

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

    def forward(self, text, audio, video, is_distill=False):
        if self.use_bert:
            text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2) # (B, D, T)
        x_v = video.transpose(1, 2)

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

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
        c_fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)

        c_proj = self.proj2_c(
            F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=self.output_dropout,
                      training=self.training))
        c_proj += c_fusion
        logits_c = self.out_layer_c(c_proj)

        # cross-modal attention
        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(s_l, s_a, s_a)  # Dimension (L, N, d_l)
        h_l_with_vs = self.trans_l_with_v(s_l, s_v, s_v)  # Dimension (L, N, d_l)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(s_a, s_l, s_l)
        h_a_with_vs = self.trans_a_with_v(s_a, s_v, s_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]

        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(s_v, s_l, s_l)
        h_v_with_as = self.trans_v_with_a(s_v, s_a, s_a)
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

        last_hs = torch.cat([last_h_l, last_h_v, last_h_a, c_fusion], dim=1)
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)

        res = {
            'logits_l_homo': logits_l_low,
            'logits_v_homo': logits_v_low,
            'logits_a_homo': logits_a_low,
            'repr_l_homo': repr_l_low,
            'repr_v_homo': repr_v_low,
            'repr_a_homo': repr_a_low,
            'origin_l': proj_x_l,
            'origin_v': proj_x_v,
            'origin_a': proj_x_a,
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
        }
        return res

    @staticmethod
    def discretize_labels(y):
        """Discretize continuous sentiment labels y ∈ [-3,3] into C=7 classes."""
        return torch.clamp(torch.round(y + 3), 0, 6).long()

    @torch.no_grad()
    def cache_features(self, output, labels):
        """Append current batch's common features to per-class caches."""
        classes = self.discretize_labels(labels.view(-1))  # (B,)
        # c_l, c_v, c_a are (T, B, D) — permute to (B, T, D)
        c_l = output['c_l'].permute(1, 0, 2).detach()  # (B, T_l, D)
        c_v = output['c_v'].permute(1, 0, 2).detach()  # (B, T_v, D)
        c_a = output['c_a'].permute(1, 0, 2).detach()  # (B, T_a, D)
        for i in range(classes.size(0)):
            c = classes[i].item()
            self.cache_l[c].append(c_l[i])  # (T_l, D)
            self.cache_v[c].append(c_v[i])
            self.cache_a[c].append(c_a[i])

    def compute_proto_loss(self, output, labels, tau=0.07):
        """
        Compute InfoNCE loss between samples and class prototypes (Eq. 4-5).
        Only active after prototypes have been initialized (epoch >= 2).
        Returns 0 during the first epoch (cache-only phase).
        """
        if not self.proto_initialized.item():
            return torch.tensor(0.0, device=output['c_l'].device)

        classes = self.discretize_labels(labels.view(-1))  # (B,)
        loss = 0.0
        count = 0

        for c_feat, proto in [
            (output['c_l'], self.proto_l),   # c_feat: (T, B, D), proto: (7, T_m, D)
            (output['c_v'], self.proto_v),
            (output['c_a'], self.proto_a),
        ]:
            feat = c_feat.permute(1, 0, 2)  # (B, T_m, D)
            # Temporal mean: μ(X_i^m) = (1/T'_m) Σ_t X_{i,t}^m
            mu_feat = feat.mean(dim=1)       # (B, D)
            # Temporal mean of each prototype: μ(P_c^m)
            mu_proto = proto.mean(dim=1)     # (7, D)

            # Valid prototypes = classes with non-zero prototype
            valid_mask = (proto.abs().sum(dim=(1, 2)) > 0)  # (7,)
            if valid_mask.sum() < 2:
                continue

            valid_indices = valid_mask.nonzero(as_tuple=True)[0]  # (R,)
            mu_proto_valid = mu_proto[valid_indices]               # (R, D)

            # Cosine similarities: s_c^m(X_i^m) = cos(μ(X_i^m), μ(P_c^m)) / τ
            sim = F.cosine_similarity(
                mu_feat.unsqueeze(1),          # (B, 1, D)
                mu_proto_valid.unsqueeze(0),   # (1, R, D)
                dim=2
            ) / tau  # (B, R)

            # Map class_id -> index in valid_indices
            class_to_valid = {int(valid_indices[j].item()): j for j in range(len(valid_indices))}

            # InfoNCE: ℓ_i^m = -log(exp(s_{c_i}) / Σ_j exp(s_j))
            batch_loss = 0.0
            n_valid = 0
            for i in range(classes.size(0)):
                ci = classes[i].item()
                if ci in class_to_valid:
                    pos_idx = class_to_valid[ci]
                    log_softmax = F.log_softmax(sim[i], dim=0)  # (R,)
                    batch_loss -= log_softmax[pos_idx]
                    n_valid += 1

            if n_valid > 0:
                loss += batch_loss / n_valid
                count += 1

        # Average over modalities (Eq. 5)
        return loss / max(count, 1)

    @torch.no_grad()
    def update_prototypes(self):
        """
        Update prototypes from caches at end of epoch (Eq. 6-7).
        Computes centroid, removes 5% farthest samples, averages remaining.
        """
        for cache, proto_buf in [
            (self.cache_l, self.proto_l),
            (self.cache_v, self.proto_v),
            (self.cache_a, self.proto_a),
        ]:
            for c in range(self.num_classes):
                S = cache[c]
                if len(S) == 0:
                    continue
                S_tensor = torch.stack(S)          # (N, T_m, D)
                mu_samples = S_tensor.mean(dim=1)  # (N, D)
                centroid = mu_samples.mean(dim=0)   # (D,)

                # Remove 5% farthest from centroid (Eq. 6)
                dists = torch.norm(mu_samples - centroid.unsqueeze(0), dim=1)  # (N,)
                k_remove = max(1, int(0.05 * len(S)))
                if len(S) > k_remove:
                    _, worst_indices = torch.topk(dists, k_remove, largest=True)
                    keep_mask = torch.ones(len(S), dtype=torch.bool, device=S_tensor.device)
                    keep_mask[worst_indices] = False
                    S_tensor = S_tensor[keep_mask]

                # Update prototype: average remaining sequences (Eq. 7)
                proto_buf[c] = S_tensor.mean(dim=0)  # (T_m, D)

        self.proto_initialized.fill_(True)

        # Clear all caches
        self.cache_l = [[] for _ in range(self.num_classes)]
        self.cache_v = [[] for _ in range(self.num_classes)]
        self.cache_a = [[] for _ in range(self.num_classes)]