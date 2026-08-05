import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .subNets2 import BertTextEncoder
from .subNets2.transformers_encoder.transformer import TransformerEncoder

class CMD(nn.Module):
    """
    Central Moment Discrepancy (CMD).
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1 - x2, 2)
        summed = torch.sum(power)
        sqrt = summed ** (0.5)
        return sqrt

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)


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

        # GMM + OT hyperparameters (for heterogeneity alignment on shared space)
        self.num_prototypes = args.nmb_prototypes
        self.gmm_em_iters = getattr(args, 'gmm_em_iters', 5)
        self.gmm_var_floor = getattr(args, 'gmm_var_floor', 1e-4)
        self.ot_reg = getattr(args, 'lambda_ot', 0.1)
        self.ot_num_iters = getattr(args, 'ot_num_iters', 50)

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

        # --- Shared-space utterance features (for CMD loss) ---
        # c_l, c_v, c_a are (T, B, D) at this point
        utt_shared_l = c_l.mean(dim=0)  # (B, D)
        utt_shared_v = c_v.mean(dim=0)  # (B, D)
        utt_shared_a = c_a.mean(dim=0)  # (B, D)

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
            'c_v': c_v,  # T, B, D
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
            'logits_c': logits_c,
            'output_logit': output,
            # Raw mean-pooled shared features for CMD loss
            'utt_shared_l': utt_shared_l,  # (B, D)
            'utt_shared_v': utt_shared_v,
            'utt_shared_a': utt_shared_a,
        }
        return res

    # ==================== GMM + Optimal Transport ====================

    def _init_gmm(self, x):
        """Initialize GMM parameters (means, diagonal variances, mixture coefficients).

        Args:
            x: (B, D) feature matrix.
        Returns:
            pi: (K,) mixture coefficients.
            mu: (K, D) means.
            var: (K, D) diagonal variances.
        """
        batch_size, dim = x.shape
        k = self.num_prototypes
        if batch_size >= k:
            indices = torch.linspace(0, batch_size - 1, steps=k, device=x.device).round().long()
            mu = x.index_select(0, indices)
        else:
            repeat = math.ceil(k / batch_size)
            mu = x.repeat(repeat, 1)[:k]

        var = x.var(dim=0, unbiased=False).clamp_min(self.gmm_var_floor)
        var = var.unsqueeze(0).expand(k, dim).clone()
        pi = torch.full((k,), 1.0 / k, device=x.device, dtype=x.dtype)
        return pi, mu, var

    def _gmm_log_prob(self, x, pi, mu, var):
        """Compute per-component log-probabilities under diagonal-covariance GMM.

        Args:
            x: (B, D) samples.
            pi: (K,) mixture coefficients.
            mu: (K, D) component means.
            var: (K, D) component variances.
        Returns:
            (B, K) log-probabilities (unnormalized by mixture).
        """
        diff = x.unsqueeze(1) - mu.unsqueeze(0)            # (B, K, D)
        mahalanobis = (diff.pow(2) / var.unsqueeze(0)).sum(dim=2)  # (B, K)
        log_det = var.log().sum(dim=1).unsqueeze(0)         # (1, K)
        log_norm = x.size(1) * math.log(2.0 * math.pi)
        return pi.clamp_min(1e-8).log().unsqueeze(0) - 0.5 * (mahalanobis + log_det + log_norm)

    def _fit_gmm_em(self, x):
        """Fit a GMM to *x* via Expectation-Maximization.

        Args:
            x: (B, D) feature matrix.
        Returns:
            dict with keys 'pi', 'mu', 'var', 'posterior'.
        """
        pi, mu, var = self._init_gmm(x)
        for _ in range(self.gmm_em_iters):
            # E-step
            gamma = F.softmax(self._gmm_log_prob(x, pi, mu, var), dim=1)  # (B, K)
            nk = gamma.sum(dim=0).clamp_min(1e-6)  # (K,)
            # M-step
            pi = nk / x.size(0)
            mu = gamma.transpose(0, 1).matmul(x) / nk.unsqueeze(1)       # (K, D)
            second_moment = gamma.transpose(0, 1).matmul(x.pow(2)) / nk.unsqueeze(1)
            var = (second_moment - mu.pow(2)).clamp_min(self.gmm_var_floor)
        # Final posterior
        gamma = F.softmax(self._gmm_log_prob(x, pi, mu, var), dim=1)
        return {'pi': pi, 'mu': mu, 'var': var, 'posterior': gamma}

    def _gaussian_cost(self, mu1, var1, mu2, var2):
        """Wasserstein-2 inspired cost between pairs of diagonal Gaussians.

        C(k_i, k_j) = ||mu_i - mu_j||^2 + Tr(Σ_i + Σ_j - 2(Σ_i Σ_j)^{1/2})
        For diagonal covariances this simplifies element-wise.

        Returns:
            (K, K) cost matrix.
        """
        diff = mu1.unsqueeze(1) - mu2.unsqueeze(0)          # (K, K, D)
        mean_cost = diff.pow(2).mean(dim=2)                  # (K, K)
        cov_cost = (
            var1.unsqueeze(1)
            + var2.unsqueeze(0)
            - 2.0 * torch.sqrt(var1.unsqueeze(1) * var2.unsqueeze(0) + 1e-8)
        ).mean(dim=2)                                        # (K, K)
        return mean_cost + cov_cost

    def _sample_to_proto_cost(self, x, mu, var):
        """Squared-distance cost from each sample to each prototype.

        Args:
            x: (B, D) samples.
            mu: (K, D) prototype means.
            var: (K, D) prototype variances (unused unless Mahalanobis).
        Returns:
            (B, K) cost matrix.
        """
        diff = x.unsqueeze(1) - mu.unsqueeze(0)  # (B, K, D)
        return diff.pow(2).mean(dim=2)

    def _multi_marginal_sinkhorn(self, cost, nu_l, nu_a, nu_v):
        """Multi-marginal Sinkhorn solver for 3-modality prototype alignment.

        Args:
            cost: (K, K, K) joint cost tensor.
            nu_l, nu_a, nu_v: (K,) marginal distributions.
        Returns:
            Scalar OT loss (transport cost + entropy regularization).
        """
        k = cost.size(0)
        kernel = torch.exp(-cost / max(self.ot_reg, 1e-6)).clamp_min(1e-30)
        u = torch.ones(k, device=cost.device, dtype=cost.dtype)
        v = torch.ones_like(u)
        w = torch.ones_like(u)

        for _ in range(self.ot_num_iters):
            u = nu_l / (torch.sum(kernel * v.view(1, k, 1) * w.view(1, 1, k), dim=(1, 2)) + 1e-8)
            v = nu_a / (torch.sum(kernel * u.view(k, 1, 1) * w.view(1, 1, k), dim=(0, 2)) + 1e-8)
            w = nu_v / (torch.sum(kernel * u.view(k, 1, 1) * v.view(1, k, 1), dim=(0, 1)) + 1e-8)

        plan = u.view(k, 1, 1) * v.view(1, k, 1) * w.view(1, 1, k) * kernel
        entropy = (plan * (plan.clamp_min(1e-8).log() - 1.0)).sum()
        return (plan * cost).sum() + self.ot_reg * entropy

    def compute_hetero_loss(self, c_l, c_a, c_v):
        """Heterogeneity alignment loss on the shared space.

        Combines:
          - L_OT:    Multi-marginal OT aligning GMM prototypes across modalities.
          - L_Proto:  Sample-to-prototype calibration (weighted by GMM posterior).

        Args:
            c_l, c_a, c_v: shared features, each (T, B, D).
        Returns:
            Scalar L_hete = L_OT + L_Proto.
        """
        # Mean-pool over temporal dimension → (B, D)
        x_l = c_l.mean(dim=0)
        x_a = c_a.mean(dim=0)
        x_v = c_v.mean(dim=0)

        # Fit per-modality GMMs
        g_l = self._fit_gmm_em(x_l)
        g_a = self._fit_gmm_em(x_a)
        g_v = self._fit_gmm_em(x_v)

        # Marginal distributions from posteriors
        nu_l = g_l['posterior'].mean(dim=0)
        nu_a = g_a['posterior'].mean(dim=0)
        nu_v = g_v['posterior'].mean(dim=0)
        nu_l = nu_l / nu_l.sum().clamp_min(1e-8)
        nu_a = nu_a / nu_a.sum().clamp_min(1e-8)
        nu_v = nu_v / nu_v.sum().clamp_min(1e-8)

        # Pairwise Gaussian costs → joint 3D cost tensor
        cost_la = self._gaussian_cost(g_l['mu'], g_l['var'], g_a['mu'], g_a['var'])  # (K, K)
        cost_lv = self._gaussian_cost(g_l['mu'], g_l['var'], g_v['mu'], g_v['var'])  # (K, K)
        cost_av = self._gaussian_cost(g_a['mu'], g_a['var'], g_v['mu'], g_v['var'])  # (K, K)
        joint_cost = cost_la.unsqueeze(2) + cost_lv.unsqueeze(1) + cost_av.unsqueeze(0)  # (K, K, K)

        # L_OT: global prototype alignment
        ot_loss = self._multi_marginal_sinkhorn(joint_cost, nu_l, nu_a, nu_v)

        # L_Proto: local sample-to-prototype calibration
        # Each source modality's samples are aligned to target modality prototypes
        local_losses = [
            (g_l['posterior'] * self._sample_to_proto_cost(x_l, g_a['mu'], g_a['var'])).sum(dim=1).mean(),
            (g_l['posterior'] * self._sample_to_proto_cost(x_l, g_v['mu'], g_v['var'])).sum(dim=1).mean(),
            (g_a['posterior'] * self._sample_to_proto_cost(x_a, g_l['mu'], g_l['var'])).sum(dim=1).mean(),
            (g_a['posterior'] * self._sample_to_proto_cost(x_a, g_v['mu'], g_v['var'])).sum(dim=1).mean(),
            (g_v['posterior'] * self._sample_to_proto_cost(x_v, g_l['mu'], g_l['var'])).sum(dim=1).mean(),
            (g_v['posterior'] * self._sample_to_proto_cost(x_v, g_a['mu'], g_a['var'])).sum(dim=1).mean(),
        ]
        local_proto_loss = torch.stack(local_losses).mean()

        return ot_loss + local_proto_loss