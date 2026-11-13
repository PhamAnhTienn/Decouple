import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .subNets2 import BertTextEncoder


__all__ = ['MIDAS']


# ──────────────────────────────────────────────────────────────────────
# Utility modules (Projector, Classifier, losses)
# ──────────────────────────────────────────────────────────────────────

class Projector(nn.Module):
    """MLP projector with GELU activations, followed by layer normalization."""

    def __init__(self, input_size, hidden_size, output_size, drop_out=0.1):
        super(Projector, self).__init__()
        module_list = []
        for i, h in enumerate(hidden_size):
            if i == 0:
                module_list.append(nn.Linear(input_size, h))
            else:
                module_list.append(nn.Linear(hidden_size[i - 1], h))
            module_list.append(nn.GELU())
        module_list.append(nn.Linear(hidden_size[-1], output_size))

        self.MLP = nn.Sequential(*module_list)
        self.layernorm = nn.LayerNorm(output_size)
        self.drop = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.drop(x)
        x = self.MLP(x)
        x = self.layernorm(x)
        return x


class Classifier(nn.Module):
    """MLP classification / regression head with GELU activations."""

    def __init__(self, input_size, hidden_size, output_size, drop_out=0.0):
        super(Classifier, self).__init__()
        module_list = []
        for i, h in enumerate(hidden_size):
            if i == 0:
                module_list.append(nn.Linear(input_size, h))
            else:
                module_list.append(nn.Linear(hidden_size[i - 1], h))
            module_list.append(nn.GELU())
        module_list.append(nn.Linear(hidden_size[-1], output_size))

        self.MLP = nn.Sequential(*module_list)

    def forward(self, x):
        x = self.MLP(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, input1, input2):
        input = torch.cat((input1, input2), dim=-1)
        output = F.relu(self.fc1(input))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output


class JSDLoss(nn.Module):
    """Jensen-Shannon divergence based mutual information estimator."""

    def __init__(self):
        super(JSDLoss, self).__init__()

    def forward(self, pos, neg):
        pos = -F.softplus(-pos)
        neg = F.softplus(neg)
        jsd_loss = (neg - pos).mean()
        return jsd_loss


class KLLoss(nn.Module):
    """KL divergence between a diagonal Gaussian and the standard normal prior."""

    def __init__(self, d_model):
        super(KLLoss, self).__init__()
        self.d_model = d_model

    def forward(self, mu, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=-1).reshape(-1)
        kl_loss = kl_loss.mean()
        return kl_loss / self.d_model


# ──────────────────────────────────────────────────────────────────────
# Position encoding & Transformer encoder for audio / vision branches
# ──────────────────────────────────────────────────────────────────────

class PositionEncodingTraining(nn.Module):
    """Construct the CLS token, position and patch embeddings.
    Uses learnable position embeddings that are dynamically sized based on input.
    """

    def __init__(self, fea_size, tf_hidden_dim, max_seq_len=512, drop_out=0.0):
        super().__init__()
        self.cls_token = nn.Parameter(torch.ones(1, 1, tf_hidden_dim))
        self.proj = nn.Linear(fea_size, tf_hidden_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_seq_len + 1, tf_hidden_dim))
        self.dropout = nn.Dropout(drop_out)

    def forward(self, embeddings):
        batch_size, seq_len, _ = embeddings.shape
        embeddings = self.proj(embeddings)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)  # (B, seq_len+1, D)
        embeddings = embeddings + self.position_embeddings[:, :seq_len + 1, :]
        embeddings = self.dropout(embeddings)
        return embeddings


class TfEncoder(nn.Module):
    """Vanilla Transformer encoder with positional encoding."""

    def __init__(self, fea_size, d_model, nhead, dim_feedforward, num_layers,
                 max_seq_len=512, dropout=0.2, activation='gelu'):
        super(TfEncoder, self).__init__()
        self.src_mask = None
        self.pos_encoder = PositionEncodingTraining(
            fea_size=fea_size,
            tf_hidden_dim=d_model,
            max_seq_len=max_seq_len,
            drop_out=dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation=activation,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, has_mask=False, src_key_padding_mask=None):
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        output = self.transformer_encoder(src, mask=self.src_mask,
                                          src_key_padding_mask=src_key_padding_mask)
        return output.transpose(0, 1)


# ──────────────────────────────────────────────────────────────────────
# Audio / Vision unimodal encoders
# ──────────────────────────────────────────────────────────────────────

class AudioEncoder(nn.Module):
    def __init__(self, fea_size, encoder_fea_dim, proj_fea_dim,
                 nhead=8, num_layers=2, drop_out=0.0, max_seq_len=512):
        super(AudioEncoder, self).__init__()
        dim_feedforward = encoder_fea_dim * 2
        self.with_projector = False
        if encoder_fea_dim != proj_fea_dim:
            self.with_projector = True
            self.projector = Projector(fea_size, [proj_fea_dim * 2], proj_fea_dim, drop_out)
        self.encoder = TfEncoder(
            fea_size=fea_size,
            d_model=encoder_fea_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            dropout=drop_out,
        )
        self.layernorm = nn.LayerNorm(encoder_fea_dim)

    def forward(self, audio, key_padding_mask):
        # Pad mask for CLS token: prepend False (CLS is never masked)
        cls_pad = torch.zeros(key_padding_mask.size(0), 1,
                              dtype=torch.bool, device=key_padding_mask.device)
        key_padding_mask_ext = torch.cat([cls_pad, key_padding_mask], dim=1)

        x = self.encoder(audio, has_mask=False, src_key_padding_mask=key_padding_mask_ext)
        x = self.layernorm(x)
        if self.with_projector:
            x = self.projector(x)
        mask_expanded = (~key_padding_mask_ext).unsqueeze(-1).expand(x.size()).float()
        x = x * mask_expanded
        x_avc = x.sum(dim=1) / mask_expanded.sum(dim=1)
        return x, x_avc


class VisionEncoder(nn.Module):
    def __init__(self, fea_size, encoder_fea_dim, proj_fea_dim,
                 nhead=8, num_layers=2, drop_out=0.0, max_seq_len=512):
        super(VisionEncoder, self).__init__()
        dim_feedforward = encoder_fea_dim * 2
        self.with_projector = False
        if encoder_fea_dim != proj_fea_dim:
            self.with_projector = True
            self.projector = Projector(fea_size, [proj_fea_dim * 2], proj_fea_dim, drop_out)
        self.encoder = TfEncoder(
            fea_size=fea_size,
            d_model=encoder_fea_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            dropout=drop_out,
        )
        self.layernorm = nn.LayerNorm(encoder_fea_dim)

    def forward(self, vision, key_padding_mask):
        # Pad mask for CLS token: prepend False (CLS is never masked)
        cls_pad = torch.zeros(key_padding_mask.size(0), 1,
                              dtype=torch.bool, device=key_padding_mask.device)
        key_padding_mask_ext = torch.cat([cls_pad, key_padding_mask], dim=1)

        x = self.encoder(vision, has_mask=False, src_key_padding_mask=key_padding_mask_ext)
        x = self.layernorm(x)
        if self.with_projector:
            x = self.projector(x)
        mask_expanded = (~key_padding_mask_ext).unsqueeze(-1).expand(x.size()).float()
        x = x * mask_expanded
        x_avc = x.sum(dim=1) / mask_expanded.sum(dim=1)
        return x, x_avc


# ──────────────────────────────────────────────────────────────────────
# Uncertainty-aware fusion
# ──────────────────────────────────────────────────────────────────────

class GateMLP(nn.Module):
    """Small MLP mapping scalar uncertainty -> (0,1) weight."""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, u):
        if u.dim() == 1:
            u = u.unsqueeze(-1)
        out = self.net(u)
        return torch.sigmoid(out).squeeze(-1)


class SimpleMultiHeadAttention(nn.Module):
    """Multi-head attention with optional key modulation by token weights."""

    def __init__(self, embed_dim, num_heads=4, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, token_weights=None, attn_mask=None, key_padding_mask=None):
        B, S, D = x.shape
        H = self.num_heads
        d_h = self.head_dim

        Q = self.q_proj(x).view(B, S, H, d_h).transpose(1, 2)
        K = self.k_proj(x).view(B, S, H, d_h).transpose(1, 2)
        V = self.v_proj(x).view(B, S, H, d_h).transpose(1, 2)

        if token_weights is not None:
            eps = 1e-6
            w = token_weights.clamp(min=eps)
            w_sqrt = torch.sqrt(w).unsqueeze(1).unsqueeze(-1)
            K = K * w_sqrt

        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_h)

        if attn_mask is not None:
            logits = logits + attn_mask
        if key_padding_mask is not None:
            mask = key_padding_mask.view(B, 1, 1, S).expand(-1, H, S, S)
            logits = logits.masked_fill(mask, float("-inf"))

        attn = torch.softmax(logits, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.layernorm1(x + out)
        out = self.layernorm2(self.out_proj(out) + out)
        return out, attn


class UncertaintyAwareFusion(nn.Module):
    """Uncertainty-aware fusion module."""

    def __init__(self, embed_dim, gate_hidden=32, tau=1.0,
                 mapping='soft_inv', uncertainty='var',
                 fusion='gated', num_heads=8):
        super().__init__()
        assert mapping in ('soft_inv', 'sigmoid', 'mlp')
        assert uncertainty in ('var', 'entropy')
        assert fusion in ('gated', 'attn')

        self.embed_dim = embed_dim
        self.mapping = mapping
        self.uncertainty = uncertainty
        self.fusion = fusion
        self.tau = tau

        if mapping == 'mlp':
            self.gate_mlps = nn.ModuleList([GateMLP(hidden=gate_hidden) for _ in range(6)])
        else:
            self.gate_mlps = None

        if fusion == 'attn':
            self.attn = SimpleMultiHeadAttention(embed_dim, num_heads=num_heads)
        else:
            self.attn = None

    @staticmethod
    def _var_uncertainty_from_logvar(logvar):
        var = torch.exp(logvar)
        u = var.mean(dim=-1)
        return u

    @staticmethod
    def _entropy_uncertainty_from_logvar(logvar):
        var = torch.exp(logvar)
        u = 0.5 * torch.log(2 * math.pi * math.e * var).sum(dim=-1)
        u = u / logvar.size(-1)
        return u

    def compute_uncertainty(self, logvar):
        if self.uncertainty == 'var':
            return self._var_uncertainty_from_logvar(logvar)
        else:
            return self._entropy_uncertainty_from_logvar(logvar)

    def map_uncertainties_to_weights(self, us):
        U = torch.stack(us, dim=1)
        if self.mapping == 'soft_inv':
            z = -self.tau * U
            W = torch.softmax(z, dim=1)
        elif self.mapping == 'sigmoid':
            W = torch.sigmoid(-self.tau * (U - U.mean(dim=1, keepdim=True)))
        else:
            Ws = []
            for i in range(U.size(1)):
                mlp = self.gate_mlps[i]
                wi = mlp(U[:, i])
                Ws.append(wi)
            W = torch.stack(Ws, dim=1)
        return W

    def forward(self, tokens, logvar_list, mask=None):
        us = [self.compute_uncertainty(logvar_list[i]) for i in range(6)]
        W = self.map_uncertainties_to_weights(us)
        B = W.size(0)
        tokens_stack = torch.stack(tokens, dim=1)

        if self.fusion == 'gated':
            Wb = W.unsqueeze(-1)
            tokens_gated = tokens_stack * Wb
            fused = tokens_gated.view(B, -1)
            out = dict(fused=fused, tokens_out=tokens_gated, weights=W,
                       uncertainties=torch.stack(us, dim=1))
            return out
        else:
            attn_out, attn_weights = self.attn(tokens_stack, token_weights=W,
                                               key_padding_mask=mask)
            fused = attn_out.view(B, -1)
            out = dict(fused=fused, tokens_out=attn_out, weights=W,
                       uncertainties=torch.stack(us, dim=1), attn=attn_weights)
            return out


# ──────────────────────────────────────────────────────────────────────
# Variational encoder
# ──────────────────────────────────────────────────────────────────────

class MI_Encoder(nn.Module):
    """Variational encoder producing the mean and log-variance of a Gaussian latent."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(MI_Encoder, self).__init__()
        self.encoder_mu = Projector(input_dim, hidden_dim, output_dim, dropout)
        self.encoder_logvar = Projector(input_dim, hidden_dim, output_dim, dropout)

    def forward(self, x):
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        return mu, logvar


# ──────────────────────────────────────────────────────────────────────
# Main MIDAS model
# ──────────────────────────────────────────────────────────────────────

class MIDAS(nn.Module):
    def __init__(self, args):
        super(MIDAS, self).__init__()
        self.args = args

        # ── Dimensions from args ──────────────────────────────────────
        text_fea_dim = args.feature_dims[0]   # 768 for BERT
        audio_fea_dim = args.feature_dims[1]
        vision_fea_dim = args.feature_dims[2]

        d_model = getattr(args, 'd_model', 128)
        encoder_fea_dim = getattr(args, 'encoder_fea_dim', 128)
        drop_out = getattr(args, 'dropout', 0.0)
        t_drop_out = getattr(args, 't_drop_out', 0.1)
        a_drop_out = getattr(args, 'a_drop_out', 0.0)
        v_drop_out = getattr(args, 'v_drop_out', 0.0)
        audio_nhead = getattr(args, 'audio_nhead', 8)
        vision_nhead = getattr(args, 'vision_nhead', 8)
        audio_tf_num_layers = getattr(args, 'audio_tf_num_layers', 2)
        vision_tf_num_layers = getattr(args, 'vision_tf_num_layers', 2)

        # Sequence lengths for position encoding buffer
        max_seq_len = getattr(args, 'max_seq_len', 512)

        # ── Modality feature extractors ───────────────────────────────
        self.text_encoder = BertTextEncoder(
            use_finetune=getattr(args, 'use_finetune', True),
            transformers=getattr(args, 'transformers', 'bert'),
            pretrained=getattr(args, 'pretrained', 'bert-base-uncased'),
        )
        self.with_text_projector = (text_fea_dim != d_model)
        if self.with_text_projector:
            self.text_projector = Projector(text_fea_dim, [d_model * 2], d_model, t_drop_out)

        self.audio_encoder = AudioEncoder(
            fea_size=audio_fea_dim,
            encoder_fea_dim=encoder_fea_dim,
            proj_fea_dim=d_model,
            nhead=audio_nhead,
            num_layers=audio_tf_num_layers,
            drop_out=a_drop_out,
            max_seq_len=max_seq_len,
        )
        self.vision_encoder = VisionEncoder(
            fea_size=vision_fea_dim,
            encoder_fea_dim=encoder_fea_dim,
            proj_fea_dim=d_model,
            nhead=vision_nhead,
            num_layers=vision_tf_num_layers,
            drop_out=v_drop_out,
            max_seq_len=max_seq_len,
        )

        # ── Feature disentanglement (shared / exclusive) ─────────────
        proj_hidden = [int(d_model * 2), int(d_model)]
        self.encoder_s = MI_Encoder(d_model, proj_hidden, int(d_model / 2), drop_out)
        self.encoder_t = MI_Encoder(d_model, proj_hidden, int(d_model / 2), drop_out)
        self.encoder_a = MI_Encoder(d_model, proj_hidden, int(d_model / 2), drop_out)
        self.encoder_v = MI_Encoder(d_model, proj_hidden, int(d_model / 2), drop_out)

        # ── Unimodal sentiment classifiers on shared latents ──────────
        self.classifier_t = nn.Linear(int(d_model / 2), 2)
        self.classifier_a = nn.Linear(int(d_model / 2), 2)
        self.classifier_v = nn.Linear(int(d_model / 2), 2)

        # ── Reconstruction projectors ─────────────────────────────────
        cls_hidden = [int(d_model / 4)]
        self.linear_t = Projector(d_model, cls_hidden, d_model, drop_out)
        self.linear_a = Projector(d_model, cls_hidden, d_model, drop_out)
        self.linear_v = Projector(d_model, cls_hidden, d_model, drop_out)

        # ── Uncertainty-aware fusion ──────────────────────────────────
        fusion_type = getattr(args, 'midas_fusion', 'attn')
        fusion_mapping = getattr(args, 'midas_mapping', 'sigmoid')
        fusion_uncertainty = getattr(args, 'midas_uncertainty', 'entropy')
        fusion_tau = getattr(args, 'midas_tau', 10.0)
        fusion_nhead = getattr(args, 'midas_fusion_nhead', 8)

        self.uncertain_fusion = UncertaintyAwareFusion(
            embed_dim=int(d_model / 2),
            mapping=fusion_mapping,
            uncertainty=fusion_uncertainty,
            fusion=fusion_type,
            num_heads=fusion_nhead,
            tau=fusion_tau,
        )

        # ── Prediction head ──────────────────────────────────────────
        classifier_hidden = [int(d_model), int(d_model / 2)]
        output_size = getattr(args, 'num_classes', 1) if getattr(args, 'train_mode', 'regression') == 'classification' else 1
        self.TVA_decoder = Classifier(int(d_model * 3), classifier_hidden, output_size, 0.0)

        # ── Loss functions (kept inside model for trainer access) ─────
        self.criterion = nn.MSELoss()
        self.kl_loss = KLLoss(d_model)
        self.discri = Discriminator(int(d_model / 2))
        self.jsd_criterion = JSDLoss()
        self.cross_entropy = nn.CrossEntropyLoss()

        # Dataset name for binary label
        self.dataset_name = getattr(args, 'dataset_name', 'mosi')

    # ── helpers ───────────────────────────────────────────────────────

    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        if self.training:
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def binary_label(self, label):
        if self.dataset_name == 'sims':
            return (label > 0).long()
        else:
            return (label >= 0).long()

    def _encode_text(self, text):
        input_mask = text[:, 1, :].float()  # (B, seq_len)
        x = self.text_encoder(text)  # (B, seq_len, 768) from subNets2
        if self.with_text_projector:
            x = self.text_projector(x)  # (B, seq_len, d_model)
        mask_expanded = input_mask.unsqueeze(-1).expand(x.size())
        x = x * mask_expanded
        x_mean = x.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        return x, x_mean

    def _make_key_padding_mask(self, lengths, max_len, device):
        """Create a bool mask: True at padding positions."""
        arange = torch.arange(max_len, device=device).unsqueeze(0)
        lengths_t = torch.as_tensor(lengths, device=device).unsqueeze(1)
        return arange >= lengths_t  # True = pad

    # ── forward ───────────────────────────────────────────────────────

    def forward(self, text, audio, video,
                text_m=None, audio_m=None, vision_m=None,
                audio_lengths=None, vision_lengths=None,
                labels=None):
        """
        Args:
            text, audio, video:   complete modality tensors
            text_m, audio_m, vision_m: (optional) missing-rate modality tensors
            audio_lengths, vision_lengths: actual sequence lengths for padding masks
            labels: ground-truth sentiment label (needed to compute losses inside model)

        Returns:
            dict with keys expected by the MIDAS trainer.
        """
        device = text.device

        # # If no missing variants supplied, treat complete as missing-complete
        # if text_m is None:
        #     text_m = text
        # if audio_m is None:
        #     audio_m = audio
        # if vision_m is None:
        #     vision_m = video

        # ── Build key-padding masks ──────────────────────────────────
        if audio_lengths is not None:
            key_padding_mask_A = self._make_key_padding_mask(
                audio_lengths, audio.size(1), device)
        else:
            key_padding_mask_A = torch.zeros(audio.size(0), audio.size(1),
                                             dtype=torch.bool, device=device)

        if vision_lengths is not None:
            key_padding_mask_V = self._make_key_padding_mask(
                vision_lengths, video.size(1), device)
        else:
            key_padding_mask_V = torch.zeros(video.size(0), video.size(1),
                                             dtype=torch.bool, device=device)

        # ── Modality encoding (missing) ──────────────────────────────
        t_embed_all_m, t_embed_m = self._encode_text(text_m)
        a_embed_all_m, a_embed_m = self.audio_encoder(audio_m, key_padding_mask=key_padding_mask_A)
        v_embed_all_m, v_embed_m = self.vision_encoder(vision_m, key_padding_mask=key_padding_mask_V)

        # ── Modality encoding (complete) ─────────────────────────────
        t_embed_all, t_embed = self._encode_text(text)
        a_embed_all, a_embed = self.audio_encoder(audio, key_padding_mask=key_padding_mask_A)
        v_embed_all, v_embed = self.vision_encoder(video, key_padding_mask=key_padding_mask_V)

        # ── Feature disentanglement ──────────────────────────────────
        mu_ts, logvar_ts = self.encoder_s(t_embed_m)
        z_ts = self.reparameterization(mu_ts, logvar_ts)
        mu_as, logvar_as = self.encoder_s(a_embed_m)
        z_as = self.reparameterization(mu_as, logvar_as)
        mu_vs, logvar_vs = self.encoder_s(v_embed_m)
        z_vs = self.reparameterization(mu_vs, logvar_vs)

        mu_te, logvar_te = self.encoder_t(t_embed_m)
        z_te = self.reparameterization(mu_te, logvar_te)
        mu_ae, logvar_ae = self.encoder_a(a_embed_m)
        z_ae = self.reparameterization(mu_ae, logvar_ae)
        mu_ve, logvar_ve = self.encoder_v(v_embed_m)
        z_ve = self.reparameterization(mu_ve, logvar_ve)

        # ── Reconstruction ───────────────────────────────────────────
        recon_t = self.linear_t(torch.cat((z_te, z_ts), dim=-1))
        recon_a = self.linear_a(torch.cat((z_ae, z_as), dim=-1))
        recon_v = self.linear_v(torch.cat((z_ve, z_vs), dim=-1))
        recon_loss_t = self.criterion(recon_t, t_embed)
        recon_loss_a = self.criterion(recon_a, a_embed)
        recon_loss_v = self.criterion(recon_v, v_embed)
        recon_loss = (recon_loss_t + recon_loss_a + recon_loss_v) / 3

        # ── KL divergence loss ───────────────────────────────────────
        kld_loss_t = self.kl_loss(mu_ts, logvar_ts) + self.kl_loss(mu_te, logvar_te)
        kld_loss_a = self.kl_loss(mu_as, logvar_as) + self.kl_loss(mu_ae, logvar_ae)
        kld_loss_v = self.kl_loss(mu_vs, logvar_vs) + self.kl_loss(mu_ve, logvar_ve)
        diff_loss = (kld_loss_t + kld_loss_a + kld_loss_v) / 6

        # ── Unimodal prediction loss ─────────────────────────────────
        logits_t = self.classifier_t(z_ts)
        logits_a = self.classifier_a(z_as)
        logits_v = self.classifier_v(z_vs)

        if labels is not None:
            label_cls = self.binary_label(labels.view(-1))
            pred_loss_t = self.cross_entropy(logits_t, label_cls)
            pred_loss_a = self.cross_entropy(logits_a, label_cls)
            pred_loss_v = self.cross_entropy(logits_v, label_cls)
            pred_loss = (pred_loss_t + pred_loss_a + pred_loss_v) / 3
        else:
            pred_loss = torch.tensor(0.0, device=device)

        # ── JSD similarity loss ──────────────────────────────────────
        pos1 = self.discri(z_ts, z_as)
        pos2 = self.discri(z_as, z_vs)
        pos3 = self.discri(z_vs, z_ts)
        neg1 = self.discri(z_ts, z_te)
        neg2 = self.discri(z_as, z_ae)
        neg3 = self.discri(z_vs, z_ve)
        sim_loss1 = self.jsd_criterion(pos1, neg1) + self.jsd_criterion(pos1, neg2)
        sim_loss2 = self.jsd_criterion(pos2, neg2) + self.jsd_criterion(pos2, neg3)
        sim_loss3 = self.jsd_criterion(pos3, neg3) + self.jsd_criterion(pos3, neg1)
        sim_loss = (sim_loss1 + sim_loss2 + sim_loss3) / 3

        # ── Uncertainty-aware fusion ─────────────────────────────────
        tokens = [z_ts, z_as, z_vs, z_te, z_ae, z_ve]
        logvar_list = [logvar_ts, logvar_as, logvar_vs, logvar_te, logvar_ae, logvar_ve]
        out = self.uncertain_fusion(tokens, logvar_list)
        pred_embed = out['fused']

        # ── Prediction ───────────────────────────────────────────────
        pred = self.TVA_decoder(pred_embed).squeeze(-1)

        if labels is not None:
            task_loss = self.criterion(pred, labels.view(-1))
        else:
            task_loss = torch.tensor(0.0, device=device)

        return {
            'M': pred.unsqueeze(-1),           # (B, 1) — same shape as other models
            'task_loss': task_loss,
            'diff_loss': diff_loss,
            'pred_loss': pred_loss,
            'sim_loss': sim_loss,
            'recon_loss': recon_loss,
            # representations for analysis
            'z_ts': z_ts, 'z_as': z_as, 'z_vs': z_vs,
            'z_te': z_te, 'z_ae': z_ae, 'z_ve': z_ve,
            't_embed': t_embed, 'a_embed': a_embed, 'v_embed': v_embed,
        }