"""Alignment Module 
@Args:      text   ([seq_len, batch_size, dt])
            audio  ([seq_len, batch_size, da])
            vision  ([seq_len, batch_size, dv])
@Returns:   text_  ([batch_size, seq_len, d ])
            audio_ ([batch_size, seq_len, d ])
            vision_ ([batch_size, seq_len, d ])
"""
import torch
import torch.nn.functional as F
from torch import nn

from ...subNets2.transformers_encoder.transformer import TransformerEncoder


class TFRTransformerEncoder(TransformerEncoder):
    """
    Subclass of TransformerEncoder supporting optional positional embeddings,
    allowing compatibility without modifying existing subNets2 code.
    """
    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False, position_embedding=True):
        super().__init__(embed_dim=embed_dim,
                         num_heads=num_heads,
                         layers=layers,
                         attn_dropout=attn_dropout,
                         relu_dropout=relu_dropout,
                         res_dropout=res_dropout,
                         embed_dropout=embed_dropout,
                         attn_mask=attn_mask)
        if not position_embedding:
            self.embed_positions = None


class CM_ATTN(nn.Module):
    def __init__(self, args):
        super(CM_ATTN, self).__init__()
        self.args = args
        self.seq_lens = getattr(args, 'seq_lens', [50, 50, 50])
        dst_feature_dims, nheads = getattr(args, 'dst_feature_dim_nheads', [30, 6])
        self.orig_d_l, self.orig_d_a, self.orig_d_v = getattr(args, 'feature_dims', [768, 5, 20])
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        args.generator_in = (dst_feature_dims * 2, dst_feature_dims * 2, dst_feature_dims * 2)

        self.num_heads = nheads
        self.layers = getattr(args, 'nlevels', 3)
        self.attn_dropout = getattr(args, 'attn_dropout', 0.2)
        self.attn_dropout_a = getattr(args, 'attn_dropout_a', 0.1)
        self.attn_dropout_v = getattr(args, 'attn_dropout_v', 0.0)
        self.relu_dropout = getattr(args, 'relu_dropout', 0.2)
        self.embed_dropout = getattr(args, 'embed_dropout', 0.2)
        self.res_dropout = getattr(args, 'res_dropout', 0.2)
        
        self.text_dropout = getattr(args, 'text_dropout', 0.2)
        self.attn_mask = getattr(args, 'attn_mask', True)

        conv1d_k_l = getattr(args, 'conv1d_kernel_size_l', 1)
        conv1d_k_a = getattr(args, 'conv1d_kernel_size_a', 5)
        conv1d_k_v = getattr(args, 'conv1d_kernel_size_v', 3)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=conv1d_k_l, padding=(conv1d_k_l - 1) // 2, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=conv1d_k_a, padding=(conv1d_k_a - 1) // 2, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=conv1d_k_v, padding=(conv1d_k_v - 1) // 2, bias=False)

        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')
    
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
    
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Intramodal Attentions
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # 4. Self Attentions
        self.trans_l_final = self.get_network(self_type='l_final', layers=3)
        self.trans_a_final = self.get_network(self_type='a_final', layers=3)
        self.trans_v_final = self.get_network(self_type='v_final', layers=3)

    def get_network(self, self_type='l', layers=-1):
        num_temporal_head = getattr(self.args, 'num_temporal_head', None)
        if num_temporal_head is None and isinstance(self.args, dict):
            num_temporal_head = self.args.get('num_temporal_head', 25)
        elif num_temporal_head is None:
            num_temporal_head = 25

        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout, num_heads, position_embedding = self.d_l, self.attn_dropout, self.num_heads, True
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout, num_heads, position_embedding = self.d_a, self.attn_dropout_a, self.num_heads, True
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout, num_heads, position_embedding = self.d_v, self.attn_dropout_v, self.num_heads, True
        elif self_type == 'l_mem':
            embed_dim, attn_dropout, num_heads, position_embedding = self.d_l, self.attn_dropout, self.num_heads, True
        elif self_type == 'a_mem':
            embed_dim, attn_dropout, num_heads, position_embedding = self.d_a, self.attn_dropout, self.num_heads, True
        elif self_type == 'v_mem':
            embed_dim, attn_dropout, num_heads, position_embedding = self.d_v, self.attn_dropout, self.num_heads, True
        elif self_type == 'l_final':
            embed_dim, attn_dropout, num_heads, position_embedding = self.seq_lens[0], self.attn_dropout, num_temporal_head, False
        elif self_type == 'a_final':
            embed_dim, attn_dropout, num_heads, position_embedding = self.seq_lens[1], self.attn_dropout, num_temporal_head, False
        elif self_type == 'v_final':
            embed_dim, attn_dropout, num_heads, position_embedding = self.seq_lens[2], self.attn_dropout, num_temporal_head, False
        else:
            raise ValueError("Unknown network type")
        
        return TFRTransformerEncoder(embed_dim=embed_dim,
                                     num_heads=num_heads,
                                     layers=max(self.layers, layers),
                                     attn_dropout=attn_dropout,
                                     relu_dropout=self.relu_dropout,
                                     res_dropout=self.res_dropout,
                                     embed_dropout=self.embed_dropout,
                                     attn_mask=self.attn_mask,
                                     position_embedding=position_embedding)

    def forward(self, text, audio, vision):
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = vision.transpose(1, 2)

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        # (V,A) --> L
        h_l = self.trans_l_mem(proj_x_l)
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        h_ls = torch.cat([h_l, h_l_with_as, h_l_with_vs], dim=2)
        h_ls_n = self.trans_l_final(h_ls.permute(1, 2, 0)).permute(0, 2, 1)

        # (L,V) --> A
        h_a = self.trans_a_mem(proj_x_a)
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a, h_a_with_ls, h_a_with_vs], dim=2)
        h_as_n = self.trans_a_final(h_as.permute(1, 2, 0)).permute(0, 2, 1)

        # (L,A) --> V
        h_v = self.trans_v_mem(proj_x_v)
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v, h_v_with_ls, h_v_with_as], dim=2)
        h_vs_n = self.trans_v_final(h_vs.permute(1, 2, 0)).permute(0, 2, 1)

        return h_ls.transpose(0, 1), h_as.transpose(0, 1), h_vs.transpose(0, 1), h_ls_n, h_as_n, h_vs_n


MODULE_MAP = {
    'crossmodal_attn': CM_ATTN, 
}


class Alignment(nn.Module):
    def __init__(self, args):
        super(Alignment, self).__init__()
        select_model = MODULE_MAP[getattr(args, 'alignmentModule', 'crossmodal_attn')]
        self.Model = select_model(args)

    def forward(self, text, audio, vision):
        return self.Model(text, audio, vision)
