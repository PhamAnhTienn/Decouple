"""
here is the main backbone for DMD containing feature decoupling and multimodal transformers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .subNets2 import BertTextEncoder
from .subNets2.transformers_encoder.transformer import TransformerEncoder

class DAR(nn.Module):
    def __init__(self, args):
        super(DAR, self).__init__()
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
        self.gate_l = nn.Sequential(
            nn.Linear(self.d_l * 2, self.d_l),
            # nn.Tanh()
            nn.Sigmoid()
        )
        self.gate_v = nn.Sequential(
            nn.Linear(self.d_v * 2, self.d_v),
            # nn.Tanh()
            nn.Sigmoid()
        )
        self.gate_a = nn.Sequential(
            nn.Linear(self.d_a * 2, self.d_a),
            # nn.Tanh()
            nn.Sigmoid()
        )
        self.gate_c_l = nn.Sequential(
            nn.Linear(self.d_l * 2, self.d_l),
            # nn.Tanh()
            nn.Sigmoid()
        )
        self.gate_c_v = nn.Sequential(
            nn.Linear(self.d_v * 2, self.d_v),
            # nn.Tanh()
            nn.Sigmoid()
        )
        self.gate_c_a = nn.Sequential(
            nn.Linear(self.d_a * 2, self.d_a),
            # nn.Tanh()
            nn.Sigmoid()
        )
        combined_dim_low = self.d_a
        combined_dim_high = 2 * self.d_a
        combined_dim = 2 * (self.d_l + self.d_a + self.d_v) + self.d_l * 3
        output_dim = 1

        # # 1. Temporal convolutional layers for initial feature
        # self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        # self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        # self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)
        seq_len_t, seq_len_a, seq_len_v = args.seq_lens
        self.dst_len = seq_len_t
        self.proj_l = nn.Sequential(
            nn.Conv1d(seq_len_t, self.dst_len, kernel_size=1, padding=0, bias=False),
            nn.Linear(self.orig_d_l, self.d_l*2),
            nn.ReLU(),
            nn.Linear(self.d_l*2, self.d_l)
        )
        self.proj_a = nn.Sequential(
            nn.Conv1d(seq_len_a, self.dst_len, kernel_size=1, padding=0, bias=False),
            nn.Linear(self.orig_d_a, self.d_a*2),
            nn.ReLU(),
            nn.Linear(self.d_a*2, self.d_a)
        )
        self.proj_v = nn.Sequential(
            nn.Conv1d(seq_len_v, self.dst_len, kernel_size=1, padding=0, bias=False),
            nn.Linear(self.orig_d_v, self.d_v*2),
            nn.ReLU(),
            nn.Linear(self.d_v*2, self.d_v)
        )

        # 2.1 Modality-specific encoder
        # self.encoder_s_l = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        # self.encoder_s_v = nn.Conv1d(self.d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        # self.encoder_s_a = nn.Conv1d(self.d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        
        self.encoder_s_l = TransformerEncoder(embed_dim=self.d_l, layers=2,num_heads=10)
        self.encoder_s_v = TransformerEncoder(embed_dim=self.d_v, layers=2,num_heads=10)
        self.encoder_s_a = TransformerEncoder(embed_dim=self.d_a, layers=2,num_heads=10)

        # 2.2 Modality-invariant encoder
        # self.encoder_c = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)

        self.encoder_c_l = TransformerEncoder(embed_dim=self.d_l, layers=2, num_heads=10)
        self.encoder_c_v = TransformerEncoder(embed_dim=self.d_l, layers=2, num_heads=10)
        self.encoder_c_a = TransformerEncoder(embed_dim=self.d_l, layers=2, num_heads=10)


        # 3. Decoder for reconstruct three modalities

        # self.decoder_l = nn.Conv1d(self.d_l * 3, self.d_l, kernel_size=1, padding=0, bias=False)
        # self.decoder_v = nn.Conv1d(self.d_v * 3, self.d_v, kernel_size=1, padding=0, bias=False)
        # self.decoder_a = nn.Conv1d(self.d_a * 3, self.d_a, kernel_size=1, padding=0, bias=False)
        self.decoder_l = nn.Sequential(
            nn.Linear(self.d_l*3, self.d_l),
            TransformerEncoder(embed_dim=self.d_l, layers=2,num_heads=10)
                                       )
        self.decoder_v = nn.Sequential(
            nn.Linear(self.d_l*3, self.d_l),
            TransformerEncoder(embed_dim=self.d_l, layers=2,num_heads=10)
                                       )
        self.decoder_a = nn.Sequential(
            nn.Linear(self.d_l*3, self.d_l),
            TransformerEncoder(embed_dim=self.d_l, layers=2,num_heads=10)
                                       )

        # for calculate cosine sim between s_x
        # self.proj_cosine_l = nn.Linear(combined_dim_low * (self.len_l), combined_dim_low)
        # self.proj_cosine_v = nn.Linear(combined_dim_low * (self.len_v), combined_dim_low)
        # self.proj_cosine_a = nn.Linear(combined_dim_low * (self.len_a), combined_dim_low)

        # # for align c_l, c_v, c_a
        # self.align_c_l = nn.Linear(combined_dim_low * (self.len_l), combined_dim_low)
        # self.align_c_v = nn.Linear(combined_dim_low * (self.len_v), combined_dim_low)
        # self.align_c_a = nn.Linear(combined_dim_low * (self.len_a), combined_dim_low)

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
        
        self.recon_sl = self.get_network(self_type='l', layers=2)
        self.recon_sv = self.get_network(self_type='v', layers=2)
        self.recon_sa = self.get_network(self_type='a', layers=2)
        
        self.recon_ca = nn.Sequential(
            nn.Linear(self.d_l*3, self.d_l),
            TransformerEncoder(embed_dim=self.d_l, layers=2,num_heads=10)
                                       )
        self.recon_cv = nn.Sequential(
            nn.Linear(self.d_l*3, self.d_l),
            TransformerEncoder(embed_dim=self.d_l, layers=2, num_heads=10)
                                       )
        self.recon_cl = nn.Sequential(
            nn.Linear(self.d_l*3, self.d_l),
            TransformerEncoder(embed_dim=self.d_l, layers=2, num_heads=10)
                                       )
        self.recon_decoder = nn.Sequential(
            nn.Linear(self.d_l*2, self.d_l),
            TransformerEncoder(embed_dim=self.d_l, layers=2,num_heads=10)
                                       )
        
        # 4. fc layers for homogeneous graph distillation
        self.proj1_l_low = self.get_network(self_type='agg',layers=3)
        # self.proj1_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.proj2_l_low = nn.Linear(combined_dim_low, combined_dim_low * (self.dst_len))
        self.out_layer_l_low = nn.Linear(combined_dim_low * (self.dst_len), output_dim)
        # self.proj1_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj1_v_low = self.get_network(self_type='agg',layers=3)

        self.proj2_v_low = nn.Linear(combined_dim_low, combined_dim_low * (self.dst_len))
        self.out_layer_v_low = nn.Linear(combined_dim_low * (self.dst_len), output_dim)
        # self.proj1_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)
        
        self.proj1_a_low = self.get_network(self_type='agg',layers=3)

        self.proj2_a_low = nn.Linear(combined_dim_low, combined_dim_low * (self.dst_len))
        self.out_layer_a_low = nn.Linear(combined_dim_low * (self.dst_len), output_dim)

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
        self.out_layer = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.ReLU(), 
            nn.Linear(combined_dim, output_dim),
        )
        # self.out_layer = nn.Linear(combined_dim, output_dim)

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
        elif self_type == 'agg':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
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

        text_c,text = text
        audio_c,audio = audio
        video_c,video = video
        if self.use_bert:
            text = self.text_model(text)
            text_c = self.text_model(text_c)
        # x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        # x_a = audio.transpose(1, 2)
        # x_v = video.transpose(1, 2)
        x_l = F.dropout(text, p=self.text_dropout, training=self.training)
        x_a = audio
        x_v = video   
        
        x_l_c = F.dropout(text_c, p=self.text_dropout, training=self.training)
        x_a_c= audio_c
        x_v_c = video_c
  

        
        proj_x_l = self.proj_l(x_l)
        proj_x_a = self.proj_a(x_a)
        proj_x_v = self.proj_v(x_v)
        
        proj_x_l = proj_x_l.transpose(0,1)
        proj_x_a = proj_x_a.transpose(0,1)
        proj_x_v = proj_x_v.transpose(0,1)
        

        proj_x_l_c = self.proj_l(x_l_c)
        proj_x_a_c = self.proj_a(x_a_c)
        proj_x_v_c = self.proj_v(x_v_c)
        
        proj_x_l_c = proj_x_l_c.transpose(0, 1)
        proj_x_a_c = proj_x_a_c.transpose(0, 1)
        proj_x_v_c = proj_x_v_c.transpose(0, 1)
        
        s_l_c = self.encoder_s_l(proj_x_l_c)
        s_v_c = self.encoder_s_v(proj_x_v_c)
        s_a_c = self.encoder_s_a(proj_x_a_c)
        
        c_l_c = self.encoder_c_l(proj_x_l_c)
        c_v_c = self.encoder_c_v(proj_x_v_c)
        c_a_c = self.encoder_c_a(proj_x_a_c)
        




        s_l = self.encoder_s_l(proj_x_l)
        s_v = self.encoder_s_v(proj_x_v)
        s_a = self.encoder_s_a(proj_x_a)
        

        
        # s_l = s_l.transpose(1, 2)
        # s_v = s_v.transpose(1, 2)
        # s_a = s_a.transpose(1, 2)


        c_l = self.encoder_c_l(proj_x_l)
        c_v = self.encoder_c_v(proj_x_v)
        c_a = self.encoder_c_a(proj_x_a)
        
        # proj_x_l = proj_x_l.transpose(1, 2)
        # proj_x_a = proj_x_a.transpose(1, 2)
        # proj_x_v = proj_x_v.transpose(1, 2)
        
        recon_sl = self.recon_sl(s_l)        
        recon_sv = self.recon_sv(s_v)
        recon_sa = self.recon_sa(s_a)
        recon_cl = self.recon_cl(torch.cat([c_l, c_v, c_a], dim=2))
        recon_cv = self.recon_cv(torch.cat([c_l, c_v, c_a], dim=2))
        recon_ca = self.recon_ca(torch.cat([c_l, c_v, c_a], dim=2))
        recon_a_c = self.recon_decoder(torch.cat([recon_ca, recon_sa], dim=2))
        recon_v_c = self.recon_decoder(torch.cat([recon_cv, recon_sv], dim=2))
        recon_l_c = self.recon_decoder(torch.cat([recon_cl, recon_sl], dim=2))
        
        # s_l = s_l.transpose(1, 2)
        # s_v = s_v.transpose(1, 2)
        
        # c_l = c_l.transpose(1, 2)
        # c_v = c_v.transpose(1, 2)
        # c_a = c_a.transpose(1, 2)
        

        c_list = [c_l, c_v, c_a]


        # c_l_sim = torch.mean(c_l, dim=1)
        # c_v_sim = torch.mean(c_v, dim=1)
        # c_a_sim = torch.mean(c_a, dim=1)
        c_l_sim = c_l
        c_v_sim = c_v
        c_a_sim = c_a

        # c_l_sim = self.align_c_l(c_l.contiguous().view(x_l.size(0), -1))
        # c_v_sim = self.align_c_v(c_v.contiguous().view(x_l.size(0), -1))
        # c_a_sim = self.align_c_a(c_a.contiguous().view(x_l.size(0), -1))

        # recon_l = self.decoder_l(torch.cat([s_l, c_list[1],c_list[2]], dim=2))
        # recon_v = self.decoder_v(torch.cat([s_v, c_list[0],c_list[2]], dim=2))
        # recon_a = self.decoder_a(torch.cat([s_a, c_list[0],c_list[1]], dim=2))
        recon_l = self.decoder_l(torch.cat([s_l, c_list[1],c_list[2]], dim=2))
        recon_v = self.decoder_v(torch.cat([s_v, c_list[0],c_list[2]], dim=2))
        recon_a = self.decoder_a(torch.cat([s_a, c_list[0],c_list[1]], dim=2))



        s_l_r = self.encoder_s_l(recon_l)
        s_v_r = self.encoder_s_v(recon_v)
        s_a_r = self.encoder_s_a(recon_a)
        
        
        ori_s_l = s_l
        ori_s_v = s_v
        ori_s_a = s_a
        ori_c_l = c_l
        ori_c_v = c_v
        ori_c_a = c_a
        
        w_s_l = self.gate_l(torch.cat([recon_sl, s_l], dim=2))
        w_s_v = self.gate_v(torch.cat([recon_sv, s_v], dim=2)) 
        w_s_a = self.gate_a(torch.cat([recon_sa, s_a], dim=2)) 
        w_c_l = self.gate_c_l(torch.cat([recon_cl, c_l], dim=2))
        w_c_v = self.gate_c_v(torch.cat([recon_cv, c_v], dim=2))
        w_c_a = self.gate_c_a(torch.cat([recon_ca, c_a], dim=2)) 
        
        s_l = s_l + w_s_l * recon_sl
        s_v = s_v + w_s_v * recon_sv
        s_a = s_a + w_s_a * recon_sa
        c_l = c_l + w_c_l * recon_cl
        c_v = c_v + w_c_v * recon_cv
        c_a = c_a + w_c_a * recon_ca
        
        # s_l = self.gate_l(torch.cat([recon_sl, s_l], dim=2)) * s_l + recon_sl
        # s_v = self.gate_v(torch.cat([recon_sv, s_v], dim=2)) * s_v + recon_sv
        # s_a = self.gate_a(torch.cat([recon_sa, s_a], dim=2)) * s_a + recon_sa
        # c_l = self.gate_c_l(torch.cat([recon_cl, c_l], dim=2)) * c_l + recon_cl
        # c_v = self.gate_c_v(torch.cat([recon_cv, c_v], dim=2)) * c_v + recon_cv
        # c_a = self.gate_c_a(torch.cat([recon_ca, c_a], dim=2)) * c_a + recon_ca
        
        # s_l =  recon_sl
        # s_v =  recon_sv
        # s_a =  recon_sa
        # c_l =  recon_cl
        # c_v =  recon_cv
        # c_a =  recon_ca

        # ori_s_l = s_l
        # ori_s_v = s_v
        # ori_s_a = s_a
        # ori_c_l = c_l
        # ori_c_v = c_v
        # ori_c_a = c_a

        hs_l_low = c_l_c.transpose(0, 1).contiguous().view(x_l.size(0), -1)
        # repr_l_low = self.proj1_l_low(hs_l_low)
        # hs_proj_l_low = self.proj2_l_low(
        #     F.dropout(F.relu(repr_l_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_low = self.proj1_l_low(c_l_c).transpose(0, 1).contiguous().view(x_l.size(0), -1)
        hs_proj_l_low += hs_l_low
        logits_l_low = self.out_layer_l_low(hs_proj_l_low)

        hs_v_low = c_v_c.transpose(0, 1).contiguous().view(x_v.size(0), -1)
        # repr_l_low = self.proj1_l_low(hs_l_low)
        # hs_proj_l_low = self.proj2_l_low(
        #     F.dropout(F.relu(repr_l_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_low = self.proj1_v_low(c_v_c).transpose(0, 1).contiguous().view(x_v.size(0), -1)
        hs_proj_v_low += hs_v_low
        logits_v_low = self.out_layer_v_low(hs_proj_v_low)

        hs_a_low = c_a_c.transpose(0, 1).contiguous().view(x_a.size(0), -1)
        # repr_l_low = self.proj1_l_low(hs_l_low)
        # hs_proj_l_low = self.proj2_l_low(
        #     F.dropout(F.relu(repr_l_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_a_low = self.proj1_a_low(c_a_c).transpose(0, 1).contiguous().view(x_a.size(0), -1)
        hs_proj_a_low += hs_a_low
        logits_a_low = self.out_layer_a_low(hs_proj_a_low)

        c_l_att = self.self_attentions_c_l(c_l)
        if type(c_l_att) == tuple:
            c_l_att = c_l_att[0]
        c_l_att = c_l_att[-1]
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
        logits_c = self.out_layer_c(c_fusion)

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

        # last_h_l = torch.sigmoid(last_h_l)
        # last_h_v = torch.sigmoid(last_h_v)
        # last_h_a = torch.sigmoid(last_h_a)
        # c_fusion = torch.sigmoid(c_fusion)

        last_hs = torch.cat([last_h_l, last_h_v, last_h_a, c_fusion], dim=1)
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)

        res = {
            'logits_l_homo': logits_l_low,
            'logits_v_homo': logits_v_low,
            'logits_a_homo': logits_a_low,
            # 'repr_l_homo': repr_l_low,
            # 'repr_v_homo': repr_v_low,
            # 'repr_a_homo': repr_a_low,
            'origin_l': proj_x_l,
            'origin_v': proj_x_v,
            'origin_a': proj_x_a,
            's_l': ori_s_l,
            's_v': ori_s_v,
            's_a': ori_s_a,
            'c_l': ori_c_l,
            'c_v': ori_c_v,
            'c_a': ori_c_a,
            's_l_r': s_l_r,
            's_v_r': s_v_r,
            's_a_r': s_a_r,
            'recon_l': recon_l,
            'recon_v': recon_v,
            'recon_a': recon_a,
            'c_l_sim': c_l_sim,
            'c_v_sim': c_v_sim,
            'c_a_sim': c_a_sim,
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
            'recon_a_c': recon_a_c,
            'recon_v_c': recon_v_c,
            'recon_l_c': recon_l_c,
            'proj_l_c':proj_x_l_c,
            'proj_v_c':proj_x_v_c,
            'proj_a_c':proj_x_a_c,
            's_l_c':s_l_c,
            's_v_c':s_v_c,
            's_a_c':s_a_c,
            'c_l_c':c_l_c,
            'c_v_c':c_v_c,
            'c_a_c':c_a_c,
            'recon_sl':recon_sl,
            'recon_sv':recon_sv,
            'recon_sa':recon_sa,
            'recon_cl':recon_cl,
            'recon_cv':recon_cv,
            'recon_ca':recon_ca
        }
        return res