import torch
from torch import nn
from .subNets2 import BertTextEncoder
from .lnln_layers import (
    Transformer, CrossTransformer,
    HhyperLearningEncoder, GradientReversalLayer
)
from einops import rearrange, repeat
from types import SimpleNamespace


def _ns_to_dict(obj):
    """Recursively convert SimpleNamespace to dict."""
    if isinstance(obj, SimpleNamespace):
        return {k: _ns_to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, list):
        return [_ns_to_dict(i) for i in obj]
    return obj


class LNLN(nn.Module):
    def __init__(self, args):
        super(LNLN, self).__init__()

        # Reconstruct the nested dict config that original LNLN expects:
        #   args['model']['feature_extractor'][...] etc.
        # The JSON config stores this under 'lnln_model' key.
        if hasattr(args, 'lnln_model'):
            model_cfg = _ns_to_dict(args.lnln_model)
        else:
            raise ValueError(
                "LNLN model requires 'lnln_model' nested config in the JSON config file. "
                "See config/lnln_mosi_config.json for reference."
            )

        cfg = {'model': model_cfg}
        fe = cfg['model']['feature_extractor']

        self.h_hyper = nn.Parameter(torch.ones(1, fe['token_length'][0], 128))
        self.h_p = nn.Parameter(torch.ones(1, fe['token_length'][0], 128))

        bert_pretrained = fe.get(
            'bert_pretrained',
            getattr(args, 'pretrained', 'bert-base-uncased')
        )
        self.bertmodel = BertTextEncoder(
            use_finetune=getattr(args, 'use_finetune', True),
            transformers=getattr(args, 'transformers', 'bert'),
            pretrained=bert_pretrained
        )

        # --- Feature extractors: proj_l (language), proj_v (video), proj_a (audio) ---
        # LNLN config dim order: [Language, Video, Audio]
        self.proj_l = nn.Sequential(
            nn.Linear(fe['input_dims'][0], fe['hidden_dims'][0]),
            Transformer(
                num_frames=fe['input_length'][0],
                save_hidden=False,
                token_len=fe['token_length'][0],
                dim=fe['hidden_dims'][0],
                depth=fe['depth'],
                heads=fe['heads'],
                mlp_dim=fe['hidden_dims'][0])
        )

        self.proj_a = nn.Sequential(
            nn.Linear(fe['input_dims'][2], fe['hidden_dims'][2]),
            Transformer(
                num_frames=fe['input_length'][2],
                save_hidden=False,
                token_len=fe['token_length'][2],
                dim=fe['hidden_dims'][2],
                depth=fe['depth'],
                heads=fe['heads'],
                mlp_dim=fe['hidden_dims'][2])
        )

        self.proj_v = nn.Sequential(
            nn.Linear(fe['input_dims'][1], fe['hidden_dims'][1]),
            Transformer(
                num_frames=fe['input_length'][1],
                save_hidden=False,
                token_len=fe['token_length'][1],
                dim=fe['hidden_dims'][1],
                depth=fe['depth'],
                heads=fe['heads'],
                mlp_dim=fe['hidden_dims'][1])
        )

        # --- Dominant Modality Completion (DMC) ---
        dmc = cfg['model']['dmc']

        self.proxy_dominate_modality_generator = Transformer(
            num_frames=dmc['proxy_dominant_feature_generator']['input_length'],
            save_hidden=False,
            token_len=dmc['proxy_dominant_feature_generator']['token_length'],
            dim=dmc['proxy_dominant_feature_generator']['input_dim'],
            depth=dmc['proxy_dominant_feature_generator']['depth'],
            heads=dmc['proxy_dominant_feature_generator']['heads'],
            mlp_dim=dmc['proxy_dominant_feature_generator']['hidden_dim'])

        self.GRL = GradientReversalLayer(alpha=1.0)

        self.effective_discriminator = nn.Sequential(
            nn.Linear(dmc['effectiveness_discriminator']['input_dim'],
                      dmc['effectiveness_discriminator']['hidden_dim']),
            nn.LeakyReLU(0.1),
            nn.Linear(dmc['effectiveness_discriminator']['hidden_dim'],
                      dmc['effectiveness_discriminator']['out_dim']),
        )

        self.completeness_check = nn.ModuleList([
            Transformer(
                num_frames=dmc['completeness_check']['input_length'],
                save_hidden=False,
                token_len=dmc['completeness_check']['token_length'],
                dim=dmc['completeness_check']['input_dim'],
                depth=dmc['completeness_check']['depth'],
                heads=dmc['completeness_check']['heads'],
                mlp_dim=dmc['completeness_check']['hidden_dim']),

            nn.Sequential(
                nn.Linear(dmc['completeness_check']['hidden_dim'],
                          int(dmc['completeness_check']['hidden_dim'] / 2)),
                nn.LeakyReLU(0.1),
                nn.Linear(int(dmc['completeness_check']['hidden_dim'] / 2), 1),
                nn.Sigmoid()),
        ])

        # --- Reconstructor ---
        rec = cfg['model']['reconstructor']
        self.reconstructor = nn.ModuleList([
            Transformer(
                num_frames=rec['input_length'],
                save_hidden=False,
                token_len=None,
                dim=rec['input_dim'],
                depth=rec['depth'],
                heads=rec['heads'],
                mlp_dim=rec['hidden_dim']) for _ in range(3)
        ])

        # --- Dominant Modality-based Multimodal Learning (DMML) ---
        dmml = cfg['model']['dmml']
        self.dmml = nn.ModuleList([
            Transformer(
                num_frames=dmml['language_encoder']['input_length'],
                save_hidden=True,
                token_len=None,
                dim=dmml['language_encoder']['input_dim'],
                depth=dmml['language_encoder']['depth'],
                heads=dmml['language_encoder']['heads'],
                mlp_dim=dmml['language_encoder']['hidden_dim']),

            HhyperLearningEncoder(
                dim=dmml['hyper_modality_learning']['input_dim'],
                dim_head=int(dmml['hyper_modality_learning']['input_dim'] /
                            dmml['hyper_modality_learning']['heads']),
                depth=dmml['hyper_modality_learning']['depth'],
                heads=dmml['hyper_modality_learning']['heads']),

            CrossTransformer(
                source_num_frames=dmml['fuison_transformer']['source_length'],
                tgt_num_frames=dmml['fuison_transformer']['tgt_length'],
                dim=dmml['fuison_transformer']['input_dim'],
                depth=dmml['fuison_transformer']['depth'],
                heads=dmml['fuison_transformer']['heads'],
                mlp_dim=dmml['fuison_transformer']['hidden_dim']),

            nn.Linear(dmml['regression']['input_dim'],
                      dmml['regression']['out_dim'])
        ])

    def forward(self, complete_input, incomplete_input):
        vision, audio, language = complete_input
        vision_m, audio_m, language_m = incomplete_input

        b = vision_m.size(0)

        h_1_v = self.proj_v(vision_m)[:, :8]
        h_1_a = self.proj_a(audio_m)[:, :8]
        h_1_l = self.proj_l(self.bertmodel(language_m))[:, :8]

        feat_tmp = self.completeness_check[0](h_1_l)[:, :1].squeeze()
        w = self.completeness_check[1](feat_tmp)  # completeness scores

        h_0_p = repeat(self.h_p, '1 n d -> b n d', b=b)
        h_1_p = self.proxy_dominate_modality_generator(
            torch.cat([h_0_p, h_1_a, h_1_v], dim=1))[:, :8]
        h_1_p = self.GRL(h_1_p)
        h_1_d = h_1_p * (1 - w.unsqueeze(-1)) + h_1_l * w.unsqueeze(-1)

        h_hyper = repeat(self.h_hyper, '1 n d -> b n d', b=b)
        h_d_list = self.dmml[0](h_1_d)
        h_hyper = self.dmml[1](h_d_list, h_1_a, h_1_v, h_hyper)
        feat = self.dmml[2](h_hyper, h_d_list[-1])
        output = self.dmml[3](torch.mean(feat[:, 1:], dim=1))

        rec_feats, complete_feats, effectiveness_discriminator_out = None, None, None
        if (vision is not None) and (audio is not None) and (language is not None):
            # Reconstruction
            rec_feat_a = self.reconstructor[0](h_1_a)
            rec_feat_v = self.reconstructor[1](h_1_v)
            rec_feat_l = self.reconstructor[2](h_1_l)
            rec_feats = torch.cat([rec_feat_a, rec_feat_v, rec_feat_l], dim=1)

            # Compute the complete features as the label of reconstruction
            complete_language_feat = self.proj_l(self.bertmodel(language))[:, :8]
            complete_vision_feat = self.proj_v(vision)[:, :8]
            complete_audio_feat = self.proj_a(audio)[:, :8]

            effective_discriminator_input = rearrange(
                torch.cat([h_1_d, complete_language_feat]),
                'b n d -> (b n) d')
            effectiveness_discriminator_out = self.effective_discriminator(
                effective_discriminator_input)

            # complete_feats as reconstruction label
            complete_feats = torch.cat(
                [complete_audio_feat, complete_vision_feat, complete_language_feat],
                dim=1)

        return {
            'sentiment_preds': output,
            'w': w,
            'effectiveness_discriminator_out': effectiveness_discriminator_out,
            'rec_feats': rec_feats,
            'complete_feats': complete_feats
        }
