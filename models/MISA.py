"""
From: https://github.com/declare-lab/MISA
Paper: MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .subNets2 import BertTextEncoder
from torch.autograd import Function
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = ['MISA']

class ReverseLayerF(Function):
    """
    Adapted from https://github.com/fungtion/DSN/blob/master/functions.py
    """
    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None

# let's define a simple model that can deal with multimodal variable length sequence
class MISA(nn.Module):
    def __init__(self, config):
        super(MISA, self).__init__()

        assert config.use_bert == True

        self.config = config
        self.text_size = config.feature_dims[0]
        self.visual_size = config.feature_dims[2]
        self.acoustic_size = config.feature_dims[1]


        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = config.num_classes if config.train_mode == "classification" else 1
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()
        
        
        rnn = nn.LSTM if self.config.rnncell == "lstm" else nn.GRU
        # defining modules - two layer bidirectional LSTM with layer norm in between
    
        if config.use_bert:
            # text subnets
            self.bertmodel = BertTextEncoder(use_finetune=config.use_finetune, transformers=config.transformers, pretrained=config.pretrained)

        self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = rnn(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True)
        
        self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2*hidden_sizes[2], hidden_sizes[2], bidirectional=True)



        ##########################################
        # mapping modalities to same sized space
        ##########################################
        if self.config.use_bert:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))
        else:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=hidden_sizes[0]*4, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=hidden_sizes[1]*4, out_features=config.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=hidden_sizes[2]*4, out_features=config.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))


        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        
        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())
        

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())


        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))



        ##########################################
        # shared space adversarial discriminator
        ##########################################
        if not self.config.use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
            self.discriminator.add_module('discriminator_layer_2', nn.Linear(in_features=config.hidden_size, out_features=len(hidden_sizes)))

        ##########################################
        # shared-private collaborative discriminator
        ##########################################

        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=4))



        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size*6, out_features=self.config.hidden_size*3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.config.hidden_size*3, out_features= output_size))

        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))


        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        ##########################################
        # prototypes for cluster loss
        ##########################################
        # self.nmb_prototypes = getattr(config, 'nmb_prototypes', 8)
        # self.prototypes = nn.Linear(config.hidden_size, self.nmb_prototypes, bias=False)

        

        
    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)

        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)
        padded_h1 = padded_h1.permute(1, 0, 2)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths, batch_first=True, enforce_sorted=False)

        if self.config.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2

    def alignment(self, text, acoustic, visual, text_m=None, audio_m=None, vision_m=None):
        # bert_sent_mask : consists of seq_len of 1, followed by padding of 0.
        bert_sent, bert_sent_mask, bert_sent_type = text[:,0,:], text[:,1,:], text[:,2,:]

        batch_size = text.size(0)
        
        if self.config.use_bert:
            bert_output = self.bertmodel(text) # [batch_size, seq_len, 768]

            # Use the mean value of bert of the front real sentence length as the final representation of text.
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
            mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)  
            bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len

            utterance_text = bert_output
            
            if text_m is None:
                utterance_text_m = utterance_text
            else:
                bert_sent_m, bert_sent_mask_m, bert_sent_type_m = text_m[:,0,:], text_m[:,1,:], text_m[:,2,:]
                
                bert_output_m = self.bertmodel(text_m)
                
                # Use the mean value of bert of the front real sentence length as the final representation of text.
                masked_output_m = torch.mul(bert_sent_mask_m.unsqueeze(2), bert_output_m)
                mask_len_m = torch.sum(bert_sent_mask_m, dim=1, keepdim=True)  
                bert_output_m = torch.sum(masked_output_m, dim=1, keepdim=False) / mask_len_m
                
                utterance_text_m = bert_output_m
                

        lengths = mask_len.squeeze().int().detach().cpu().view(-1)
        # extract features from visual modality
        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        
        if vision_m is None:
            utterance_video_m = utterance_video
        else:
            final_h1v_m, final_h2v_m = self.extract_features(vision_m, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
            utterance_video_m = torch.cat((final_h1v_m, final_h2v_m), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # extract features from acoustic modality
        final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        if audio_m is None:
            utterance_audio_m = utterance_audio
        else:
            final_h1a_m, final_h2a_m = self.extract_features(audio_m, lengths, self.arnn1, self.arnn2, self.alayer_norm)
            utterance_audio_m = torch.cat((final_h1a_m, final_h2a_m), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        
        # Shared-private encoders
        self.shared_private(utterance_text, utterance_video, utterance_audio, utterance_text_m, utterance_video_m, utterance_audio_m)


        if not self.config.use_cmd_sim:
            # discriminator
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.config.reverse_grad_weight)
            reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.config.reverse_grad_weight)
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.config.reverse_grad_weight)

            self.domain_label_t = self.discriminator(reversed_shared_code_t)
            self.domain_label_v = self.discriminator(reversed_shared_code_v)
            self.domain_label_a = self.discriminator(reversed_shared_code_a)
        else:
            self.domain_label_t = None
            self.domain_label_v = None
            self.domain_label_a = None


        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator( (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a)/3.0 )
        
        # For reconstruction
        self.reconstruct()
        
        # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v,  self.utt_shared_a), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        o = self.fusion(h)
        return o
    
    def reconstruct(self,):

        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)


    def shared_private(self, utterance_t, utterance_v, utterance_a, utterance_t_m=None, utterance_v_m=None, utterance_a_m=None):
        # Projecting to same sized space
        self.utt_t_orig = self.project_t(utterance_t)
        if utterance_t_m is None:
            self.utt_t_input = self.utt_t_orig
        else:
            self.utt_t_input = self.project_t(utterance_t_m)
            
        self.utt_v_orig = self.project_v(utterance_v)
        if utterance_v_m is None:
            self.utt_v_input = self.utt_v_orig
        else:
            self.utt_v_input = self.project_v(utterance_v_m)
        
        self.utt_a_orig = self.project_a(utterance_a)
        if utterance_a_m is None:
            self.utt_a_input = self.utt_a_orig
        else:
            self.utt_a_input = self.project_a(utterance_a_m)


        # Private-shared components
        self.utt_private_t = self.private_t(self.utt_t_input)
        self.utt_private_v = self.private_v(self.utt_v_input)
        self.utt_private_a = self.private_a(self.utt_a_input)

        self.utt_shared_t = self.shared(self.utt_t_input)
        self.utt_shared_v = self.shared(self.utt_v_input)
        self.utt_shared_a = self.shared(self.utt_a_input)


    def forward(self, text, audio, video, text_m=None, audio_m=None, vision_m=None):
        output = self.alignment(text, audio, video, text_m, audio_m, vision_m)
        tmp = {
            "M": output
        }
        return tmp

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

    def compute_cluster_loss(self, tau=0.1, epsilon=0.05, sinkhorn_iterations=3):
        """
        Compute L_cluster: OT-based cross-entropy between soft assignments Q*
        and per-modality cluster predictions.
        Uses self.utt_shared_t/v/a (already utterance-level, shape (B, D)).
        """
        # L2-normalize shared features
        h_c_t = F.normalize(self.utt_shared_t, dim=1, p=2)  # (B, D)
        h_c_v = F.normalize(self.utt_shared_v, dim=1, p=2)  # (B, D)
        h_c_a = F.normalize(self.utt_shared_a, dim=1, p=2)  # (B, D)

        # Fused shared representation (average then L2-norm)
        h_shared = F.normalize((h_c_t + h_c_v + h_c_a) / 3.0, dim=1, p=2)  # (B, D)

        # Prototype weights (K, D)
        C = self.prototypes.weight  # (K, D)

        # --- Compute OT target Q* from fused representation ---
        S_bar = h_shared @ C.t()  # (B, K)
        Q_star = self.sinkhorn_knopp(S_bar, epsilon=epsilon,
                                      sinkhorn_iterations=sinkhorn_iterations)  # (B, K)

        # --- Per-modality cluster predictions ---
        loss_cluster = 0.0
        for h_m in [h_c_t, h_c_v, h_c_a]:
            s_m = h_m @ C.t()  # (B, K)
            log_p_m = F.log_softmax(s_m / tau, dim=1)  # (B, K)
            loss_cluster -= torch.mean(torch.sum(Q_star * log_p_m, dim=1))

        loss_cluster /= 3.0  # average over 3 modalities
        return loss_cluster

    def compute_push_loss(self, t=1.0):
        """
        Compute L_push: prototype separation loss.
        Encourages prototypes to be far apart from each other.
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

        # L_push = log( (1 / K(K-1)) * sum_{i!=j} exp(-t * ||c_i - c_j||^2) )
        loss_push = torch.logsumexp(-t * sq_dists_off, dim=0) - math.log(K * (K - 1))
        return loss_push