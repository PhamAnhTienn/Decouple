import torch
from torch import nn

from ...subNets2.BertTextEncoder import BertTextEncoder
from .alignment import Alignment
from .fusion import Fusion
from .generator import Generator


# CMD Loss
class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """
    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments=3):
        x1 = x1.view(-1, x1.shape[-1])
        x2 = x2.view(-1, x2.shape[-1])
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
        sqrt = (summed + 1e-12) ** 0.5
        return sqrt

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)


class RECLoss(nn.Module):
    def __init__(self, args):
        super(RECLoss, self).__init__()
        self.register_buffer('eps', torch.FloatTensor([1e-4]))
        self.args = args

        recloss_type = getattr(args, 'recloss_type', 'combine')
        if recloss_type == 'SmoothL1Loss':
            self.loss = nn.SmoothL1Loss(reduction='sum')
        elif recloss_type == 'MSELoss':
            self.loss = nn.MSELoss(reduction='sum')
        elif recloss_type == 'cmd':
            self.loss = CMD()
        elif recloss_type == 'combine':
            self.loss = nn.SmoothL1Loss(reduction='sum')
            self.loss_cmd = CMD()
        else:
            self.loss = nn.SmoothL1Loss(reduction='sum')

    def forward(self, pred, target, mask):
        """
            pred, target -> batch, seq_len, d
            mask -> batch, seq_len
        """
        mask = mask.unsqueeze(-1).expand(pred.shape[0], pred.shape[1], pred.shape[2])

        loss = self.loss(pred * mask, target * mask) / (torch.sum(mask) + self.eps)

        recloss_type = getattr(self.args, 'recloss_type', 'combine')
        weight_sim_loss = getattr(self.args, 'weight_sim_loss', 0)
        if recloss_type == 'combine' and weight_sim_loss != 0:
            loss += (weight_sim_loss * self.loss_cmd(pred * mask, target * mask) / (torch.sum(mask) + self.eps))
        return loss


class TFR_NET(nn.Module):
    def __init__(self, args):
        super(TFR_NET, self).__init__()
        self.args = args
        
        self.use_bert = getattr(args, 'use_bert', True)
        if self.use_bert:
            use_finetune = getattr(args, 'use_bert_finetune', getattr(args, 'use_finetune', False))
            pretrained = getattr(args, 'pretrained', 'bert-base-uncased')
            transformers = getattr(args, 'transformers', 'bert')
            self.text_model = BertTextEncoder(use_finetune=use_finetune,
                                              transformers=transformers,
                                              pretrained=pretrained)

        self.align_subnet = Alignment(args)

        without_generator = getattr(args, 'without_generator', False)
        if not without_generator:
            self.generator_t = Generator(args, modality='text')
            self.generator_a = Generator(args, modality='audio')
            self.generator_v = Generator(args, modality='vision')
            self.gen_loss = RECLoss(args)
   
        dst_dim = getattr(args, 'dst_feature_dim_nheads', [30, 6])[0] * 3
        args.fusion_t_in = args.fusion_a_in = args.fusion_v_in = dst_dim

        self.fusion_subnet = Fusion(args)

    def forward(self, text, audio, vision):
        # Unpack text inputs
        if isinstance(text, (tuple, list)):
            text_orig, text_m, missing_mask_t = text
        else:
            text_orig, text_m, missing_mask_t = text, text, None

        # Unpack audio inputs
        if isinstance(audio, (tuple, list)):
            if len(audio) == 4:
                audio_orig, audio_m, audio_mask, missing_mask_a = audio
            elif len(audio) == 2:
                audio_orig, audio_m = audio
                audio_mask, missing_mask_a = None, None
            else:
                audio_orig = audio_m = audio[0]
                audio_mask, missing_mask_a = None, None
        else:
            audio_orig = audio_m = audio
            audio_mask, missing_mask_a = None, None

        # Unpack vision inputs
        if isinstance(vision, (tuple, list)):
            if len(vision) == 4:
                vision_orig, vision_m, vision_mask, missing_mask_v = vision
            elif len(vision) == 2:
                vision_orig, vision_m = vision
                vision_mask, missing_mask_v = None, None
            else:
                vision_orig = vision_m = vision[0]
                vision_mask, missing_mask_v = None, None
        else:
            vision_orig = vision_m = vision
            vision_mask, missing_mask_v = None, None

        # Process text mask and text representations
        if self.use_bert:
            text_mask = text_orig[:, 1, :]
            text_m = self.text_model(text_m)
            text_orig = self.text_model(text_orig)
        else:
            text_mask = torch.ones((text_orig.shape[0], text_orig.shape[1]), device=text_orig.device)

        # Ensure masks are tensors on the correct device
        if audio_mask is None:
            audio_mask = torch.ones((audio_orig.shape[0], audio_orig.shape[1]), device=audio_orig.device)
        else:
            audio_mask = torch.as_tensor(audio_mask, device=audio_orig.device, dtype=torch.float32)

        if vision_mask is None:
            vision_mask = torch.ones((vision_orig.shape[0], vision_orig.shape[1]), device=vision_orig.device)
        else:
            vision_mask = torch.as_tensor(vision_mask, device=vision_orig.device, dtype=torch.float32)

        if missing_mask_t is None:
            missing_mask_t = torch.zeros_like(text_mask)
        else:
            missing_mask_t = torch.as_tensor(missing_mask_t, device=text_orig.device, dtype=torch.float32)

        if missing_mask_a is None:
            missing_mask_a = torch.zeros_like(audio_mask)
        else:
            missing_mask_a = torch.as_tensor(missing_mask_a, device=audio_orig.device, dtype=torch.float32)

        if missing_mask_v is None:
            missing_mask_v = torch.zeros_like(vision_mask)
        else:
            missing_mask_v = torch.as_tensor(missing_mask_v, device=vision_orig.device, dtype=torch.float32)

        text_h, audio_h, vision_h, text_h_g, audio_h_g, vision_h_g = self.align_subnet(text_m, audio_m, vision_m)

        without_generator = getattr(self.args, 'without_generator', False)
        if not without_generator:
            text_ = self.generator_t(text_h_g)
            audio_ = self.generator_a(audio_h_g)
            vision_ = self.generator_v(vision_h_g)

            text_gen_loss = self.gen_loss(text_, text_orig, text_mask - missing_mask_t)
            audio_gen_loss = self.gen_loss(audio_, audio_orig, audio_mask - missing_mask_a)
            vision_gen_loss = self.gen_loss(vision_, vision_orig, vision_mask - missing_mask_v)

            prediction = self.fusion_subnet((text_h, text_mask), (audio_h, audio_mask), (vision_h, vision_mask))

            w = getattr(self.args, 'weight_gen_loss', [5, 2, 20])
            total_gen_loss = w[0] * text_gen_loss + w[1] * audio_gen_loss + w[2] * vision_gen_loss
            return prediction, total_gen_loss
        else:
            prediction = self.fusion_subnet((text_h, text_mask), (audio_h, audio_mask), (vision_h, vision_mask))
            zero_loss = prediction.new_tensor(0.0)
            return prediction, zero_loss
