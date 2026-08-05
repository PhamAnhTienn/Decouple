import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils import MetricsTop, dict_to_str

logger = logging.getLogger('DecAlign')

import torch
import torch.nn as nn


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def compute_cosine(self, x, y):
        # x = self.compute_compact_s(x)
        # y = self.compute_compact_s(y)
        x_norm = torch.sqrt(torch.sum(torch.pow(x, 2), 1)+1e-8)
        x_norm = torch.max(x_norm, 1e-8*torch.ones_like(x_norm))
        y_norm = torch.sqrt(torch.sum(torch.pow(y, 2), 1)+1e-8)
        y_norm = torch.max(y_norm, 1e-8*torch.ones_like(y_norm))
        cosine = torch.sum(x * y, 1) / (x_norm * y_norm)
        return cosine

    def forward(self, ids, feats, margin=0.1):
        B, F = feats.shape

        s = feats.repeat(1, B).view(-1, F) # B**2 X F
        s_ids = ids.view(B, 1).repeat(1, B) # B X B
        
        t = feats.repeat(B, 1) # B**2 X F
        t_ids = ids.view(1, B).repeat(B, 1) # B X B 

        cosine = self.compute_cosine(s, t) # B**2
        equal_mask = torch.eye(B, dtype=torch.bool) # B X B
        s_ids = s_ids[~equal_mask].view(B, B-1) # B X (B-1)
        t_ids = t_ids[~equal_mask].view(B, B-1) # B X (B-1)
        cosine = cosine.view(B, B)[~equal_mask].view(B, B-1) # B X (B-1)

        sim_mask = (s_ids == t_ids) # B X (B-1)
        margin = 0.15 * abs(s_ids - t_ids)#[~sim_mask].view(B, B - 3)

        loss = 0
        loss_num = 0
        
        for i in range(B):
            sim_num = sum(sim_mask[i])
            dif_num = B - 1 - sim_num
            if not sim_num or not dif_num:
                continue
            sim_cos = cosine[i, sim_mask[i]].reshape(-1, 1).repeat(1, dif_num)
            dif_cos = cosine[i, ~sim_mask[i]].reshape(-1, 1).repeat(1, sim_num).transpose(0, 1)
            t_margin = margin[i, ~sim_mask[i]].reshape(-1, 1).repeat(1, sim_num).transpose(0, 1)

            loss_i = torch.max(torch.zeros_like(sim_cos), t_margin - sim_cos + dif_cos).mean()
            loss += loss_i
            loss_num += 1

        if loss_num == 0:
            loss_num = 1

        loss = loss / loss_num
        return loss


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class DMD():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()
        self.sim_loss = HingeLoss()

    def do_train(self, model, dataloader, return_epoch_results=False):

        # 0: DMD model, 1: Homo GD, 2: Hetero GD
        params = list(model[0].parameters()) + \
                 list(model[1].parameters()) + \
                 list(model[2].parameters())

        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        net = []
        net_dmd = model[0]
        net_distill_homo = model[1]
        net_distill_hetero = model[2]
        net.append(net_dmd)
        net.append(net_distill_homo)
        net.append(net_distill_hetero)
        model = net

        while True:
            epochs += 1
            y_pred, y_true = [], []
            for mod in model:
                mod.train()

            train_loss = 0.0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    text = batch_data['text'].to(self.args.device)
                    text_m = batch_data['text_m'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    audio_m = batch_data['audio_m'].to(self.args.device)
                    vision = batch_data['vision'].to(self.args.device)
                    vision_m = batch_data['vision_m'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)

                    logits_homo, reprs_homo, logits_hetero, reprs_hetero = [], [], [], []

                    output = model[0](text_m, audio_m, vision_m, is_distill=True)

                    # logits for homo GD
                    logits_homo.append(output['logits_l_homo'])
                    logits_homo.append(output['logits_v_homo'])
                    logits_homo.append(output['logits_a_homo'])

                    # reprs for homo GD
                    reprs_homo.append(output['repr_l_homo'])
                    reprs_homo.append(output['repr_v_homo'])
                    reprs_homo.append(output['repr_a_homo'])

                    # logits for hetero GD
                    logits_hetero.append(output['logits_l_hetero'])
                    logits_hetero.append(output['logits_v_hetero'])
                    logits_hetero.append(output['logits_a_hetero'])

                    # reprs for hetero GD
                    reprs_hetero.append(output['repr_l_hetero'])
                    reprs_hetero.append(output['repr_v_hetero'])
                    reprs_hetero.append(output['repr_a_hetero'])

                    logits_homo = torch.stack(logits_homo)
                    reprs_homo = torch.stack(reprs_homo)

                    logits_hetero = torch.stack(logits_hetero)
                    reprs_hetero = torch.stack(reprs_hetero)

                    # edges for homo distill
                    edges_homo, edges_origin_homo = model[1](logits_homo, reprs_homo)

                    # edges for hetero distill
                    edges_hetero, edges_origin_hetero = model[2](logits_hetero, reprs_hetero)

                    # task loss
                    loss_task_all = self.criterion(output['output_logit'], labels)
                    loss_task_l_homo = self.criterion(output['logits_l_homo'], labels)
                    loss_task_v_homo = self.criterion(output['logits_v_homo'], labels)
                    loss_task_a_homo = self.criterion(output['logits_a_homo'], labels)
                    loss_task_l_hetero = self.criterion(output['logits_l_hetero'], labels)
                    loss_task_v_hetero = self.criterion(output['logits_v_hetero'], labels)
                    loss_task_a_hetero = self.criterion(output['logits_a_hetero'], labels)
                    loss_task_c = self.criterion(output['logits_c'], labels)
                    loss_task = loss_task_all + loss_task_l_homo + loss_task_v_homo + loss_task_a_homo + loss_task_l_hetero + loss_task_v_hetero + loss_task_a_hetero + loss_task_c

                    # reconstruction loss
                    loss_recon_l = self.MSE(output['recon_l'], output['origin_l'])
                    loss_recon_v = self.MSE(output['recon_v'], output['origin_v'])
                    loss_recon_a = self.MSE(output['recon_a'], output['origin_a'])
                    loss_recon = loss_recon_l + loss_recon_v + loss_recon_a

                    # cycle consistency loss between s_x and s_x_r
                    loss_sl_slr = self.MSE(output['s_l'].permute(1, 2, 0), output['s_l_r'])
                    loss_sv_slv = self.MSE(output['s_v'].permute(1, 2, 0), output['s_v_r'])
                    loss_sa_sla = self.MSE(output['s_a'].permute(1, 2, 0), output['s_a_r'])
                    loss_s_sr = loss_sl_slr + loss_sv_slv + loss_sa_sla

                    # ort loss
                    s_l = output['s_l'].transpose(0, 1).contiguous().view(labels.size(0), -1)
                    c_l = output['c_l'].transpose(0, 1).contiguous().view(labels.size(0), -1)

                    s_v = output['s_v'].transpose(0, 1).contiguous().view(labels.size(0), -1)
                    c_v = output['c_v'].transpose(0, 1).contiguous().view(labels.size(0), -1)

                    s_a = output['s_a'].transpose(0, 1).contiguous().view(labels.size(0), -1)
                    c_a = output['c_a'].transpose(0, 1).contiguous().view(labels.size(0), -1)

                    target = -torch.ones(labels.size(0), device=labels.device)

                    cosine_similarity_s_c_l = self.cosine(s_l, c_l, target)
                    cosine_similarity_s_c_v = self.cosine(s_v, c_v, target)
                    cosine_similarity_s_c_a = self.cosine(s_a, c_a, target)

                    loss_ort = cosine_similarity_s_c_l + cosine_similarity_s_c_v + cosine_similarity_s_c_a

                    # margin loss
                    c_l, c_v, c_a = output['c_l_sim'], output['c_v_sim'], output['c_a_sim']
                    ids, feats = [], []
                    for i in range(labels.size(0)):
                        feats.append(c_l[i].view(1, -1))
                        feats.append(c_v[i].view(1, -1))
                        feats.append(c_a[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                    feats = torch.cat(feats, dim=0)
                    ids = torch.cat(ids, dim=0)
                    loss_sim = self.sim_loss(ids, feats)

                    # homo GD loss
                    loss_reg_homo, loss_logit_homo, loss_repr_homo = \
                        model[1].distillation_loss(logits_homo, reprs_homo, edges_homo)
                    graph_distill_loss_homo = 0.05 * (loss_logit_homo + loss_reg_homo)

                    # hetero GD loss
                    loss_reg_hetero, loss_logit_hetero, loss_repr_hetero = \
                        model[2].distillation_loss(logits_hetero, reprs_hetero, edges_hetero)
                    graph_distill_loss_hetero = 0.05 * (loss_logit_hetero + loss_repr_hetero + loss_reg_hetero)

                    combined_loss = loss_task + \
                                    graph_distill_loss_homo + graph_distill_loss_hetero + \
                                    (loss_s_sr + loss_recon + (loss_sim+loss_ort) * 0.1) * 0.1

                    # combined_loss = loss_task + \
                    #                 graph_distill_loss_homo + graph_distill_loss_hetero + \
                    #                 (loss_s_sr + loss_recon + (loss_ort) * 0.1) * 0.1
                    combined_loss.backward()


                    if self.args.grad_clip != -1.0:
                        params = list(model[0].parameters()) + \
                                 list(model[1].parameters()) + \
                                 list(model[2].parameters())
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)

                    train_loss += combined_loss.item()

                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(model[0], dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model[0].state_dict(), self.args.model_save_path)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred, y_true = [], []

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    text = batch_data['text'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    vision = batch_data['vision'].to(self.args.device)
                    text_m = batch_data['text_m'].to(self.args.device)
                    audio_m = batch_data['audio_m'].to(self.args.device)
                    vision_m = batch_data['vision_m'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    output = model(text_m, audio_m, vision_m, is_distill=True)

                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            # for k in features.keys():
            #     features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results