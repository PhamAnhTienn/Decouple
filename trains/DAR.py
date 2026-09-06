from calendar import c
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils import MetricsTop, dict_to_str
from .network.club import CLUBLoss
logger = logging.getLogger('DecAlign')

### Contrastive

# 非线性投影头（MLP）
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第一层全连接 + ReLU
        x = self.fc2(x)          # 第二层全连接
        # x = self.fc2(x)          # 第二层全连接

        return x

# 修改后的对比学习损失函数，加入非线性投影
# def contrastive_loss_with_projection(T1, T2, T3, temperature=0.07):
#     # 通过投影头，将每个模态特征投影到新的空间

#     T1_proj = F.normalize(T1, dim=-1)
#     T2_proj = F.normalize(T2, dim=-1)
#     T3_proj = F.normalize(T3, dim=-1)
    
#     T1_proj = T1_proj.reshape(-1, T1_proj.size(-1))
#     T2_proj = T2_proj.reshape(-1, T2_proj.size(-1))
#     T3_proj = T3_proj.reshape(-1, T3_proj.size(-1))
#     # 计算模态之间的相似性


#     sim_T1_T2 = torch.matmul(T1_proj, T2_proj.T) / temperature  # T1和T2的相似性
#     sim_T1_T3 = torch.matmul(T1_proj, T3_proj.T) / temperature  # T1和T3的相似性
    
#     # 生成标签（正样本对的标签：对角线为1，其他为0）
#     batch_size = T1.size(0)
#     labels = torch.arange(batch_size).to(T1.device)
    
#     # 损失1：T1和T2对比
#     loss_T1_T2 = F.cross_entropy(sim_T1_T2, labels)
    
#     # 损失2：T1和T3对比
#     loss_T1_T3 = F.cross_entropy(sim_T1_T3, labels)
    
#     # 总损失
#     loss = (loss_T1_T2 + loss_T1_T3) / 2
    
#     return loss


def contrastive_loss_with_projection(a, b, c, temperature=0.07):

    a=a.transpose(0,1)
    b=b.transpose(0,1)
    c=c.transpose(0,1)

    a= torch.mean(a, dim=1,keepdim=True)
    b= torch.mean(b, dim=1,keepdim=True)
    c= torch.mean(c, dim=1,keepdim=True)
    batch_size, seq_len, dim = a.shape
    
    # 将特征展平为 (b*l, d)
    a_flat = a.reshape(-1, dim)
    b_flat = b.reshape(-1, dim)
    c_flat = c.reshape(-1, dim)
    
    # 计算对比损失1：同一batch内的特征对为正样本
    # 构建所有可能的特征对
    features = torch.cat([a_flat, b_flat, c_flat], dim=0)  # (3*b*l, d)
    
    # 计算相似度矩阵
    sim_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2) / temperature
    
    # 构建标签矩阵（标识正样本对）
    labels = torch.zeros_like(sim_matrix)
    total_len = batch_size * seq_len
    
    # 标记同一batch的a-b对和a-c对为正样本
    for i in range(batch_size):
        start_idx = i * seq_len
        end_idx = (i + 1) * seq_len
        # a-b对
        labels[start_idx:end_idx, total_len+start_idx:total_len+end_idx] = 1
        labels[total_len+start_idx:total_len+end_idx, start_idx:end_idx] = 1
        # a-c对
        labels[start_idx:end_idx, 2*total_len+start_idx:2*total_len+end_idx] = 1
        labels[2*total_len+start_idx:2*total_len+end_idx, start_idx:end_idx] = 1
    
    # 移除对角线上的自身对比
    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    labels[mask] = 0
    sim_matrix = sim_matrix - mask.float() * 1e9
    
    # 计算对比损失1
    loss1 = F.cross_entropy(sim_matrix, labels)
    
    # 计算对比损失2：来自同一来源的特征为正样本

    
    # 总损失
    total_loss = loss1
    
    return total_loss


def contrastive_loss_with_projection_s(a, b, c, temperature=0.07):
    a=a.transpose(0,1)
    b=b.transpose(0,1)
    c=c.transpose(0,1)
    a= torch.mean(a, dim=1,keepdim=True)
    b= torch.mean(b, dim=1,keepdim=True)
    c= torch.mean(c, dim=1,keepdim=True)
    batch_size, seq_len, dim = a.shape
    
    # 将特征展平为 (b*l, d)
    a_flat = a.reshape(-1, dim)
    b_flat = b.reshape(-1, dim)
    c_flat = c.reshape(-1, dim)
    
    # 计算对比损失1：同一batch内的特征对为正样本
    # 构建所有可能的特征对
    features = torch.cat([a_flat, b_flat, c_flat], dim=0)  # (3*b*l, d)
    
    # 计算相似度矩阵
    sim_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2) / temperature
    source_labels = torch.zeros_like(sim_matrix)
    # a与a之间的对比
    labels = torch.zeros_like(sim_matrix)
    total_len = batch_size * seq_len
    source_labels[:total_len, :total_len] = 1
    # b与b之间的对比
    source_labels[total_len:2*total_len, total_len:2*total_len] = 1
    # c与c之间的对比
    source_labels[2*total_len:, 2*total_len:] = 1
    
    # 移除自身对比
    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)

    source_labels[mask] = 0
    
    # 计算对比损失2
    loss2 = F.cross_entropy(sim_matrix, source_labels)

    return loss2


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class DAR():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()

    def do_train(self, model, dataloader, return_epoch_results=False):

        # 0: DMD model, 1: Homo GD, 2: Hetero GD
        params = list(model.parameters()) 


        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.args.patience)
        
        # optimizer = torch.optim.AdamW(model[0].parameters(),
        #                          lr=self.args.learning_rate,
        #                          weight_decay=self.args.weight_decay)
        # scheduler = get_scheduler(optimizer)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss','MAE'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        best_test = 1e8 if min_or_max == 'min' else 0

        while True:
            epochs += 1
            y_pred, y_true = [], []
            model.train()

            club_loss_l = CLUBLoss(d=self.args.dst_feature_dim_nheads[0]).to(self.args.device)
            club_loss_v = CLUBLoss(d=self.args.dst_feature_dim_nheads[0]).to(self.args.device)
            club_loss_a = CLUBLoss(d=self.args.dst_feature_dim_nheads[0]).to(self.args.device)
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    miss_vision = batch_data['vision_m'].to(self.args.device)
                    miss_audio = batch_data['audio_m'].to(self.args.device)
                    miss_text = batch_data['text_m'].to(self.args.device)
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)

                    output = model((text,miss_text), (audio,miss_audio), (vision,miss_vision), is_distill=True)

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
                    loss_sl_slr = self.MSE(output['s_l'], output['s_l_r'])
                    loss_sv_slv = self.MSE(output['s_v'], output['s_v_r'])
                    loss_sa_sla = self.MSE(output['s_a'], output['s_a_r'])
                    loss_s_sr = loss_sl_slr + loss_sv_slv + loss_sa_sla
                    
                    complete_loss_l = self.MSE(output['recon_a_c'], output['proj_l_c'])
                    complete_loss_v = self.MSE(output['recon_a_c'], output['proj_v_c'])
                    complete_loss_a = self.MSE(output['recon_a_c'], output['proj_a_c'])
                    complete_loss_s_l = self.MSE(output['s_l_c'], output['recon_sl'])
                    complete_loss_s_v = self.MSE(output['s_v_c'], output['recon_sv'])
                    complete_loss_s_a = self.MSE(output['s_a_c'], output['recon_sa'])
                    complete_loss_c_l = self.MSE(output['c_l_c'], output['recon_cl'])
                    complete_loss_c_v = self.MSE(output['c_v_c'], output['recon_cv'])
                    complete_loss_c_a = self.MSE(output['c_a_c'], output['recon_ca'])
                    complete_loss = complete_loss_l + complete_loss_v + complete_loss_a+ complete_loss_s_l + complete_loss_s_v + complete_loss_s_a + complete_loss_c_l + complete_loss_c_v + complete_loss_c_a

                    _ = club_loss_l.learning_loss(output['s_l'].clone().detach(), output['c_l'].clone().detach())
                    _ = club_loss_v.learning_loss(output['s_v'].clone().detach(), output['c_v'].clone().detach())
                    _ = club_loss_a.learning_loss(output['s_a'].clone().detach(), output['c_a'].clone().detach())
                    with torch.no_grad():
                        loss_l_ort = club_loss_l(output['s_l'], output['c_l'])
                        loss_v_ort = club_loss_v(output['s_v'], output['c_v'])
                        loss_a_ort = club_loss_a(output['s_a'], output['c_a'])
                    loss_ort = loss_l_ort + loss_v_ort + loss_a_ort

                    # margin loss
                    c_l, c_v, c_a = output['c_l_sim'], output['c_v_sim'], output['c_a_sim']
                    
                    loss_sim1 = contrastive_loss_with_projection(c_l,c_a,c_v)
                    loss_sim2 = contrastive_loss_with_projection_s(output['s_l'],output['s_v'],output['s_a'])
                    loss_sim = loss_sim1 + loss_sim2

                    combined_loss = loss_task + \
                                    (loss_s_sr + loss_recon + (loss_sim+loss_ort)*0.7) * 0.1+\
                                            complete_loss * 0.1

                    combined_loss.backward()


                    if self.args.grad_clip != -1.0:
                        params = list(model.parameters()) 
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
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.state_dict(), self.args.model_save_path)

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

                    output = model((text,text_m), (audio,audio_m), (vision,vision_m), is_distill=True)
                    
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