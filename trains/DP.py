import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import wandb
from utils import MetricsTop, dict_to_str

logger = logging.getLogger('DecAlign')

LAMBDA_RECON_GROUP = 0.1
LAMBDA_ORT_INNER   = 0.1
LAMBDA_CLUSTER     = 0.1
LAMBDA_PUSH        = 0.1


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class DP():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()

    def do_train(self, model, dataloader, return_epoch_results=False):
        # --- W&B Init ---
        wandb.init(
            entity="anhtien130805-vietnam-national-university-hanoi",
            project="Decouple",
            name=f"{getattr(self.args,'dataset_name','NA')}_seed{getattr(self.args,'cur_seed','NA')}_lc{LAMBDA_CLUSTER}_lp{LAMBDA_PUSH}",
            group=getattr(self.args, 'model_name', None),
            config={
                **(vars(self.args) if hasattr(self.args, '__dict__') else {}),
                "lambda_recon_group": LAMBDA_RECON_GROUP,
                "lambda_ort_inner": LAMBDA_ORT_INNER,
                "lambda_cluster": LAMBDA_CLUSTER,
                "lambda_push": LAMBDA_PUSH,
            },
            reinit="finish_previous",
        )
        wandb.watch(model.prototypes, log='gradients', log_freq=50)

        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)
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

        # Number of initial iterations to freeze prototype weights
        freeze_prototypes_niters = getattr(self.args, 'freeze_prototypes_niters', -1)
        global_iter = 0

        try:
            while True:
                epochs += 1
                y_pred, y_true = [], []
                model.train()

                train_loss = 0.0
                # Epoch-level accumulators for each loss component
                epoch_loss_task = 0.0
                epoch_loss_recon = 0.0
                epoch_loss_s_sr = 0.0
                epoch_loss_ort = 0.0
                epoch_loss_cluster = 0.0
                epoch_loss_push = 0.0
                epoch_loss_cmd = 0.0
                num_batches = 0

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

                        output = model(text, audio, vision, is_distill=True)

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

                        # # edges for homo distill
                        # edges_homo, edges_origin_homo = model[1](logits_homo, reprs_homo)

                        # # edges for hetero distill
                        # edges_hetero, edges_origin_hetero = model[2](logits_hetero, reprs_hetero)

                        # normalize the prototypes
                        with torch.no_grad():
                            w = model.prototypes.weight.data.clone()
                            w = F.normalize(w, dim=1, p=2)
                            model.prototypes.weight.copy_(w)

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

                        # CPA loss (Cross-modal Prototype Alignment)
                        loss_cluster = model.compute_cpa_loss(
                            output, tau=0.1, epsilon=0.05, sinkhorn_iterations=3
                        )

                        # prototype separation loss
                        loss_push = model.compute_push_loss(margin=2.0)

                        combined_loss = loss_task + \
                                        (loss_s_sr + loss_recon + loss_ort * LAMBDA_ORT_INNER) * LAMBDA_RECON_GROUP + \
                                        loss_cluster * LAMBDA_CLUSTER + loss_push * LAMBDA_PUSH
                        combined_loss.backward()

                        # --- Step-level W&B logging ---
                        _loss_task_val = loss_task.item()
                        _loss_recon_val = loss_recon.item()
                        _loss_s_sr_val = loss_s_sr.item()
                        _loss_ort_val = loss_ort.item()
                        _loss_cluster_val = loss_cluster.item()
                        _loss_push_val = loss_push.item()
                        _combined_val = combined_loss.item()

                        wandb.log({
                            # Raw (unweighted) loss values
                            "step/loss_task": _loss_task_val,
                            "step/loss_recon": _loss_recon_val,
                            "step/loss_s_sr": _loss_s_sr_val,
                            "step/loss_ort": _loss_ort_val,
                            "step/loss_cluster": _loss_cluster_val,
                            "step/loss_push": _loss_push_val,
                            "step/combined_loss": _combined_val,
                            # Weighted contributions to combined_loss
                            "step/weighted_recon": _loss_recon_val * LAMBDA_RECON_GROUP,
                            "step/weighted_s_sr": _loss_s_sr_val * LAMBDA_RECON_GROUP,
                            "step/weighted_ort": _loss_ort_val * LAMBDA_RECON_GROUP * LAMBDA_ORT_INNER,
                            "step/weighted_cluster": _loss_cluster_val * LAMBDA_CLUSTER,
                            "step/weighted_push": _loss_push_val * LAMBDA_PUSH
                        })

                        # Accumulate for epoch-level averages
                        epoch_loss_task += _loss_task_val
                        epoch_loss_recon += _loss_recon_val
                        epoch_loss_s_sr += _loss_s_sr_val
                        epoch_loss_ort += _loss_ort_val
                        epoch_loss_cluster += _loss_cluster_val
                        epoch_loss_push += _loss_push_val
                        num_batches += 1

                        # Freeze prototypes during warmup iterations
                        if freeze_prototypes_niters > 0 and global_iter < freeze_prototypes_niters:
                            for name, p in model.named_parameters():
                                if "prototypes" in name:
                                    p.grad = None

                        if self.args.grad_clip != -1.0:
                            nn.utils.clip_grad_value_(model.parameters(), self.args.grad_clip)

                        global_iter += 1

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

                # --- Epoch-level W&B logging ---
                n = max(num_batches, 1)
                avg_task = epoch_loss_task / n
                avg_recon = epoch_loss_recon / n
                avg_s_sr = epoch_loss_s_sr / n
                avg_ort = epoch_loss_ort / n
                avg_cluster = epoch_loss_cluster / n
                avg_push = epoch_loss_push / n
                avg_cmd = epoch_loss_cmd / n

                epoch_log = {
                    "epoch": epochs,
                    # Average raw losses per epoch
                    "epoch/loss_task": avg_task,
                    "epoch/loss_recon": avg_recon,
                    "epoch/loss_s_sr": avg_s_sr,
                    "epoch/loss_ort": avg_ort,
                    "epoch/loss_cluster": avg_cluster,
                    "epoch/loss_push": avg_push,
                    "epoch/loss_cmd": avg_cmd,
                    "epoch/combined_loss": train_loss,
                    # Weighted contributions (epoch avg)
                    "epoch/weighted_recon_group": (avg_s_sr + avg_recon + avg_ort * 0.1) * 0.1,
                    "epoch/weighted_cluster": avg_cluster * LAMBDA_CLUSTER,
                    "epoch/weighted_push": avg_push * LAMBDA_PUSH,
                    # Relative magnitudes to task loss
                    "epoch/cluster_to_task_ratio": (avg_cluster * LAMBDA_CLUSTER) / (avg_task + 1e-8),
                    "epoch/push_to_task_ratio": (avg_push * LAMBDA_PUSH) / (avg_task + 1e-8),
                    # Learning rate
                    "epoch/learning_rate": optimizer.param_groups[0]['lr'],
                }
                for k, v in train_results.items():
                    epoch_log[f"train/{k}"] = v

                # Validation metrics
                for k, v in val_results.items():
                    epoch_log[f"val/{k}"] = v

                # Prototype statistics
                with torch.no_grad():
                    C = F.normalize(model.prototypes.weight.data, dim=1, p=2)
                    dots = C @ C.t()
                    K = C.shape[0]
                    mask = ~torch.eye(K, dtype=torch.bool, device=C.device)
                    pairwise_cosines = dots[mask]
                    epoch_log["epoch/proto_cosine_mean"] = pairwise_cosines.mean().item()
                    epoch_log["epoch/proto_cosine_max"] = pairwise_cosines.max().item()
                    epoch_log["epoch/proto_cosine_min"] = pairwise_cosines.min().item()

                wandb.log(epoch_log)

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
        finally:
            wandb.finish()

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
                    output = model(text, audio, vision, is_distill=True)

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