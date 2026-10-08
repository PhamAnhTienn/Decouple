import logging
from pathlib import Path

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
LAMBDA_PRESERVE    = 0.1   # weight for specific preservation loss L_p


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse


class WeightedMSE(nn.Module):
    """Per-sample weighted MSE: L_m = Σ_i w_i ℓ_i / Σ_i w_i."""
    def __init__(self):
        super(WeightedMSE, self).__init__()

    def forward(self, pred, real, weights):
        """
        Args:
            pred: (B, D, T) or (B, ...)
            real: same shape as pred
            weights: (B,) per-sample reliability weights
        Returns:
            scalar weighted MSE loss
        """
        # Per-sample MSE: average over all dims except batch
        per_sample_mse = (pred - real).pow(2).flatten(1).mean(dim=1)  # (B,)
        return (weights * per_sample_mse).sum() / weights.sum().clamp(min=1e-8)

class DP():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()

    def _weighted_recon_loss(self, x_tilde, x_orig, w):
        """
        Weighted cross-reconstruction loss (Eq. 8).
        L_m = Σ_i w_i * ||X̃_m^i - X_m^i||^2 / Σ_i w_i
        
        Args:
            x_tilde: (B, d, T) cross-reconstructed feature
            x_orig:  (B, d, T) original projected feature
            w:       (B,) per-sample CAW weights
        """
        per_sample = (x_tilde - x_orig).pow(2).flatten(1).mean(dim=1)  # (B,)
        return (w * per_sample).sum() / w.sum().clamp(min=1e-8)

    def _cosine_similarity_abs(self, h, p):
        """
        Soft orthogonality loss (Eq. 2).
        L_o = (1/B) Σ_i |cos(h_i, p_i)|

        Args:
            h: (B, d, T) shared features
            p: (B, d, T) specific features
        """
        h_flat = h.flatten(1)  # (B, d*T)
        p_flat = p.flatten(1)
        cos = F.cosine_similarity(h_flat, p_flat, dim=1)  # (B,)
        return cos.abs().mean()

    def _specific_preservation_loss(self, p_tilde, p_orig):
        """
        Specific preservation loss (Eq. 10).
        L_p = (1/B) Σ_i ||P̃_m^i - sg(P_m^i)||^2
        Ensures cross-reconstruction preserves modality-specific information.

        Args:
            p_tilde: (B, d, T) re-disentangled specific features
            p_orig:  (B, d, T) original specific features (stop-gradient applied)
        """
        return F.mse_loss(p_tilde, p_orig.detach())

    def do_train(self, model, dataloader, return_epoch_results=False):
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
        # How often to rebuild the train-set KNN bank (in epochs).
        # bank_update_freq=1 → every epoch (most accurate but expensive)
        # bank_update_freq=N → every N epochs (cheaper, slightly stale)
        bank_update_freq = getattr(self.args, 'bank_update_freq', 1)
        global_iter = 0

        while True:
            epochs += 1
            y_pred, y_true = [], []
            model.train()

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

                    output = model(text_m, audio_m, vision_m, text, audio, vision)

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
                    # loss_task_shared = self.criterion(output['logits_shared'], labels)  # CRM shared branch — commented out
                    loss_task = loss_task_all + loss_task_l_homo + loss_task_v_homo + loss_task_a_homo + loss_task_l_hetero + loss_task_v_hetero + loss_task_a_hetero + loss_task_c

                    # reconstruction loss (OLD uncertainty-reweighted — commented out)
                    # loss_recon_l = self.MSE(output['recon_l'], output['origin_l'])
                    # loss_recon_v = self.MSE(output['recon_v'], output['origin_v'])
                    # loss_recon_a = self.MSE(output['recon_a'], output['origin_a'])
                    # loss_recon = loss_recon_l + loss_recon_v + loss_recon_a

                    # --- L_recon: cross-reconstruction loss (using CRM x_tilde features) ---
                    # # CAW-weighted version (commented out, CAW disabled):
                    # loss_recon_l = self._weighted_recon_loss(
                    #     output['x_tilde_l'], output['origin_l'], recon_w['l'])
                    # loss_recon_v = self._weighted_recon_loss(
                    #     output['x_tilde_v'], output['origin_v'], recon_w['v'])
                    # loss_recon_a = self._weighted_recon_loss(
                    #     output['x_tilde_a'], output['origin_a'], recon_w['a'])
                    loss_recon_l = self.MSE(output['x_tilde_l'], output['origin_l'])
                    loss_recon_v = self.MSE(output['x_tilde_v'], output['origin_v'])
                    loss_recon_a = self.MSE(output['x_tilde_a'], output['origin_a'])
                    loss_recon = loss_recon_l + loss_recon_v + loss_recon_a

                    # # cycle consistency loss (OLD — commented out, replaced by L_p)
                    # loss_sl_slr = self.MSE(output['s_l'].permute(1, 2, 0).detach(), output['s_l_r'])
                    # loss_sv_slv = self.MSE(output['s_v'].permute(1, 2, 0).detach(), output['s_v_r'])
                    # loss_sa_sla = self.MSE(output['s_a'].permute(1, 2, 0).detach(), output['s_a_r'])
                    # loss_s_sr = loss_sl_slr + loss_sv_slv + loss_sa_sla

                    # --- L_p (Eq. 10): specific preservation loss (replaces cycle consistency) ---
                    # P̃_m should preserve original P_m after cross-reconstruction
                    loss_p_l = self._specific_preservation_loss(output['p_tilde_l'], output['s_l'].permute(1, 2, 0))
                    loss_p_v = self._specific_preservation_loss(output['p_tilde_v'], output['s_v'].permute(1, 2, 0))
                    loss_p_a = self._specific_preservation_loss(output['p_tilde_a'], output['s_a'].permute(1, 2, 0))
                    loss_p = loss_p_l + loss_p_v + loss_p_a

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

                    combined_loss = loss_task + \
                                        (loss_p + loss_recon + loss_ort * LAMBDA_ORT_INNER) * LAMBDA_RECON_GROUP \
                                        + loss_cluster * LAMBDA_CLUSTER
                    
                    # combined_loss = loss_task + \
                    #                     (loss_s_sr + loss_recon + loss_ort * LAMBDA_ORT_INNER) * LAMBDA_RECON_GROUP \
                    #                     + loss_cluster * LAMBDA_CLUSTER
                    
                    combined_loss.backward()

                    # Freeze prototypes during warmup iterations
                    if freeze_prototypes_niters > 0 and global_iter < freeze_prototypes_niters:
                        for name, p in model.named_parameters():
                            if "prototypes" in name:
                                p.grad = None

                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_(model.parameters(), self.args.grad_clip)

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

                # --- Update train-set memory bank for KNN neighbor uncertainty ---
                # This rebuilds z_c / z_s for the full training set (eval mode, no_grad)
                # so that both train and test samples use the same neighbor pool.
                # Controlled by bank_update_freq: update every N epochs to save compute.
                if epochs % bank_update_freq == 0:
                    logger.info(f">> Epoch {epochs}: rebuilding train-set KNN bank (freq={bank_update_freq})...")
                    model.update_train_bank(
                        model_forward_fn=model,
                        train_dataloader=dataloader['train'],
                        device=self.args.device,
                    )
                    logger.info(f">> Train bank updated: {model._train_bank_z_c['l'].shape[0]} samples.")

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
                    output = model(text_m, audio_m, vision_m, text, audio, vision)

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

    def do_test_uncertainty(self, model, dataloader, save_dir, K=10):
        """
        Compute and save all uncertainty metrics on a dataset split.

        Args:
            model: trained model (already loaded with best weights)
            dataloader: DataLoader for the target split (typically test)
            save_dir: root directory to save results
            K: number of neighbors for the specific neighbor metric
        """
        from utils.uncertainty_metrics import UncertaintyMetrics

        missing_rate = getattr(self.args, 'missing_rate_eval_test', 0.0)
        seed = getattr(self.args, 'cur_seed', -1)
        dataset = self.args.dataset_name

        save_path = (
            Path(save_dir) / 'uncertainty' /
            f'uncertainty_{dataset}_mr{missing_rate:.1f}_seed{seed}.pt'
        )

        um = UncertaintyMetrics(model, dataloader, self.args, K=K)
        return um.compute_and_save(str(save_path))