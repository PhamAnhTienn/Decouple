import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils import MetricsTop, dict_to_str

logger = logging.getLogger('DecAlign')


class CACR():
    """
    Training logic for the CACR (Conflict-Aware Adaptive Cross-Reconstruction) model.
    
    Loss function (Eq. 14):
        L_total = L_MSA + L_recon + L_o + L_p
    
    where:
        L_MSA   = α * L_f + β * L_TH̃ + Σ L_TP'm     (Eq. 13, hierarchical prediction)
        L_recon = Σ_m L_m                               (Eq. 8, weighted cross-reconstruction)
        L_o     = Σ_m |cos(H_m, P_m)|                   (Eq. 2, soft orthogonality)
        L_p     = Σ_m ‖P̃_m - P_m‖²                     (Eq. 10, specific preservation)
    """

    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

        # CACR hyperparameters from config
        self.alpha = getattr(args, 'alpha', 1.0)
        self.beta = getattr(args, 'beta', 2.0)

    @staticmethod
    def _cosine_similarity_abs(x, y):
        """
        Compute |cos(x, y)| for the soft orthogonality loss (Eq. 2).
        x, y: (B, d, T) or flattened to (B, -1)
        Returns scalar.
        """
        x_flat = x.reshape(x.size(0), -1)  # (B, d*T)
        y_flat = y.reshape(y.size(0), -1)
        cos_sim = F.cosine_similarity(x_flat, y_flat, dim=-1)  # (B,)
        return cos_sim.abs().mean()

    @staticmethod
    def _weighted_recon_loss(x_tilde, x_orig, weights):
        """
        Compute per-sample weighted cross-reconstruction loss (Eq. 8).
        
        L_m = Σ_i (ω_m^(i) * ℓ_m^(i)) / Σ_i ω_m^(i)
        where ℓ_m^(i) = ‖X̃_m^(i) - X_m^(i)‖²
        
        Args:
            x_tilde: (B, d, T) reconstructed features
            x_orig:  (B, d, T) original features
            weights: (B,) per-sample weights from CAW
        Returns:
            scalar weighted reconstruction loss
        """
        # Per-sample MSE: (B,)
        per_sample_loss = ((x_tilde - x_orig) ** 2).mean(dim=(1, 2))
        # Weighted average
        weighted_loss = (weights * per_sample_loss).sum() / weights.sum()
        return weighted_loss

    @staticmethod
    def _specific_preservation_loss(p_tilde, p_orig):
        """
        Specific preservation loss (Eq. 10): ‖P̃_m - P_m‖²
        
        Args:
            p_tilde: (B, d, T) specific features from reconstructed
            p_orig:  (B, d, T) original specific features
        Returns:
            scalar loss
        """
        return ((p_tilde - p_orig) ** 2).mean()

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

                    # Forward pass
                    output = model(text_m, audio_m, vision_m, is_distill=True)

                    # ==========================================================
                    # Loss computation (Eq. 14): L_total = L_MSA + L_recon + L_o + L_p
                    # ==========================================================

                    # --- L_MSA (Eq. 13): α * L_f + β * L_TH̃ + Σ L_TP'm ---
                    loss_f = self.criterion(output['output_logit'], labels)
                    loss_shared = self.criterion(output['logits_shared'], labels)
                    loss_tp_l = self.criterion(output['logits_tp_l'], labels)
                    loss_tp_v = self.criterion(output['logits_tp_v'], labels)
                    loss_tp_a = self.criterion(output['logits_tp_a'], labels)

                    loss_msa = (self.alpha * loss_f
                                + self.beta * loss_shared
                                + loss_tp_l + loss_tp_v + loss_tp_a)

                    # --- L_recon (Eq. 8): weighted cross-reconstruction loss ---
                    loss_recon_l = self._weighted_recon_loss(
                        output['x_tilde_l'], output['origin_l'], output['w_l'])
                    loss_recon_v = self._weighted_recon_loss(
                        output['x_tilde_v'], output['origin_v'], output['w_v'])
                    loss_recon_a = self._weighted_recon_loss(
                        output['x_tilde_a'], output['origin_a'], output['w_a'])
                    loss_recon = loss_recon_l + loss_recon_v + loss_recon_a

                    # --- L_o (Eq. 2): soft orthogonality loss ---
                    loss_ort_l = self._cosine_similarity_abs(output['h_l'], output['p_l'])
                    loss_ort_v = self._cosine_similarity_abs(output['h_v'], output['p_v'])
                    loss_ort_a = self._cosine_similarity_abs(output['h_a'], output['p_a'])
                    loss_ort = loss_ort_l + loss_ort_v + loss_ort_a

                    # --- L_p (Eq. 10): specific preservation loss ---
                    loss_p_l = self._specific_preservation_loss(output['p_tilde_l'], output['p_l'])
                    loss_p_v = self._specific_preservation_loss(output['p_tilde_v'], output['p_v'])
                    loss_p_a = self._specific_preservation_loss(output['p_tilde_a'], output['p_a'])
                    loss_p = loss_p_l + loss_p_v + loss_p_a

                    # --- Total loss (Eq. 14) ---
                    combined_loss = loss_msa + loss_recon + loss_ort + loss_p

                    combined_loss.backward()

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
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results