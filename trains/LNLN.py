import logging

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

from utils import MetricsTop, dict_to_str

logger = logging.getLogger('DecAlign')

class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up (increasing) learning rate in optimizer.

    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer: Wrapped optimizer.
        multiplier: target lr = base lr * multiplier (must be >= 1.0).
        total_epoch: target lr is reached at total_epoch.
        after_scheduler: scheduler to use after warmup phase.
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater than or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
                for base_lr in self.base_lrs
            ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


# ---------------------------------------------------------------------------
# Multi-component loss (from lnln/core/losses.py)
# ---------------------------------------------------------------------------
class LNLNLoss(nn.Module):
    """Multi-component loss for LNLN model.

    Components:
        l_cc  : completeness check loss (MSE)
        l_adv : effectiveness adversarial loss (CrossEntropy)
        l_rec : reconstruction loss (MSE)
        l_sp  : sentiment prediction loss (MSE)

    Total = alpha * l_cc + beta * l_adv + gamma * l_rec + sigma * l_sp
    """

    def __init__(self, alpha=0.9, beta=0.8, gamma=0.1, sigma=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.CE_Fn = nn.CrossEntropyLoss()
        self.MSE_Fn = nn.MSELoss()

    def forward(self, out, label):
        l_cc = (self.MSE_Fn(out['w'], label['completeness_labels'])
                if out['w'] is not None else 0)

        l_adv = (self.CE_Fn(out['effectiveness_discriminator_out'],
                            label['effectiveness_labels'])
                 if out['effectiveness_discriminator_out'] is not None else 0)

        l_rec = (self.MSE_Fn(out['rec_feats'], out['complete_feats'])
                 if out['rec_feats'] is not None and out['complete_feats'] is not None
                 else 0)

        l_sp = self.MSE_Fn(out['sentiment_preds'], label['sentiment_labels'])

        loss = (self.alpha * l_cc + self.beta * l_adv
                + self.gamma * l_rec + self.sigma * l_sp)

        return {
            'loss': loss, 'l_sp': l_sp,
            'l_cc': l_cc, 'l_adv': l_adv, 'l_rec': l_rec
        }


class LNLN():
    def __init__(self, args):
        assert getattr(args, 'data_missing', False), (
            "LNLN requires 'data_missing: true' in config to enable "
            "missing modality simulation."
        )
        self.args = args
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.loss_fn = LNLNLoss(
            alpha=getattr(args, 'lnln_alpha', 0.9),
            beta=getattr(args, 'lnln_beta', 0.8),
            gamma=getattr(args, 'lnln_gamma', 0.1),
            sigma=getattr(args, 'lnln_sigma', 1.0),
        )
        self.eval_criterion = nn.MSELoss()

    def do_train(self, model, dataloader, return_epoch_results=False):
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=getattr(self.args, 'weight_decay', 0.0001)
        )

        n_epochs = self.args.num_epochs
        warmup_epochs = max(int(0.1 * n_epochs), 1)
        scheduler_cosine = CosineAnnealingLR(
            optimizer, T_max=int(0.9 * n_epochs))
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1,
            total_epoch=warmup_epochs,
            after_scheduler=scheduler_cosine
        )

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {'train': [], 'valid': [], 'test': []}

        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        while True:
            epochs += 1
            y_pred, y_true = [], []
            model.train()
            train_loss = 0.0

            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    optimizer.zero_grad()

                    # Complete modality inputs (for reconstruction during training)
                    text = batch_data['text'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    vision = batch_data['vision'].to(self.args.device)

                    # Incomplete (missing) modality inputs
                    text_m = batch_data['text_m'].to(self.args.device)
                    audio_m = batch_data['audio_m'].to(self.args.device)
                    vision_m = batch_data['vision_m'].to(self.args.device)

                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    # LNLN forward expects tuples: (vision, audio, language)
                    complete_input = (vision, audio, text)
                    incomplete_input = (vision_m, audio_m, text_m)

                    # Build label dict for LNLN loss
                    completeness_labels = (
                        1.0 - batch_data['labels']['missing_rate_l']
                              .to(self.args.device)
                    )
                    effectiveness_labels = torch.cat([
                        torch.ones(len(labels) * 8),
                        torch.zeros(len(labels) * 8)
                    ]).long().to(self.args.device)

                    label_dict = {
                        'sentiment_labels': labels,
                        'completeness_labels': completeness_labels,
                        'effectiveness_labels': effectiveness_labels
                    }

                    out = model(complete_input, incomplete_input)
                    loss_dict = self.loss_fn(out, label_dict)
                    loss_dict['loss'].backward()

                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_(
                            model.parameters(), self.args.grad_clip)

                    optimizer.step()
                    train_loss += loss_dict['loss'].item()

                    y_pred.append(out['sentiment_preds'].cpu())
                    y_true.append(labels.cpu())

            scheduler.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN-({self.args.model_name}) "
                f"[{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )

            # Validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]

            # Save best model
            isBetter = (cur_valid <= (best_valid - 1e-6)
                        if min_or_max == 'min'
                        else cur_valid >= (best_valid + 1e-6))
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                torch.save(model.state_dict(), self.args.model_save_path)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(
                    model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)

            # Early stop
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
                "Feature_t": [], "Feature_a": [],
                "Feature_v": [], "Feature_f": [],
            }

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    # During evaluation, only use incomplete (missing) inputs
                    text_m = batch_data['text_m'].to(self.args.device)
                    audio_m = batch_data['audio_m'].to(self.args.device)
                    vision_m = batch_data['vision_m'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    # LNLN eval: complete_input = (None, None, None)
                    complete_input = (None, None, None)
                    incomplete_input = (vision_m, audio_m, text_m)

                    out = model(complete_input, incomplete_input)

                    loss = self.eval_criterion(
                        out['sentiment_preds'], labels)
                    eval_loss += loss.item()

                    y_pred.append(out['sentiment_preds'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(
            f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results
