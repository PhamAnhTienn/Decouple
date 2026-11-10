import logging

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from utils import MetricsTop, dict_to_str

logger = logging.getLogger('DecAlign')


class MIDAS:
    def __init__(self, args):
        self.args = args
        self.criterion = nn.MSELoss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    def do_train(self, model, dataloader, return_epoch_results=False):
        # ── Optimizer setup ───────────────────────────────────────────
        bert_finetune = getattr(self.args, 'use_finetune', True)

        if bert_finetune:
            lr_bert = getattr(self.args, 'bert_learning_rate', 1e-5)
            lr_other = getattr(self.args, 'learning_rate', 2e-3)
            decay_bert = getattr(self.args, 'weight_decay_bert', 1e-4)
            decay_other = getattr(self.args, 'weight_decay', 1e-3)

            bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            bert_params = list(model.text_encoder.named_parameters())

            bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
            bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
            model_params_other = [p for n, p in list(model.named_parameters()) if 'text_encoder' not in n]

            optimizer_grouped_parameters = [
                {'params': bert_params_decay, 'weight_decay': decay_bert, 'lr': lr_bert},
                {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': lr_bert},
                {'params': model_params_other, 'weight_decay': decay_other, 'lr': lr_other},
            ]
            optimizer = optim.Adam(optimizer_grouped_parameters)
        else:
            lr_other = getattr(self.args, 'learning_rate', 2e-3)
            decay_other = getattr(self.args, 'weight_decay', 1e-3)
            optimizer = optim.Adam(model.parameters(), lr=lr_other, weight_decay=decay_other)

        scheduler = ReduceLROnPlateau(
            optimizer, mode='min',
            factor=getattr(self.args, 'factor', 0.5),
            patience=getattr(self.args, 'patience', 5),
        )

        # ── Loss weights ─────────────────────────────────────────────
        w_kld = getattr(self.args, 'w_kld', 0.05)
        w_pred = getattr(self.args, 'w_pred', 0.1)
        total_epoch = getattr(self.args, 'num_epochs', 100)

        # ── Training loop ────────────────────────────────────────────
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

            left_epochs = getattr(self.args, 'update_epochs', 1)

            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == getattr(self.args, 'update_epochs', 1):
                        optimizer.zero_grad()
                    left_epochs -= 1

                    text = batch_data['text'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    vision = batch_data['vision'].to(self.args.device)
                    text_m = batch_data.get('text_m', text).to(self.args.device)
                    audio_m = batch_data.get('audio_m', audio).to(self.args.device)
                    vision_m = batch_data.get('vision_m', vision).to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)

                    # Audio / vision lengths for padding masks
                    audio_lengths = batch_data.get('audio_lengths', None)
                    vision_lengths = batch_data.get('vision_lengths', None)

                    if self.args.train_mode == 'classification':
                        labels_flat = labels.view(-1).long()
                    else:
                        labels_flat = labels.view(-1, 1)

                    # Forward
                    outputs = model(
                        text, audio, vision,
                        text_m=text_m, audio_m=audio_m, vision_m=vision_m,
                        audio_lengths=audio_lengths, vision_lengths=vision_lengths,
                        labels=labels.view(-1),
                    )

                    # ── Compose loss ──────────────────────────────────
                    task_loss = outputs['task_loss']
                    kld_loss = outputs['diff_loss']
                    pred_loss = outputs['pred_loss']
                    sim_loss = outputs['sim_loss']
                    recon_loss = outputs['recon_loss']

                    # Warm up sim / recon in first 10% of epochs
                    if epochs < 0.1 * total_epoch:
                        w_sim = 0.1 * epochs
                        w_rec = 0.1 * epochs
                    else:
                        w_sim = 1.0
                        w_rec = 1.0

                    loss = task_loss + w_kld * kld_loss + w_pred * pred_loss + \
                           w_sim * sim_loss + w_rec * recon_loss

                    loss.backward()

                    grad_clip = getattr(self.args, 'grad_clip', -1.0)
                    if grad_clip != -1.0:
                        nn.utils.clip_grad_value_(
                            [p for p in model.parameters() if p.requires_grad],
                            grad_clip,
                        )

                    train_loss += loss.item()
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(labels_flat.cpu())

                    if not left_epochs:
                        optimizer.step()
                        left_epochs = getattr(self.args, 'update_epochs', 1)

                if not left_epochs:
                    optimizer.step()

            train_loss /= len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f"TRAIN-({self.args.model_name}) "
                f"[{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> loss: {round(train_loss, 4)} {dict_to_str(train_results)}"
            )

            # Validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])

            isBetter = (cur_valid <= (best_valid - 1e-6)) if min_or_max == 'min' \
                else (cur_valid >= (best_valid + 1e-6))

            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)

            # Early stop
            early_stop = getattr(self.args, 'early_stop', 8)
            if epochs - best_epoch >= early_stop:
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
                    labels = batch_data['labels']['M'].to(self.args.device)

                    text_m = batch_data.get('text_m', text)
                    audio_m = batch_data.get('audio_m', audio)
                    vision_m = batch_data.get('vision_m', vision)
                    if isinstance(text_m, torch.Tensor):
                        text_m = text_m.to(self.args.device)
                    if isinstance(audio_m, torch.Tensor):
                        audio_m = audio_m.to(self.args.device)
                    if isinstance(vision_m, torch.Tensor):
                        vision_m = vision_m.to(self.args.device)

                    audio_lengths = batch_data.get('audio_lengths', None)
                    vision_lengths = batch_data.get('vision_lengths', None)

                    if self.args.train_mode == 'classification':
                        labels_flat = labels.view(-1).long()
                    else:
                        labels_flat = labels.view(-1, 1)

                    outputs = model(
                        text, audio, vision,
                        text_m=text_m, audio_m=audio_m, vision_m=vision_m,
                        audio_lengths=audio_lengths, vision_lengths=vision_lengths,
                        labels=labels.view(-1),
                    )

                    loss = self.criterion(outputs['M'], labels_flat)
                    eval_loss += loss.item()
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(labels_flat.cpu())

                    if return_sample_results:
                        ids.extend(batch_data['id'])
                        all_labels.extend(labels_flat.cpu().detach().tolist())
                        preds = outputs['M'].cpu().detach().numpy()
                        sample_results.extend(preds.squeeze())

        eval_loss /= len(dataloader)
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

