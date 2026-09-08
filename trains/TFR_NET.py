import logging
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import dict_to_str, MetricsTop

logger = logging.getLogger('DecAlign')


class TFR_NET():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetrics(args.dataset_name)

    def do_train(self, model, dataloader, return_epoch_results=False):
        use_bert_finetune = getattr(self.args, 'use_bert_finetune', getattr(self.args, 'use_finetune', False))
        if use_bert_finetune:
            bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            actual_model = getattr(model, 'Model', model)
            bert_params = list(actual_model.text_model.named_parameters())

            bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
            bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
            model_params_other = [p for n, p in list(model.named_parameters()) if 'text_model' not in n]

            lr_bert = getattr(self.args, 'learning_rate_bert', 2e-6)
            lr_other = getattr(self.args, 'learning_rate_other', getattr(self.args, 'learning_rate', 0.001))
            wd_bert = getattr(self.args, 'weight_decay_bert', 0.0001)
            wd_other = getattr(self.args, 'weight_decay_other', getattr(self.args, 'weight_decay', 0.001))

            optimizer_grouped_parameters = [
                {'params': bert_params_decay, 'weight_decay': wd_bert, 'lr': lr_bert},
                {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': lr_bert},
                {'params': model_params_other, 'weight_decay': wd_other, 'lr': lr_other}
            ]
            optimizer = optim.Adam(optimizer_grouped_parameters)
        else:
            lr_other = getattr(self.args, 'learning_rate_other', getattr(self.args, 'learning_rate', 0.001))
            wd_other = getattr(self.args, 'weight_decay_other', getattr(self.args, 'weight_decay', 0.001))
            optimizer = optim.Adam(model.parameters(), lr=lr_other, weight_decay=wd_other)

        patience = getattr(self.args, 'patience', 5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, verbose=True, patience=patience)

        epochs, best_epoch = 0, 0
        key_eval = getattr(self.args, 'KeyEval', 'Loss')
        min_or_max = 'min' if key_eval in ['Loss', 'MAE'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }

        update_epochs = getattr(self.args, 'update_epochs', 4)
        early_stop = getattr(self.args, 'early_stop', 6)
        grad_clip = getattr(self.args, 'grad_clip', 0.8)

        while True:
            epochs += 1
            y_pred, y_true = [], []
            model.train()
            train_loss, predict_loss, generate_loss = 0.0, 0.0, 0.0
            left_epochs = update_epochs

            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    
                    text = batch_data['text'].to(self.args.device)
                    text_m = batch_data['text_m'].to(self.args.device) if 'text_m' in batch_data else text
                    text_missing_mask = batch_data['text_missing_mask'].to(self.args.device) if 'text_missing_mask' in batch_data else None

                    audio = batch_data['audio'].to(self.args.device)
                    audio_m = batch_data['audio_m'].to(self.args.device) if 'audio_m' in batch_data else audio
                    audio_mask = batch_data['audio_mask']
                    if isinstance(audio_mask, torch.Tensor):
                        audio_mask = audio_mask.to(self.args.device)
                    audio_missing_mask = batch_data['audio_missing_mask'].to(self.args.device) if 'audio_missing_mask' in batch_data else None

                    vision = batch_data['vision'].to(self.args.device)
                    vision_m = batch_data['vision_m'].to(self.args.device) if 'vision_m' in batch_data else vision
                    vision_mask = batch_data['vision_mask']
                    if isinstance(vision_mask, torch.Tensor):
                        vision_mask = vision_mask.to(self.args.device)
                    vision_missing_mask = batch_data['vision_missing_mask'].to(self.args.device) if 'vision_missing_mask' in batch_data else None

                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)

                    prediction, gen_loss = model(
                        (text, text_m, text_missing_mask),
                        (audio, audio_m, audio_mask, audio_missing_mask),
                        (vision, vision_m, vision_mask, vision_missing_mask)
                    )

                    pred_loss = self.criterion(prediction, labels)
                    if epochs > 1:
                        loss = pred_loss + gen_loss
                    else:
                        loss = pred_loss

                    loss.backward()
                    
                    if grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], grad_clip)

                    optimizer.step()
                    train_loss += loss.item()
                    predict_loss += pred_loss.item()
                    generate_loss += gen_loss.item()

                    y_pred.append(prediction.cpu())
                    y_true.append(labels.cpu())

                    if not left_epochs:
                        optimizer.step()
                        left_epochs = update_epochs

                if not left_epochs:
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            predict_loss = predict_loss / len(dataloader['train'])
            generate_loss = generate_loss / len(dataloader['train'])
            
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f(pred: %.4f; gen: %.4f) %s" % (
                self.args.model_name,
                epochs - best_epoch, epochs, getattr(self.args, 'cur_seed', 0),
                train_loss, predict_loss, generate_loss, dict_to_str(train_results)
            ))
            
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results.get(key_eval, val_results.get('Loss', 0))
            scheduler.step(val_results.get('Loss', 0))

            is_better = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if is_better:
                best_valid, best_epoch = cur_valid, epochs
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)

            if epochs - best_epoch >= early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred, y_true = [], []
        eval_loss, predict_loss, generate_loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    text = batch_data['text'].to(self.args.device)
                    text_m = batch_data['text_m'].to(self.args.device) if 'text_m' in batch_data else text
                    text_missing_mask = batch_data['text_missing_mask'].to(self.args.device) if 'text_missing_mask' in batch_data else None

                    audio = batch_data['audio'].to(self.args.device)
                    audio_m = batch_data['audio_m'].to(self.args.device) if 'audio_m' in batch_data else audio
                    audio_mask = batch_data['audio_mask']
                    if isinstance(audio_mask, torch.Tensor):
                        audio_mask = audio_mask.to(self.args.device)
                    audio_missing_mask = batch_data['audio_missing_mask'].to(self.args.device) if 'audio_missing_mask' in batch_data else None

                    vision = batch_data['vision'].to(self.args.device)
                    vision_m = batch_data['vision_m'].to(self.args.device) if 'vision_m' in batch_data else vision
                    vision_mask = batch_data['vision_mask']
                    if isinstance(vision_mask, torch.Tensor):
                        vision_mask = vision_mask.to(self.args.device)
                    vision_missing_mask = batch_data['vision_missing_mask'].to(self.args.device) if 'vision_missing_mask' in batch_data else None

                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)

                    outputs, gen_loss = model(
                        (text, text_m, text_missing_mask),
                        (audio, audio_m, audio_mask, audio_missing_mask),
                        (vision, vision_m, vision_mask, vision_missing_mask)
                    )

                    pred_loss = self.criterion(outputs, labels)
                    loss = pred_loss

                    eval_loss += loss.item()
                    predict_loss += pred_loss.item()
                    generate_loss += gen_loss.item()

                    y_pred.append(outputs.cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)

        logger.info("%s-(%s) >> %s" % (mode, self.args.model_name, dict_to_str(eval_results)))
        return eval_results
