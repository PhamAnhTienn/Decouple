import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from utils import MetricsTop, dict_to_str

logger = logging.getLogger('DecAlign')

__all__ = ['LMF']


class LMF():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    def do_train(self, model, dataloader, return_epoch_results=False):
        factor_params = [model.audio_factor, model.video_factor, model.text_factor]
        factor_param_ids = {id(p) for p in factor_params}
        other_params = [p for p in model.parameters() if id(p) not in factor_param_ids]

        optimizer = optim.Adam([
            {"params": factor_params, "lr": getattr(self.args, 'factor_lr', self.args.learning_rate)},
            {"params": other_params, "lr": self.args.learning_rate}
        ], weight_decay=getattr(self.args, 'weight_decay', 0.0))

        # initialize results
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        key_eval = getattr(self.args, 'KeyEval', 'Loss')
        min_or_max = 'min' if key_eval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        update_epochs = getattr(self.args, 'update_epochs', 1)
        grad_clip = getattr(self.args, 'grad_clip', -1.0)
        early_stop = getattr(self.args, 'early_stop', 8)

        # loop until early stop or max epochs
        max_epochs = getattr(self.args, 'num_epochs', None)
        while True:
            epochs += 1
            if max_epochs is not None and epochs > max_epochs:
                break

            # train
            y_pred, y_true = [], []
            model.train()
            train_loss = 0.0
            left_epochs = update_epochs

            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1

                    # Support missing modality simulation if enabled
                    if getattr(self.args, 'data_missing', False) and 'text_m' in batch_data:
                        vision = batch_data['vision_m'].to(self.args.device)
                        audio = batch_data['audio_m'].to(self.args.device)
                        text = batch_data['text_m'].to(self.args.device)
                    else:
                        vision = batch_data['vision'].to(self.args.device)
                        audio = batch_data['audio'].to(self.args.device)
                        text = batch_data['text'].to(self.args.device)

                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)

                    # forward
                    outputs = model(text, audio, vision)['M']
                    # compute loss
                    loss = self.criterion(outputs, labels)
                    # backward
                    loss.backward()

                    if grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], grad_clip)

                    # store results
                    train_loss += loss.item()
                    y_pred.append(outputs.cpu())
                    y_true.append(labels.cpu())

                    if not left_epochs:
                        optimizer.step()
                        left_epochs = update_epochs

                if not left_epochs:
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] >> loss: {round(train_loss, 4)} {dict_to_str(train_results)}"
            )

            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[key_eval]

            # save best model
            is_better = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if is_better:
                best_valid, best_epoch = cur_valid, epochs
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)

            # epoch results
            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)

            # early stop
            if epochs - best_epoch >= early_stop:
                logger.info(f"Early stopping at epoch {epochs} (best epoch: {best_epoch})")
                return epoch_results if return_epoch_results else None

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
                    if getattr(self.args, 'data_missing', False) and 'text_m' in batch_data:
                        vision = batch_data['vision_m'].to(self.args.device)
                        audio = batch_data['audio_m'].to(self.args.device)
                        text = batch_data['text_m'].to(self.args.device)
                    else:
                        vision = batch_data['vision'].to(self.args.device)
                        audio = batch_data['audio'].to(self.args.device)
                        text = batch_data['text'].to(self.args.device)

                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)

                    outputs = model(text, audio, vision)

                    if return_sample_results:
                        ids.extend(batch_data['id'])
                        for item in features.keys():
                            features[item].append(outputs[item].cpu().detach().numpy())
                        all_labels.extend(labels.cpu().detach().tolist())
                        preds = outputs["M"].cpu().detach().numpy()
                        sample_results.extend(preds.squeeze())

                    loss = self.criterion(outputs['M'], labels)
                    eval_loss += loss.item()
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results
