from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .standard.models import FedNCFModel
from fedlib.data import FedDataModule
from stats import TimeStats
import math
import optree
from .param_tree import tree_sub_

class Client:
    def __init__(
        self,
        cid,
        model: FedNCFModel,
        datamodule,
        loss_fn,
        central_train=False,
    ) -> None:
        self._cid = cid
        self.datamodule = datamodule
        self._model = model
        if not central_train:
            self._private_params = self._model.get_private_params()
        self.loss_fn = loss_fn

    @property
    def cid(self):
        return self._cid

    def get_parameters(self, config):
        private_params, sharable_params = self._model._get_splited_params()
        self._private_params = private_params
        return sharable_params

    def set_parameters(self, global_params: List[np.ndarray]) -> None:
        self._model._set_state_from_splited_params([self._private_params, global_params])
    
    def prepare_dataloader_mp(self, config):
        # print(f'*Preparing client {self.cid}')
        train_loader = self.datamodule.train_dataloader([self.cid])
        return train_loader
    
    def prepare_dataloader(self, config):
        self.train_loader = self.datamodule.train_dataloader([self.cid])
        return len(self.train_loader.dataset)

    def fit(
        self, 
        server_params: dict, 
        local_epochs: int,
        config: Dict[str, str], 
        device, 
        stats_logger: TimeStats,
        **forward_kwargs
    ) -> Tuple[dict, int, Dict]:
        # Preparing train dataloader
        try:
            train_loader = self.train_loader
        except AttributeError as e:
            print("Please call prepare_dataloader() first. CID: %d" % self._cid)
            raise e

        # Set model parameters, train model, return updated model parameters
        with torch.no_grad():
            if server_params is not None:
                with stats_logger.timer('set_parameters'):
                    self.set_parameters(server_params)
        
        item_emb_params, params_1  = self._model._get_splited_params_for_optim()
        opt_params = [
                {'params': item_emb_params, 'lr': config.TRAIN.lr},
                {'params': params_1, 'lr': config.TRAIN.lr},
        ]
        if config.TRAIN.optimizer == 'sgd':
            optimizer = torch.optim.SGD(opt_params, lr=config.TRAIN.lr,) # weight_decay=config.TRAIN.weight_decay)
        elif config.TRAIN.optimizer == 'adam':
            optimizer = torch.optim.Adam(opt_params, lr=config.TRAIN.lr,) # weight_decay=config.TRAIN.weight_decay)

        with stats_logger.timer('fit'):
            metrics = self._fit(train_loader, 
                                optimizer, 
                                self.loss_fn, 
                                num_epochs=local_epochs, 
                                device=device, 
                                base_lr=config.TRAIN.lr, 
                                wd=config.TRAIN.weight_decay, 
                                **forward_kwargs)
        
        with torch.no_grad():
            with stats_logger.timer('get_parameters'):
                sharable_param_tree = self.get_parameters(None)
                if server_params is not None:
                    sharable_param_tree = tree_sub_(sharable_param_tree, server_params)
                    # if config.FED.compression_kwargs.method != "none":
                    #     with stats_logger.timer('compress', max_agg=True):
                    #         sharable_param_tree.compress(**config.FED.compression_kwargs)

        # stats_logger.stats_transfer_params(cid=self._cid, stat_dict=self._model.stat_transfered_params(update_tree))
        return sharable_param_tree, len(train_loader.dataset), metrics
    
    def _scale_lr(self, base_lr, item, optimizer):
        scale_lr_item_emb = len(set(item.tolist()))
        if self._model.is_lora:
            scale_lr_item_emb *= self._model.lora_scale_lr
        optimizer.param_groups[0]['lr'] = base_lr * scale_lr_item_emb

    def _fit(self, train_loader, optimizer, loss_fn, num_epochs, device, base_lr, wd, mask_zero_user_index=False):
        self._model.train() # Enable dropout (if have).
        loss_hist = []
        for e in range(num_epochs):
            total_loss, count_example = 0, 0
            for user, item, label in train_loader:
                user, item, label = user.to(device), item.to(device), label.float().to(device)
                if mask_zero_user_index:
                    user *= 0
                self._scale_lr(base_lr, item, optimizer)
 
                optimizer.zero_grad()
                prediction = self._model(user, item)
                loss = loss_fn(prediction, label)
                # if wd > 0:
                #     reg_loss = self._model.reg_loss(item, user[:1]*0, scale_item_reg=1/scale_lr_item_emb)
                #     loss += reg_loss * wd * 0.5
                loss.backward()
                optimizer.step()

                count_example += 1
                total_loss += loss.item()
            total_loss /= count_example
            loss_hist.append(total_loss)

        with torch.no_grad():
            if self._model.is_lora:
                lora_comp = (self._model.embed_item_GMF.lora_A @ self._model.embed_item_GMF.lora_B) * self._model.embed_item_GMF.lora_scaling
                item_emb_weight = self._model.embed_item_GMF.weight.data + lora_comp
            else:
                item_emb_weight = self._model.embed_item_GMF.weight.data
            item_emb = item_emb_weight[item]
            item_avg_norm = item_emb.norm(dim=1).mean().item() 
            if mask_zero_user_index:
                user_emb = self._model.embed_user_GMF.weight[0]
                user_norm = user_emb.norm().item()
            else:
                user_norm = self._model.embed_user_GMF.weight[user].norm(dim=1).mean().item()
        # raise RuntimeError()
        return {
            "loss": loss_hist,
            "item_avg_norm": item_avg_norm,
            "user_norm": user_norm
        }