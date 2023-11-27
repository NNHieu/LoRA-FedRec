from typing import List, Tuple
from .client import Client, Client
import torch.nn
import numpy as np
import random
import torch
import tqdm
import logging
import rec.evaluate as evaluate
from stats import TimeStats
from fedlib.comm import AvgAggregator, ClientSampler
import time
import optree
from .param_tree import treedict2statedict

class SimpleServer:
    def __init__(self, cfg, model, client_sampler: ClientSampler):
        self.cfg = cfg
        self.client_sampler = client_sampler
        self.model = model
        self._timestats = TimeStats()
    
    def _step_server_optim(self, delta_params):
        # with self._timestats.timer("server_compress_time"):
        #     delta_params.compress(**self.cfg.FED.compression_kwargs)
        # with self._timestats.timer("server_decompress_time"):
        #     delta_params.decompress()
        with self._timestats.timer("server_time"):
            optree.tree_map_(lambda x, y: x.add_(y), self.server_params, delta_params)
            # self.server_params.server_step_(delta_params)
        with self._timestats.timer("server_unroleAB_time"):
            self.model._set_state_from_splited_params([self._dummy_private_params, treedict2statedict(self.server_params)])
    
    def prepare(self):
        # reinit
        if self.cfg.net.get('server_prepare_kwargs', None) is not None:
            self.model.server_prepare(**self.cfg.net.server_prepare_kwargs)
        else:
            self.model.server_prepare(init_B_strategy='random')
        self._dummy_private_params, self.server_params = self.model._get_splited_params(server_init=True)

    def train_round(self, epoch_idx: int = 0):
        '''
        Flow:
        1. Sample clients & Prepare dataloader for each client
        2. Prepare parameter
        3. Train each client
        4. Aggregate updates
        5. Update server model

        '''

        # 1. Sample clients
        with self._timestats.timer("sampling_clients"):
            participants, all_data_size = self.client_sampler.next_round(self.cfg.FED.num_clients)

        # 2. Prepare parameter
        self.prepare()
        aggregator = AvgAggregator(self.server_params, strategy=self.cfg.FED.aggregation)
        
        # 3. Train each client
        total_loss = 0
        self._timestats.set_aggregation_epoch(epoch_idx)
        pbar = tqdm.tqdm(participants, desc='Training', disable=True)
        update_numel = 0
        item_norm = 0
        user_norm = 0
        for client in pbar:
            with self._timestats.timer("client_time", max_agg=True):
                update, data_size, metrics = client.fit(self.server_params, 
                                                        local_epochs=self.cfg.FED.local_epochs, 
                                                        config=self.cfg, 
                                                        device=self.cfg.TRAIN.device, 
                                                        stats_logger=self._timestats,
                                                        mask_zero_user_index=True)
                
            # Monitor the update norm
            update_norms = optree.tree_map(lambda x: x.norm().item() if x is not None else 0., update)
            update_norms_dict = treedict2statedict(update_norms)

                # time.sleep(0.5)
            # print(self._timestats._time_dict, data_size)
            # self._timestats.reset()
            # print(self._timestats._time_dict['set_parameters'], self._timestats._time_dict['get_parameters'], self._timestats._time_dict['fit'])
            # update_norm += torch.linalg.norm((update['embed_item_GMF.lora_A'] @ update['embed_item_GMF.lora_B'])*update["embed_item_GMF.lora_scaling"]).item()
            # update_norm += torch.linalg.norm(update['embed_item_GMF.weight']).item()
            
            # update_numel += sum([t.numel() for t in update.values()])
            update_numel += optree.tree_reduce(lambda x, y: x + y.numel(), update, initial=0)
            with self._timestats.timer("server_time"):
                aggregator.collect(update, weight=(data_size/all_data_size))
            
            client_loss = np.mean(metrics['loss'])
            log_dict = {"client_loss": client_loss}
            total_loss += client_loss
            item_norm += metrics["item_avg_norm"]
            user_norm += metrics["user_norm"]
            pbar.set_postfix(log_dict)
        with self._timestats.timer("server_time"):
            aggregated_update = aggregator.finallize()
        self._step_server_optim(aggregated_update)
        # B_1 = self.server_params['embed_item_GMF.lora_B'].clone()
        # print(torch.linalg.norm(B_1 - B_0))
        # print("update norm", update_norm / len(participants))
        return {"client_loss": total_loss / len(participants), 
                "update_numel": update_numel / len(participants), 
                "data_size": all_data_size,
                "item_norm": item_norm / len(participants), 
                "user_norm": user_norm / len(participants),
                "update_norms_dict": update_norms_dict, }

    
    @torch.no_grad()
    def evaluate(self, val_loader, test_loader, train_loader=None):
        sorted_client_set = self.client_sampler.sorted_client_set
        metrics = {}
        with self._timestats.timer("evaluate"):
            eval_model = self.model.merge_client_params(sorted_client_set, self.server_params, self.model, self.cfg.TRAIN.device)
            if train_loader is not None:
                train_loss = evaluate.cal_loss(eval_model, train_loader, loss_function=torch.nn.BCEWithLogitsLoss(),device=self.cfg.TRAIN.device)
                metrics['train/loss'] = train_loss
            eval_model.eval()
            if test_loader is not None:
                HR, NDCG = evaluate.metrics(eval_model, test_loader, self.cfg.EVAL.topk, device=self.cfg.TRAIN.device)
                metrics['test/HR'] = HR
                metrics['test/NDCG'] = NDCG
            if val_loader is not None:
                HR_val, NDCG_val = evaluate.metrics(eval_model, val_loader, self.cfg.EVAL.topk, device=self.cfg.TRAIN.device)
                metrics['val/HR'] = HR_val
                metrics['val/NDCG'] = NDCG_val
        return metrics
