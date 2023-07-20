from typing import List, Any, Tuple
from client import Client, NCFClient
import torch.nn
import os
import pandas as pd
import numpy as np
from pathlib import Path
import collections
import random
import torch
from torch.utils.data import DataLoader
import copy
import tqdm
import logging
from models import FedNCFModel
from data import FedMovieLen1MDataset
import evaluate
from stats import TimeStats

class SimpleServer:
    def __init__(self, clients: List[Client], cfg, model: FedNCFModel, train_dataset: FedMovieLen1MDataset):
        self.client_set = clients
        self.model = model
        _, self.server_params = self.model._get_splited_params()
        self.cfg = cfg
        self.train_dataset = train_dataset
        self._circulated_client_count = 0
        self._timestats = TimeStats()

    def sample_clients(
        self,
    ) -> Tuple[List[Client], List[Client]]:
        """
        :param clients: list of all available clients
        :param num_clients: number of clients to sample

        sample `num_clients` clients and return along with their respective data
        """
        num_clients = self.cfg.FED.num_clients
        sample = self.client_set[:num_clients]
        # rotate the list by `num_clients`
        self.client_set =  self.client_set[num_clients:] + sample

        self._circulated_client_count += num_clients
        if self._circulated_client_count >= len(self.client_set):
            print("Resample negative items")
            self.train_dataset.sample_negatives()
            self._circulated_client_count = 0

        return sample

    def train_round(self):
        participants: List[Client] = self.sample_clients()
        aggregated_weights = [np.zeros_like(p) for p in self.server_params['weights']]
        pbar = tqdm.tqdm(participants, desc='Training')
        for client in pbar:
            self.train_dataset.set_client(client.cid)
            train_loader = DataLoader(self.train_dataset, **self.cfg.DATALOADER)
            client_params, data_size, metrics = client.fit(train_loader, self.server_params, self.cfg, self.cfg.TRAIN.device, self._timestats)

            aggregated_weights = [(p0 + p1) for p0, p1 in zip(aggregated_weights, client_params['weights'])]
            log_dict = {"loss": np.mean(metrics['loss'])}
            pbar.set_postfix(log_dict)
        aggregated_weights = [p / len(participants) for p in aggregated_weights]
        self.server_params['weights'] = aggregated_weights

    
    @torch.no_grad()
    def evaluate(self, test_loader):
        self._timestats.mark_start("evaluate")
        client_weights = [c._private_params['weights'] for c in self.client_set]
        client_weights = list(zip(*client_weights))
        client_weights = [torch.tensor(np.concatenate(w, axis=0)).to(cfg.TRAIN.device) for w in client_weights]
        client_weights = {k: v for k,v in zip(self.client_set[0]._private_params['keys'], client_weights)}
        eval_model = copy.deepcopy(self.model)
        eval_model._set_state_from_splited_params([self.client_set[0]._private_params, self.server_params])
        eval_model.embed_user_GMF = torch.nn.Embedding.from_pretrained(client_weights['embed_user_GMF.weight'])
        eval_model.embed_user_MLP = torch.nn.Embedding.from_pretrained(client_weights['embed_user_MLP.weight'])
        # evaluate the model
        eval_model.eval()
        HR, NDCG = evaluate.metrics(eval_model, test_loader, cfg.EVAL.topk, device=self.cfg.TRAIN.device)
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
        self._timestats.mark_end("evaluate")

def run_server(
    cfg,
) -> torch.nn.Module:
    """
    defines server side ncf model and initiates the training process
    saves the trained model at indicated path
    evaluates the trained model and prints results

    :param dataset: dataset class to load train/test data
    :param num_clients: number of clients to sample during each global training epoch
    :param epochs: number of global training epochs
    :param path: path where pretrained model is stored
    :param save: boolean parameter which indicates if trained model should be stored in `path`
    :param local_epochs: number local training epochs
    :param learning_rate: learning rate for client models for local training
    :return: trained server model
    """

    ############################## PREPARE DATASET ##########################
    train_dataset = FedMovieLen1MDataset(cfg.DATA.root, train=True, num_negatives=cfg.DATA.num_negatives)
    test_dataset = FedMovieLen1MDataset(cfg.DATA.root, train=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.DATA.test_num_ng+1, shuffle=False, num_workers=0)
    # define server side model
    print("Init model")
    model = FedNCFModel(
        train_dataset.num_items,
        factor_num=cfg.MODEL.factor_num,
        num_layers=cfg.MODEL.num_layers,
        dropout=cfg.MODEL.dropout,
        model="NCF-pre"
        # use_lora=cfg.MODEL.use_lora,
        # lora_r=cfg.MODEL.lora_r,
        # lora_alpha=cfg.MODEL.lora_alpha,
    )
    # summary(server_model, *[torch.LongTensor((1,1)), torch.LongTensor((1,1)), None], layer_modules=(lora.Embedding, torch.nn.Parameter))
    model.to(cfg.TRAIN.device)
    print("Init clients")
    clients = initialize_clients(cfg, model, train_dataset.num_users)
    print("Init server")
    server = SimpleServer(clients, cfg, model, train_dataset)
    for epoch in range(cfg.FED.aggregation_epochs):
        print(f"Aggregation Epoch: {epoch}")
        server.train_round()
        print("Evaluate model")
        server.evaluate(test_loader)
        print(server._timestats)
        server._timestats.reset()


def initialize_clients(cfg, model, num_users) -> List[Client]:
    """
    creates `Client` instance for each `client_id` in dataset
    :param dataset: `Dataset` object to load train data
    :return: list of `Client` objects
    """
    clients = list()
    for client_id in range(num_users):
        c = NCFClient(client_id, model=model)
        model._reinit_private_params()
        clients.append(c)
    return clients


if __name__ == '__main__':
    # from pyrootutils import setup_root
    # setup_root(__file__, ".git", pythonpath=True)

    from config import setup_cfg, get_parser
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    run_server(cfg)
