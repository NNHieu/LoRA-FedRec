from typing import List, Any, Tuple
import hydra
import torch.nn
import os
import pandas as pd
import numpy as np
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader
import copy
import tqdm
import logging
import rec
from stats import TimeStats, log_hyperparameters
import torch.nn.functional as F
import fedlib

import wandb

os.environ['EXP_DIR'] = str(Path.cwd())

class Logger():
    def __init__(self, cfg, model, wandb=True) -> None:
        self.wandb = wandb
        if wandb:
            self.run = self.init_wandb(cfg, model)
        self.hist = []

    def log(self, log_dict):
        if self.wandb:
            wandb.log(log_dict)
        self.hist.append(log_dict)
        logging.info(log_dict)
        
    def finish(self, **kwargs):
        if self.wandb:
            self.run.finish(**kwargs)
        # pca_var_df = pd.DataFrame(data=server._timestats._pca_vars)
        hist_df = pd.DataFrame(self.hist)
        
        return hist_df


    @classmethod
    def init_wandb(cls, cfg, model):
        hparams = log_hyperparameters({"cfg": cfg, "model": model, "trainer": None})
        # start a new wandb run to track this script
        run = wandb.init(
            # set the wandb project where this run will be logged
            project="lowrank-fedrec",
            
            # track hyperparameters and run metadata
            config=hparams,
            reinit=True
        )
        return run

def run_server(
    cfg,
) -> pd.DataFrame:

    ############################## PREPARE DATASET ##########################
    feddm = fedlib.data.FedDataModule(cfg)
    feddm.setup()
    num_items = feddm.num_items
    num_users = feddm.num_users
    test_loader = feddm.test_dataloader()
    logging.info("Num users: %d" % num_users)
    logging.info("Num items: %d" % num_items)
    
    # define server side model
    logging.info("Init model")
    model = hydra.utils.instantiate(cfg.net.init_params, item_num=num_items)
    mylogger = Logger(cfg, model, wandb=cfg.TRAIN.wandb)
    
    model.to(cfg.TRAIN.device)

    logging.info("Init clients")
    clients = fedlib.server.initialize_clients(cfg, model, num_users)
    logging.info("Init server")
    server = fedlib.server.SimpleServer(clients, cfg, model, feddm)

    for epoch in range(cfg.FED.aggregation_epochs):
        log_dict = {"epoch": epoch}
        logging.info(f"Aggregation Epoch: {epoch}")
        log_dict.update(server.train_round(epoch_idx=epoch))
        logging.info("Evaluate model")
        test_metrics = server.evaluate(test_loader)
        log_dict.update(test_metrics)
        log_dict.update(server._timestats._time_dict)
        server._timestats.reset()
        mylogger.log(log_dict)
    hist_df = mylogger.finish(quiet=True)
    pca_var_df = pd.DataFrame(data=server._timestats._pca_vars)
    return hist_df, pca_var_df

@hydra.main(config_path=str(Path.cwd() / 'configs'), config_name='fedtrain.yaml', version_base="1.2")
def main(cfg):
    logging.info(cfg)
    out_dir = Path(cfg.paths.output_dir)
    hist_df, pca_var_df = run_server(cfg)
    hist_df.to_csv(out_dir / "hist.csv", index=False)
    pca_var_df.to_csv(out_dir / "pca_var.csv", index=False)

if __name__ == '__main__':
    main()