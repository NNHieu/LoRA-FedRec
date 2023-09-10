from typing import List, Any, Tuple
import hydra
from omegaconf import OmegaConf
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
from stats import TimeStats, Logger
import torch.nn.functional as F
import fedlib
from fedlib.comm import ClientSampler

import wandb

os.environ['EXP_DIR'] = str(Path.cwd())

def run_server(
    cfg,
) -> pd.DataFrame:

    ############################## PREPARE DATASET ##########################
    feddm = fedlib.data.FedDataModule(cfg)
    feddm.setup()
    num_items = feddm.num_items
    num_users = feddm.num_users
    all_train_loader = feddm.train_dataloader()
    test_loader = feddm.test_dataloader()
    logging.info("Num users: %d" % num_users)
    logging.info("Num items: %d" % num_items)
    
    # define server side model
    logging.info("Init model")
    model = hydra.utils.instantiate(cfg.net.init, item_num=num_items)
    mylogger = Logger(cfg, model, wandb=cfg.TRAIN.wandb)
    
    model.to(cfg.TRAIN.device)

    logging.info("Init clients")
    client_sampler = ClientSampler(feddm.num_users)
    client_sampler.initialize_clients(model, feddm, shuffle_seed=42)
    logging.info("Init server")
    server = fedlib.server.SimpleServer(cfg, model, client_sampler)

    for epoch in range(cfg.FED.agg_epochs):
        log_dict = {"epoch": epoch}
        logging.info(f"Aggregation Epoch: {epoch}")
        log_dict.update(server.train_round(epoch_idx=epoch))
        logging.info("Evaluate model")
        if (epoch % cfg.EVAL.interval == 0) or (epoch == cfg.FED.agg_epochs - 1):
            test_metrics = server.evaluate(test_loader, all_train_loader)
            log_dict.update(test_metrics)
        log_dict.update(server._timestats._time_dict)
        mylogger.log(log_dict)
        logging.info(server._timestats._pca_vars)
        server._timestats.reset()
    hist_df = mylogger.finish(quiet=True)
    pca_var_df = pd.DataFrame(data=server._timestats._pca_vars)
    return hist_df, pca_var_df

@hydra.main(config_path=str(Path.cwd() / 'configs'), config_name='fedtrain.yaml', version_base="1.2")
def main(cfg):
    OmegaConf.resolve(cfg)
    logging.info(cfg)
    out_dir = Path(cfg.paths.output_dir)
    hist_df, pca_var_df = run_server(cfg)
    hist_df.to_csv(out_dir / "hist.csv", index=False)
    pca_var_df.to_csv(out_dir / "pca_var.csv", index=False)

if __name__ == '__main__':
    main()