from typing import Dict, List, Any, Optional, Tuple
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
import math

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
    all_train_loader = feddm.train_dataloader(for_eval=True)
    
    val_loader = feddm.val_dataloader()
    test_loader = feddm.test_dataloader()

    logging.info("Num users: %d" % num_users)
    logging.info("Num items: %d" % num_items)
    
    # define server side model
    logging.info("Init model")
    model = hydra.utils.instantiate(cfg.net.init, item_num=num_items)
    mylogger = Logger(cfg, model, wandb=False)
    
    model.to(cfg.TRAIN.device)

    # loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
    loss_fn = torch.nn.BCEWithLogitsLoss()

    logging.info("Init clients")
    client_sampler = ClientSampler(feddm.num_users, n_workers=1)
    client_sampler.initialize_clients(model, feddm, loss_fn=loss_fn, shuffle_seed=42)
    client_sampler.prepare_dataloader(n_clients_per_round=cfg.FED.num_clients*10)
    client_time = 0
    server_time = 0
    try:
        logging.info("Init server")
        server = fedlib.server.SimpleServer(cfg, model, client_sampler)

        for epoch in range(cfg.FED.agg_epochs):
            log_dict = {"epoch": epoch}
            log_dict.update(server.train_round(epoch_idx=epoch))
            time_log = {f"time/{k}": v for k, v in server._timestats._time_dict.items()}
            client_time = client_time + time_log['time/client_time']
            server_time = server_time + time_log['time/server_time']
            # client_time = time_log['time/client_time']
            # server_time = time_log['time/server_time']

            log_dict.update(time_log)
            # if (epoch % cfg.TRAIN.log_interval == 0) or (epoch == cfg.FED.agg_epochs - 1):
            #     mylogger.log(log_dict, term_out=True)
            if cfg.FED.compression_kwargs.method != "none":
                server_time += time_log['time/server_compress_time']
                client_time += time_log['time/server_decompress_time']
                # print('*', time_log['time/server_compress_time'], time_log['time/server_decompress_time'])
            if model.is_lora:
                client_time += time_log['time/server_unroleAB_time']
            server._timestats.reset('client_time', 'server_time', 'server_compress_time', 'server_decompress_time', 'compress', 'server_unroleAB_time')
            # print(epoch, server_time, client_time, time_log['time/compress'], time_log['time/compress']/client_time)
            print(epoch, server_time, client_time, time_log['time/compress'], time_log['time/compress']/client_time, time_log['time/fit'])
        
    except KeyboardInterrupt:
        logging.info("Interrupted")
    except Exception as e:
        logging.exception(e)
    finally:
        client_sampler.close()
        hist_df = mylogger.finish(quiet=True)
        pca_var_df = pd.DataFrame(data=server._timestats._pca_vars)
    return hist_df, pca_var_df, log_dict

@hydra.main(config_path=str(Path.cwd() / 'configs'), config_name='fedtrain.yaml', version_base="1.2")
def main(cfg):
    OmegaConf.resolve(cfg)
    logging.info(cfg)
    out_dir = Path(cfg.paths.output_dir)
    hist_df, pca_var_df, log_dict = run_server(cfg)
    # hist_df.to_csv(out_dir / "hist.csv", index=False)
    # pca_var_df.to_csv(out_dir / "pca_var.csv", index=False)
    
    # safely retrieve metric value for hydra-based hyperparameter optimization
    # metric_value = get_metric_value(
    #     metric_dict=log_dict, metric_name=cfg.get("optimized_metric")
    # )

if __name__ == '__main__':
    main()


# 0 0.0037240982055664062 0.0660710334777832 0 0.0
# 1 0.002226114273071289 0.03639674186706543 0 0.0
# 2 0.002263307571411133 0.03381061553955078 0 0.0
# 3 0.0022072792053222656 0.037168025970458984 0 0.0
# 4 0.002229452133178711 0.03371620178222656 0 0.0
# 5 0.0022284984588623047 0.020023584365844727 0 0.0
# 6 0.0022695064544677734 0.0232546329498291 0 0.0
# 7 0.0023224353790283203 0.02009272575378418 0 0.0
# 8 0.0023000240325927734 0.02769780158996582 0 0.0
# 9 0.002267122268676758 0.026363849639892578 0 0.0

# 0 0.009896278381347656 0.09058690071105957 0.02400493621826172 0.26499345964853177
# 1 0.007302522659301758 0.03931760787963867 0.0021817684173583984 0.055490873809956945
# 2 0.007277965545654297 0.03896760940551758 0.0021779537200927734 0.05589138654690961
# 3 0.007342338562011719 0.041031837463378906 0.0021677017211914062 0.05282975014526438
# 4 0.007292270660400391 0.036443471908569336 0.0021505355834960938 0.05901017303980897
# 5 0.007326602935791016 0.021593809127807617 0.0021457672119140625 0.099369555376445
# 6 0.007237434387207031 0.026180028915405273 0.0020754337310791016 0.07927545602739351
# 7 0.007253170013427734 0.02250385284423828 0.002146005630493164 0.09536169852099843
# 8 0.007332801818847656 0.030155181884765625 0.002064228057861328 0.06845351043643264
# 9 0.0072591304779052734 0.02795267105102539 0.0020639896392822266 0.07383872673615258

# 0 0.004796504974365234 0.07875490188598633 0 0.0
# 1 0.0033714771270751953 0.04137372970581055 0 0.0
# 2 0.003372669219970703 0.04038667678833008 0 0.0
# 3 0.0033609867095947266 0.04482865333557129 0 0.0
# 4 0.003363370895385742 0.03804421424865723 0 0.0
# 5 0.003358602523803711 0.023171424865722656 0 0.0
# 6 0.003450155258178711 0.026766061782836914 0 0.0
# 7 0.0033860206604003906 0.022943496704101562 0 0.0
# 8 0.0034744739532470703 0.03199124336242676 0 0.0
# 9 0.0033936500549316406 0.02987217903137207 0 0.0

# 499 2.5974018573760986 9.9449942111969 0 0.0
# 499 5.979083776473999 10.956307172775269 0.001262664794921875 0.00011524547231200327