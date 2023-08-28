import os
import time
import numpy as np
import hydra
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from rec.models import NCF
import rec.evaluate as evaluate
from fedlib.data import FedDataModule
from tqdm import tqdm
import random


# args = config.get_parser().parse_args()
# cfg = config.setup_cfg(args)

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True
device = torch.device("cuda")
os.environ['EXP_DIR'] = str(Path.cwd())

@hydra.main(config_path=str(Path.cwd() / 'configs'), config_name='fedtrain.yaml', version_base="1.2")
def main(cfg):
	print(cfg)
	############################## PREPARE DATASET ##########################
	# train_dataset = FedMovieLen1MDataset(cfg.DATA.root, train=True, num_negatives=cfg.DATA.num_negatives)
	dm = FedDataModule(cfg)
	dm.setup()


	# test_dataset = dm.test_dataloader
	# train_loader = data.DataLoader(train_dataset,
	# 		batch_size=cfg.DATALOADER.batch_size, shuffle=True, num_workers=0)
	test_loader = dm.test_dataloader()

	########################### CREATE MODEL #################################

	print("Num users", dm.num_users)
	print("Num items", dm.num_items)
	# model = NCF(dm.num_users, dm.num_items,
	# 			gmf_emb_size=cfg.MODEL.gmf_emb_size,
	# 			mlp_emb_size=cfg.MODEL.mlp_emb_size,
	# 			mlp_layer_dims=cfg.MODEL.mlp_layer_dims,
	# 			dropout=cfg.MODEL.dropout,)
	model = hydra.utils.instantiate(cfg.net.init_params, item_num=dm.num_items, user_num=dm.num_users)
	print(model)
	model.to(device)	
	loss_function = nn.BCEWithLogitsLoss()


	# writer = SummaryWriter() # for visualization

	def log_time(fn):
		start = time.time()
		result = fn()
		return result, time.time() - start

	########################### TRAINING #####################################
	count, best_hr = 0, 0
	pbar = tqdm(range(cfg.FED.aggregation_epochs))
	client_set = list(range(dm.num_users))
	random.shuffle(client_set)
	for epoch in pbar:
		# _, sample_time = log_time(lambda : d.sample_negatives())
		# print("Sampling neg time", sample_time)
		
		client_losses = []
		client_sample = client_set[:cfg.FED.num_clients]
		client_set = client_set[cfg.FED.num_clients:] + client_sample

		for uid in tqdm(client_sample, leave=False):
			optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.lr)
			model.train() # Enable dropout (if have).
			train_loader = dm.train_dataloader(uid)
			
			# _, sample_time = log_time(lambda : train_loader.dataset.ng_sample())
			total_loss = 0
			for _ in range(1):
				for batch_idx, (user, item, label) in enumerate(train_loader):
					user = user.to(device)
					item = item.to(device)
					label = label.float().to(device)

					optimizer.zero_grad()
					prediction = model(user, item, mask_zero_user_index=False)
					loss = loss_function(prediction, label)
					loss.backward()
					# tmp = model.embed_item_MLP.weight.grad.data.norm()
					# print((model.embed_item_GMF.lora_A.grad.data.abs() > 0).sum(), len(train_loader.dataset))
					# print(tmp)
					optimizer.step()
					# writer.add_scalar('data/loss', loss.item(), count)
					count += 1
					total_loss += loss.item()
			with torch.no_grad():
				# print(model.embed_item_GMF.lora_A.data.norm())
				model._merge_all_lora_weights()
				model._reset_all_lora_weights(to_zero=False)
			total_loss /= len(train_loader)
			client_losses.append(total_loss)


		# elapsed_time = time.time() - start_time
		# print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
		#		time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
		if epoch % cfg.EVAL.every_agg_epochs == 0:
			with torch.no_grad():
				model.eval()
				HR, NDCG = evaluate.metrics(model, test_loader, cfg.EVAL.topk, device=device)
				# print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
				pbar.set_postfix({"HR": np.mean(HR), "NDCG": np.mean(NDCG), "loss": np.mean(client_losses)})

		# if HR > best_hr:
		# 	best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
		# 	if args.out:
		# 		if not os.path.exists(config.model_path):
		# 			os.mkdir(config.model_path)
		# 		torch.save(model, 
		# 			'{}{}.pth'.format(config.model_path, config.model))

	# print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
	# 									best_epoch, best_hr, best_ndcg))
main()