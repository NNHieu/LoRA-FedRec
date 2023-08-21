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
from rec.data import MovieLen1MDataset, FedMovieLen1MDataset
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
	train_dataset = FedMovieLen1MDataset(cfg.DATA.root, train=True, num_negatives=cfg.DATA.num_negatives)

	test_dataset = MovieLen1MDataset(cfg.DATA.root, train=False)
	# train_loader = data.DataLoader(train_dataset,
	# 		batch_size=cfg.DATALOADER.batch_size, shuffle=True, num_workers=0)
	test_loader = data.DataLoader(test_dataset,
			batch_size=2048, shuffle=False, num_workers=0)

	########################### CREATE MODEL #################################

	print("Num users", train_dataset.num_users)
	print("Num items", train_dataset.num_items)
	model = NCF(train_dataset.num_users, train_dataset.num_items,
				gmf_emb_size=cfg.MODEL.gmf_emb_size,
				mlp_emb_size=cfg.MODEL.mlp_emb_size,
				mlp_layer_dims=cfg.MODEL.mlp_layer_dims,
				dropout=cfg.MODEL.dropout,)
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
	client_set = list(range(train_dataset.num_users))
	random.shuffle(client_set)
	for epoch in pbar:
		_, sample_time = log_time(lambda : train_dataset.sample_negatives())
		# print("Sampling neg time", sample_time)
		
		client_losses = []
		client_sample = client_set[:cfg.FED.num_clients]
		client_set = client_set[cfg.FED.num_clients:] + client_sample

		for uid in client_sample:
			optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.lr)
			model.train() # Enable dropout (if have).
			train_dataset.set_client(uid)
			train_loader = data.DataLoader(train_dataset,
				batch_size=cfg.DATALOADER.batch_size, shuffle=True, num_workers=0)
			
			# _, sample_time = log_time(lambda : train_loader.dataset.ng_sample())
			total_loss = 0
			for batch_idx, (user, item, label) in enumerate(train_loader):
				user = user.to(device)
				item = item.to(device)
				label = label.float().to(device)

				optimizer.zero_grad()
				prediction = model(user, item)
				loss = loss_function(prediction, label)
				loss.backward()
				# tmp = model.embed_item_MLP.weight.grad.data.norm()
				# print(tmp)
				optimizer.step()
				# writer.add_scalar('data/loss', loss.item(), count)
				count += 1
				total_loss += loss.item()
			total_loss /= len(train_loader)
			client_losses.append(total_loss)


		# elapsed_time = time.time() - start_time
		# print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
		#		time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
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