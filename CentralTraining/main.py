import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import model
import config
import evaluate
import data_utils
from tqdm import tqdm
import data


args = config.get_parser().parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
# cudnn.benchmark = True
device = "cuda:0"


############################## PREPARE DATASET ##########################
# train_dataset = data_utils.MovieLen1MDataset(config.main_path, train=True, num_negatives=args.num_ng)
# test_dataset = data_utils.MovieLen1MDataset(config.main_path, train=False)
# train_dataset = data_utils.LastFM2kDataset(config.main_path, train=True, num_negatives=args.num_ng)
# test_dataset = data_utils.LastFM2kDataset(config.main_path, train=False)
# train_dataset = data_utils.AmazonVideoDataset(config.main_path, train=True, num_negatives=args.num_ng)
# test_dataset = data_utils.AmazonVideoDataset(config.main_path, train=False)
# train_dataset = data_utils.DoubanMovieDataset(config.main_path, train=True, num_negatives=args.num_ng)
# test_dataset = data_utils.DoubanMovieDataset(config.main_path, train=False)
from omegaconf import DictConfig

data_cfg = DictConfig({'DATA': { 'name': 'pinterest',
								'num_negatives': 4,
								'root': "../dataset",
								'test_num_ng': 99},
						'DATALOADER': {
							'batch_size': 256, 'num_workers': 0, 'shuffle': True
						}})

dm = data.get_datamodule(data_cfg)
dm.setup()
# train_dataset = dm.train_dataset()
test_dataset = dm.test_dataset()

# train_dataset = data_utils.PinterestDataset(config.main_path, train=True, num_negatives=args.num_ng)
# test_dataset = data_utils.PinterestDataset(config.main_path, train=False)
# train_loader = data.DataLoader(train_dataset,
# 		batch_size=args.batch_size, shuffle=True, num_workers=4)
# test_loader = data.DataLoader(test_dataset,
# 		batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)

# num_users = train_dataset.num_users
# num_items = train_dataset.num_items

num_users = dm.num_users
num_items = dm.num_items

########################### CREATE MODEL #################################
if config.model == 'NeuMF-pre':
	assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
	assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
	GMF_model = torch.load(config.GMF_model_path)
	MLP_model = torch.load(config.MLP_model_path)
else:
	GMF_model = None
	MLP_model = None



print("Num users", num_users)
print("Num items", num_items)
# model = model.NCF(num_users, num_items, args.factor_num, args.num_layers, 
# 						args.dropout, config.model, GMF_model, MLP_model)
model = model.MF(num_users, num_items, 64)
print("Using device", device)
model.to(device)	
loss_function = nn.BCEWithLogitsLoss(reduction='sum')

# optimizer = optim.Adam(model.parameters(), lr=args.lr)
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)

def log_time(fn):
	start = time.time()
	result = fn()
	return result, time.time() - start

########################### TRAINING #####################################
count, best_hr = 0, 0
for epoch in range(args.epochs):
	model.train() # Enable dropout (if have).
	start_time = time.time()
	# _, sample_time = log_time(lambda : train_loader.dataset.sample_negatives())
	# train_dataset.sample_negatives()
	train_dataset = dm.train_dataset()
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
	total_loss = 0
	count = 0
	for batch_idx, (user, item, label) in tqdm(enumerate(train_loader), total=len(train_loader), leave=True, disable=False):
		user = user.to(device)
		item = item.to(device)
		label = label.float().to(device)

		optimizer.zero_grad()
		prediction = model(user, item)
		loss = loss_function(prediction, label)
		loss.backward()
		optimizer.step()
		# writer.add_scalar('data/loss', loss.item(), count)
		total_loss += loss.item()
		count += 1
	

	model.eval()
	with torch.no_grad():
		HR, NDCG = evaluate.metrics(model, test_loader, args.top_k, device=device)

	elapsed_time = time.time() - start_time
	print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
			time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
	print("HR: {:.3f}\tNDCG: {:.3f}\tLoss: {:.4f}".format(np.mean(HR), np.mean(NDCG), total_loss/count))

	if HR > best_hr:
		best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
		if args.out:
			if not os.path.exists(config.model_path):
				os.mkdir(config.model_path)
			torch.save(model, 
				'{}{}.pth'.format(config.model_path, config.model))

print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
									best_epoch, best_hr, best_ndcg))
