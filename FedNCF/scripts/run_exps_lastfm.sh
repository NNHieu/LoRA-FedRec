# python server.py --config-file configs/ncf16.yaml  --opt TRAIN.device cpu 
# python server.py --config-file configs/lora4_ncf16.yaml  --opt TRAIN.device cpu
# python server.py --config-file configs/lora8_ncf16.yaml  --opt TRAIN.device cpu DATALOADER.batch_size 128 TRAIN.lr 0.005
# python server.py --config-file configs/lora2_ncf16.yaml  --opt TRAIN.device cpu DATALOADER.batch_size 128 TRAIN.lr 0.005
# python server.py --config-file configs/lora4_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.003
# python server.py --config-file configs/lora8_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.003
# python server.py --config-file configs/lora2_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.003
# python server.py --config-file configs/pinterest/lora4_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.01 FED.num_clients 200
# CUDA_VISIBLE_DEVICES=1 python server.py --config-file configs/pinterest/lora2_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.01 FED.num_clients 200
# CUDA_VISIBLE_DEVICES=1 python server.py --config-file configs/pinterest/lora8_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.01 FED.num_clients 200

# CUDA_VISIBLE_DEVICES=1 python server.py --config-file configs/pinterest/lora2_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.01 FED.num_clients 200
# CUDA_VISIBLE_DEVICES=1 python server.py --config-file configs/pinterest/lora8_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.01 FED.num_clients 200

CUDA_VISIBLE_DEVICES=2 python server.py --config-file configs/lastfm/ncf16.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.003 FED.num_clients 50 # FED.aggregation_epochs 800 FED.local_epochs 4
CUDA_VISIBLE_DEVICES=2 python server.py --config-file configs/lastfm/lora4_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.003 FED.num_clients 50 # FED.aggregation_epochs 800 FED.local_epochs 4
CUDA_VISIBLE_DEVICES=2 python server.py --config-file configs/lastfm/lora2_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.003 FED.num_clients 50 # FED.aggregation_epochs 800 FED.local_epochs 4
CUDA_VISIBLE_DEVICES=2 python server.py --config-file configs/lastfm/lora8_ncf16_freezeB.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.003 FED.num_clients 50 # FED.aggregation_epochs 800 FED.local_epochs 4
# CUDA_VISIBLE_DEVICES=2 python server.py --config-file configs/lastfm/ncf12.yaml  --opt TRAIN.device cuda DATALOADER.batch_size 128 TRAIN.lr 0.003 FED.num_clients 50
