# python fedtrain.py data=ml-1m net=fedncf64-lora-fb net.init.lora_rank=1 task_name=tuning FED.agg_epochs=1000 TRAIN.log_interval=100 EVAL.interval=100 DATALOADER.batch_size=32 FED.local_epochs=1 TRAIN.lr=0.1 TRAIN.weight_decay=0 FED.num_clients=60 EXP.project=lfedrec-ncf-v4 TRAIN.wandb=True
# python fedtrain.py data=ml-1m net=fedncf64-lora-fb net.init.lora_rank=2 task_name=tuning FED.agg_epochs=1000 TRAIN.log_interval=100 EVAL.interval=100 DATALOADER.batch_size=32 FED.local_epochs=1 TRAIN.lr=0.1 TRAIN.weight_decay=0 FED.num_clients=60 EXP.project=lfedrec-ncf-v4 TRAIN.wandb=True
# python fedtrain.py data=ml-1m net=fedncf64-lora-fb net.init.lora_rank=4 task_name=tuning FED.agg_epochs=1000 TRAIN.log_interval=100 EVAL.interval=100 DATALOADER.batch_size=32 FED.local_epochs=1 TRAIN.lr=0.1 TRAIN.weight_decay=0 FED.num_clients=60 EXP.project=lfedrec-ncf-v4 TRAIN.wandb=True
# python fedtrain.py data=ml-1m net=fedncf64-lora-fb net.init.lora_rank=8 task_name=tuning FED.agg_epochs=1000 TRAIN.log_interval=100 EVAL.interval=100 DATALOADER.batch_size=32 FED.local_epochs=1 TRAIN.lr=0.1 TRAIN.weight_decay=0 FED.num_clients=60 EXP.project=lfedrec-ncf-v4 TRAIN.wandb=True
# python fedtrain.py data=ml-1m net=fedncf64-lora-fb net.init.lora_rank=16 task_name=tuning FED.agg_epochs=1000 TRAIN.log_interval=100 EVAL.interval=100 DATALOADER.batch_size=32 FED.local_epochs=1 TRAIN.lr=0.1 TRAIN.weight_decay=0 FED.num_clients=60 EXP.project=lfedrec-ncf-v4 TRAIN.wandb=True
# python fedtrain.py data=ml-1m net=fedncf64-lora-fb net.init.lora_rank=32 task_name=tuning FED.agg_epochs=1000 TRAIN.log_interval=100 EVAL.interval=100 DATALOADER.batch_size=32 FED.local_epochs=1 TRAIN.lr=0.1 TRAIN.weight_decay=0 FED.num_clients=60 EXP.project=lfedrec-ncf-v4 TRAIN.wandb=True

# python fedtrain.py data=ml-1m net=fedncf64-lora-on-fb net.init.lora_rank=1 task_name=tuning FED.agg_epochs=1000 TRAIN.log_interval=100 EVAL.interval=100 DATALOADER.batch_size=32 FED.local_epochs=1 TRAIN.lr=0.1 TRAIN.weight_decay=0 FED.num_clients=60 EXP.project=lfedrec-ncf-v4 TRAIN.wandb=True
# python fedtrain.py data=ml-1m net=fedncf64-lora-on-fb net.init.lora_rank=2 task_name=tuning FED.agg_epochs=1000 TRAIN.log_interval=100 EVAL.interval=100 DATALOADER.batch_size=32 FED.local_epochs=1 TRAIN.lr=0.1 TRAIN.weight_decay=0 FED.num_clients=60 EXP.project=lfedrec-ncf-v4 TRAIN.wandb=True
# python fedtrain.py data=ml-1m net=fedncf64-lora-on-fb net.init.lora_rank=4 task_name=tuning FED.agg_epochs=1000 TRAIN.log_interval=100 EVAL.interval=100 DATALOADER.batch_size=32 FED.local_epochs=1 TRAIN.lr=0.1 TRAIN.weight_decay=0 FED.num_clients=60 EXP.project=lfedrec-ncf-v4 TRAIN.wandb=True
# python fedtrain.py data=ml-1m net=fedncf64-lora-on-fb net.init.lora_rank=8 task_name=tuning FED.agg_epochs=1000 TRAIN.log_interval=100 EVAL.interval=100 DATALOADER.batch_size=32 FED.local_epochs=1 TRAIN.lr=0.1 TRAIN.weight_decay=0 FED.num_clients=60 EXP.project=lfedrec-ncf-v4 TRAIN.wandb=True
# python fedtrain.py data=ml-1m net=fedncf64-lora-on-fb net.init.lora_rank=16 task_name=tuning FED.agg_epochs=1000 TRAIN.log_interval=100 EVAL.interval=100 DATALOADER.batch_size=32 FED.local_epochs=1 TRAIN.lr=0.1 TRAIN.weight_decay=0 FED.num_clients=60 EXP.project=lfedrec-ncf-v4 TRAIN.wandb=True
# python fedtrain.py data=ml-1m net=fedncf64-lora-on-fb net.init.lora_rank=32 task_name=tuning FED.agg_epochs=1000 TRAIN.log_interval=100 EVAL.interval=100 DATALOADER.batch_size=32 FED.local_epochs=1 TRAIN.lr=0.1 TRAIN.weight_decay=0 FED.num_clients=60 EXP.project=lfedrec-ncf-v4 TRAIN.wandb=True


python fedtrain.py data=ml-1m net=fedncf64 task_name=init_small_0.05d FED.agg_epochs=1000 TRAIN.log_interval=50 EVAL.interval=50 DATALOADER.batch_size=32 FED.local_epochs=1 TRAIN.lr=0.1 TRAIN.weight_decay=0 FED.num_clients=60 EXP.project=lfedrec-ncf-v10 TRAIN.wandb=True
# python fedtrain.py data=ml-1m net=fedncf32 task_name=init_small_0.05d FED.agg_epochs=1000 TRAIN.log_interval=50 EVAL.interval=50 DATALOADER.batch_size=32 FED.local_epochs=1 TRAIN.lr=0.1 TRAIN.weight_decay=0 FED.num_clients=60 EXP.project=lfedrec-ncf-v9 TRAIN.wandb=True
# python fedtrain.py data=ml-1m net=fedncf16 task_name=init_small_0.05d FED.agg_epochs=1000 TRAIN.log_interval=50 EVAL.interval=50 DATALOADER.batch_size=32 FED.local_epochs=1 TRAIN.lr=0.1 TRAIN.weight_decay=0 FED.num_clients=60 EXP.project=lfedrec-ncf-v9 TRAIN.wandb=True
# python fedtrain.py data=ml-1m net=fedncf8 task_name=init_small_0.05d FED.agg_epochs=1000 TRAIN.log_interval=50 EVAL.interval=50 DATALOADER.batch_size=32 FED.local_epochs=1 TRAIN.lr=0.1 TRAIN.weight_decay=0 FED.num_clients=60 EXP.project=lfedrec-ncf-v9 TRAIN.wandb=True

python fedtrain.py data=ml-1m net=fedncf64-lora-on-fb net.init.lora_rank=16 task_name=tuning FED.agg_epochs=1000 TRAIN.log_interval=50 EVAL.interval=50 DATALOADER.batch_size=32 FED.local_epochs=1 TRAIN.lr=0.1 TRAIN.weight_decay=0 FED.num_clients=60 EXP.project=lfedrec-ncf-v10 TRAIN.wandb=True
# python fedtrain.py data=ml-1m net=fedncf64-lora-on-fb net.init.lora_rank=32 task_name=tuning FED.agg_epochs=1000 TRAIN.log_interval=50 EVAL.interval=50 DATALOADER.batch_size=32 FED.local_epochs=1 TRAIN.lr=0.1 TRAIN.weight_decay=0 FED.num_clients=60 EXP.project=lfedrec-ncf-v9 TRAIN.wandb=True
# python fedtrain.py data=ml-1m net=fedncf64-lora-on-fb net.init.lora_rank=8 task_name=tuning FED.agg_epochs=1000 TRAIN.log_interval=50 EVAL.interval=50 DATALOADER.batch_size=32 FED.local_epochs=1 TRAIN.lr=0.1 TRAIN.weight_decay=0 FED.num_clients=60 EXP.project=lfedrec-ncf-v9 TRAIN.wandb=True
# python fedtrain.py data=ml-1m net=fedncf64-lora-on-fb net.init.lora_rank=4 task_name=tuning FED.agg_epochs=1000 TRAIN.log_interval=50 EVAL.interval=50 DATALOADER.batch_size=32 FED.local_epochs=1 TRAIN.lr=0.1 TRAIN.weight_decay=0 FED.num_clients=60 EXP.project=lfedrec-ncf-v9 TRAIN.wandb=True

