from typing import Dict, List, Tuple
import random
from fedlib.client import Client
from multiprocessing import Process, Queue
import pickle
from .param_tree import tree_add_, tree_div_, tree_zeros_like

def _prepare_dataloader(participants, pid, n_workers, queue):
    i = pid
    step_size = n_workers
    n_participants = len(participants)
    while True:
        client_permuted_index = i % n_participants
        client = participants[client_permuted_index]
        # print(f'Preparing client {client.cid}')
        train_loader = client.prepare_dataloader_mp(None)
        train_loader = pickle.dumps(train_loader)
        queue.put((client_permuted_index, client.cid, train_loader))
        del train_loader   # save memory
        i += step_size

    # for cid in range(pid, len(participants), n_workers):
    #     client = participants[cid]
    #     # print(f'Preparing client {client.cid}')
    #     train_loader = client.prepare_dataloader_mp(None)
    #     queue.put(train_loader)

class ClientSampler:
    def __init__(self, num_users, n_workers=1) -> None:
        # self._client_set = client_set
        self.num_users = num_users
        self._round_count = 0
        self._client_count = 0
        self._n_workers = 1 # Currently only support 1 worker
    
    def initialize_clients(self, model, dm, loss_fn, shuffle_seed, reinit=True, central_train=False) -> None:
        """
        creates `Client` instance for each `client_id` in dataset
        :param cfg: configuration dict
        :return: list of `Client` objects
        """
        clients = list()
        for client_id in range(self.num_users):
            c = Client(client_id, model=model, datamodule=dm, loss_fn=loss_fn, central_train=central_train)
            if reinit:
                model._reinit_private_params()
            clients.append(c)
        self._client_set = clients
        self._suffle_client_set(shuffle_seed)

    def _suffle_client_set(self, seed):
        random.seed(seed)
        random.shuffle(self._client_set)
        self.sorted_client_set = sorted(self._client_set, key=lambda t: t.cid)

    def next_round(self, num_clients) -> List[Client]:
        participants = self._client_set[:num_clients]
        # rotate the list by `num_clients`
        self._client_set =  self._client_set[num_clients:] + participants
        # self._client_count += num_clients
        self._round_count += 1

        total_ds_sizes = 0
        for i in range(len(participants)):
            worker_id = self._client_count % self._n_workers
            client_permuted_index, cid, train_loader = self.queue[worker_id].get()
            assert participants[i].cid == cid
            train_loader = pickle.loads(train_loader)
            participants[i].train_loader = train_loader
            total_ds_sizes += len(train_loader.dataset)
            self._client_count += 1
            # yield participants[i]
        return participants, total_ds_sizes

    def prepare_dataloader(self, n_clients_per_round) -> None:
        self.processors = []
        self._n_workers = 1
        self.queue = [Queue(maxsize=n_clients_per_round) for _ in range(self._n_workers)]
        # for i in range(self._n_workers):
        for i in range(self._n_workers):
            print(f'Starting worker {i}')
            process = Process(
                target=_prepare_dataloader,
                args=(self._client_set, i, self._n_workers, self.queue[i])
            )
            self.processors.append(process)
            process.daemon = True
            process.start()
            # process.join()
        # total_ds_sizes = 0
        # for i in range(len(participants)):
        #     train_loader = queue.get()
        #     participants[i].train_loader = train_loader
        #     total_ds_sizes += len(train_loader.dataset)
    
    def close(self):
        for process in self.processors:
            process.terminate()
            process.join()

class AvgAggregator:
    def __init__(self, sample_param_tree: dict, strategy='fedavg') -> None:
        self.aggregated_param_tree = tree_zeros_like(sample_param_tree)
        self.count = 0.
        self.strategy = strategy
    
    def collect(self, param_tree: dict, weight=1):
        if 'private_inter_mask' in param_tree:
            interaction_mask = param_tree['private_inter_mask']
        else:
            interaction_mask = None

        if self.strategy == 'fedavg':
            tree_add_(self.aggregated_param_tree, param_tree, interaction_mask, weight)
            self.count += weight
        elif self.strategy == 'simpleavg':
            tree_add_(self.aggregated_param_tree, param_tree, interaction_mask, 1.)
            self.count += 1.
        else:
            raise NotImplementedError(f'Aggregation strategy {self.strategy} not implemented')

    def finallize(self):
        if 'private_inter_mask' in self.aggregated_param_tree:
            interaction_mask = self.aggregated_param_tree['private_inter_mask']
        else:
            interaction_mask = None
        tree_div_(self.aggregated_param_tree, interaction_mask, self.count)
        return self.aggregated_param_tree