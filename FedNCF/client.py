from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from abc import ABC, abstractmethod
from models import FedNCFModel

class Client(ABC):
    @abstractmethod
    def get_parameters(self, config):
        pass

    @abstractmethod
    def set_parameters(self, config):
        pass

    @abstractmethod
    def fit(self, train_loader: DataLoader,  parameters: List[np.ndarray], config: Dict[str, str], device):
        pass

class NCFClient(Client):
    """Flower client implementing CIFAR-10 image classification using
    PyTorch."""

    def __init__(
        self,
        cid,
        model: FedNCFModel,
    ) -> None:
        self._cid = cid
        self._model = model
        self._private_params = self._model._get_splited_params()[0]

    @property
    def cid(self):
        return self._cid

    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        private_params, sharable_params = self._model._get_splited_params()
        # print(self._private_params['keys'])
        # print("Update on private params", private_params['weights'][0].shape, np.linalg.norm(private_params['weights'][1] - self._private_params['weights'][1]))
        # print("Update on private params", private_params['weights'][1].shape, np.linalg.norm(private_params['weights'][0] - self._private_params['weights'][0]))
        self._private_params = private_params

        return sharable_params

    def set_parameters(self, global_params: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self._model._set_state_from_splited_params([self._private_params, global_params])

    def fit(
        self, train_loader: DataLoader,  parameters: List[np.ndarray], config: Dict[str, str], device, timestats
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        
        timestats.mark_start("set_parameters")
        self.set_parameters(parameters)
        timestats.mark_end("set_parameters")
        optimizer = torch.optim.Adam(self._model.parameters(), lr=config.TRAIN.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        timestats.mark_start("fit")
        metrics = self._fit(train_loader, optimizer, loss_fn, num_epochs=config.FED.local_epochs, device=device)
        timestats.mark_end("fit")

        timestats.mark_start("get_parameters")
        sharable_params = self.get_parameters(None)
        timestats.mark_end("get_parameters")

        return sharable_params, len(train_loader.dataset), metrics

    # def evaluate(
    #     self, parameters: List[np.ndarray], config: Dict[str, str]
    # ) -> Tuple[float, int, Dict]:
    #     # Set model parameters, evaluate model on local test dataset, return result
    #     self.set_parameters(parameters)
    #     return

    def _fit(self, train_loader, optimizer, loss_fn, num_epochs, device):
        self._model.train() # Enable dropout (if have).
        pbar = tqdm(range(num_epochs), leave=False)
        loss_hist = []
        for e in pbar:
            total_loss = 0
            count_example = 0
            for user, item, label in train_loader:
                user = user.to(device)
                item = item.to(device)
                label = label.float().to(device)

                optimizer.zero_grad()
                prediction = self._model(user, item)
                loss = loss_fn(prediction, label)
                loss.backward()
                # tmp = self._model.embed_user_GMF.weight.detach().clone()
                # print(self._model.embed_user_GMF.weight.grad.data)
                optimizer.step()
                # print(torch.linalg.norm(self._model.embed_user_GMF.weight.detach().cpu() - torch.tensor(self._private_params['weights'][0])))
                count_example += label.shape[0]
                total_loss += loss.item()* label.shape[0]
            total_loss /= count_example
            loss_hist.append(total_loss)
        return {
            "loss": loss_hist
        }