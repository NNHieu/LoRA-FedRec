import copy
import torch
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F 
from rec.models import MF, NCF
from stats import cal_explain_variance_ratio
from fedlib.compression import SVDMess
from fedlib.compresors.compressors import Compressor
from .. import param_tree
from functools import partial

class FedParamSpliter:
    def __init__(self, item_num) -> None:
        self.register_buffer('private_inter_mask', torch.zeros(item_num, dtype=torch.long))
        pass

    def get_server_params(self, **kwargs):
        return self._get_splited_params()[1]
    
    def get_private_params(self, **kwarfs):
        return self._get_splited_params()[0]

    def _get_splited_params(self, **kwarfs):
        submit_params = {}  
        private_params = {}
        for key, val in self.state_dict().items():
            if 'user' in key:
                private_params[key] = val.detach().clone()
            elif key == "private_inter_mask":
                submit_params[key] = val.detach().clone().clamp_(max=1)
            else:
                submit_params[key] = val.detach().clone()
        submit_param_tree = param_tree.statedict_to_treedict(submit_params)
        submit_param_tree = param_tree.parse_lora_node(submit_param_tree, freeze_B=False, ignore_weight=False)
        return private_params, submit_param_tree
    
    def _get_splited_params_for_optim(self, **kwarfs):
        submit_params = []
        private_params = []
        for key, val in self.named_parameters():
            if 'item' in key and 'emb' in key:
                submit_params.append(val)
            else:
                private_params.append(val)
        return submit_params, private_params 
    
    def _set_state_from_splited_params(self, splited_params):
        private_params, submit_param_tree = splited_params
        submit_params = param_tree.treedict2statedict(submit_param_tree)
        params_dict = dict(private_params, **submit_params)
        state_dict = OrderedDict(params_dict)
        self.load_state_dict(state_dict, strict=True)
        self.private_inter_mask = torch.zeros_like(self.private_inter_mask)

class FedMF(MF, FedParamSpliter):
    def __init__(self, item_num, gmf_emb_size=16, user_num=1):
        MF.__init__(self, user_num, item_num, gmf_emb_size)
        FedParamSpliter.__init__(self, item_num)

    def forward(self, user, item):
        self.private_inter_mask[item] = 1
        return super().forward(user, item)
    
    def _reinit_private_params(self):
        self._init_weight_()
    
    def server_prepare(self, **kwargs):
        return
    
    @classmethod
    def merge_client_params(cls, clients, server_params, model, device):
        client_weights = [c._private_params.values() for c in clients]
        client_weights = [torch.cat(w, dim=0).to(device) for w in zip(*client_weights)]
        client_weights = {k: v for k,v in zip(clients[0]._private_params.keys(), client_weights)}
        eval_model = copy.deepcopy(model)
        eval_model._set_state_from_splited_params([clients[0]._private_params, server_params])
        eval_model.embed_user_GMF = torch.nn.Embedding.from_pretrained(client_weights['embed_user_GMF.weight'])
        return eval_model
    
    @classmethod
    def stat_transfered_params(cls, transfer_params):
        # item_emb = transfer_params['embed_item_GMF.weight']
        # explain_variance_ratio = cal_explain_variance_ratio(item_emb)
        # return {"mf_item_emb_explain_variance_ratio": explain_variance_ratio}
        return {}

    def reg_loss(self, item, user, scale_item_reg=1):
        reg_loss = 0
        for name, param in self.named_parameters():
            if "emb" in name:
                continue
            else:
                reg_loss += (param**2).sum()
        gmf_item_emb = self.embed_item_GMF(item)
        gmf_user_emb = self.embed_user_GMF(user)
        
        item_emb_reg = (gmf_item_emb**2).sum() * scale_item_reg
        user_emb_reg = (gmf_user_emb**2).sum()

        # item_emb_reg *= self._model.lora_scale_lr

        reg_loss += item_emb_reg + user_emb_reg
        return reg_loss

class FedNCFModel(NCF, FedParamSpliter):
    def __init__(self, item_num, gmf_emb_size=16, mlp_emb_size=64, mlp_layer_dims=[128, 64, 32, 16], dropout=0., user_num=1):
        # ItemEmbedding = partial(nn.Embedding, scale_grad_by_freq=True)
        # NCF.__init__(self, user_num, item_num, gmf_emb_size, mlp_emb_size, mlp_layer_dims, dropout, ItemEmbedding=ItemEmbedding)
        NCF.__init__(self, user_num, item_num, gmf_emb_size, mlp_emb_size, mlp_layer_dims, dropout)
        FedParamSpliter.__init__(self, item_num)
        self.user_num = user_num

    def forward(self, user, item):
        self.private_inter_mask[item] = 1
        return super().forward(user, item)
    
    def _reinit_private_params(self):
        self._init_weight_()
    
    def server_prepare(self, **kwargs):
        return
    
    @classmethod
    def merge_client_params(cls, clients, server_params, model, device):
        client_weights = [c._private_params.values() for c in clients]
        client_weights = [torch.cat(w, dim=0).to(device) for w in zip(*client_weights)]
        client_weights = {k: v for k,v in zip(clients[0]._private_params.keys(), client_weights)}
        eval_model = copy.deepcopy(model)
        eval_model._set_state_from_splited_params([clients[0]._private_params, server_params])
        eval_model.embed_user_GMF = torch.nn.Embedding.from_pretrained(client_weights['embed_user_GMF.weight'])
        eval_model.embed_user_MLP = torch.nn.Embedding.from_pretrained(client_weights['embed_user_MLP.weight'])
        return eval_model

    def reg_loss(self, item, user, scale_item_reg=1):
        reg_loss = 0
        for name, param in self.named_parameters():
            if "emb" in name:
                continue
            else:
                reg_loss += (param**2).sum()
        gmf_item_emb = self.embed_item_GMF(item)
        gmf_user_emb = self.embed_user_GMF(user)
        mlp_item_emb = self.embed_item_MLP(item)
        mlp_user_emb = self.embed_user_MLP(user)
        
        item_emb_reg = (gmf_item_emb**2).sum() * scale_item_reg
        item_emb_reg += (mlp_item_emb**2).sum() * scale_item_reg
        user_emb_reg = (gmf_user_emb**2).sum()
        user_emb_reg += (mlp_user_emb**2).sum()

        # item_emb_reg *= self._model.lora_scale_lr

        reg_loss += item_emb_reg + user_emb_reg
        return reg_loss