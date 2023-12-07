import copy
import torch
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F 
import lora
from rec.models import LoraNCF, LoraMF
from stats import cal_explain_variance_ratio
from .. import param_tree
# from fedlib.compression import TopKCompressor

class FedLoraParamsSplitter:
    def __init__(self, item_num) -> None:
        self.register_buffer('private_inter_mask', torch.zeros(item_num, dtype=torch.long))
        pass

    def get_server_params(self, **kwargs):
        return self._get_splited_params()[1]
    
    def get_private_params(self, **kwarfs):
        return self._get_splited_params()[0]
        
    def pre_extract_weight(self):
        raise NotImplementedError

    def _get_splited_params(self, server_init=False, **kwarfs):
        '''

        Return: private_params, submit_params
        '''
        self.pre_extract_weight()
        submit_params = {}
        private_params = {}
        for key, val in self.state_dict().items():
            if 'user' in key:
                private_params[key] = val.detach().clone()
            else:
                submit_params[key] = val.detach().clone()
        submit_param_tree = param_tree.statedict_to_treedict(submit_params)
        if server_init:
            submit_param_tree = param_tree.parse_lora_node(submit_param_tree, freeze_B=False, ignore_weight=False)
        else:
            submit_param_tree = param_tree.parse_lora_node(submit_param_tree, freeze_B=self.freeze_B, ignore_weight=self.freeze_B)
        return private_params, submit_param_tree

    def _set_state_from_splited_params(self, splited_params):
        private_params, submit_param_tree = splited_params
        submit_params = param_tree.treedict2statedict(submit_param_tree)
        params_dict = dict(private_params, **submit_params)
        state_dict = OrderedDict(params_dict)
        self.load_state_dict(state_dict, strict=True)
        if self.freeze_B:
            self._merge_all_lora_weights()
            self._reset_all_lora_weights(init_B_strategy="random-scaling", keep_B=self.freeze_B)
        else:
            self._reset_all_lora_weights(init_B_strategy=self.init_B_strategy, keep_B=self.freeze_B)
        # self._reset_all_lora_weights(init_B_strategy="random-scaling", keep_B=False)
        self.private_inter_mask = torch.zeros_like(self.private_inter_mask)

    def _get_splited_params_for_optim(self, **kwarfs):
        submit_params = []
        private_params = []
        for key, val in self.named_parameters():
            if val.requires_grad:
                if 'item' in key and 'emb' in key:
                    submit_params.append(val)
                else:
                    private_params.append(val)
        return submit_params, private_params 

class FedLoraNCF(LoraNCF, FedLoraParamsSplitter):
    def __init__(self, item_num, gmf_emb_size=16, mlp_emb_size=64, mlp_layer_dims=[128, 64, 32, 16], dropout=0., lora_rank=4, lora_alpha=4, freeze_B=False, user_num=1):
        LoraNCF.__init__(self, user_num, item_num, gmf_emb_size, mlp_emb_size, mlp_layer_dims, dropout, lora_rank, lora_alpha, freeze_B)
        self.freeze_B = freeze_B
        if self.freeze_B:
            self.embed_item_GMF.lora_B.requires_grad = False
            self.embed_item_MLP.lora_B.requires_grad = False

        FedLoraParamsSplitter.__init__(self, item_num)
    
    def forward(self, user, item):
        self.private_inter_mask[item] = 1
        return super().forward(user, item)
    
    def server_prepare(self, **kwargs):
        '''
        This function is called after the aggregation step at the central server. \n
        If freeze_B, then multiplying all aggregated lora_A weights with the pre-defined B and merging to the base weight.
        After that, reset all CoLS weights (init new B, set A to be a zero matrix.).
        '''
        self.init_B_strategy = kwargs['init_B_strategy']
        if self.freeze_B:
            self._merge_all_lora_weights()
        self._reset_all_lora_weights(keep_B=False, **kwargs)
    
    def _reinit_private_params(self):
        self._init_weight_()
    
    def pre_extract_weight(self):
        # if not self.freeze_B:
        #     self._merge_all_lora_weights()
        pass
    
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
    
class FedLoraMF(LoraMF, FedLoraParamsSplitter):
    def __init__(self, item_num, gmf_emb_size=16, lora_rank=4, lora_alpha=4, freeze_B=False, user_num=1):
        LoraMF.__init__(self, user_num, item_num, gmf_emb_size, lora_rank, lora_alpha, freeze_B)
        FedLoraParamsSplitter.__init__(self, item_num)
        if self.freeze_B:
            self.embed_item_GMF.lora_B.requires_grad = False
    
    def pre_extract_weight(self):
        # if not self.freeze_B:
        #     self._merge_all_lora_weights()
        pass
    
    def forward(self, user, item):
        self.private_inter_mask[item] = 1
        return super().forward(user, item)
    
    def server_prepare(self, **kwargs):
        self.init_B_strategy = kwargs['init_B_strategy']
        self._reset_all_lora_weights(keep_B=False, **kwargs)
    
    def _reinit_private_params(self):
        self._init_weight_()
    
    # def _reinit_B(self):
    #     print("Reinit B")
    #     nn.init.normal_(self.embed_item_GMF.lora_B)
    
    @classmethod
    def merge_client_params(cls, clients, server_params, model, device):
        client_weights = [c._private_params.values() for c in clients]
        client_weights = [torch.cat(w, dim=0).to(device) for w in zip(*client_weights)]
        client_weights = {k: v for k,v in zip(clients[0]._private_params.keys(), client_weights)}
        eval_model = copy.deepcopy(model)
        eval_model._set_state_from_splited_params([clients[0]._private_params, server_params])
        eval_model.embed_user_GMF = torch.nn.Embedding.from_pretrained(client_weights['embed_user_GMF.weight'])
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
        
        item_emb_reg = (gmf_item_emb**2).sum() * (scale_item_reg / self.lora_scale_lr)
        user_emb_reg = (gmf_user_emb**2).sum()

        # item_emb_reg *= self._model.lora_scale_lr

        reg_loss += item_emb_reg + user_emb_reg
        return reg_loss