#!/usr/bin/env python
# coding: utf-8
from collections import OrderedDict
import copy

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from . import param_tree


class FedParamSpliter:
    def __init__(self, item_num) -> None:
        # self.register_buffer('private_inter_mask', torch.zeros(item_num, dtype=torch.long))
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
            # elif key == "private_inter_mask":
            #     submit_params[key] = val.detach().clone().clamp_(max=1)
            else:
                submit_params[key] = val.detach().clone()
        submit_param_tree = param_tree.statedict_to_treedict(submit_params)
        submit_param_tree = param_tree.parse_lora_node(submit_param_tree, freeze_B=False, ignore_weight=False)
        return private_params, submit_param_tree
    
    def _get_splited_params_for_optim(self, **kwarfs):
        submit_params = []
        private_params = self.parameters()
        # for key, val in self.named_parameters():
        #     # if 'item' in key and 'emb' in key:
        #     #     submit_params.append(val)
        #     # else:
        #     private_params.append(val)
        return submit_params, private_params 
    
    def _set_state_from_splited_params(self, splited_params):
        private_params, submit_param_tree = splited_params
        submit_params = param_tree.treedict2statedict(submit_param_tree)
        params_dict = dict(private_params, **submit_params)
        state_dict = OrderedDict(params_dict)
        self.load_state_dict(state_dict, strict=True)
        # self.private_inter_mask = torch.zeros_like(self.private_inter_mask)

class MetaRecommender(nn.Module):#in fact, it's not a hypernetwork
    def __init__(self, user_num, item_num, item_emb_size=32, item_mem_num=8, user_emb_size=32, mem_size=128, hidden_size=512):#note that we have many users and each user has many layers
        super(MetaRecommender, self).__init__()
        self.item_num = item_num
        self.item_emb_size = item_emb_size
        self.item_mem_num = item_mem_num
        #For each user
        self.user_embedding = nn.Embedding(user_num, user_emb_size)
        self.memory = Parameter(nn.init.xavier_normal_(torch.Tensor(user_emb_size, mem_size)), requires_grad=True)
        #For each layer
        self.hidden_layer_1, self.weight_layer_1, self.bias_layer_1 = self.define_one_layer(mem_size, hidden_size, item_emb_size, int(item_emb_size/4))
        self.hidden_layer_2, self.weight_layer_2, self.bias_layer_2 = self.define_one_layer(mem_size, hidden_size, int(item_emb_size/4), 1)
        self.hidden_layer_3, self.emb_layer_1, self.emb_layer_2 = self.define_item_embedding(item_num, item_emb_size, item_mem_num, mem_size, hidden_size)
    
    def define_one_layer(self, mem_size, hidden_size, int_size, out_size):#define one layer in MetaMF
        hidden_layer = nn.Linear(mem_size, hidden_size)
        weight_layer = nn.Linear(hidden_size, int_size*out_size)
        bias_layer = nn.Linear(hidden_size, out_size)
        return hidden_layer, weight_layer, bias_layer
    
    def define_item_embedding(self, item_num, item_emb_size, item_mem_num, mem_size, hidden_size):
        hidden_layer = nn.Linear(mem_size, hidden_size)
        emb_layer_1 = nn.Linear(hidden_size, item_num*item_mem_num)
        emb_layer_2 = nn.Linear(hidden_size, item_mem_num*item_emb_size)
        return hidden_layer, emb_layer_1, emb_layer_2 
            
    def forward(self, user_id):
        #collaborative memory module
        user_emb = self.user_embedding(user_id)#input_user=[batch_size, user_emb_size]
        cf_vec = torch.matmul(user_emb, self.memory)#cf_vec=[batch_size, mem_size]
        #collaborative memory module
        #meta recommender module
        output_weight = []
        output_bias = []
        
        weight, bias = self.get_one_layer(self.hidden_layer_1, self.weight_layer_1, self.bias_layer_1, cf_vec, self.item_emb_size, int(self.item_emb_size/4))
        output_weight.append(weight)
        output_bias.append(bias) 
                
        weight, bias = self.get_one_layer(self.hidden_layer_2, self.weight_layer_2, self.bias_layer_2, cf_vec, int(self.item_emb_size/4), 1)
        output_weight.append(weight)
        output_bias.append(bias)
        
        item_embedding = self.get_item_embedding(self.hidden_layer_3, self.emb_layer_1, self.emb_layer_2, cf_vec, self.item_num, self.item_mem_num, self.item_emb_size)
        #meta recommender module
        return output_weight, output_bias, item_embedding, cf_vec#([len(layer_list)+1, batch_size, *, *], [len(layer_list)+1, batch_size, 1, *], [batch_size, item_num, item_emb_size], [batch_size, mem_size])
    
    def get_one_layer(self, hidden_layer, weight_layer, bias_layer, cf_vec, int_size, out_size):#get one layer in MetaMF
        hid = hidden_layer(cf_vec)#hid=[batch_size, hidden_size]
        hid = F.relu(hid)
        weight = weight_layer(hid)#weight=[batch_size, self.layer_list[i-1]*self.layer_list[i]]
        bias = bias_layer(hid)#bias=[batch_size, self.layer_list[i]] 
        weight = weight.view(-1, int_size, out_size)
        bias = bias.view(-1, 1, out_size)
        return weight, bias
    
    def get_item_embedding(self, hidden_layer, emb_layer_1, emb_layer_2, cf_vec, item_num, item_mem_num, item_emb_size):
        hid = hidden_layer(cf_vec)#hid=[batch_size, hidden_size]
        hid = F.relu(hid)
        emb_left = emb_layer_1(hid)#emb_left=[batch_size, item_num*item_mem_num]
        emb_right = emb_layer_2(hid)#emb_right=[batch_size, item_mem_num*item_emb_size]
        emb_left = emb_left.view(-1, item_num, item_mem_num)#emb_left=[batch_size, item_num, item_mem_num]
        emb_right = emb_right.view(-1, item_mem_num, item_emb_size)#emb_right=[batch_size, item_mem_num, item_emb_size]
        item_embedding = torch.matmul(emb_left, emb_right)#item_embedding=[batch_size, item_num, item_emb_size]
        return item_embedding

class MetaMF(nn.Module, FedParamSpliter):
    def __init__(self, item_num, user_num=1, item_emb_size=32, item_mem_num=8, user_emb_size=32, mem_size=128, hidden_size=512):
        super(MetaMF, self).__init__()
        FedParamSpliter.__init__(self, item_num)
        self.item_num = item_num
        self.metarecommender = MetaRecommender(user_num, item_num, item_emb_size, item_mem_num, user_emb_size, mem_size, hidden_size)
        
    def forward(self, user_id, item_id):
        #prediction module
        model_weight, model_bias, item_embedding, _ = self.metarecommender(user_id)
        item_id = item_id.view(-1, 1)#item_id=[batch_size, 1]
        item_one_hot = torch.zeros(len(item_id), self.item_num, device=item_id.device)#we generate it dynamically, and default device is cpu
        item_one_hot.scatter_(1, item_id, 1)#item_one_hot=[batch_size, item_num]
        item_one_hot = torch.unsqueeze(item_one_hot, 1)#item_one_hot=[batch_size, 1, item_num]
        item_emb = torch.matmul(item_one_hot, item_embedding)#out=[batch_size, 1, item_emb_size]
        out = torch.matmul(item_emb, model_weight[0])#out=[batch_size, 1, item_emb_size/4]
        out = out+model_bias[0]#out=[batch_size, 1, item_emb_size/4]
        out = F.relu(out)#out=[batch_size, 1, item_emb_size/4]
        out = torch.matmul(out, model_weight[1])#out=[batch_size, 1, 1]
        out = out+model_bias[1]#out=[batch_size, 1, 1]
        out = torch.squeeze(out, dim=(1, 2))#out=[batch_size]
        #prediction module
        return out

    def _reinit_private_params(self):
        self.metarecommender.user_embedding.reset_parameters()
    
    def server_prepare(self, **kwargs):
        return
    
    @classmethod
    def merge_client_params(cls, clients, server_params, model, device):
        client_weights = [c._private_params.values() for c in clients]
        client_weights = [torch.cat(w, dim=0).to(device) for w in zip(*client_weights)]
        client_weights = {k: v for k,v in zip(clients[0]._private_params.keys(), client_weights)}
        eval_model = copy.deepcopy(model)
        eval_model._set_state_from_splited_params([clients[0]._private_params, server_params])
        eval_model.metarecommender.user_embedding = torch.nn.Embedding.from_pretrained(client_weights['metarecommender.user_embedding.weight'])
        return eval_model

    def loss(self, prediction, rating):
        #regularizer = torch.sum(torch.matmul(self.metarecommender.memory, self.metarecommender.memory.t()))
        return torch.mean(torch.pow(prediction-rating,2))#+self._lambda*regularizer
    
    @property
    def is_lora(self):
        return False

