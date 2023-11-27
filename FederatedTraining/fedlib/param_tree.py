import optree
import torch

@optree.register_pytree_node_class(namespace='fedlib')
class LoRANode:
    def __init__(self, lora_A, lora_B, lora_scaling, weight, freeze_B, aggregate_weight=False) -> None:
        self.weight = weight
        self.lora_A = lora_A
        self.lora_B = lora_B
        self.lora_scaling = lora_scaling
        self.merged = False
        self.freeze_B = freeze_B
        self.aggregate_weight = aggregate_weight

    def tree_flatten(self):  # -> (children, metadata, entries)
        return (
            [self.lora_A, self.lora_B, self.lora_scaling, self.weight, self.freeze_B],  # children
            ['lora_A', 'lora_B', 'lora_scaling', 'weight', 'freeze_B'],  # metadata
            ['lora_A', 'lora_B', 'lora_scaling', 'weight', 'freeze_B'],  # entries
        )

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(*children)

    def add_(self, o, alpha=1.):
        assert isinstance(o, LoRANode)
        if not self.freeze_B:
            if not o.aggregate_weight:
                update_weight = o.update_lora_weights()
            else:
                update_weight = o.weight
            if isinstance(alpha, InteractionMask):
                self.weight.add_(update_weight * alpha.mask.unsqueeze(-1))
            else:
                self.weight.add_(update_weight, alpha=alpha)
            return self
        else:
            if isinstance(alpha, InteractionMask):
                self.lora_A.add_(o.lora_A * alpha.mask.unsqueeze(-1))
            else:
                self.lora_A.add_(o.lora_A, alpha=alpha)
            return self
        # if isinstance(o, LoRANode):
            # self.lora_A.add_(o.lora_A)
            # if self.lora_B is not None and o.lora_B is not None:
            #     self.lora_B.add_(o.lora_B)
            # self.lora_scaling.add_(o.lora_scaling)
            # if self.weight is not None and o.weight is not None:
            #     self.weight.add_(o.weight)
            # return self
        # elif isinstance(o, int):
        #     return LoRANode(self.lora_A + o, self.lora_B + o, self.lora_scaling + o, self.weight + o)
        # else:
        #     raise ValueError
    
    def sub_(self, o):
        assert isinstance(o, LoRANode)
        # self.weight.sub_(o.weight)
        return self

    def div_(self, alpha):
        if self.freeze_B:
            if isinstance(alpha, InteractionMask):
                mask = alpha.mask
                mask[mask == 0] = 1
                self.lora_A.div_(mask.unsqueeze(-1))
            else:
                self.lora_A.div_(alpha)
        else:
            if isinstance(alpha, InteractionMask):
                mask = alpha.mask
                mask[mask == 0] = 1
                self.weight.div_(mask.unsqueeze(-1))
            else:
                self.weight.div_(alpha)
            return self
        return self
    
    def new_zeros(self):
        if self.freeze_B:
            return LoRANode(torch.zeros_like(self.lora_A), 
                            None, 
                            None, 
                            None,
                            self.freeze_B)
        else:
            return LoRANode(torch.zeros_like(self.lora_A), 
                            torch.zeros_like(self.lora_B), 
                            torch.zeros_like(self.lora_scaling), 
                            torch.zeros_like(self.weight),
                            self.freeze_B,
                            aggregate_weight=True)

    def __str__(self):
        return f"LoRANode(lora_A={self.lora_A}, lora_B={self.lora_B}, lora_scaling={self.lora_scaling}, weight={self.weight}, freeze_B={self.freeze_B})"
    
    def to_dict(self):
        return {
            'lora_A': self.lora_A,
            'lora_B': self.lora_B,
            'lora_scaling': self.lora_scaling,
            'weight': self.weight,
        }
    
    def update_lora_weights(self):
            return self.lora_A @ self.lora_B * self.lora_scaling
    
    def numel(self):
        if self.freeze_B:
            numel = self.lora_A.numel()
        else:
            numel = self.lora_A.numel() + self.lora_B.numel()
        # if self.lora_B is not None:
        #     numel += self.lora_B.numel()
        # if self.lora_scaling is not None:
        #     numel += self.lora_scaling.numel()
        # if self.weight is not None:
        #     numel += self.weight.numel()
        return numel

    def norm(self):
        update_weight = self.update_lora_weights()
        update_weight_norm = update_weight.norm(dim=-1)
        mean_weight_norm = update_weight_norm.sum() / (update_weight_norm > 0).sum()
        return mean_weight_norm
        

@optree.register_pytree_node_class(namespace='fedlib')
class EmbNode:
    def __init__(self, weight) -> None:
        self.weight = weight
    
    def tree_flatten(self):  # -> (children, metadata, entries)
        return (
            [self.weight],  # children
            ['weight'],  # metadata
            ['weight'],  # entries
        )

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(*children)

    def add_(self, o, alpha=1.):
        if isinstance(alpha, InteractionMask):
            self.weight.add_(o.weight * alpha.mask.unsqueeze(-1))
        else:
            self.weight.add_(o.weight, alpha=alpha)
        return self
    
    def sub_(self, o):
        self.weight.sub_(o.weight)
        return self

    def div_(self, alpha):
        if isinstance(alpha, InteractionMask):
            mask = alpha.mask
            mask[mask == 0] = 1
            self.weight.div_(mask.unsqueeze(-1))
        else:
            self.weight.div_(alpha)
        return self
    
    def new_zeros(self):
        return EmbNode(torch.zeros_like(self.weight))

    def to_dict(self):
        return {'weight': self.weight}

    def numel(self):
        return self.weight.numel()

    def norm(self):
        norms = self.weight.norm(dim=-1)
        return norms.sum() / (norms > 0).sum()

@optree.register_pytree_node_class(namespace='fedlib')
class InteractionMask:
    def __init__(self, mask) -> None:
        self.mask = mask
    
    def tree_flatten(self):  # -> (children, metadata, entries)
        return (
            [self.mask],  # children
            ['mask'],  # metadata
            ['mask'],  # entries
        )

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(*children)

    def add_(self, other, alpha=1):
        self.mask.add_(other.mask)
        return self

    def div_(self, other):
        return self
    
    def new_zeros(self):
        return InteractionMask(torch.zeros_like(self.mask))
    
    def sub_(self, o):
        return self

    def numel(self):
        return self.mask.numel()

    def norm(self):
        return self.mask.sum()

def statedict_to_treedict(state_dict):
    root = {}
    for k, v in state_dict.items():
        d = root
        path = k.split('.')
        for i, node in enumerate(path):
            if i == len(path) - 1:
                d[node] = v
            else:
                if node not in d:
                    d[node] = {}
                d = d[node]
    return root

def treedict2statedict(root):
    state_dict = {}
    def dfs(d, path):
        for k, v in d.items():
            if isinstance(v, dict):
                dfs(v, path + [k])
            elif isinstance(v, LoRANode):
                dfs(v.to_dict(), path + [k])
            elif isinstance(v, EmbNode):
                dfs(v.to_dict(), path + [k])
            elif isinstance(v, InteractionMask):
                state_dict['.'.join(path + [k])] = v.mask
            else:
                state_dict['.'.join(path + [k])] = v
    dfs(root, [])
    return state_dict

def parse_lora_node(root, freeze_B, ignore_weight):
    def dfs(d, path):
        for k, v in d.items():
            if isinstance(v, dict):
                is_lora_node = dfs(v, path + [k])                    
                if is_lora_node:
                    d[k] = LoRANode(
                        lora_A=v['lora_A'],
                        lora_B=v['lora_B'],
                        lora_scaling=v['lora_scaling'],
                        weight=v['weight'] if not ignore_weight else None,
                        freeze_B=freeze_B
                    )
                elif "item" in k and "emb" in k:
                    d[k] = EmbNode(weight=v['weight'])
            else:
                if k == 'private_inter_mask':
                    d[k] = InteractionMask(v)
                elif 'lora' in k:
                    return True
        return False
    dfs(root, [])
    return root

def tree_sub_(a, b):
    return optree.tree_map_(lambda x, y: x.sub_(y), a, b)


def add_fn(p, x, y, interaction_mask, weight):
    # print(p)
    x.add_(y, alpha=interaction_mask if isinstance(x, (EmbNode, LoRANode)) else weight)

def tree_add_(a, b, interaction_mask, weight):
    optree.tree_map_with_path_(lambda p, x, y: add_fn(p, x, y, interaction_mask, weight), a, b)

def div_fn(x, interaction_mask, weight):
    # print(p)
    x.div_(interaction_mask if isinstance(x, (EmbNode, LoRANode)) else weight)

def tree_div_(a, interaction_mask, weight):
    optree.tree_map_(lambda x: div_fn(x, interaction_mask, weight), a)

def new_zeros_fn(p, x):
    # print(p, type(x))
    if isinstance(x, torch.Tensor):
        return torch.zeros_like(x)
    elif x is None:
        return None
    else:
        return x.new_zeros()

def tree_zeros_like(a):
    return optree.tree_map(lambda x: new_zeros_fn(None, x), a)

if __name__ == "__main__":
    root = {
        'a': 1,
        'b': 2,
        'c': {
            'd': 3,
            'e': 4,
            'f': {
                'lora_A': 5,
                'lora_B': 6,
                'lora_scaling': 7,
                'weight': 8,
            }
        }
    }
    root = parse_lora_node(root)
    print(root)
    state_dict = treedict2statedict(root)
    print(state_dict)
    # root = statedict_to_treedict(state_dict)
    # print(root)

    root_2 = optree.tree_map(lambda x: x+1, root)
    root_2 = parse_lora_node(root_2)
    print(treedict2statedict(root_2))
    
    
    root_3 = optree.tree_map(lambda x, y: x+y, root, root_2)
    state_dict3 = treedict2statedict(root_3)
    print(state_dict3)

    