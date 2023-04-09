import torch
import torch.nn as nn

import torch_pruning as tp
from torch_pruning import BasePruningFunc, ops

from copy import deepcopy
from functools import reduce
from operator import mul

from typing import Callable, Sequence, Tuple, Dict

class RMSNormPrunner(BasePruningFunc):

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        #print("Pruning RMSNorm Layer: {}".format(layer))
        keep_idxs = list(set(range(layer.weight.size(0))) - set(idxs))
        keep_idxs.sort()
        
        layer.weight = torch.nn.Parameter(
            layer.weight[keep_idxs]
        )
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.weight.size(0)

    def get_in_channels(self, layer):
        return layer.weight.size(0)

class AttentionPrunner(BasePruningFunc):

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        assert len(idxs) % layer.n_heads == 0
        #print("Pruning Attention Layer: {}".format(layer))
        
        for sub_layer in [layer.wq, layer.wk, layer.wv, layer.wo]:
            keep_idxs = list(set(range(sub_layer.out_features)) - set(idxs))
            keep_idxs.sort()
            sub_layer.out_features = sub_layer.out_features-len(idxs)

            sub_layer.weight = torch.nn.Parameter(sub_layer.weight.data.cpu().clone()[keep_idxs])
            if sub_layer.bias is not None:
                sub_layer.bias = torch.nn.Parameter(sub_layer.bias.data.cpu().clone()[keep_idxs])
            
            keep_idxs = list(set(range(sub_layer.in_features)) - set(idxs))
            keep_idxs.sort()
            sub_layer.in_features = sub_layer.in_features-len(idxs)
            sub_layer.weight = torch.nn.Parameter(
                sub_layer.weight.data.cpu().clone()[:, keep_idxs]
            )
        
        layer.dim = layer.dim - len(idxs)
        layer.head_dim = layer.dim // layer.n_heads
        layer.cache_k.data = layer.cache_k.data.cpu().clone()[..., :layer.head_dim]
        layer.cache_v = layer.cache_v.data.cpu().clone()[..., :layer.head_dim]
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.dim

    def get_in_channels(self, layer):
        return layer.dim