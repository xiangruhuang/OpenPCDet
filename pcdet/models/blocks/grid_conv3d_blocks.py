import torch
from torch import nn

import torch.nn.functional as F
import torch_cluster
from torch_scatter import scatter
from functools import partial

from .block_templates import (
    DownBlockTemplate,
    MessagePassingBlockTemplate,
    UpBlockTemplate
)
from .message_passing_v2 import message_passing

def grid_assign_3x3(relative_xyz):
    kernel_index = torch.zeros(relative_xyz.shape[0], dtype=torch.long,
                               device=relative_xyz.device)
    for i in range(3):
        zero_mask = (relative_xyz[:, i] < 1e-5) & (relative_xyz[:, i] > -1e-5)
        sign = relative_xyz[:, i].sign()
        sign[zero_mask] = 0
        offset = sign + 1
        kernel_index = kernel_index * 3 + offset
        
    return kernel_index


class GridConvFlatBlock(DownBlockTemplate):
    def __init__(self, block_cfg, graph_cfg):
        super().__init__(block_cfg, None, graph_cfg)
        input_channel = block_cfg["INPUT_CHANNEL"]
        output_channel = block_cfg["OUTPUT_CHANNEL"]

        kernel_weights = torch.randn(27, input_channel, output_channel)
        nn.init.xavier_normal_(kernel_weights)
        self.kernel_weights = nn.Parameter(kernel_weights, requires_grad=True)

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.norm = norm_fn(output_channel)
        self.aggr = 'mean'
        self.kernel_assigner = grid_assign_3x3
        
    def forward(self, ref_bxyz, ref_feat):
        query_bxyz = ref_bxyz

        e_ref, e_query = self.graph(ref_bxyz, query_bxyz)
        
        e_kernel = self.kernel_assigner(ref_bxyz[e_ref] - query_bxyz[e_query]) # in range [0, 27)

        query_feat = message_passing(self.kernel_weights, ref_feat, e_kernel, e_ref, e_query, query_bxyz.shape[0])

        return query_bxyz, query_feat, e_ref, e_query

    def extra_repr(self):
        return f"kernel={list(self.kernel_weights.shape)}"

class GridConvDownBlock(DownBlockTemplate):
    def __init__(self, block_cfg, sampler_cfg, graph_cfg):
        super().__init__(block_cfg, sampler_cfg, graph_cfg)
        self.num_kernel_points = block_cfg.get("NUM_KERNEL_POINTS", 16)
        self.grid_size = block_cfg.get("GRID_SIZE", 0.1)

        input_channel = block_cfg["INPUT_CHANNEL"]
        output_channel = block_cfg["OUTPUT_CHANNEL"]

        kernel_weights = torch.randn(27, input_channel, output_channel)
        nn.init.xavier_normal_(kernel_weights)
        self.kernel_weights = nn.Parameter(kernel_weights, requires_grad=True)

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.norm = norm_fn(output_channel)
        self.aggr = 'mean'
        self.kernel_assigner = grid_assign_3x3
        
    def forward(self, ref_bxyz, ref_feat):
        if self.sampler is not None:
            query_bxyz = self.sampler(ref_bxyz)
        else:
            query_bxyz = ref_bxyz

        e_ref, e_query = self.graph(ref_bxyz, query_bxyz)
        
        e_kernel = self.kernel_assigner(ref_bxyz[e_ref] - query_bxyz[e_query]) # in range [0, 27)

        query_feat = message_passing(self.kernel_weights, ref_feat, e_kernel, e_ref, e_query, query_bxyz.shape[0])

        return query_bxyz, query_feat, e_ref, e_query

    def extra_repr(self):
        return f"kernel={list(self.kernel_weights.shape)}"


class GridConvUpBlock(UpBlockTemplate):
    def __init__(self, block_cfg, graph_cfg):
        super().__init__(block_cfg, graph_cfg)
        self.num_kernel_points = block_cfg.get("NUM_KERNEL_POINTS", 16)
        self.grid_size = block_cfg.get("GRID_SIZE", 0.1)

        input_channel = block_cfg["INPUT_CHANNEL"]
        output_channel = block_cfg["OUTPUT_CHANNEL"]

        kernel_weights = torch.randn(27, input_channel, output_channel)
        nn.init.xavier_normal_(kernel_weights)
        self.kernel_weights = nn.Parameter(kernel_weights, requires_grad=True)

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.norm = norm_fn(output_channel)
        self.aggr = 'mean'
        self.kernel_assigner = grid_assign_3x3
        
    def forward(self, ref_bxyz, ref_feat, query_bxyz):
        e_ref, e_query = self.graph(ref_bxyz, query_bxyz)
        
        e_kernel = self.kernel_assigner(ref_bxyz[e_ref, 1:] - query_bxyz[e_query, 1:]) # in range [0, 27)

        query_feat = message_passing(self.kernel_weights, ref_feat, e_kernel, e_ref, e_query, query_bxyz.shape[0])

        return query_feat, e_ref, e_query
    
    def extra_repr(self):
        return f"kernel={list(self.kernel_weights.shape)}"
