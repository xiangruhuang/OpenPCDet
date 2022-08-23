import torch
import numpy as np
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
from .message_passing_v2 import MessagePassingBlock
import time

def grid_assign_3x3(relative_bxyz):
    assert relative_bxyz.shape[-1] == 4
    relative_xyz = relative_bxyz[:, 1:4]
    kernel_index = torch.zeros(relative_xyz.shape[0], dtype=torch.long,
                               device=relative_xyz.device)
    for i in range(3):
        zero_mask = (relative_xyz[:, i] < 1e-3) & (relative_xyz[:, i] > -1e-3)
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
        self.relu = block_cfg.get("RELU", True)
        self.key = block_cfg['KEY']

        self.message_passing = MessagePassingBlock(input_channel, output_channel, self.key)

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.norm = norm_fn(output_channel)
        self.kernel_assigner = grid_assign_3x3
        
    def forward(self, ref_bxyz, ref_feat, conv_dict):
        query_bxyz = ref_bxyz

        if f'{self.key}_graph' in conv_dict:
            e_ref, e_query, e_kernel = conv_dict[f'{self.key}_graph']
        else:
            e_ref, e_query = self.graph(ref_bxyz, query_bxyz)
            e_kernel = self.kernel_assigner(ref_bxyz[e_ref] - query_bxyz[e_query]) # in range [0, 27)
            conv_dict[f'{self.key}_graph'] = e_ref, e_query, e_kernel

        query_feat, conv_dict = self.message_passing(
                                    ref_feat, e_kernel, e_ref, e_query,
                                    query_bxyz.shape[0], conv_dict)

        query_feat = self.norm(query_feat)

        if self.relu:
            query_feat = F.relu(query_feat)

        return query_bxyz, query_feat, conv_dict


class GridConvDownBlock(DownBlockTemplate):
    def __init__(self, block_cfg, sampler_cfg, graph_cfg):
        super().__init__(block_cfg, sampler_cfg, graph_cfg)
        input_channel = block_cfg["INPUT_CHANNEL"]
        output_channel = block_cfg["OUTPUT_CHANNEL"]
        self.key = block_cfg['KEY']
        
        self.message_passing = MessagePassingBlock(input_channel, output_channel, self.key)

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.norm = norm_fn(output_channel)
        self.kernel_assigner = grid_assign_3x3
        
    def forward(self, ref_bxyz, ref_feat, conv_dict):
        if self.sampler is not None:
            query_bxyz = self.sampler(ref_bxyz)
        else:
            query_bxyz = ref_bxyz

        if f'{self.key}_graph' in conv_dict:
            e_ref, e_query, e_kernel = conv_dict[f'{self.key}_graph']
        else:
            e_ref, e_query = self.graph(ref_bxyz, query_bxyz)
            e_kernel = self.kernel_assigner(ref_bxyz[e_ref] - query_bxyz[e_query]) # in range [0, 27)
            conv_dict[f'{self.key}_graph'] = e_ref, e_query, e_kernel

        query_feat, conv_dict = self.message_passing(
                                    ref_feat, e_kernel, e_ref, e_query,
                                    query_bxyz.shape[0], conv_dict)

        query_feat = self.norm(query_feat)

        query_feat = F.relu(query_feat)

        return query_bxyz, query_feat, conv_dict


class GridConvUpBlock(UpBlockTemplate):
    def __init__(self, block_cfg, graph_cfg):
        super().__init__(block_cfg, graph_cfg)
        input_channel = block_cfg["INPUT_CHANNEL"]
        output_channel = block_cfg["OUTPUT_CHANNEL"]
        self.key = block_cfg['KEY']

        self.message_passing = MessagePassingBlock(input_channel, output_channel, self.key)

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.norm = norm_fn(output_channel)
        self.kernel_assigner = grid_assign_3x3
        
    def forward(self, ref_bxyz, ref_feat, query_bxyz, conv_dict):
        if f'{self.key}_graph' in conv_dict:
            e_query, e_ref, e_kernel = conv_dict[f'{self.key}_graph']
        else:
            e_ref, e_query = self.graph(ref_bxyz, query_bxyz)
            e_kernel = self.kernel_assigner(ref_bxyz[e_ref] - query_bxyz[e_query]) # in range [0, 27)
            conv_dict[f'{self.key}_graph'] = e_ref, e_query, e_kernel

        query_feat, conv_dict = self.message_passing(
                                    ref_feat, e_kernel, e_ref, e_query,
                                    query_bxyz.shape[0], conv_dict)
        
        query_feat = self.norm(query_feat)

        query_feat = F.relu(query_feat)

        return query_feat, conv_dict
