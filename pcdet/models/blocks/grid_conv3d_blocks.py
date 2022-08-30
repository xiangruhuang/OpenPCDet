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

def grid_assign_3x3(ref, query, e_ref, e_query):
    relative_bcoords = ref.bcoords[e_ref] - query.bcoords[e_query]
    assert relative_bcoords.shape[-1] == 4, relative_bcoords.dtype==torch.long
    relative_coord = relative_bcoords[:, 1:4]
    kernel_index = torch.zeros(relative_coord.shape[0], dtype=torch.long,
                               device=relative_coord.device)
    for i in range(3):
        sign = relative_coord[:, i].sign()
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

        self.message_passing = MessagePassingBlock(input_channel, output_channel, 27, self.key)

        self.kernel_assigner = grid_assign_3x3
        
    def forward(self, ref, conv_dict):
        query = ref

        if f'{self.key}_graph' in conv_dict:
            e_ref, e_query, e_weight, e_kernel = conv_dict[f'{self.key}_graph']
        else:
            e_ref, e_query, e_weight = self.graph(ref, query)
            e_kernel = self.kernel_assigner(ref, query, e_ref, e_query) # in range [0, 27)
            conv_dict[f'{self.key}_graph'] = e_ref, e_query, e_weight, e_kernel

        query.feat, conv_dict = self.message_passing(
                                    ref.feat, e_kernel, e_ref, e_query,
                                    query.bcenter.shape[0], conv_dict, e_weight)

        if self.norm:
            query.feat = self.norm(query.feat)

        if self.relu:
            if self.act:
                query.feat = self.act(query.feat)

        return query, conv_dict


class GridConvDownBlock(DownBlockTemplate):
    def __init__(self, block_cfg, sampler_cfg, graph_cfg):
        super().__init__(block_cfg, sampler_cfg, graph_cfg)
        input_channel = block_cfg["INPUT_CHANNEL"]
        output_channel = block_cfg["OUTPUT_CHANNEL"]
        self.key = block_cfg['KEY']
        
        self.message_passing = MessagePassingBlock(input_channel, output_channel, 27, self.key)

        self.kernel_assigner = grid_assign_3x3
        
    def forward(self, ref, conv_dict):
        if self.sampler is not None:
            query = self.sampler(ref)
        else:
            query = ref

        if f'{self.key}_graph' in conv_dict:
            e_ref, e_query, e_weight, e_kernel = conv_dict[f'{self.key}_graph']
        else:
            e_ref, e_query, e_weight = self.graph(ref, query)
            e_kernel = self.kernel_assigner(ref, query, e_ref, e_query) # in range [0, 27)
            conv_dict[f'{self.key}_graph'] = e_ref, e_query, e_weight, e_kernel

        query.feat, conv_dict = self.message_passing(
                                    ref.feat, e_kernel, e_ref, e_query,
                                    query.bcenter.shape[0], conv_dict, e_weight)

        if self.norm:
            query.feat = self.norm(query.feat)

        if self.act:
            query.feat = self.act(query.feat)

        return query, conv_dict


class GridConvUpBlock(UpBlockTemplate):
    def __init__(self, block_cfg, graph_cfg):
        super().__init__(block_cfg, graph_cfg)
        input_channel = block_cfg["INPUT_CHANNEL"]
        output_channel = block_cfg["OUTPUT_CHANNEL"]
        self.key = block_cfg['KEY']

        self.message_passing = MessagePassingBlock(input_channel, output_channel, 27, self.key)

        self.kernel_assigner = grid_assign_3x3
        
    def forward(self, ref, query, conv_dict):
        assert f'{self.key}_graph' in conv_dict
        e_query, e_ref, e_weight, e_kernel = conv_dict[f'{self.key}_graph']

        query.feat, conv_dict = self.message_passing(
                                    ref.feat, e_kernel, e_ref, e_query,
                                    query.bcenter.shape[0], conv_dict, e_weight)
        
        if self.norm:
            query.feat = self.norm(query.feat)

        if self.act:
            query.feat = self.act(query.feat)

        return query, conv_dict
