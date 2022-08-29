import torch
import numpy as np
from torch import nn
from easydict import EasyDict

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
from pcdet.models.model_utils.volume_utils import VOLUMES

def grid_assign_3x3(relative_bcoords):
    assert relative_bcoords.shape[-1] == 4, relative_bcoords.dtype==torch.long
    relative_coord = relative_bcoords[:, 1:4]
    kernel_index = torch.zeros(relative_coord.shape[0], dtype=torch.long,
                               device=relative_coord.device)
    for i in range(3):
        sign = relative_coord[:, i].sign()
        offset = sign + 1
        kernel_index = kernel_index * 3 + offset
        
    return kernel_index


class VolumeConvFlatBlock(DownBlockTemplate):
    def __init__(self, block_cfg, graph_cfg, volume_cfg):
        super().__init__(block_cfg, None, graph_cfg)
        input_channel = block_cfg["INPUT_CHANNEL"]
        output_channel = block_cfg["OUTPUT_CHANNEL"]
        self.relu = block_cfg.get("RELU", True)
        self.key = block_cfg['KEY']
        
        self.volume = VOLUMES[volume_cfg["TYPE"]](
                        None,
                        volume_cfg,
                      )

        self.message_passing = MessagePassingBlock(input_channel, output_channel, self.key)

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.norm = norm_fn(output_channel)
        self.kernel_assigner = grid_assign_3x3
        
    def forward(self, ref, runtime_dict):
        query = EasyDict(ref.copy())

        if f'{self.key}_graph' in runtime_dict:
            e_ref, e_query, e_kernel, e_weight = runtime_dict[f'{self.key}_graph']
        else:
            ref = self.volume(ref, runtime_dict)
            query = self.volume(query, runtime_dict)
            e_ref, e_query, e_weight = self.graph(ref, query)
            e_kernel = self.kernel_assigner(ref.bcoords[e_ref] - query.bcoords[e_query]) # in range [0, 27)
            runtime_dict[f'{self.key}_graph'] = e_ref, e_query, e_kernel, e_weight

        query.feat, runtime_dict = self.message_passing(
                                    ref.feat, e_kernel, e_ref, e_query,
                                    query.bcoords.shape[0], runtime_dict, e_weight)

        query.feat = self.norm(query.feat)

        if self.relu:
            query.feat = F.relu(query.feat)

        return query, runtime_dict


class VolumeConvDownBlock(DownBlockTemplate):
    def __init__(self, block_cfg, sampler_cfg, graph_cfg, volume_cfg):
        super().__init__(block_cfg, sampler_cfg, graph_cfg)
        input_channel = block_cfg["INPUT_CHANNEL"]
        output_channel = block_cfg["OUTPUT_CHANNEL"]
        self.key = block_cfg['KEY']

        self.volume = VOLUMES[volume_cfg["TYPE"]](
                        None,
                        volume_cfg,
                      )
        
        self.message_passing = MessagePassingBlock(input_channel, output_channel, self.key)

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.norm = norm_fn(output_channel)
        self.kernel_assigner = grid_assign_3x3
        
    def forward(self, ref, runtime_dict):
        if self.sampler is not None:
            query = self.sampler(ref, runtime_dict)
        else:
            query = EasyDict(ref.copy())

        if f'{self.key}_graph' in runtime_dict:
            e_ref, e_query, e_kernel, e_weight = runtime_dict[f'{self.key}_graph']
        else:
            ref = self.volume(ref, runtime_dict)
            query = self.volume(query, runtime_dict)
            e_ref, e_query, e_weight = self.graph(ref, query)
            e_kernel = self.kernel_assigner(ref.bcoords[e_ref] - query.bcoords[e_query]) # in range [0, 27)
            runtime_dict[f'{self.key}_graph'] = e_ref, e_query, e_kernel, e_weight

        query.feat, runtime_dict = self.message_passing(
                                    ref.feat, e_kernel, e_ref, e_query,
                                    query.bcoords.shape[0], runtime_dict, e_weight)

        query.feat = self.norm(query.feat)

        query.feat = F.relu(query.feat)

        return query, runtime_dict


class VolumeConvUpBlock(UpBlockTemplate):
    def __init__(self, block_cfg, graph_cfg):
        super().__init__(block_cfg, graph_cfg)
        input_channel = block_cfg["INPUT_CHANNEL"]
        output_channel = block_cfg["OUTPUT_CHANNEL"]
        self.key = block_cfg['KEY']
        
        self.message_passing = MessagePassingBlock(input_channel, output_channel, self.key)

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.norm = norm_fn(output_channel)
        self.kernel_assigner = grid_assign_3x3
        
    def forward(self, ref, query, runtime_dict):
        assert f'{self.key}_graph' in runtime_dict
        e_query, e_ref, e_kernel, e_weight = runtime_dict[f'{self.key}_graph']

        query.feat, runtime_dict = self.message_passing(
                                    ref.feat, e_kernel, e_ref, e_query,
                                    query.bcoords.shape[0], runtime_dict, e_weight)
        
        query.feat = self.norm(query.feat)

        query.feat = F.relu(query.feat)

        return query, runtime_dict