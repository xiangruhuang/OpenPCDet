import torch
import torch.nn as nn
from collections import defaultdict
import time

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_modules_stack
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack

#from .pointnet2_utils import (
#    PointNetSetAbstraction,
#    PointNetFeaturePropagation,
#)
from ...utils import common_utils
from .post_processors import build_post_processor
from .graphconv_utils import GraphConvDown, GraphConvUp

class GraphConv(nn.Module):
    def __init__(self, runtime_cfg, model_cfg, **kwargs):
        super(GraphConv, self).__init__()
        input_channels = runtime_cfg.get("num_point_features", None)
        if model_cfg.get("INPUT_KEY", None) is not None:
            self.input_key = model_cfg.get("INPUT_KEY", None)
        else:
            self.input_key = runtime_cfg.get("input_key", 'point')
        
        self.samplers = model_cfg.get("SAMPLERS", None)
        self.groupers = model_cfg.get("GROUPERS", None)
        self.blocks = model_cfg.get("BLOCKS", None)
        self.num_blocks = model_cfg["NUM_BLOCKS"]
        
        T = runtime_cfg.get("scale", 1)
        #fp_channels = model_cfg["FP_CHANNELS"]
        self.output_key = model_cfg.get("OUTPUT_KEY", None)
        self.down_modules = nn.ModuleList()

        cur_channel = input_channels
        channel_stack = []
        for i in range(self.num_blocks):
            sampler_cfg = common_utils.indexing_list_elements(self.samplers, i)
            grouper_cfg = common_utils.indexing_list_elements(self.groupers, i)
            block_cfg = common_utils.indexing_list_elements(self.blocks, i)
            down_module = GraphConvDown(cur_channel, sampler_cfg,
                                        grouper_cfg, block_cfg)
            self.down_modules.append(down_module)
            channel_stack.append(cur_channel)
            cur_channel = block_cfg['DOWN_CHANNEL']

        self.up_modules = nn.ModuleList()
        for i in range(self.num_blocks):
            block_cfg = common_utils.indexing_list_elements(self.blocks, (-1 - i))
            #up_module = PointNetFeaturePropagation(cur_channel, channel_stack.pop(), up_channel)
            up_module = GraphConvUp(cur_channel, channel_stack.pop(), block_cfg)
            self.up_modules.append(up_module)
            cur_channel = block_cfg['UP_CHANNEL']

        self.num_point_features = cur_channel
        
        runtime_cfg['input_key'] = self.output_key
        self.post_processor = build_post_processor(model_cfg.get("POST_PROCESSING_CFG", {}),
                                                   runtime_cfg)
        if self.post_processor:
            self.num_point_features = self.post_processor.num_point_features

        self.timer = defaultdict(lambda : 0.0)

    def forward(self, batch_dict):
        point_bxyz = batch_dict[f'{self.input_key}_bxyz']
        point_feat = batch_dict[f'{self.input_key}_feat']

        data_stack = []
        data_stack.append([point_bxyz, point_feat])
        
        for i, down_module in enumerate(self.down_modules):
            self.timer[f'down_{i}'] -= time.time()
            point_bxyz, point_feat = down_module(point_bxyz, point_feat)
            self.timer[f'down_{i}'] += time.time()
            data_stack.append([point_bxyz, point_feat])
            key = f'graphconv_sa{self.num_blocks-i}_out'
            batch_dict[f'{key}_bxyz'] = point_bxyz
            batch_dict[f'{key}_feat'] = point_feat
            #print(f'DownBlock({i}): memory={torch.cuda.memory_allocated()/2**30}')

        point_bxyz, point_feat = data_stack.pop()
        for i, up_module in enumerate(self.up_modules):
            self.timer[f'up_{i}'] -= time.time()
            point_bxyz_cur, point_feat_cur = data_stack.pop()
            point_feat_cur = up_module(point_bxyz, point_feat,
                                       point_bxyz_cur, point_feat_cur)
            self.timer[f'up_{i}'] += time.time()
            point_bxyz, point_feat = point_bxyz_cur, point_feat_cur
            key = f'graphconv_fp{i+1}_out'
            batch_dict[f'{key}_bxyz'] = point_bxyz 
            batch_dict[f'{key}_feat'] = point_feat
            #print(f'UpBlock({i}): memory={torch.cuda.memory_allocated()/2**30}')

        if self.output_key is not None:
            batch_dict[f'{self.output_key}_bxyz'] = point_bxyz
            batch_dict[f'{self.output_key}_feat'] = point_feat

        if self.post_processor:
            self.timer[f'postprocessor'] -= time.time()
            batch_dict = self.post_processor(batch_dict)
            self.timer[f'postprocessor'] += time.time()
            #print(f'PostProcessor: memory={torch.cuda.memory_allocated()/2**30}')

        return batch_dict
