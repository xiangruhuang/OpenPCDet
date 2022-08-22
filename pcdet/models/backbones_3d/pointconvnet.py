import torch
import torch.nn as nn

from pcdet.utils import common_utils
from pcdet.models.model_utils import graph_utils
from .post_processors import build_post_processor
from pcdet.models.blocks import (
    GridConvDownBlock,
    GridConvFlatBlock,
    GridConvUpBlock,
)


class PointConvNet(nn.Module):
    def __init__(self, runtime_cfg, model_cfg, **kwargs):
        super(PointConvNet, self).__init__()
        input_channels = runtime_cfg.get("num_point_features", None)
        if model_cfg.get("INPUT_KEY", None) is not None:
            self.input_key = model_cfg.get("INPUT_KEY", None)
        else:
            self.input_key = runtime_cfg.get("input_key", 'point')
        
        self.samplers = model_cfg.get("SAMPLERS", None)
        self.graphs = model_cfg.get("GRAPHS", None)
        self.sa_channels = model_cfg.get("SA_CHANNELS", None)
        self.fp_channels = model_cfg.get("FP_CHANNELS", None)
        self.num_global_channels = model_cfg.get("NUM_GLOBAL_CHANNELS", 0)
        
        self.scale = runtime_cfg.get("scale", 1)
        #fp_channels = model_cfg["FP_CHANNELS"]
        self.output_key = model_cfg.get("OUTPUT_KEY", None)

        self.down_modules = nn.ModuleList()
        self.down_flat_modules = nn.ModuleList()

        cur_channel = input_channels
        channel_stack = []
        for i, sa_channels in enumerate(self.sa_channels):
            sampler_cfg = common_utils.indexing_list_elements(self.samplers, i)
            graph_cfg = graph_utils.select_graph(self.graphs, i)
            prev_graph_cfg = graph_utils.select_graph(self.graphs, max(i-1, 0))
            sa_channels = [int(self.scale*c) for c in sa_channels]
            
            down_module = nn.ModuleList()
            for j, sc in enumerate(sa_channels):
                block_cfg = dict(
                    INPUT_CHANNEL=cur_channel,
                    OUTPUT_CHANNEL=sc,
                )
                if j == 0:
                    down_module_j = GridConvDownBlock(block_cfg,
                                                      sampler_cfg,
                                                      prev_graph_cfg)
                else:
                    down_module_j = GridConvFlatBlock(block_cfg,
                                                      graph_cfg)
                down_module.append(down_module_j)

                cur_channel = sc

            self.down_modules.append(down_module)
            channel_stack.append(cur_channel)
        
        self.up_modules = nn.ModuleList()
        self.skip_modules = nn.ModuleList()
        self.merge_modules = nn.ModuleList()
        for i, fp_channels in enumerate(self.fp_channels):
            graph_cfg = graph_utils.select_graph(self.graphs, -i-1)
            prev_graph_cfg = graph_utils.select_graph(self.graphs, max(-i-2, 0))
            fc0, fc1, fc2 = [int(self.scale*c) for c in fp_channels]
            skip_channel = channel_stack.pop()
            self.skip_modules.append(
                GridConvFlatBlock(
                    dict(
                        INPUT_CHANNEL=skip_channel,
                        OUTPUT_CHANNEL=fc0,
                    ),
                    graph_cfg,
                ))
            
            self.up_modules.append(
                GridConvUpBlock(
                    dict(
                        INPUT_CHANNEL=fc1,
                        OUTPUT_CHANNEL=fc2,
                    ),
                    graph_cfg=prev_graph_cfg
                ))
            
            self.merge_modules.append(
                GridConvFlatBlock(
                    dict(
                        INPUT_CHANNEL=fc0*2,
                        OUTPUT_CHANNEL=fc1,
                    ),
                    graph_cfg,
                ))
            cur_channel = fc2

        self.num_point_features = cur_channel
        
        runtime_cfg['input_key'] = self.output_key
        self.post_processor = build_post_processor(model_cfg.get("POST_PROCESSING_CFG", {}),
                                                   runtime_cfg)
        if self.post_processor:
            self.num_point_features = self.post_processor.num_point_features

    def forward(self, batch_dict):
        point_bxyz = batch_dict[f'{self.input_key}_bcenter']
        point_feat = batch_dict[f'{self.input_key}_feat']

        data_stack = []
        data_stack.append([point_bxyz, point_feat])
        
        for i, down_module in enumerate(self.down_modules):
            key = f'pointnet2_down{len(self.sa_channels)-i}_out'
            for j, down_module_j in enumerate(down_module):
                point_bxyz, point_feat, down_ref, down_query = down_module_j(point_bxyz, point_feat)
                if j == 0:
                    batch_dict[f'{key}_edges'] = torch.stack([down_query, down_ref], dim=0)
                elif j == 1:
                    batch_dict[f'{key}_flat_edges'] = torch.stack([down_query, down_ref], dim=0)
            batch_dict[f'{key}_ref'] = point_bxyz
            batch_dict[f'{key}_query'] = point_bxyz
            data_stack.append([point_bxyz, point_feat])
            batch_dict[f'{key}_bxyz'] = point_bxyz
            batch_dict[f'{key}_feat'] = point_feat

        point_bxyz_ref, point_feat_ref = data_stack.pop()
        #for i, global_module in enumerate(self.global_modules):
        #    point_feat_ref = global_module(point_bxyz_ref, point_feat_ref)

        point_skip_feat_ref = point_feat_ref
        for i, (up_module, skip_module, merge_module) in enumerate(zip(self.up_modules, self.skip_modules, self.merge_modules)):
            key = f'pointnet2_up{i+1}_out'
            # skip transformation and merging
            _, point_skip_feat_ref, skip_ref, skip_query = \
                    skip_module(point_bxyz_ref, point_skip_feat_ref)
            batch_dict[f'{key}_ref'] = point_bxyz_ref

            point_concat_feat_ref = torch.cat([point_feat_ref, point_skip_feat_ref], dim=-1)
            _, point_merge_feat_ref, merge_ref, merge_query = \
                    merge_module(point_bxyz_ref, point_concat_feat_ref)
            num_ref_points = point_bxyz_ref.shape[0]
            point_feat_ref = point_merge_feat_ref \
                             + point_concat_feat_ref.view(num_ref_points, -1, 2).sum(dim=2)

            # upsampling
            point_bxyz_query, point_skip_feat_query = data_stack.pop()
            point_feat_query, up_ref, up_query = up_module(point_bxyz_ref, point_feat_ref,
                                                           point_bxyz_query)
            point_bxyz_ref, point_feat_ref = point_bxyz_query, point_feat_query
            point_skip_feat_ref = point_skip_feat_query

            batch_dict[f'{key}_bxyz'] = point_bxyz_ref
            batch_dict[f'{key}_feat'] = point_feat_ref
            batch_dict[f'{key}_skip_edges'] = torch.stack([skip_query, skip_ref], dim=0)
            batch_dict[f'{key}_merge_edges'] = torch.stack([merge_query, merge_ref], dim=0)
            batch_dict[f'{key}_up_edges'] = torch.stack([up_query, up_ref], dim=0)

        if self.output_key is not None:
            batch_dict[f'{self.output_key}_bxyz'] = point_bxyz_ref
            batch_dict[f'{self.output_key}_feat'] = point_feat_ref

        if self.post_processor:
            batch_dict = self.post_processor(batch_dict)

        return batch_dict
