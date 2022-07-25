import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_modules_stack
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack

from .pointnet2_utils import (
    PointNetSetAbstraction,
    PointNetFeaturePropagation,
)
from ...utils import common_utils
from .post_processors import build_post_processor

class PointNet2(nn.Module):
    def __init__(self, runtime_cfg, model_cfg, **kwargs):
        super(PointNet2, self).__init__()
        input_channels = runtime_cfg.get("num_point_features", None)
        if model_cfg.get("INPUT_KEY", None) is not None:
            self.input_key = model_cfg.get("INPUT_KEY", None)
        else:
            self.input_key = runtime_cfg.get("input_key", 'point')
        
        self.samplers = model_cfg.get("SAMPLERS", None)
        self.groupers = model_cfg.get("GROUPERS", None)
        self.pos_encoders = model_cfg.get("POS_ENCODERS", None)
        
        T = runtime_cfg.get("scale", 1)
        sa_channels = model_cfg["SA_CHANNELS"]
        fp_channels = model_cfg["FP_CHANNELS"]
        self.output_key = model_cfg.get("OUTPUT_KEY", None)
        self.sa_modules = nn.ModuleList()

        cur_channel = input_channels
        channel_stack = []
        for i, sa_channel in enumerate(sa_channels):
            sampler_cfg = common_utils.filter_dict(self.samplers, i, ignore_keys=['TYPE'])
            grouper_cfg = common_utils.filter_dict(self.groupers, i, ignore_keys=['TYPE'])
            pos_encoder_cfg = common_utils.filter_dict(self.pos_encoders, i, ignore_keys=['TYPE'])
            sa_channel = [int(c*T) for c in sa_channel]
            sa_module = PointNetSetAbstraction(cur_channel, sa_channel,
                                               sampler_cfg, grouper_cfg,
                                               pos_encoder_cfg)
            self.sa_modules.append(sa_module)
            channel_stack.append(cur_channel)
            cur_channel = sa_channel[-1]

        self.fp_modules = nn.ModuleList()
        for i, fp_channel in enumerate(fp_channels):
            fp_channel = [int(c*T) for c in fp_channel]
            fp_module = PointNetFeaturePropagation(cur_channel, channel_stack.pop(), fp_channel)
            self.fp_modules.append(fp_module)
            cur_channel = fp_channel[-1]

        self.num_point_features = cur_channel
        
        runtime_cfg['input_key'] = self.output_key
        self.post_processor = build_post_processor(model_cfg.get("POST_PROCESSING_CFG", {}),
                                                   runtime_cfg)
        if self.post_processor:
            self.num_point_features = self.post_processor.num_point_features

    def convert_to_bxyz(self, pos_feat_off):
        xyz = pos_feat_off[0]
        assert xyz.shape[-1] == 3, f"expecting xyz to have shape [..., 3], got {xyz.shape}"
        offset = pos_feat_off[2]
        batch_idx = []
        last_offset = 0
        for i, offset_i in enumerate(offset):
            batch_idx.append(torch.full([offset_i-last_offset, 1], i).long().to(xyz.device))
            last_offset = offset_i
        batch_idx = torch.cat(batch_idx, dim=0)
        bxyz = torch.cat([batch_idx, xyz], dim=-1)

        return bxyz

    def forward(self, batch_dict):

        point_bxyz = batch_dict[f'{self.input_key}_bxyz']
        point_feat = batch_dict[f'{self.input_key}_feat']

        data_stack = []
        data_stack.append([point_bxyz, point_feat])
        for i, sa_module in enumerate(self.sa_modules):
            point_bxyz, point_feat = sa_module(point_bxyz, point_feat)
            data_stack.append([point_bxyz, point_feat])
            key = f'pointnet2_sa{len(self.sa_modules)-i}_out'
            batch_dict[f'{key}_bxyz'] = point_bxyz
            batch_dict[f'{key}_feat'] = point_feat

        point_bxyz, point_feat = data_stack.pop()
        for i, fp_module in enumerate(self.fp_modules):
            point_bxyz_cur, point_feat_cur = data_stack.pop()
            point_feat_cur = fp_module(point_bxyz, point_feat,
                                       point_bxyz_cur, point_feat_cur)
            point_bxyz, point_feat = point_bxyz_cur, point_feat_cur
            key = f'pointnet2_fp{i+1}_out'
            batch_dict[f'{key}_bxyz'] = point_bxyz 
            batch_dict[f'{key}_feat'] = point_feat

        if self.output_key is not None:
            batch_dict[f'{self.output_key}_bxyz'] = point_bxyz
            batch_dict[f'{self.output_key}_feat'] = point_feat

        if self.post_processor:
            batch_dict = self.post_processor(batch_dict)

        return batch_dict
