import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_modules_stack
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack

from .pointnet2_utils import (
    PointNetSetAbstractionCN2Nor,
    PointNetFeaturePropagationCN2,
    batch_index_to_offset
)
from .post_processors import build_post_processor

class PointNet2MSG(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3

        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules.PointnetSAModuleMSG(
                    npoint=self.model_cfg.SA_CONFIG.NPOINTS[k],
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(
                pointnet2_modules.PointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
                )
            )

        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )  # (B, C, N)

        point_features = l_features[0].permute(0, 2, 1).contiguous()  # (B, N, C)
        batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1])
        batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0].view(-1, 3)), dim=1)
        return batch_dict

class PointNet2RepSurf(nn.Module):
    def __init__(self, runtime_cfg, model_cfg, **kwargs):
        super(PointNet2RepSurf, self).__init__()
        input_channels = runtime_cfg.get("num_point_features", None)
        if model_cfg.get("INPUT_KEY", None) is not None:
            self.input_key = model_cfg.get("INPUT_KEY", None)
        else:
            self.input_key = runtime_cfg.get("input_key", 'point')
        max_num_points = [runtime_cfg.get("max_num_points", None)]
        
        return_polar = model_cfg.get("RETURN_POLAR", False)
        self.strides = model_cfg.get("STRIDES", None)
        T = runtime_cfg.get("scale", 1)
        sa_channels = model_cfg["SA_CHANNELS"]
        fp_channels = model_cfg["FP_CHANNELS"]
        num_sectors = model_cfg["NUM_SECTORS"]
        num_neighbors = model_cfg.get("NUM_NEIGHBORS", 32)
        self.output_key = model_cfg.get("OUTPUT_KEY", None)
        self.sa_modules = nn.ModuleList()

        cur_channel = input_channels
        channel_stack = []
        for i, sa_channel in enumerate(sa_channels):
            sa_channel = [int(c*T) for c in sa_channel]
            sa_module = PointNetSetAbstractionCN2Nor(self.strides[i], num_neighbors, cur_channel, sa_channel, return_polar, num_sectors=num_sectors[i])
            max_num_points.append(max_num_points[-1] // self.strides[i])
            self.sa_modules.append(sa_module)
            channel_stack.append(cur_channel)
            cur_channel = sa_channel[-1]

        self.fp_modules = nn.ModuleList()
        for i, fp_channel in enumerate(fp_channels):
            fp_channel = [int(c*T) for c in fp_channel]
            fp_module = PointNetFeaturePropagationCN2(cur_channel, channel_stack.pop(), fp_channel)
            max_num_points.pop()
            self.fp_modules.append(fp_module)
            cur_channel = fp_channel[-1]

        self.max_num_points = max_num_points[-1]
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
        if 'memory' not in batch_dict:
            batch_dict['memory'] = {}

        pos = batch_dict[f'{self.input_key}_bxyz'][:, 1:4].contiguous()
        feat = batch_dict[f'{self.input_key}_feat']
        batch_index = batch_dict[f'{self.input_key}_bxyz'][:, 0].round().long()
        offset = batch_index_to_offset(batch_index)

        data_stack = []
        pos_feat_off = [pos, feat, offset]
        data_stack.append(pos_feat_off)
        for i, sa_module in enumerate(self.sa_modules):
            memory_before = torch.cuda.memory_allocated()/2**30
            pos_feat_off = sa_module(pos_feat_off)
            memory_after = torch.cuda.memory_allocated()/2**30
            data_stack.append(pos_feat_off)
            key = f'pointnet2_sa{len(self.sa_modules)-i}_out'
            batch_dict[f'{key}_bxyz'] = self.convert_to_bxyz(pos_feat_off)
            batch_dict[f'{key}_feat'] = pos_feat_off[1]
            batch_dict['memory'][f'{key}'] = memory_after - memory_before

        pos_feat_off = data_stack.pop()
        for i, fp_module in enumerate(self.fp_modules):
            pos_feat_off_cur = data_stack.pop()
            memory_before = torch.cuda.memory_allocated()/2**30
            pos_feat_off_cur[1] = fp_module(pos_feat_off_cur, pos_feat_off)
            memory_after = torch.cuda.memory_allocated()/2**30
            pos_feat_off = pos_feat_off_cur
            key = f'pointnet2_fp{i+1}_out'
            batch_dict[f'{key}_bxyz'] = self.convert_to_bxyz(pos_feat_off)
            batch_dict[f'{key}_feat'] = pos_feat_off[1]
            batch_dict['memory'][f'{key}'] = memory_after - memory_before

        if self.output_key is not None:
            batch_dict[f'{self.output_key}_bxyz'] = self.convert_to_bxyz(pos_feat_off)
            batch_dict[f'{self.output_key}_feat'] = pos_feat_off_cur[1]

        if self.post_processor:
            batch_dict = self.post_processor(batch_dict)

        return batch_dict
