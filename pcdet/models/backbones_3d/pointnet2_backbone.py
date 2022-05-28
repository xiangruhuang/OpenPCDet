import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_modules_stack
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack

from .pointnet2_utils import PointNetSetAbstractionCN2Nor, PointNetFeaturePropagationCN2

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
    def __init__(self, model_cfg, input_channels, **kwargs):
        super(PointNet2RepSurf, self).__init__()
        return_polar = model_cfg.get("RETURN_POLAR", False)
        self.sa1 = PointNetSetAbstractionCN2Nor(4, 32, input_channels - 3, [64, 64, 128], return_polar)
        self.sa2 = PointNetSetAbstractionCN2Nor(4, 32, 128, [128, 128, 256], return_polar)
        self.sa3 = PointNetSetAbstractionCN2Nor(4, 32, 256, [256, 256, 512], return_polar)
        self.sa4 = PointNetSetAbstractionCN2Nor(4, 32, 512, [512, 512, 1024], return_polar)

        self.fp4 = PointNetFeaturePropagationCN2(1024, 512, [512, 512])
        self.fp3 = PointNetFeaturePropagationCN2(512, 256, [512, 512])
        self.fp2 = PointNetFeaturePropagationCN2(512, 128, [512, 256])
        self.fp1 = PointNetFeaturePropagationCN2(256, None, [256, 256, 256])

        #self.classifier = nn.Sequential(
        #    nn.Linear(256, 256),
        #    nn.BatchNorm1d(256),
        #    nn.ReLU(inplace=True),
        #    nn.Dropout(0.5),
        #    nn.Linear(256, num_class),
        #)
        self.num_point_features = 256

    def forward(self, batch_dict):
        pos = batch_dict['points'][:, 1:4].contiguous()
        feat = batch_dict['points'][:, 4:].contiguous()
        batch_index = batch_dict['points'][:, 0]
        offset = []
        for i in range(batch_dict['batch_size']):
            offset.append((batch_index == i).sum().int())
        offset = torch.tensor(offset).int().cuda()
        pos_feat_off0 = (pos, feat, offset)
        pos_feat_off1 = self.sa1(pos_feat_off0)
        pos_feat_off2 = self.sa2(pos_feat_off1)
        pos_feat_off3 = self.sa3(pos_feat_off2)
        pos_feat_off4 = self.sa4(pos_feat_off3)

        pos_feat_off3[1] = self.fp4(pos_feat_off3, pos_feat_off4)
        pos_feat_off2[1] = self.fp3(pos_feat_off2, pos_feat_off3)
        pos_feat_off1[1] = self.fp2(pos_feat_off1, pos_feat_off2)
        pos_feat_off0[1] = self.fp1([pos_feat_off0[0], None, pos_feat_off0[2]], pos_feat_off1)

        import ipdb; ipdb.set_trace()
        return batch_dict
        return pos_feat_off0[1]

        #feature = self.classifier(pos_feat_off0[1])
        # feature = F.log_softmax(feature, dim=1)

        #return feature
