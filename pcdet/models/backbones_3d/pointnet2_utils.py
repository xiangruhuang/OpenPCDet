import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from torch_scatter import scatter

from ...utils.polar_utils import xyz2sphere, xyz2cylind, xyz2sphere_aug
from ...ops.pointops.functions import pointops, pointops_utils

from pcdet.models.model_utils.sampler_utils import SAMPLERS
from pcdet.models.model_utils.grouper_utils import GROUPERS
from pcdet.models.model_utils.basic_blocks import PointConv

class PointNetSetAbstraction(nn.Module):
    """
    SA Module (normal input) with CN (pre-bn; xyz and normal separate)

    """

    def __init__(self, in_channel, mlp,
                 sampler_cfg=None, grouper_cfg=None,
                 pos_encoder_cfg=None):
        super(PointNetSetAbstraction, self).__init__()
        #self.stride = stride
        #self.return_polar = return_polar
        #self.nsample = nsample
        self.pos_channel = 3

        self.conv0 = PointConv(in_channel, mlp[0], pos_encoder_cfg)
        #self.mlp_l0 = nn.Linear(self.pos_channel, mlp[0], bias=False)
        #self.norm_l0 = nn.BatchNorm1d(mlp[0])
        #if in_channel > 0:
        #    self.mlp_f0 = nn.Linear(in_channel, mlp[0], bias=False)
        #    self.norm_f0 = nn.BatchNorm1d(mlp[0])

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = mlp[0]
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Linear(last_channel, out_channel))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

        if sampler_cfg is not None:
            sampler = SAMPLERS[sampler_cfg.pop("TYPE")]
            self.sampler = sampler(
                               runtime_cfg=None,
                               model_cfg=sampler_cfg,
                           )
        
        if grouper_cfg is not None:
            grouper = GROUPERS[grouper_cfg.pop("TYPE")]
            self.grouper = grouper(
                               runtime_cfg=None,
                               model_cfg=grouper_cfg,
                           )

        #self.pos_encoder = nn.Linear(3, mlp_channel)
        #self.pos_encoder = []
        #self.div_factor = 4
        #self.num_kernels = 15
        #self.num_act_kernels = 3
        #self.kernel_pos = torch.
        #for 

    def forward(self, point_bxyz, point_feat):
        """
        Input:
            point_bxyz [N, 4]: input points, first dimension indicates batch index
            point_feat [N, C]: per-point feature vectors
        Return:
            new_bxyz: sampled points [M, 4]
            new_feat: per-sampled-point feature vector [M, C_out]
        """
        if self.sampler:
            new_bxyz = self.sampler(point_bxyz)
        else:
            new_bxyz = point_bxyz
        
        if self.grouper:
            e_point, e_new = self.grouper(point_bxyz, new_bxyz)
        
        new_feat = self.conv0(point_bxyz, point_feat, new_bxyz, e_point, e_new)
        # position encoding
        #pos_tokens = self.pos_encoder(point_bxyz[e_point, 1:4] - new_bxyz[e_new, 1:4]) # [E, act_k]

        #fps_idx, (e_new, e_point) = pointops_utils.knn_graph(
        #                                point_bxyz, self.stride, self.nsample,
        #                                num_sectors=self.num_sectors
        #                            )

        #new_bxyz = point_bxyz[fps_idx]
        #point_out_feat = self.norm_f0(self.mlp_f0(point_feat)) # [N, C_out]
        #new_feat = scatter(point_out_feat[e_point], e_new, dim=0,
        #                   dim_size=new_bxyz.shape[0], reduce='mean') # [M, C_out]

        ### new_xyz: sampled points position data, [M, 3]
        ### new_points: sampled points data, [M, nsample, C+3]
        #new_points = new_points.transpose(1, 2).contiguous()  # [M, 3+C, nsample]

        ## init layer
        #loc = self.norm_l0(self.mlp_l0(new_points[:, :self.pos_channel]))
        #if points is not None:
        #    feat = self.norm_f0(self.mlp_f0(new_points[:, self.pos_channel:]))
        #    new_points = F.relu(loc + feat, inplace=False)
        #else:
        #    new_points = F.relu(loc, inplace=False)

        ## mlp
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feat = conv(new_feat)
            new_feat = F.relu(bn(new_feat), inplace=False)
        #new_feat = torch.max(new_feat, 2)[0]

        return new_bxyz, new_feat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, prev_channel, skip_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.skip = skip_channel is not None

        self.mlp_f0 = nn.Linear(prev_channel, mlp[0])
        self.norm_f0 = nn.BatchNorm1d(mlp[0])
        if skip_channel is not None:
            self.mlp_s0 = nn.Linear(skip_channel, mlp[0])
            self.norm_s0 = nn.BatchNorm1d(mlp[0])

        last_channel = mlp[0]
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Linear(last_channel, out_channel))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        self.grouper = GROUPERS['KNNGrouper'](
                           runtime_cfg=None,
                           model_cfg=dict(
                               num_neighbors=3,
                           )
                       )

    def forward(self, ref_bxyz, ref_feat, query_bxyz, query_skip_feat):
        """
        Input:

        Return:
            query_feat
        """
        #xyz1, points1, offset1 = pos_feat_off1  # [N, 3], [N, C], [B]
        #xyz2, points2, offset2 = pos_feat_off2  # [M, 3], [M, C], [B]
        num_queries = query_bxyz.shape[0]

        e_ref, e_query = self.grouper(ref_bxyz, query_bxyz)
        diff = (ref_bxyz[e_ref]  - query_bxyz[e_query])[:, 1:4]
        dist = diff.norm(p=2, dim=-1)
        dist_recip = 1.0 / (dist + 1e-8)  # [M, 3]
        query_weight = scatter(dist_recip, e_query, dim=0,
                               dim_size=num_queries, reduce='sum')

        query_feat = scatter(ref_feat[e_ref], e_query, dim=0,
                             dim_size=num_queries, reduce='sum')
        query_feat /= query_weight[:, None]

        # interpolation
        #idx, dist = pointops.knnquery(3, xyz2, xyz1, offset2, offset1)  # [M, 3], [M, 3]
        #norm = torch.sum(dist_recip, dim=1, keepdim=True)
        #weight = dist_recip / norm  # [M, 3]

        query_feat = self.norm_f0(self.mlp_f0(query_feat))
        #interpolated_points = torch.cuda.FloatTensor(xyz1.shape[0], points2.shape[1]).zero_()
        #for i in range(3):
        #    interpolated_points += points2[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1)

        # init layer
        if self.skip:
            query_skip_feat = self.norm_s0(self.mlp_s0(query_skip_feat))
            query_feat = F.relu(query_feat + query_skip_feat, inplace=False)
        else:
            query_feat = F.relu(query_feat, inplace=False)

        # mlp
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            query_feat = F.relu(bn(conv(query_feat)), inplace=False)

        return query_feat
