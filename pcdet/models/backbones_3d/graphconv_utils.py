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
from pcdet.models.model_utils.graphconv_blocks import GraphConvBlock

class GraphConvDown(nn.Module):
    def __init__(self, in_channel, 
                 sampler_cfg, grouper_cfg, block_cfg):
        super(GraphConvDown, self).__init__()
        self.conv = GraphConvBlock(in_channel,
                                   block_cfg['DOWN_CHANNEL'],
                                   block_cfg)
        self.reuse_graph = block_cfg.get("REUSE_GRAPH", False)

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

    def forward(self, point_bxyz, point_feat, e_point=None, e_new=None):
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
        t0 = time()
        if self.grouper and (not self.reuse_graph):
            e_point, e_new = self.grouper(point_bxyz, new_bxyz)
            #print(f"num_edges={e_point.shape[0] / new_bxyz.shape[0]}", self.grouper)
        
        new_feat = self.conv(point_bxyz, point_feat, new_bxyz, e_point, e_new)

        return new_bxyz, new_feat, e_point, e_new


class GraphConvUp(nn.Module):
    def __init__(self, input_channel, skip_channel, block_cfg):
        super(GraphConvUp, self).__init__()

        up_channel = block_cfg['UP_CHANNEL']

        self.conv = GraphConvBlock(input_channel, up_channel, block_cfg)

        self.skip_mlp = nn.Linear(skip_channel, up_channel)
        self.skip_norm = nn.BatchNorm1d(up_channel)

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
            query_feat [M, C_out]
        """

        e_ref, e_query = self.grouper(ref_bxyz, query_bxyz)
        query_feat = self.conv(ref_bxyz, ref_feat, query_bxyz, e_ref, e_query)
        query_skip_feat = self.skip_norm(self.skip_mlp(query_skip_feat))
        query_feat = F.relu(query_feat + query_skip_feat, inplace=False)

        return query_feat
