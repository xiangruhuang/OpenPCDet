import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from .vfe_template import VFETemplate
from pcdet.models.blocks import MLPBlock
from pcdet.ops.voxel import VoxelGraph

class DynamicVFE(VFETemplate):
    def __init__(self, model_cfg, runtime_cfg):
        super().__init__(model_cfg=model_cfg, runtime_cfg=runtime_cfg)
        self.model_cfg = model_cfg
        self.runtime_cfg = runtime_cfg

        self.use_volume = model_cfg.get("USE_VOLUME", False)

        self.point_feature_cfg = self.model_cfg.get("POINT_FEATURE_CFG", [])
        
        num_point_features = runtime_cfg.get("num_point_features", None)
        num_point_features += 3
        for key, size in self.point_feature_cfg.items():
            num_point_features += size
        self.scale = runtime_cfg.get("scale", 1.0)

        self.mlp_channels = self.model_cfg.get("MLP_CHANNELS", None)
        self.mlp_channels = [int(self.scale*c) for c in self.mlp_channels]
        assert len(self.mlp_channels) > 0
        mlp_channels = [num_point_features] + list(self.mlp_channels)

        self.norm_cfg = self.model_cfg.get("NORM_CFG", None)

        self.vfe_layers = nn.ModuleList()
        for i in range(len(mlp_channels) - 1):
            in_channel = mlp_channels[i]

            if i > 0:
                in_channel *= 2
            out_channel = mlp_channels[i + 1]
            self.vfe_layers.append(
                MLPBlock(in_channel, out_channel, self.norm_cfg, activation=nn.ReLU(), bias=False)
            )

        self.voxel_graph_cfg = model_cfg.get("VOXEL_GRAPH_CFG", None)
        self.runtime_cfg.update(self.voxel_graph_cfg)
        self.voxel_graph = VoxelGraph(model_cfg=self.voxel_graph_cfg,
                                      runtime_cfg=self.runtime_cfg)

        self.num_point_features = self.mlp_channels[-1]
        runtime_cfg['input_channels'] = self.mlp_channels[-1]
        self.output_key = 'voxel'

    def get_output_feature_dim(self):
        return self.num_point_features

    def process_point_features(self, voxel_wise_dict, batch_dict, voxel_index, out_of_boundary_mask):
        """
        Args:
            voxel_wise_dict: attributes that has shape [V, ?]
            batch_dict: input data
            voxel_index [N] : the voxel index of each point
        Returns:
            point_features [N, C_out]
        """
        point_xyz = batch_dict['point_bxyz'][~out_of_boundary_mask, 1:4].contiguous()
        point_feat = batch_dict['point_feat'][~out_of_boundary_mask]
        
        feature_list = [point_xyz, point_feat]
        if 'offset_to_voxel_xyz' in self.point_feature_cfg:
            feature_list.append(point_xyz-voxel_wise_dict['voxel_xyz'][voxel_index])
        if 'offset_to_voxel_center' in self.point_feature_cfg:
            feature_list.append(point_xyz-voxel_wise_dict['voxel_center'][voxel_index])

        return torch.cat(feature_list, dim=-1)

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            point_bxyz [N, 4] input point coordinates
            point_feat [N, C] input point features
        Returns:
            voxel_features [V, C] output feature per voxel
            voxel_coords [V, 4] integer coordinate of each voxel
        """
        point_bxyz = batch_dict['point_bxyz'] # (batch_idx, x, y, z)
        point_feat = batch_dict['point_feat'] # (i, e)
        point_wise_mean_dict=dict(
            point_bxyz=point_bxyz,
            point_feat=point_feat,
        )

        voxel_wise_dict, voxel_index, num_voxels, out_of_boundary_mask = \
                self.voxel_graph(point_wise_mean_dict,
                                 point_wise_median_dict=dict(
                                     segmentation_label = batch_dict['segmentation_label']
                                 ))
        if self.use_volume:
            point_xyz = point_bxyz[:, 1:]
            voxel_volume = scatter(point_xyz.new_ones(point_xyz.shape[0]), voxel_index,
                                   dim=0, dim_size=num_voxels, reduce='sum')
            assert (voxel_volume > 0.5).all()
            voxel_xyz = scatter(point_xyz, voxel_index, dim=0,
                                dim_size=num_voxels, reduce='mean')
            point_d = point_xyz - voxel_xyz[voxel_index]
            point_ddT = point_d.unsqueeze(-1) * point_d.unsqueeze(-2)
            voxel_ddT = scatter(point_ddT, voxel_index, dim=0,
                                dim_size=num_voxels, reduce='mean')

            import ipdb; ipdb.set_trace()
            voxel_eigvals, voxel_eigvecs = torch.linalg.eigh(voxel_ddT) # eigvals in ascending order
            voxel_wise_dict['voxel_eigvals'] = voxel_eigvals
            voxel_wise_dict['voxel_eigvecs'] = voxel_eigvecs
                                 
        point_features = self.process_point_features(voxel_wise_dict, batch_dict,
                                                     voxel_index, out_of_boundary_mask)

        for i, vfe_layer in enumerate(self.vfe_layers):
            point_features = vfe_layer(point_features)
            voxel_features = scatter(point_features, voxel_index, dim_size=num_voxels, dim=0, reduce='mean')
            if i != len(self.vfe_layers) - 1:
                point_features = torch.cat([point_features, voxel_features[voxel_index]], dim=-1)

        voxel_wise_dict['voxel_feat'] = voxel_features
        batch_dict.update(voxel_wise_dict)

        return batch_dict
