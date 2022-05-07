import torch

from .vfe_template import VFETemplate


class HybridVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        import ipdb; ipdb.set_trace()
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points'].long()
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()
        voxel_seg_labels = []
        indices0 = torch.arange(voxel_features.shape[0]
                               ).unsqueeze(-1).to(voxel_num_points) # [N, 1]
        indices1 = torch.arange(voxel_features.shape[1]
                               ).unsqueeze(0).to(voxel_num_points) # [1, C]
        indices1 = indices1 % voxel_num_points.unsqueeze(-1) # [1, C] % [N, 1] = [N, C]
        voxel_seg_labels = batch_dict['voxel_point_seg_labels'][:, :, 1] # [N, C]
        voxel_seg_labels_median = voxel_seg_labels[(indices0, indices1)].median(-1)[0]

        batch_dict['voxel_seg_labels'] = voxel_seg_labels_median
        return batch_dict
