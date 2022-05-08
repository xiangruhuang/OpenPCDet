import torch

from .vfe_template import VFETemplate
from ..grid_sampling import GridSampling3D
from ....ops.hybrid_geop import PrimitiveFitting

class HybridVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features
        self.grid_size = model_cfg.get("GRID_SIZE", [0.1, 0.1, 0.15])
        max_num_points = model_cfg.get("MAX_NUM_POINTS", 800000)
        self.grid_sampler = GridSampling3D(self.grid_size)
        self.pf = PrimitiveFitting(self.grid_size, max_num_points)

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
        points = batch_dict['points'][:, :4].contiguous() # [N, 4]
        #grid_points = self.grid_sampler(points) # [M, 4]
        
        mu, R = self.pf(points)

        primitives = torch.cat([mu, R.reshape(-1, 9)], dim=-1)
        batch_dict['primitives'] = primitives
        
        return batch_dict
