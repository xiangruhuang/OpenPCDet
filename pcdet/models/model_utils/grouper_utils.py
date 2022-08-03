from torch import nn
import torch
from torch_cluster import grid_cluster
import numpy as np
from torch_scatter import scatter

class GrouperTemplate(nn.Module):
    def __init__(self, runtime_cfg, model_cfg):
        super(GrouperTemplate, self).__init__()
        self.model_cfg = model_cfg

    def forward(self, point_bxyz):
        raise NotImplementedError


class VoxelGrouper(GrouperTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        """
        Args in model_cfg:
            grid_size (int or list of three int)
        """
        super(VoxelGrouper, self).__init__(runtime_cfg, model_cfg)

        grid_size = model_cfg.get("GRID_SIZE", None)
        self._grid_size = grid_size
        if isinstance(grid_size, list):
            grid_size = torch.tensor([1]+grid_size).float()
        else:
            grid_size = torch.tensor([1]+[grid_size for i in range(3)]).float()
        assert grid_size.shape[0] == 4, "Expecting 4D grid size." 

        self.register_buffer("grid_size", grid_size)

    def forward(self, point_bxyz):
        """Partition into groups via voxelization
        Args:
            point_bxyz [N, 4] points. (first dimension is batch index)
        Returns:
            group_id [N] indicate the group id of each point
        """
        start = point_bxyz.min(0)[0]
        start[0] -= 0.5
        end = point_bxyz.max(0)[0]
        end[0] += 0.5

        cluster = grid_cluster(point_bxyz, self.grid_size, start=start, end=end)
        _, group_ids = torch.unique(cluster, sorted=True, return_inverse=True)

        return group_ids

    def __repr__(self):
        return "VoxelGrouper(grid size={})".format(self._grid_size)


GROUPERS = dict(
    VoxelGrouper=VoxelGrouper,
)
