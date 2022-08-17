import torch
from torch import nn

from torch_scatter import scatter
from torch_cluster import grid_cluster

from pcdet.ops.pointops.functions.pointops import (
    furthestsampling,
    sectorized_fps,
)
from pcdet.ops.voxel.voxel_modules import VoxelGraph

@torch.no_grad()
def bxyz_to_xyz_index_offset(point_bxyz):
    num_points = []
    batch_size = point_bxyz[:, 0].max().round().long().item() + 1
    for i in range(batch_size):
        num_points.append((point_bxyz[:, 0].round().long() == i).int().sum())
    num_points = torch.stack(num_points, dim=0).reshape(-1).int()
    _, indices = torch.sort(point_bxyz[:, 0])
    offset = num_points.cumsum(dim=0).int()
    point_xyz = point_bxyz[indices, 1:4].contiguous()
    return point_xyz, indices.long(), offset

class SamplerTemplate(nn.Module):
    def __init__(self, runtime_cfg, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg

    def forward(self, point_bxyz):
        assert NotImplementedError


class VoxelCenterSampler(SamplerTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        super(VoxelCenterSampler, self).__init__(
                                     runtime_cfg=runtime_cfg,
                                     model_cfg=model_cfg,
                                 )
        voxel_size = model_cfg.get("VOXEL_SIZE", None)
        self._voxel_size = voxel_size
        if isinstance(voxel_size, list):
            voxel_size = torch.tensor([1]+voxel_size).float()
        else:
            voxel_size = torch.tensor([1]+[voxel_size for i in range(3)]).float()
        assert voxel_size.shape[0] == 4, "Expecting 4D voxel size." 
        self.register_buffer("voxel_size", voxel_size)

        stride = model_cfg.get("STRIDE", 1)
        if not isinstance(stride, list):
            stride = [stride for i in range(3)]
        stride = torch.tensor(stride, dtype=torch.int64)
        self.register_buffer('stride', stride)
        
        self.z_padding = model_cfg.get("Z_PADDING", 1)

        model_cfg_cp = {}
        model_cfg_cp.update(model_cfg)
        model_cfg_cp['VOXEL_SIZE'] = [voxel_size[1+i] / stride[i] for i in range(3)]
        self.voxel_graph = VoxelGraph(model_cfg=model_cfg_cp, runtime_cfg=runtime_cfg)
        
        #point_cloud_range = model_cfg.get("POINT_CLOUD_RANGE", None)
        #if point_cloud_range is None:
        #    self.point_cloud_range = None
        #else:
        #    point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        #    self.register_buffer("point_cloud_range", point_cloud_range)
        
    def forward(self, point_bxyz):
        """
        Args:
            point_bxyz [N, 4]: (b,x,y,z), first dimension is batch index
        Returns:
            voxel_center: [V, 4] sampled centers of voxels
        """

        with torch.no_grad():
            point_bxyz_list = []
            for dx in range(-self.stride[2]+1, self.stride[2]):
                for dy in range(-self.stride[2]+1, self.stride[2]):
                    for dz in range(-self.stride[2]+1, self.stride[2]):
                        dr = torch.tensor([dx / self.stride[0], dy / self.stride[1], dz / self.stride[2]]).to(self.voxel_size)
                        #dr = torch.tensor([dx, dy, dz]).to(self.voxel_size)
                        point_bxyz_this = point_bxyz.clone()
                        point_bxyz_this[:, 1:4] += dr * self.voxel_size[1:]
                        point_bxyz_list.append(point_bxyz_this)
            point_bxyz = torch.cat(point_bxyz_list, dim=0)
            
        point_wise_mean_dict = dict(
            point_bxyz=point_bxyz,
        )

        voxel_wise_dict, voxel_index, num_voxels, _ = self.voxel_graph(point_wise_mean_dict)

        vc = voxel_wise_dict['voxel_coords']
        #mask = (vc[:, 1] % self.stride[0] == (self.stride[0] - 1)) & (vc[:, 2] % self.stride[1] == (self.stride[1] - 1)) & (vc[:, 3] % self.stride[2] == (self.stride[2] - 1))
        if self.z_padding == 1:
            mask = (vc[:, 1] % self.stride[0] == 0) & (vc[:, 2] % self.stride[1] == 0) & (vc[:, 3] % self.stride[2] == 0)
        else:
            mask = (vc[:, 1] % self.stride[0] == 0) & (vc[:, 2] % self.stride[1] == 0) & (vc[:, 3] % self.stride[2] == 1) & (vc[:, 3] != vc[:, 3].max())

        voxel_bcenter = voxel_wise_dict['voxel_bcenter'][mask]
        voxel_bcenter += (self.voxel_size - self.voxel_graph.voxel_size) / 2.0
        voxel_bcenter[:, -1] -= (1.0-self.z_padding) * self.voxel_graph.voxel_size[-1]

        return voxel_bcenter

    def __repr__(self):
        return f"VoxelCenterSampler(voxel_graph={self.voxel_graph})"


class GridSampler(SamplerTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        super(GridSampler, self).__init__(
                               runtime_cfg=runtime_cfg,
                               model_cfg=model_cfg,
                           )
        grid_size = model_cfg.get("GRID_SIZE", None)
        self._grid_size = grid_size
        if isinstance(grid_size, list):
            grid_size = torch.tensor([1]+grid_size).float()
        else:
            grid_size = torch.tensor([1]+[grid_size for i in range(3)]).float()
        assert grid_size.shape[0] == 4, "Expecting 4D grid size." 
        self.register_buffer("grid_size", grid_size)
        
        point_cloud_range = model_cfg.get("POINT_CLOUD_RANGE", None)
        if point_cloud_range is None:
            self.point_cloud_range = None
        else:
            point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
            self.register_buffer("point_cloud_range", point_cloud_range)
        
    def forward(self, point_bxyz):
        """
        Args:
            point_bxyz [N, 4]: (b,x,y,z), first dimension is batch index
        Returns:
            new_bxyz: [M, 4] sampled points, M roughly equals (N // self.stride)
        """

        if self.point_cloud_range is not None:
            start = self.point_cloud_range.new_zeros(4)
            end = self.point_cloud_range.new_zeros(4)
            start[1:4] = self.point_cloud_range[:3]
            end[1:4] = self.point_cloud_range[3:]
            start[0] = point_bxyz[:, 0].min() - 0.5
            end[0] = point_bxyz[:, 0].max() + 0.5
        else:
            start = point_bxyz.min(0)[0]
            start[0] -= 0.5
            end = point_bxyz.max(0)[0]
            end[0] += 0.5

        cluster = grid_cluster(point_bxyz, self.grid_size, start=start, end=end)
        unique, inv = torch.unique(cluster, sorted=True, return_inverse=True)
        #perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
        #perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
        num_grids = unique.shape[0]
        sampled_bxyz = scatter(point_bxyz, inv, dim=0, dim_size=num_grids, reduce='mean')

        return sampled_bxyz

    def __repr__(self):
        return f"GridSampler(stride={self._grid_size}, point_cloud_range={list(self.point_cloud_range)})"

        
class FPSSampler(SamplerTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        super(FPSSampler, self).__init__(
                              runtime_cfg=runtime_cfg,
                              model_cfg=model_cfg,
                          )
        self.stride = model_cfg.get("STRIDE", 1)
        self.num_sectors = model_cfg.get("NUM_SECTORS", 1)
        
    def forward(self, point_bxyz):
        """
        Args:
            point_bxyz [N, 4]: (b,x,y,z), first dimension is batch index
        Returns:
            new_bxyz: [M, 4] sampled points, M roughly equals (N // self.stride)
        """
        if self.stride == 1:
            return point_bxyz

        point_xyz, point_indices, offset = bxyz_to_xyz_index_offset(point_bxyz)

        # sample
        new_offset = [(offset[0].item() + self.stride - 1) // self.stride]
        sample_idx = (offset[0].item() + self.stride - 1) // self.stride
        for i in range(1, offset.shape[0]):
            sample_idx += (offset[i].item() - offset[i - 1].item() + self.stride - 1) // self.stride
            new_offset.append(sample_idx)
        new_offset = torch.cuda.IntTensor(new_offset)
        if (self.num_sectors > 1) and (point_xyz.shape[0] > 100):
            fps_idx = sectorized_fps(point_xyz, offset, new_offset, self.num_sectors) # [M]
        else:
            fps_idx = furthestsampling(point_xyz, offset, new_offset) # [M]
        fps_idx = point_indices[fps_idx.long()]

        return point_bxyz[fps_idx]

    def __repr__(self):
        return f"FPSSampler(stride={self.stride})"


SAMPLERS = {
    'FPSSampler': FPSSampler,
    'GridSampler': GridSampler,
    'VoxelCenterSampler': VoxelCenterSampler,
}

