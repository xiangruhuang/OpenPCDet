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
        self.required_attributes = model_cfg.get("REQUIRED_ATTRIBUTES", ['bxyz'])
        if not isinstance(self.required_attributes, list):
            self.required_attributes = [self.required_attributes]

    def sample(self, point_bxyz):
        raise NotImplementedError

    def forward(self, point_bxyz):
        result_dict = self.sample(point_bxyz)
        results = []
        for attr in self.required_attributes:
            if attr not in result_dict:
                raise ValueError(f"{self}: Required attribute {attr} not in sample results")
            results.append(result_dict[attr])
        if len(results) == 1:
            return results[0]
        else:
            return results

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
        downsample_times = model_cfg.get("DOWNSAMPLE_TIMES", 1)
        if not isinstance(downsample_times, list):
            downsample_times = [downsample_times for i in range(3)]
        self.downsample_times = downsample_times
        downsample_times = torch.tensor(downsample_times, dtype=torch.float32)
        model_cfg_cp = {}
        model_cfg_cp.update(model_cfg)
        model_cfg_cp['VOXEL_SIZE'] = [voxel_size[1+i] / downsample_times[i] for i in range(3)]
        self.voxel_graph = VoxelGraph(model_cfg=model_cfg_cp, runtime_cfg=runtime_cfg)
        
    def sample(self, point_bxyz):
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
                        point_bxyz_this = point_bxyz.clone()
                        point_bxyz_this[:, 1:4] += dr * self.voxel_size[1:]
                        point_bxyz_list.append(point_bxyz_this)
            point_bxyz = torch.cat(point_bxyz_list, dim=0)
            
        point_wise_mean_dict = dict(
            point_bxyz=point_bxyz,
        )

        voxel_wise_dict, voxel_index, num_voxels, _ = self.voxel_graph(point_wise_mean_dict)

        vc = voxel_wise_dict['voxel_coords']
        if self.z_padding == -1:
            mask = (vc[:, 3] % self.downsample_times[2] == 0)
            for i in range(2):
                mask &= (vc[:, i+1] % self.downsample_times[i] == 0)
        else:
            mask = (vc[:, 3] % self.downsample_times[2] == self.z_padding)
            for i in range(2):
                mask &= (vc[:, i+1] % self.downsample_times[i] == 0)

        voxel_bcenter = voxel_wise_dict['voxel_bcenter'][mask]

        return dict(
            bxyz=voxel_bcenter
        )


class VolumeSampler(SamplerTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        super(VolumeSampler, self).__init__(
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
        downsample_times = model_cfg.get("DOWNSAMPLE_TIMES", 1)
        if not isinstance(downsample_times, list):
            downsample_times = [downsample_times for i in range(3)]
        self.downsample_times = downsample_times
        downsample_times = torch.tensor(downsample_times, dtype=torch.float32)
        model_cfg_cp = {}
        model_cfg_cp.update(model_cfg)
        model_cfg_cp['VOXEL_SIZE'] = [voxel_size[1+i] / downsample_times[i] for i in range(3)]
        self.voxel_graph = VoxelGraph(model_cfg=model_cfg_cp, runtime_cfg=runtime_cfg)
        
    def sample(self, point_bxyz):
        """
        Args:
            point_bxyz [N, 4]: (b,x,y,z), first dimension is batch index
        Returns:
            voxel_center: [V, 4] sampled centers of voxels
        """

        with torch.no_grad():
            point_bxyz_list = []
            point_volume_list = []
            for dx in range(-self.stride[2]+1, self.stride[2]):
                for dy in range(-self.stride[2]+1, self.stride[2]):
                    for dz in range(-self.stride[2]+1, self.stride[2]):
                        volume = 1 if (dx == 0) and (dy == 0) and (dz == 0) else 0
                        dr = torch.tensor([dx / self.stride[0], dy / self.stride[1], dz / self.stride[2]]).to(self.voxel_size)
                        point_bxyz_this = point_bxyz.clone()
                        point_volume_this = point_bxyz.new_full(point_bxyz.shape[0:1], volume)
                        point_bxyz_this[:, 1:4] += dr * self.voxel_size[1:]
                        point_bxyz_list.append(point_bxyz_this)
                        point_volume_list.append(point_volume_this)
            point_bxyz = torch.cat(point_bxyz_list, dim=0)
            point_volume = torch.cat(point_volume_list, dim=0)
            
        point_wise_mean_dict = dict(
            point_bxyz=point_bxyz,
        )

        voxel_wise_dict, voxel_index, num_voxels, out_of_boundary_mask = self.voxel_graph(point_wise_mean_dict)
        point_bxyz = point_bxyz[~out_of_boundary_mask]
        point_volume = point_volume[~out_of_boundary_mask]

        point_bxyz = point_bxyz[point_volume > 0.5]
        voxel_index = voxel_index[point_volume > 0.5]
        point_volume = point_volume[point_volume > 0.5]
        point_xyz = point_bxyz[:, 1:]

        voxel_volume = scatter(point_volume, voxel_index, dim=0,
                               dim_size=num_voxels, reduce='sum')
        is_empty = voxel_volume < 0.5
        voxel_xyz = scatter(point_xyz, voxel_index, dim=0,
                            dim_size=num_voxels, reduce='sum')
        voxel_xyz[~is_empty] = voxel_xyz[~is_empty] / voxel_volume[~is_empty].unsqueeze(-1)
        voxel_xyz[is_empty] = voxel_wise_dict['voxel_center'][is_empty]
        point_d = point_xyz - voxel_xyz[voxel_index]
        point_ddT = point_d.unsqueeze(-1) * point_d.unsqueeze(-2)
        voxel_ddT = scatter(point_ddT, voxel_index, dim=0,
                            dim_size=num_voxels, reduce='sum')
        voxel_ddT[~is_empty] = voxel_ddT[~is_empty] / voxel_volume[~is_empty].unsqueeze(-1).unsqueeze(-1)

        voxel_eigvals, voxel_eigvecs = torch.linalg.eigh(voxel_ddT) # eigvals in ascending order

        vc = voxel_wise_dict['voxel_coords']
        if self.z_padding == -1:
            mask = (vc[:, 3] % self.downsample_times[2] == 0)
            for i in range(2):
                mask &= (vc[:, i+1] % self.downsample_times[i] == 0)
        else:
            mask = (vc[:, 3] % self.downsample_times[2] == self.z_padding)
            for i in range(2):
                mask &= (vc[:, i+1] % self.downsample_times[i] == 0)

        voxel_bcenter = voxel_wise_dict['voxel_bcenter'][mask]
        voxel_wise_dict['voxel_bcenter'] = voxel_bcenter
        voxel_wise_dict['voxel_eigvals'] = voxel_eigvals
        voxel_wise_dict['voxel_eigvecs'] = voxel_eigvecs
        ret_voxel_wise_dict = {}
        for key in voxel_wise_dict.keys():
            if key.startswith('voxel_'):
                ret_key = key.split('voxel_')[-1]
                ret_voxel_wise_dict[ret_key] = voxel_wise_dict[key]

        return ret_voxel_wise_dict


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
        
    def sample(self, point_bxyz):
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

        return dict(bxyz=sampled_bxyz)

    def extra_repr(self):
        return f"stride={self._grid_size}, point_cloud_range={list(self.point_cloud_range)}"

        
class FPSSampler(SamplerTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        super(FPSSampler, self).__init__(
                              runtime_cfg=runtime_cfg,
                              model_cfg=model_cfg,
                          )
        self.stride = model_cfg.get("STRIDE", 1)
        self.num_sectors = model_cfg.get("NUM_SECTORS", 1)
        
    def sample(self, point_bxyz):
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

    def extra_repr(self):
        return f"stride={self.stride}"


SAMPLERS = {
    'FPSSampler': FPSSampler,
    'GridSampler': GridSampler,
    'VoxelCenterSampler': VoxelCenterSampler,
    'VolumeSampler': VolumeSampler,
}

