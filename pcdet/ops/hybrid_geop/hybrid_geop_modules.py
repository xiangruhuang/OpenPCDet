import torch
from torch import nn
from .hybrid_geop_cuda import (
    hash_insert_gpu,
    hybrid_geop_gpu
)

class PrimitiveFitting(nn.Module):
    """Fit hybrid geometric primitives to point cloud.
    Args:
        points [N, 4]
        edge_indices [2, E]

    Returns:
        
    """
    def __init__(self,
                 grid_size,
                 max_num_points=400000,
                 ):
        super(PrimitiveFitting, self).__init__()
        self.ndim = 3
        self.max_num_points = max_num_points
        keys = torch.zeros(max_num_points).long()
        values = torch.zeros(max_num_points, self.ndim+1).float()
        reverse_indices = torch.zeros(max_num_points).long()
        self.register_buffer("keys", keys, persistent=False)
        self.register_buffer("values", values, persistent=False)
        self.register_buffer("reverse_indices", reverse_indices, persistent=False)

        if isinstance(grid_size, list):
            voxel_size = torch.tensor([1] + grid_size).float()
        else:
            voxel_size = torch.tensor([1, grid_size, grid_size, grid_size]).float()
        self.register_buffer('voxel_size', voxel_size)
        self.decay_radius = self.voxel_size[1:].norm(p=2).item() / 2

        mu = torch.zeros(max_num_points, self.ndim+1)
        sigma = torch.zeros(max_num_points, 3, 3)
        qmin = torch.tensor([0, -1, -1, -1]).long()
        qmax = torch.tensor([0,  1,  1,  1]).long()
        self.register_buffer('mu', mu, persistent=False)
        self.register_buffer('sigma', sigma, persistent=False)
        self.register_buffer('qmin', qmin, persistent=False)
        self.register_buffer('qmax', qmax, persistent=False)
    
    @torch.no_grad()
    def forward(self, points):
        assert points.shape[0] * 2 <= self.max_num_points, \
               f"Too many points, shape={points.shape[0]}"
        assert points.shape[1] == self.ndim + 1, "points must have {self.ndim+1} dimensions"

        pc_range_min = (points.min(0)[0] - self.voxel_size*2).cuda()
        pc_range_max = (points.max(0)[0] + self.voxel_size*2).cuda()

        points = points.cuda()
        voxel_coors = torch.round((points-pc_range_min) / self.voxel_size).long() + 1

        dims = torch.round((pc_range_max - pc_range_min) / self.voxel_size).long()+3
        
        self.keys[:] = -1

        hash_insert_gpu(
            self.keys,
            self.values,
            self.reverse_indices,
            dims,
            voxel_coors,
            points)
        
        # find unique voxel coordinates
        voxel_coors1d = torch.zeros_like(voxel_coors[:, 0]).long()
        for i in range(self.ndim+1):
            voxel_coors1d = voxel_coors1d * dims[i] + voxel_coors[:, i]
        query_coors1d = voxel_coors1d.unique()
        query_coors = query_coors1d.new_zeros(query_coors1d.shape[0], self.ndim+1)
        for i in range(self.ndim, -1, -1):
            query_coors[:, i] = query_coors1d % dims[i]
            query_coors1d = torch.div(query_coors1d, dims[i], rounding_mode='floor').long()
        hybrid_geop_gpu(
            self.keys,
            self.values,
            self.reverse_indices,
            dims,
            self.qmin,
            self.qmax,
            query_coors,
            self.mu,
            self.sigma,
            self.decay_radius
        )
        mu = self.mu[:query_coors.shape[0]]
        sigma = self.sigma[:query_coors.shape[0]]
        sigma[:] = (sigma + sigma.transpose(1, 2))/2.0
        R, S, V = sigma.svd()
        S = S.sqrt().clamp(min=1e-6)
        det = R.det()
        flip_mask = (det == -1)
        R[flip_mask, :, 2] *= -1
        R = R * S[:, None, :]
        
        return mu, R 

if __name__ == '__main__':
    n = 1000
    points = torch.randn(n, 3).cuda()
    points[:, 2] = 0
    zeros = torch.zeros(n).cuda()
    points = torch.cat([zeros[:, None], points], axis=-1)
    pf = PrimitiveFitting(0.1)
    pf = pf.cuda()
    pf(points)
