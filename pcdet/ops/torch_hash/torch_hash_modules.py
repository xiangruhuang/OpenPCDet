import torch
from torch import nn
from .torch_hash_cuda import (
    hash_insert_gpu,
    correspondence,
    voxel_graph_gpu,
    points_in_radius_gpu
)

class RadiusGraph(nn.Module):
    def __init__(self,
                 max_num_neighbors=32,
                 ndim=3,
                 max_num_points=200000):
        super().__init__()
        self.max_num_neighbors = max_num_neighbors
        self.ndim = ndim
        
        # store hash table
        self.max_num_points = max_num_points
        keys = torch.zeros(max_num_points).long()
        values = torch.zeros(max_num_points, ndim+1).float()
        reverse_indices = torch.zeros(max_num_points).long()
        corres = torch.zeros(max_num_points*max_num_neighbors).long()
        self.register_buffer("keys", keys, persistent=False)
        self.register_buffer("values", values, persistent=False)
        self.register_buffer("reverse_indices", reverse_indices, persistent=False)
        self.register_buffer("corres", corres, persistent=False)

        # dummy variable
        qmin = torch.tensor([0] + [-1 for i in range(ndim)]).int()
        qmax = torch.tensor([0] + [1 for i in range(ndim)]).int()
        self.register_buffer("qmin", qmin, persistent=False)
        self.register_buffer("qmax", qmax, persistent=False)

    def clear(self):
        self.keys[:] = -1

    def forward(self, ref, query, radius, num_neighbors, sort_by_dist):
        """
        Args:
            ref [N, 4] the first dimension records batch index
            query [M, 4] ..
            radius (float)
            num_neighbors (int)

        Returns:
            edge_indices [2, E] each column represent edge (idx_of_ref, idx_of_query)
        """
        assert ref.shape[0] * 2 <= self.max_num_points, f"Too many points, shape={ref.shape[0]}"

        voxel_size = torch.tensor([1] + [radius for i in range(self.ndim)]).to(ref.device)
        assert ref.shape[1] == self.ndim + 1, "points must have {self.ndim+1} dimensions"
        all_points = torch.cat([ref, query], axis=0)
        pc_range_min = (all_points.min(0)[0] - voxel_size*2).cuda()
        pc_range_max = (all_points.max(0)[0] + voxel_size*2).cuda()
        
        ref = ref.cuda()
        query = query.cuda()
        voxel_coors_ref = torch.round((ref-pc_range_min) / voxel_size).long() + 1
        voxel_coors_query = torch.round((query-pc_range_min) / voxel_size).long() + 1
        dims = torch.round((pc_range_max - pc_range_min) / voxel_size).long()+3

        self.clear()
        
        hash_insert_gpu(
            self.keys,
            self.values,
            self.reverse_indices,
            dims,
            voxel_coors_ref,
            ref)

        self.corres[:(query.shape[0]*num_neighbors)] = -1

        voxel_graph_gpu(
            self.keys,
            self.values,
            self.reverse_indices,
            dims,
            voxel_coors_query,
            query,
            self.qmin,
            self.qmax,
            num_neighbors,
            radius,
            sort_by_dist,
            self.corres)
        
        corres = self.corres[:(query.shape[0]*num_neighbors)]
        corres = corres.view(-1, num_neighbors)
        mask = (corres != -1)
        corres0, corres1 = torch.where(mask)
        # (query, ref)
        corres = torch.stack([corres[(corres0, corres1)], corres0], dim=0)

        return corres
