import torch
from torch import nn
from .torch_hash_cuda import (
    hash_insert_gpu,
    correspondence,
    radius_graph_gpu,
    points_in_radius_gpu
)

class RadiusGraph(nn.Module):
    def __init__(self,
                 max_num_points=400000,
                 ndim=3):
        super().__init__()
        self.ndim = ndim
        
        # store hash table
        self.max_num_points = max_num_points
        keys = torch.zeros(max_num_points).long()
        values = torch.zeros(max_num_points, ndim+1).float()
        reverse_indices = torch.zeros(max_num_points).long()
        self.register_buffer("keys", keys, persistent=False)
        self.register_buffer("values", values, persistent=False)
        self.register_buffer("reverse_indices", reverse_indices, persistent=False)

        # dummy variable
        qmin = torch.tensor([0] + [-1 for i in range(ndim)]).int()
        qmax = torch.tensor([0] + [1 for i in range(ndim)]).int()
        self.register_buffer("qmin", qmin, persistent=False)
        self.register_buffer("qmax", qmax, persistent=False)

    def clear(self):
        self.keys[:] = -1

    def forward(self, ref, query, radius, num_neighbors, sort_by_dist=False):
        """
        Args:
            ref [N, 1+dim] the first dimension records batch index
            query [M, 1+dim] ..
            radius (float)
            num_neighbors (int)
            sort_by_dist 

        Returns:
            edge_indices [2, E] each column represent edge (idx_of_ref, idx_of_query)
        """
        assert ref.shape[0] * 2 <= self.max_num_points, f"Too many points, shape={ref.shape[0]}"

        if isinstance(radius, float):
            radius = query.new_zeros(query.shape[0]) + radius
        elif isinstance(radius, int):
            radius = query.new_zeros(query.shape[0]) + radius

        voxel_size = torch.tensor([1] + [radius.max().item() for i in range(self.ndim)]).to(ref.device)
        assert ref.shape[1] == self.ndim + 1, f"points must have {self.ndim+1} dimensions"
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

        edges = radius_graph_gpu(
                    self.keys,
                    self.values,
                    self.reverse_indices,
                    dims,
                    voxel_coors_query,
                    query,
                    self.qmin,
                    self.qmax,
                    radius,
                    num_neighbors,
                    sort_by_dist).T

        return edges

    def __repr__(self):
        return f"RadiusGraph(ndim={self.ndim}, max_npoints={self.max_num_points})"

if __name__ == '__main__':
    #rg = RadiusGraph().cuda()
    #points = torch.randn(100, 4).cuda() * 3
    #points[:, 0] = 0
    #er, eq = rg(points, points, 2, -1)
    #assert (points[er] - points[eq]).norm(p=2, dim=-1).max() < 2
    #print('Test 1 okay')
    #
    #rg = RadiusGraph(ndim=2).cuda()
    #points = torch.randn(100, 3).cuda() * 3
    #points[:, 0] = 0
    #er, eq = rg(points, points, 2, -1)
    #assert (points[er] - points[eq]).norm(p=2, dim=-1).max() < 2
    #print('Test 2 okay')

    rg = RadiusGraph(ndim=2).cuda()
    points = torch.tensor([[0, 0.0, 0.0], [0, 0.1, 0.1], [0, 0.2, 0.2]], dtype=torch.float32).cuda()
    er, eq = rg(points, points, 0.15, 1, sort_by_dist=True)
    import ipdb; ipdb.set_trace()
    #print(points[er])
    #print(points[eq])

    from sklearn.neighbors import NearestNeighbors as NN
    import numpy as np
    data = np.random.randn(1000, 4)
    data[:, 0] = 0
    tree = NN(n_neighbors=2).fit(data)
    dists, indices = tree.kneighbors(data)
    import ipdb; ipdb.set_trace()
    rg = RadiusGraph(ndim=3).cuda()
    er, eq = rg(data, data, 0.5, 2, sort_by_dist=True)
    

    NN(n_neighbors=1).fit()
    pass

