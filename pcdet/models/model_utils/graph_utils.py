import torch
from torch import nn
from torch_scatter import scatter

from pcdet.ops.pointops.functions.pointops import (
    knnquery
)
from pcdet.ops.torch_hash.torch_hash_cuda import (
    hash_insert_gpu,
    correspondence,
    radius_graph_gpu,
    points_in_radius_gpu
)
from .sampler_utils import bxyz_to_xyz_index_offset
from pcdet.utils import common_utils

def select_graph(graph_cfgs, i):
    if isinstance(graph_cfgs, list):
        idx = i
        for j in range(len(graph_cfgs)):
            if idx >= graph_cfgs[j]['COUNT']:
                idx -= graph_cfgs[j]['COUNT']
            else:
                graph_cfg = common_utils.indexing_list_elements(graph_cfgs[j], idx)
                break
    else:
        graph_cfg = common_utils.indexing_list_elements(graph_cfgs, i)
    return graph_cfg


class GraphTemplate(nn.Module):
    def __init__(self, runtime_cfg, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg

    def forward(self, ref_bxyz, query_bxyz):
        assert NotImplementedError


class KNNGraph(GraphTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        super(KNNGraph, self).__init__(
                                    runtime_cfg=runtime_cfg,
                                    model_cfg=model_cfg,
                                )
        self.k = model_cfg.get("NUM_NEIGHBORS", 32)
    
    def forward(self, ref_bxyz, query_bxyz):
        """Build knn graph from ref point cloud to query point cloud,
            each query point connects to k ref points.
        Args:
            ref_bxyz [N, 4]: ref point cloud
            query_bxyz [M, 4]: query point cloud
        Returns:
            edge_idx [2, M*K]: (idx_of_ref, idx_of_query)
        """
        
        ref_xyz, ref_indices, ref_offset = bxyz_to_xyz_index_offset(ref_bxyz)
        query_xyz, query_indices, query_offset = bxyz_to_xyz_index_offset(query_bxyz)
        
        # building a bipartite graph from [N] to [M]
        #N, M = ref_xyz.shape[0], query_xyz.shape[0]
        ref_idx, _ = knnquery(self.k, ref_xyz, query_xyz,
                              ref_offset.int(), query_offset.int()) # [M, k] -> [N]
        ref_idx = ref_idx.long()

        query_idx = torch.arange(query_xyz.shape[0])[:, None].expand(-1, self.k).to(ref_idx) # [M, k] -> [M]

        edge_index = torch.stack([ref_indices[ref_idx],
                                  query_indices[query_idx]], dim=0).reshape(2, -1)

        return edge_index
    
    def __repr__(self):
        return f"KNNGraph(num_neighbors={self.k})"


class RadiusGraph(GraphTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        super(RadiusGraph, self).__init__(
                                       runtime_cfg=runtime_cfg,
                                       model_cfg=model_cfg,
                                   )
        self.radius = model_cfg.get("RADIUS", None)
        self.max_num_neighbors = model_cfg.get("MAX_NUM_NEIGHBORS", 32)
        self.sort_by_dist = model_cfg.get("SORT_BY_DIST", False)
        self.util_ratio = 0.5

        qmin = torch.tensor([0] + [-1 for i in range(3)]).int()
        qmax = torch.tensor([0] + [1 for i in range(3)]).int()
        self.register_buffer("qmin", qmin, persistent=False)
        self.register_buffer("qmax", qmax, persistent=False)
    
    def forward(self, ref_bxyz, query_bxyz):
        """Build knn graph from source point cloud to target point cloud,
            each target point connects to k source points.
        Args:
            ref_bxyz [N, 4]: source point cloud
            query_bxyz [M, 4]: target point cloud
        Returns:
            edge_idx [2, M*K]: (idx_of_ref, idx_of_query)
        """
        assert ref_bxyz.shape[-1] == 4

        # find data range, voxel size
        radius = query_bxyz.new_zeros(query_bxyz.shape[0]) + self.radius
        voxel_size = torch.tensor([1-1e-3] + [radius.max().item() for i in range(3)]).to(ref_bxyz.device)
        all_points = torch.cat([ref_bxyz, query_bxyz], axis=0)
        pc_range_min = (all_points.min(0)[0] - voxel_size*2).cuda()
        pc_range_max = (all_points.max(0)[0] + voxel_size*2).cuda()
        voxel_coors_ref = torch.round((ref_bxyz-pc_range_min) / voxel_size).long() + 1
        voxel_coors_query = torch.round((query_bxyz-pc_range_min) / voxel_size).long() + 1
        dims = torch.round((pc_range_max - pc_range_min) / voxel_size).long()+3

        # allocate memory
        hashtable_size = int(ref_bxyz.shape[0] / self.util_ratio)

        keys = voxel_coors_ref.new_zeros(hashtable_size) - 1
        values = ref_bxyz.new_empty(hashtable_size, 4)
        reverse_indices = voxel_coors_ref.new_zeros(hashtable_size)

        # hashing
        hash_insert_gpu(
            keys,
            values,
            reverse_indices,
            dims,
            voxel_coors_ref,
            ref_bxyz)

        edges = radius_graph_gpu(
                    keys,
                    values,
                    reverse_indices,
                    dims,
                    voxel_coors_query,
                    query_bxyz,
                    self.qmin,
                    self.qmax,
                    radius,
                    self.max_num_neighbors,
                    self.sort_by_dist).T

        return edges
    
    def __repr__(self):
        return f"RadiusGraph(radius={self.radius}, max_ngbrs={self.max_num_neighbors}, sort={self.sort_by_dist})"


class VoxelGraph(GraphTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        super(VoxelGraph, self).__init__(
                                   runtime_cfg=runtime_cfg,
                                   model_cfg=model_cfg,
                               )
        self.voxel_size = model_cfg.get("VOXEL_SIZE", None)
        self.kernel_offset = model_cfg.get("KERNEL_OFFSET", None)
        self.util_ratio = 0.5

        qmin = torch.tensor([0] + [-self.kernel_offset for i in range(3)]).int()
        qmax = torch.tensor([0] + [self.kernel_offset for i in range(3)]).int()
        self.register_buffer("qmin", qmin, persistent=False)
        self.register_buffer("qmax", qmax, persistent=False)
    
    def forward(self, ref_bxyz, query_bxyz):
        """Build knn graph from source point cloud to target point cloud,
            each target point connects to k source points.
        Args:
            ref_bxyz [N, 4]: source point cloud
            query_bxyz [M, 4]: target point cloud
        Returns:
            edge_idx [2, M*K]: (idx_of_ref, idx_of_query)
        """
        assert ref_bxyz.shape[-1] == 4

        # find data range, voxel size
        radius = query_bxyz.new_zeros(query_bxyz.shape[0]) + 1e5
        voxel_size = torch.tensor([1-1e-3] + self.voxel_size).to(ref_bxyz.device)
        all_points = torch.cat([ref_bxyz, query_bxyz], axis=0)
        pc_range_min = (all_points.min(0)[0] - voxel_size*2).cuda()
        pc_range_max = (all_points.max(0)[0] + voxel_size*2).cuda()
        voxel_coors_ref = torch.round((ref_bxyz-pc_range_min) / voxel_size).long() + 1
        voxel_coors_query = torch.round((query_bxyz-pc_range_min) / voxel_size).long() + 1
        dims = torch.round((pc_range_max - pc_range_min) / voxel_size).long() + 3

        # allocate memory
        hashtable_size = int(ref_bxyz.shape[0] / self.util_ratio)

        keys = voxel_coors_ref.new_zeros(hashtable_size) - 1
        values = ref_bxyz.new_empty(hashtable_size, 4)
        reverse_indices = voxel_coors_ref.new_zeros(hashtable_size)

        # hashing
        hash_insert_gpu(
            keys,
            values,
            reverse_indices,
            dims,
            voxel_coors_ref,
            ref_bxyz)

        edges = radius_graph_gpu(
                    keys,
                    values,
                    reverse_indices,
                    dims,
                    voxel_coors_query,
                    query_bxyz,
                    self.qmin,
                    self.qmax,
                    radius,
                    32,
                    False).T

        return edges
    
    def __repr__(self):
        return f"VoxelGraph(voxel_size={self.voxel_size}, kernel_offset={self.kernel_offset})"

GRAPHS = {
    'KNNGraph': KNNGraph,
    'RadiusGraph': RadiusGraph,
    'VoxelGraph': VoxelGraph,
}

