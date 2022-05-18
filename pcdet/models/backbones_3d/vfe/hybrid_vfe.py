import torch
from torch import nn
from .vfe_template import VFETemplate
from ..grid_sampling import GridSampling3D
from ....ops.torch_hash import RadiusGraph
from torch_scatter import scatter
import numpy as np

class HybridVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features
        self.grid_size = model_cfg.get("GRID_SIZE", None)
        max_num_points = model_cfg.get("MAX_NUM_POINTS", 800000)
        self.radius_graph = RadiusGraph(max_num_points)
        self.grid_sampler = nn.ModuleList()
        for grid_size in self.grid_size:
            self.grid_sampler.append(GridSampling3D(grid_size))
        self.min_fitness = model_cfg.get("MIN_FITNESS", None)
        self.min_point_llh = model_cfg.get("MIN_POINT_LLH", None)
        decay_radius = model_cfg.get("DECAY_RADIUS", None)
        self.decay_radius2 = [d**2 for d in decay_radius]
        self.num_class = kwargs.get("num_class", 6)
        self.NA = - 1
        self.radius = model_cfg.get("RADIUS", None)
        self.decay = model_cfg.get("DECAY", None)
        self.gain = model_cfg.get("GAIN", None)

    def get_output_feature_dim(self):
        return self.num_point_features

    def fit_primitive(self, points, voxels, ep, ev, decay, gain):
        """
        Args:
            points [N, 3+D]
            voxels [V, 4]
            ep, ev [E]
            mu [V, 4]

        Returns:
            mu
            cov_inv
            llh
            fitness
        """

        num_voxels = voxels.shape[0]
        edge_weight = torch.ones_like(ep).float()
        for itr in range(3):
            # fit a primitive
            mu = scatter(points[ep]*edge_weight[:, None], ev,
                         dim=0, dim_size=num_voxels, reduce='sum') # [V, 6]
            weight_sum = scatter(edge_weight, ev, dim=0, dim_size=num_voxels, reduce='sum') # [V]
            mu = mu / weight_sum[:, None]
            d = (points[ep] - mu[ev])[:, 1:4] # [E, 3]
            #edge_weight = self.decay_radius2 / (d.square().sum(dim=-1) + self.decay_radius2) # [E]
            ddT = (d.unsqueeze(-1) @ d.unsqueeze(-2)).view(-1, 9) # [E, 9]
            cov = scatter(ddT * edge_weight[:, None], ev,
                          dim=0, dim_size=num_voxels, reduce='sum').view(-1, 3, 3) # [V, 3, 3]
            weight_sum = scatter(edge_weight, ev, dim=0, dim_size=num_voxels, reduce='sum') # [V]
            cov = cov / weight_sum[:, None, None].clamp(min=1)

            # compute validity of primitives
            # compute weight of each edge
            cov = cov + torch.eye(3).to(cov).repeat(num_voxels, 1, 1) * 1e-4
            cov_inv = cov.inverse()
            #valid_mask = (weight_sum >= 4).float()
            
            llh = (d.unsqueeze(-2) @ cov_inv[ev] @ d.unsqueeze(-1)).squeeze(-1).squeeze(-1) # [E, 1, 3] @ [E, 3, 3] @ [E, 3, 1] = [E]
            llh = (-0.5*llh / 3).exp()# * valid_mask[ev] # / ((2*np.pi)**3*cov_det[ev]).sqrt()

            edge_weight = llh

        llh = llh * (weight_sum[ev] >= 4).float()
        fitness_sum = scatter(llh, ev, dim=0, dim_size=num_voxels, reduce='sum')
        fitness_mean = scatter(llh, ev, dim=0, dim_size=num_voxels, reduce='mean')
        fitness = (fitness_sum/decay).clamp(max=gain) + fitness_mean
        
        #S, R = torch.linalg.eigh(cov)
        #R = R * S[:, None, :].sqrt()

        primitives = torch.cat([mu, cov.reshape(-1, 9), fitness.reshape(-1, 1)], dim=-1)

        return primitives, fitness, llh

    def get_loss(self):
        """
        Args in forward_ret_dict:
            points [N, 4]
            voxels [V, 4]
            ep, ev [E]
            edge_weight [E]
            seg_labels [N]

        Returns:
            
        """
        import ipdb; ipdb.set_trace()

        ep, eh = self.forward_ret_dict['edges']
        edge_weight = self.forward_ret_dict['edge_weight']
        point_seg_cls_labels = self.forward_ret_dict['point_seg_cls_labels']
        hybrid_seg_cls_labels = self.forward_ret_dict['hybrid_seg_cls_labels']
        
        num_primitives = self.forward_ret_dict['hybrid_seg_cls_labels'].shape[0]
        
        valid_mask = (point_seg_cls_labels[ep] != self.NA) & (hybrid_seg_cls_labels[eh] != self.NA)
        consistency = (hybrid_seg_cls_labels[eh] == point_seg_cls_labels[ep]) & valid_mask
        neg_consistency = (hybrid_seg_cls_labels[eh] != point_seg_cls_labels[ep]) & valid_mask

        for th in np.linspace(0, 1, 100):
            mask = edge_weight > th
            iou1 = (mask & consistency).sum() / consistency.sum()
            neg_mask = edge_weight < th
            iou2 = (neg_mask & neg_consistency).sum() / neg_consistency.sum()
            print(f'th={th}, prec_pos={iou1}, prec_neg={iou2}')

        return loss

    def merge_seg_label(self, seg_cls_labels, seg_inst_labels):
        """
        Args:
            seg_cls_labels range [-1, 5]
            seg_inst_labels range [0, N]
        Returns:
            seg_labels range [-1, N*7+5]
        """
        seg_labels = seg_inst_labels * (self.num_class + 1) + seg_cls_labels
        return seg_labels

    def propagate_seg_labels(self, seg_labels, ep, ev, num_voxels):
        """
        Args:
            seg_labels range [-1, N*7+5]

        Returns:
            primitive_seg_labels [-1, N*7+5]
        """
        seg_labels_nz = seg_labels + 1 # [0, N*7+6]

        num_seg_label = seg_labels_nz.max().long().item() + 1 # [1, N*7+7]
        keys = ev * num_seg_label + seg_labels_nz[ep] # (ev, seg_labels_nz)
        sorted_keys = torch.sort(keys)[0] % num_seg_label
        degree = scatter(torch.ones_like(ep), ev, reduce='sum', dim_size=num_voxels, dim=0) # [V]
        offset = torch.cumsum(degree, dim=0) - degree # [V]
        primitive_seg_labels = sorted_keys[offset + torch.div(degree, 2, rounding_mode='trunc')] - 1 # [-1, N*6+6]

        return primitive_seg_labels

    def seg_label_to_cls_label(self, seg_labels):
        """
        Args:
            seg_labels [-1, N*7+5]
            
        Returns:
            seg_cls_labels [-1, 5]
        """
        valid_mask = seg_labels != -1
        seg_cls_labels = seg_labels.clone()
        seg_cls_labels[valid_mask] = (seg_cls_labels[valid_mask] + 1) % (self.num_class + 1) - 1
        return seg_cls_labels

    def summarize_primitive(self, batch_dict, level):
        points = batch_dict['sp_points']
        point_indices = batch_dict['sp_point_indices']
        points4d = points[:, :4].contiguous()
        voxels = self.grid_sampler[level](points4d) # [V, 4]
        num_voxels = voxels.shape[0]

        ep, ev = self.radius_graph(points4d, voxels, self.radius[level], -1) # [2, E], all neighbors
        
        # propagate segmentation labels to primitive
        point_seg_labels = batch_dict['sp_point_seg_labels']
        primitive_seg_labels = self.propagate_seg_labels(
                                   point_seg_labels,
                                   ep, ev, num_voxels)

        primitives, fitness, edge_weight = self.fit_primitive(points, voxels, ep, ev, self.decay[level], self.gain[level])
        valid_mask = (fitness > self.min_fitness[level])
        edge_fitness = valid_mask.float()[ev] * edge_weight
        point_llh = scatter(edge_fitness, ep, dim=0,
                            dim_size=points.shape[0], reduce='max')
        point_remain_mask = point_llh < self.min_point_llh[level]
        
        # select valid primitives, remove covered points, update edge weight
        primitive_index_map = torch.zeros(primitives.shape[0]).long().to(primitives.device) - 1
        primitive_index_map[valid_mask] = torch.arange(valid_mask.sum().long()).to(primitive_index_map)
        primitives = primitives[valid_mask]
        valid_primitive_seg_labels = primitive_seg_labels[valid_mask]
        sp_points = points[point_remain_mask]
        sp_point_indices = point_indices[point_remain_mask]
        sp_point_seg_labels = point_seg_labels[point_remain_mask]
        batch_dict['sp_point_llh'] = point_llh[point_remain_mask]

        edge_mask = primitive_index_map[ev] != -1
        ep = point_indices[ep[edge_mask]] # to full point set indices
        ev = primitive_index_map[ev[edge_mask]]
        edge_weight = edge_weight[edge_mask]

        # merge primitives and points
        batch_dict['primitives'].append(primitives)
        batch_dict['primitive_edges'].append(torch.stack([ep, ev], dim=0))
        batch_dict['primitive_edge_weight'].append(edge_weight)
        batch_dict['primitive_seg_labels'].append(valid_primitive_seg_labels)
        batch_dict['primitive_seg_cls_labels'].append(self.seg_label_to_cls_label(valid_primitive_seg_labels))
        batch_dict[f'primitives_{level}'] = primitives
        batch_dict[f'primitive_seg_labels_{level}'] = valid_primitive_seg_labels
        batch_dict[f'primitive_seg_cls_labels_{level}'] = self.seg_label_to_cls_label(valid_primitive_seg_labels)
        batch_dict['sp_points'] = sp_points
        batch_dict['sp_point_indices'] = sp_point_indices
        batch_dict['sp_point_seg_labels'] = sp_point_seg_labels
        batch_dict['sp_point_seg_cls_labels'] = self.seg_label_to_cls_label(sp_point_seg_labels)

        return batch_dict
        

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
        points = batch_dict['points'] # [N, 4]
        batch_dict['sp_points'] = points.clone()
        batch_dict['sp_point_indices'] = torch.arange(points.shape[0]).long().to(points.device)
        batch_dict['sp_point_seg_labels'] = self.merge_seg_label(
                                                batch_dict['seg_cls_labels'],
                                                batch_dict['seg_inst_labels'])
        batch_dict['sp_point_seg_cls_labels'] = self.seg_label_to_cls_label(batch_dict['sp_point_seg_labels'])
        batch_dict['primitives'] = []
        batch_dict['primitive_seg_labels'] = []
        batch_dict['primitive_edge_weight'] = []
        batch_dict['primitive_edges'] = []
        batch_dict['primitive_seg_cls_labels'] = []
        for level in range(len(self.radius)):
            batch_dict = self.summarize_primitive(batch_dict, level)
        
        # merge primitives and points
        primitives = torch.cat(batch_dict['primitives'], dim=0)
        primitive_sizes = torch.tensor([p.shape[0] for p in batch_dict['primitives']]).long()
        primitive_offset = torch.cumsum(primitive_sizes, dim=0) - primitive_sizes
        primitive_ep = torch.cat([pe[0] for pe in batch_dict['primitive_edges']])
        primitive_ev = torch.cat([pe[1] + primitive_offset[i] for i, pe in enumerate(batch_dict['primitive_edges'])])
        primitive_edges = torch.stack([primitive_ep, primitive_ev], dim=0)
        primitive_edge_weight = torch.cat(batch_dict['primitive_edge_weight'], dim=0)

        # recover point to primitive correspondence
        primitive_seg_cls_labels = torch.cat(batch_dict['primitive_seg_cls_labels'], dim=0)
        sp_points = batch_dict['sp_points']
        sp_points = torch.cat([sp_points,
                               sp_points.new_zeros(sp_points.shape[0],
                                                   primitives.shape[-1] - sp_points.shape[-1])
                              ], dim=-1) # hybrid points and primitives
        hybrid = torch.cat([primitives, sp_points], dim=0)
        sp_point_indices = batch_dict['sp_point_indices']
        sp_point_edges = torch.stack([sp_point_indices, torch.arange(sp_points.shape[0]).to(sp_point_indices) + primitives.shape[0]], dim=0)
        hybrid_edges = torch.cat([primitive_edges, sp_point_edges], dim=1)
        hybrid_edge_weight = torch.cat([primitive_edge_weight, primitive_edge_weight.new_ones(sp_points.shape[0])], dim=0)
        hybrid_seg_cls_labels = torch.cat([primitive_seg_cls_labels, batch_dict['sp_point_seg_cls_labels']], dim=0)
        
        # recover point to hybrid correspondence
        #points4d = points[:, :4].contiguous()
        #hybrid_centers = hybrid[:, :4].contiguous()
        #hybrid_edges = self.radius_graph(hybrid_centers, points4d, max(self.radius)+0.1, 1)
        batch_dict['hybrid_edges'] = hybrid_edges
        batch_dict['hybrid_edge_weight'] = hybrid_edge_weight

        batch_dict['hybrid'] = hybrid
        batch_dict['batch_idx'] = points[:, 0].round().long()
        batch_dict['hybrid_seg_cls_labels'] = hybrid_seg_cls_labels
        
        # save variables for computing separation loss
        ret_dict = dict(
            hybrid_seg_cls_labels=hybrid_seg_cls_labels,
            point_seg_cls_labels=batch_dict['seg_cls_labels'],
            edges=hybrid_edges,
            edge_weight=hybrid_edge_weight,
        )
        self.forward_ret_dict = ret_dict
        
        return batch_dict
