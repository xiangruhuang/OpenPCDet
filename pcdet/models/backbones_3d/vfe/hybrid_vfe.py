import torch

from .vfe_template import VFETemplate
from ..grid_sampling import GridSampling3D
from ....ops.torch_hash import RadiusGraph
from torch_scatter import scatter
import numpy as np

class HybridVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features
        self.grid_size = model_cfg.get("GRID_SIZE", [0.1, 0.1, 0.15])
        max_num_points = model_cfg.get("MAX_NUM_POINTS", 800000)
        self.radius_graph = RadiusGraph(max_num_points)
        self.grid_sampler = GridSampling3D(self.grid_size)
        self.min_fitness = model_cfg.get("MIN_FITNESS", 0.3)
        #self.pf = PrimitiveFitting(self.grid_size, max_num_points)
        self.decay_radius2 = 0.2**2
        self.num_class = kwargs.get("num_class", 1)
        self.NA = -1 * self.num_class - 1

    def get_output_feature_dim(self):
        return self.num_point_features

    def fit_primitive(self, points, voxels, ep, ev):
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
        for itr in range(1):
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
            cov_det = cov.det()
            cov_inv = cov.inverse()
            valid_mask = (weight_sum >= 6).float()
            
            llh = (d.unsqueeze(-2) @ cov_inv[ev] @ d.unsqueeze(-1)).squeeze(-1).squeeze(-1) # [E, 1, 3] @ [E, 3, 3] @ [E, 3, 1] = [E]
            llh = (-0.5*llh / 3).exp() * valid_mask[ev] # / ((2*np.pi)**3*cov_det[ev]).sqrt()

            edge_weight = llh
            fitness_sum = scatter(llh, ev, dim=0, dim_size=num_voxels, reduce='sum')
            fitness_mean = scatter(llh, ev, dim=0, dim_size=num_voxels, reduce='mean')
            fitness = (fitness_sum/20.0).clamp(max=0.2) + fitness_mean
            fitness = fitness_mean
        
        S, R = torch.linalg.eigh(cov)
        R = R * S[:, None, :].sqrt()

        primitives = torch.cat([mu, R.reshape(-1, 9), fitness.reshape(-1, 1)], dim=-1)

        return primitives, fitness

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
        #points = self.forward_ret_dict['points']
        #voxels = self.forward_ret_dict['voxels']
        ep, ev = self.forward_ret_dict['edges']
        edge_weight = self.forward_ret_dict['edge_weight']
        point_seg_labels = self.forward_ret_dict['point_seg_labels']
        primitive_seg_labels = self.forward_ret_dict['primitive_seg_labels']
        num_voxels = self.forward_ret_dict['num_voxels']
        
        valid_mask = (point_seg_labels[ep] != self.NA) & (primitive_seg_labels[ev] != self.NA)
        consistency = (primitive_seg_labels[ev] == point_seg_labels[ep]) & valid_mask
        neg_consistency = (primitive_seg_labels[ev] != point_seg_labels[ep]) & valid_mask

        for th in np.linspace(0, 1, 100):
            mask = edge_weight > th
            iou1 = (mask & consistency).sum() / consistency.sum()
            neg_mask = edge_weight < th
            iou2 = (neg_mask & neg_consistency).sum() / neg_consistency.sum()
            print(f'th={th}, prec_pos={iou1}, prec_neg={iou2}')

        import ipdb; ipdb.set_trace()

        return loss

    def propagate_seg_labels(self, seg_cls_labels, seg_inst_labels, ep, ev, num_voxels):
        seg_labels = seg_inst_labels * self.num_class + seg_cls_labels
        max_seg_label = seg_labels.max().long().item() + 1
        keys = ev * max_seg_label + seg_labels[ep]
        sorted_keys = torch.sort(keys)[0] % max_seg_label
        degree = scatter(torch.ones_like(ep), ev, reduce='sum', dim_size=num_voxels, dim=0) # [V]
        offset = torch.cumsum(degree, dim=0) - degree # [V]
        primitive_seg_labels = seg_labels[ep][offset + torch.div(degree, 2, rounding_mode='trunc')]
        point_seg_labels = seg_labels

        return point_seg_labels, primitive_seg_labels

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
        points4d = points[:, :4].contiguous()
        
        voxels = self.grid_sampler(points4d) # [V, 4] 
        num_voxels = voxels.shape[0]

        ep, ev = self.radius_graph(points4d, voxels, 0.3, -1) # [2, E]
        
        # propagate segmentation labels to primitive
        point_seg_labels, primitive_seg_labels = \
            self.propagate_seg_labels(batch_dict['seg_cls_labels'], batch_dict['seg_inst_labels'],
                                      ep, ev, num_voxels)
        #ignored_mask = (seg_labels[ep] == self.NA) | (primitive_seg_labels[ev] == self.NA)
        #consistency = (primitive_seg_labels[ev] == seg_labels[ep]) | ignored_mask
        #ret_dict['consistency'] = consistency.float()

        primitives, fitness = self.fit_primitive(points, voxels, ep, ev)
        valid_mask = (fitness > self.min_fitness)
        point_validity = scatter(valid_mask.float()[ev], ep, dim=0,
                                 dim_size=points.shape[0], reduce='max')
        
        # select valid primitives, remove covered points
        primitives = primitives[valid_mask]
        valid_primitive_seg_labels = primitive_seg_labels[valid_mask]
        sp_points = points[point_validity == 0]
        sp_points = torch.cat([sp_points,
                               sp_points.new_zeros(sp_points.shape[0],
                                                   primitives.shape[-1] - sp_points.shape[-1])
                              ], dim=-1) # hybrid points and primitives
        sp_point_seg_labels = point_seg_labels[point_validity == 0]

        # merge primitives and points
        hybrid = torch.cat([primitives, sp_points], dim=0)
        hybrid_seg_labels = torch.cat([valid_primitive_seg_labels, sp_point_seg_labels] , dim=0)
        batch_dict['hybrid'] = hybrid
        batch_dict['batch_idx'] = hybrid[:, 0].long()
        batch_dict['gt_seg_cls_labels'] = hybrid_seg_labels % self.num_class
        
        return batch_dict
