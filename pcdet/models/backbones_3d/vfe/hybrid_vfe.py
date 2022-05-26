import torch
from torch import nn
from .vfe_template import VFETemplate
from ..grid_sampling import GridSampling3D
from ....ops.torch_hash import RadiusGraph
from torch_scatter import scatter
import numpy as np
from ...model_utils.basic_blocks import MLP
from collections import defaultdict

class Dummy(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([w], dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        return self.w.clamp(min=0) * x

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
        self.num_class = kwargs.get("num_class", 6)
        self.NA = - 1
        self.radius = model_cfg.get("RADIUS", None)
        self.decay = model_cfg.get("DECAY", None)
        self.gain = model_cfg.get("GAIN", None)
        self.theta0 = Dummy(1.0/3)
        self.theta1 = 1e-4
        self.fitness_regress = nn.Sequential(
                                   nn.Linear(2, 1),
                                   nn.Sigmoid()
                               )

        self.K = 4
        self.eigval_transform = nn.Sequential(
                                    MLP([self.K*2, 16, 16, 16, 16, 1]),
                                    #nn.ReLU(),
                                )

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
        degree = scatter(edge_weight, ev, dim=0, dim_size=num_voxels, reduce='sum') # [V]
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
            cov = cov + torch.diag_embed(torch.tensor([1, 1, 1]).float()).to(cov).repeat(num_voxels, 1, 1) * self.theta1
            eigvecs, eigvals, _ = torch.linalg.svd(cov) # [V, 3, 3], [V, 3], [V, 3, 3]
            #eigvecs = eigvecs.detach()
            eigvals_ext = []
            for k in range(self.K):
                eigvals_p = eigvals * (10**k) * 2 * np.pi
                eigvals_sin = eigvals_p.sin()
                eigvals_cos = eigvals_p.cos()
                eigvals_ext.append(eigvals_sin)
                eigvals_ext.append(eigvals_cos)
            eigvals_ext = torch.stack(eigvals_ext, dim=-1).reshape(-1, self.K*2) # [V*3, K*2]

            eigvals_out = (self.eigval_transform(eigvals_ext).reshape(-1, 3)).exp()*0.001 + eigvals # [V, 3]
            #print('min={}, max={}, nan?{}, inf?{}'.format(eigvals_out.min(0)[0].data, eigvals_out.max(0)[0].data, eigvals_out.isnan().any(), eigvals_out.isinf().any()))

            cov_inv = eigvecs @ torch.diag_embed(1.0/eigvals_out) @ eigvecs.transpose(1, 2)
            #valid_mask = (weight_sum >= 4).float()
            
            llh = (d.unsqueeze(-2) @ cov_inv[ev] @ d.unsqueeze(-1)).squeeze(-1).squeeze(-1) # [E, 1, 3] @ [E, 3, 3] @ [E, 3, 1] = [E, 1]
            llh = self.theta0(-0.5*llh).exp() # * valid_mask[ev] # / ((2*np.pi)**3*cov_det[ev]).sqrt()

            edge_weight = llh

        llh = llh * (degree[ev] >= 4).float()
        llh_sum = scatter(llh, ev, dim=0, dim_size=num_voxels, reduce='sum')
        llh_mean = scatter(llh, ev, dim=0, dim_size=num_voxels, reduce='mean')
        llh_vec = torch.stack([llh_sum, llh_mean], axis=-1)
        fitness = self.fitness_regress(llh_vec).squeeze(-1)
        
        #S, R = torch.linalg.eigh(cov)
        #R = R * S[:, None, :].sqrt()

        primitives = torch.cat([mu, cov.reshape(-1, 9), fitness.reshape(-1, 1)], dim=-1)

        return primitives, fitness, llh

    def get_loss(self, tb_dict=None):
        """
        Args in forward_ret_dict:
            points [N, 4]
            voxels [V, 4]
            ep, ev [E]
            edge_weight [E]
            seg_labels [N]

        Returns:
            
        """
        gt_edge_weight = torch.cat(self.forward_dict['gt_edge_weight'], dim=0)
        edge_weight = torch.cat(self.forward_dict['edge_weight'], dim=0)
        gt_fitness = torch.cat(self.forward_dict['gt_fitness'], dim=0)
        fitness = torch.cat(self.forward_dict['fitness'], dim=0)
        assert edge_weight.shape == gt_edge_weight.shape
        assert fitness.shape == gt_fitness.shape
        
        pos_mask = gt_edge_weight == 1
        if pos_mask.any():
            pos_loss = (0.7 - edge_weight[pos_mask]).clamp(min=0).square().sum()
        else:
            pos_loss = 0.0
        neg_mask = gt_edge_weight != 1
        if neg_mask.any():
            neg_loss = (edge_weight[neg_mask]-0.3).clamp(min=0).square().sum()
        else:
            neg_loss = 0.0
        pos_pmask = gt_fitness > 0.7
        if pos_pmask.any():
            pos_ploss = (0.7 - fitness[pos_pmask]).clamp(min=0).square().sum()
        else:
            pos_ploss = 0.0
        neg_pmask = gt_fitness < 0.3
        if neg_pmask.any():
            neg_ploss = (fitness[neg_pmask] - 0.3).clamp(min=0).square().sum()
        else:
            neg_ploss = 0.0
        loss = (pos_loss + neg_loss) / max(gt_edge_weight.shape[0], 1) + (pos_ploss + neg_ploss) / max(gt_fitness.shape[0], 1)

        if tb_dict is not None:
            if pos_mask.any():
                tb_dict['positive_prec_0.5'] = (edge_weight[pos_mask] > 0.5).float().mean().item()
            if pos_pmask.any():
                tb_dict['positive_primitive_prec_0.5'] = (fitness[pos_pmask] > 0.5).float().mean().item()
            if neg_mask.any():
                tb_dict['negative_prec_0.5'] = (edge_weight[neg_mask] < 0.5).float().mean().item()
            if neg_pmask.any():
                tb_dict['negitive_primitive_prec_0.5'] = (fitness[neg_pmask] < 0.5).float().mean().item()
            tb_dict['num_pos'] = pos_mask.sum().long().item()
            tb_dict['num_neg'] = neg_mask.sum().long().item()
            tb_dict['num_pos_primitive'] = pos_pmask.sum().long().item()
            tb_dict['num_neg_primitive'] = neg_pmask.sum().long().item()
            tb_dict['theta0'] = self.theta0.w.data[0].item()

        #ep, eh = self.forward_ret_dict['edges']
        #edge_weight = self.forward_ret_dict['edge_weight']
        #point_seg_cls_labels = self.forward_ret_dict['point_seg_cls_labels']
        #hybrid_seg_cls_labels = self.forward_ret_dict['hybrid_seg_cls_labels']
        #
        #num_primitives = self.forward_ret_dict['hybrid_seg_cls_labels'].shape[0]
        #
        #valid_mask = (point_seg_cls_labels[ep] != self.NA) & (hybrid_seg_cls_labels[eh] != self.NA)
        #consistency = (hybrid_seg_cls_labels[eh] == point_seg_cls_labels[ep]) & valid_mask
        #neg_consistency = (hybrid_seg_cls_labels[eh] != point_seg_cls_labels[ep]) & valid_mask

        #for th in np.linspace(0, 1, 100):
        #    mask = edge_weight > th
        #    iou1 = (mask & consistency).sum() / consistency.sum()
        #    neg_mask = edge_weight < th
        #    iou2 = (neg_mask & neg_consistency).sum() / neg_consistency.sum()
        #    print(f'th={th}, prec_pos={iou1}, prec_neg={iou2}')

        return loss, tb_dict

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
        primitive_seg_cls_labels = self.seg_label_to_cls_label(primitive_seg_labels)

        primitives, fitness, edge_weight = self.fit_primitive(points, voxels, ep, ev, self.decay[level], self.gain[level])
        pcoords = (primitives[:, 1:4] - points.min(0)[0][1:4]) // self.grid_sampler[0].grid_size[1:4]
        vcoords = (voxels[:, 1:4] - points.min(0)[0][1:4]) // self.grid_sampler[0].grid_size[1:4]
        devi_mask = (vcoords == pcoords).all(-1)
        #devi_mask = (primitives[:, 1:4] - voxels[:, 1:4]).norm(p=2, dim=-1) < 0.1
        self.forward_dict['edge_weight'].append(edge_weight)
        gt_edge_weight = (primitive_seg_cls_labels[ev] == batch_dict['seg_cls_labels'][ep]).long()
        gt_primitive_fitness = scatter(gt_edge_weight.float(), ev, reduce='mean', dim=0, dim_size=num_voxels)
        self.forward_dict['gt_edge_weight'].append((primitive_seg_cls_labels[ev] == batch_dict['seg_cls_labels'][ep]).long())
        self.forward_dict['gt_fitness'].append(gt_primitive_fitness)
        self.forward_dict['fitness'].append(fitness)

        valid_mask = (fitness > self.min_fitness[level]) & devi_mask
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
        primitive_seg_cls_labels = self.seg_label_to_cls_label(valid_primitive_seg_labels)
        gt_edge_weight = (primitive_seg_cls_labels[ev] == batch_dict['seg_cls_labels'][ep]).long()

        # merge primitives and points
        batch_dict['primitives'].append(primitives)
        batch_dict['primitive_edges'].append(torch.stack([ep, ev], dim=0))
        batch_dict['primitive_edge_weight'].append(edge_weight)
        batch_dict['primitive_seg_labels'].append(valid_primitive_seg_labels)
        batch_dict['primitive_seg_cls_labels'].append(primitive_seg_cls_labels)
        batch_dict[f'primitives_{level}'] = primitives
        batch_dict[f'primitive_edges_{level}'] = torch.stack([ep, ev], dim=0)
        batch_dict[f'primitive_seg_labels_{level}'] = valid_primitive_seg_labels
        batch_dict[f'primitive_seg_cls_labels_{level}'] = self.seg_label_to_cls_label(valid_primitive_seg_labels)
        batch_dict[f'primitive_gt_edge_weight_{level}'] = gt_edge_weight
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
        self.forward_dict = defaultdict(list)
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
