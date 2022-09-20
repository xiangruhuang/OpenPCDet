import torch
from easydict import EasyDict
from torch_scatter import scatter
from torch_cluster import knn

from ...utils import common_utils

def ransac(point_xyz, e_plane, num_planes, sigma, stopping_delta=1e-2):
    """
    Args:
        point_xyz [N, 3]: point coordinates
        e_plane [N]: partition group id of each point
        num_planes (integer): number of partitions (planes), (denoted as P)
        sigma2: reweighting parameters
        stopping_delta: algorithmic parameter

    Returns:
        points: point-wise dictionary {
            weight [N]: indicating the likelihood of this point belong to a plane,
                        higher means more likely
            coords [N, 3]: the local rank coordinates
        }
        planes: plane-wise dictionary {
            eigvals [P, 3]: per plane PCA eigenvalues
            eigvecs [P, 3, 3]: per plane PCA eigenvectors
            normal [P, 3]: per plane normal vector
        }
    """
    point_weight = torch.ones(point_xyz.shape[0], dtype=torch.float32)
    sigma2 = sigma*sigma
    plane_degree = scatter(torch.ones_like(point_weight).long(), e_plane, dim=0, dim_size=num_planes, reduce='sum')
    
    for itr in range(100):
        # compute plane center
        plane_xyz = scatter(point_xyz*point_weight[:, None], e_plane, dim=0, dim_size=num_planes, reduce='sum')
        plane_weight_sum = scatter(point_weight, e_plane, dim=0, dim_size=num_planes, reduce='sum')
        plane_xyz = plane_xyz / (plane_weight_sum[:, None] + 1e-6)

        # compute
        point_d = point_xyz - plane_xyz[e_plane]
        point_ddT = point_d[:, None, :] * point_d[:, :, None] * point_weight[:, None, None]
        plane_ddT = scatter(point_ddT, e_plane, dim=0, dim_size=num_planes, reduce='mean')
        eigvals, eigvecs = torch.linalg.eigh(plane_ddT)
        plane_normal = eigvecs[:, :, 0]
        p2plane_dist = (point_d * plane_normal[e_plane]).sum(-1).abs()
        new_point_weight = sigma2 / (p2plane_dist ** 2 +sigma2)
        delta_max = (new_point_weight - point_weight).abs().max()
        point_weight = new_point_weight
        if delta_max < stopping_delta:
            break
    
    point_coords = torch.stack([torch.ones_like(point_weight),
                                (eigvecs[e_plane, :, 1] * point_d).sum(-1),
                                (eigvecs[e_plane, :, 2] * point_d).sum(-1)], dim=-1)
    point_coords[:, 1:] = point_coords[:, 1:] - point_coords[:, 1:].min(0)[0]
    point_coords[:, 1:] /= point_coords[:, 1:].max(0)[0].clamp(min=1e-5)
    
    plane_max0 = scatter((point_d * eigvecs[e_plane, :, 0]).sum(-1), e_plane, dim=0, dim_size=num_planes, reduce='max')
    plane_min0 = scatter((point_d * eigvecs[e_plane, :, 0]).sum(-1), e_plane, dim=0, dim_size=num_planes, reduce='min')
    plane_max1 = scatter((point_d * eigvecs[e_plane, :, 1]).sum(-1), e_plane, dim=0, dim_size=num_planes, reduce='max')
    plane_min1 = scatter((point_d * eigvecs[e_plane, :, 1]).sum(-1), e_plane, dim=0, dim_size=num_planes, reduce='min')
    plane_max2 = scatter((point_d * eigvecs[e_plane, :, 2]).sum(-1), e_plane, dim=0, dim_size=num_planes, reduce='max')
    plane_min2 = scatter((point_d * eigvecs[e_plane, :, 2]).sum(-1), e_plane, dim=0, dim_size=num_planes, reduce='min')

    l1_proj_min = torch.stack([plane_min0, plane_min1, plane_min2], dim=-1)
    l1_proj_max = torch.stack([plane_max0, plane_max1, plane_max2], dim=-1)

    points = EasyDict(
        weight=point_weight,
        coords=point_coords,
        plane_dist=p2plane_dist,
    )

    planes = EasyDict(
        xyz=plane_xyz,
        degree=plane_degree,
        eigvals=eigvals,
        eigvecs=eigvecs,
        normal=plane_normal,
        l1_proj_min=l1_proj_min,
        l1_proj_max=l1_proj_max,
    )

    return points, planes

def plane_analysis(points, planes, e_plane, num_planes, dist_thresh, count_gain, decision_thresh):
    # number of points within distance threshold `dist_thresh`
    valid_mask = (points.plane_dist < dist_thresh).float()
    plane_count = scatter(valid_mask, e_plane, dim=0, dim_size=num_planes, reduce='sum')

    # fitting error (weighted)
    plane_error = scatter(points.plane_dist*points.weight, e_plane, dim=0, dim_size=num_planes, reduce='sum')
    plane_weight_sum = scatter(points.weight, e_plane, dim=0, dim_size=num_planes, reduce='sum')
    plane_mean_error = plane_error / (plane_weight_sum + 1e-5)

    # compute fitness
    plane_fitness = (plane_count * count_gain).clamp(max=0.55) + (decision_thresh / (decision_thresh + plane_mean_error)).clamp(max=0.55)

    planes.fitness = plane_fitness
    planes.mean_error = plane_mean_error
    
    return points, planes

def pca_fitting(point_xyz, stride, k, dist_thresh, count_gain, sigma, decision_thresh):
    point_xyz = torch.from_numpy(point_xyz)
    sampled_xyz = point_xyz[::stride]
    num_planes = sampled_xyz.shape[0]
    e_point, e_plane = knn(sampled_xyz, point_xyz, k=1)
    assert (e_point - torch.arange(point_xyz.shape[0])).abs().max() < 1e-2
    del e_point
    
    # plane fitting
    points, planes = ransac(point_xyz, e_plane, num_planes, sigma)
    
    # evaluate fitness of planes
    points, planes = plane_analysis(points, planes, e_plane, num_planes, dist_thresh, count_gain, decision_thresh)
    plane_mask = planes.fitness > 1.0
    point_mask = planes.fitness[e_plane] > 1.0
    point_mask &= points.weight > 0.5
    
    # transform plane id
    map2new_id = torch.zeros(num_planes, dtype=torch.long) - 1
    map2new_id[plane_mask] = torch.arange(plane_mask.long().sum())
    points.plane_id = map2new_id[e_plane]

    planes = common_utils.apply_to_dict(planes, lambda x: x.numpy())
    planes = common_utils.filter_dict(planes, plane_mask)
    planes = common_utils.transform_name(planes, lambda name: 'plane_'+name)

    points.pop('weight')
    points.pop('plane_dist')
    points = common_utils.apply_to_dict(points, lambda x: x.numpy())
    points = common_utils.transform_name(points, lambda name: 'point_'+name)
    return points, planes
