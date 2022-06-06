import argparse
import torch
import numpy as np
#from visualizer_utils import PolyScopeVisualizer as psv
import pickle
import multiprocessing
from lib.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu

from torch_scatter import scatter
import yaml
import joblib
import os


def to_torch_cuda(d):
    if isinstance(d, np.ndarray):
        return torch.from_numpy(d).cuda()
    elif isinstance(d, torch.Tensor):
        return d.cuda()
    else:
        raise ValueError("Unrecognied Class")

def to_numpy_cpu(d):
    if isinstance(d, np.ndarray):
        return d
    elif isinstance(d, torch.Tensor):
        return d.detach().cpu().numpy()
    elif d is None:
        return None
    else:
        raise ValueError("Unrecognied Class")


def fps_clustering(points, stopping_diameter):
    points = points[:, :2]
    seed = points[0]
    dist = (points - seed).norm(p=2, dim=-1)
    cluster_ids = torch.zeros(points.shape[0], dtype=torch.long).to(points.device)
    num_clusters = 1
    while dist.max() > stopping_diameter:
        next_idx = dist.argmax()
        seed = points[next_idx]
        new_dist = (points - seed).norm(p=2, dim=-1)
        mask = new_dist < dist
        cluster_ids[mask] = num_clusters
        dist = dist.min(new_dist)
        num_clusters += 1

    return cluster_ids, num_clusters
        
def filter_clusters(points, cluster_ids, valid_mask, num_clusters, cluster_args=[], point_args=[]):
    point_valid_mask = valid_mask[cluster_ids]
    # remove points, keep dummy clusters
    new_points = points[point_valid_mask]
    cluster_ids = cluster_ids[point_valid_mask]
    # remove dummy clusters
    cluster_idx_map = cluster_ids.new_zeros(num_clusters)
    num_new_clusters = valid_mask.sum().long()
    cluster_idx_map[valid_mask] = torch.arange(num_new_clusters).to(cluster_idx_map)
    new_cluster_ids = cluster_idx_map[cluster_ids]

    returns = [new_points, new_cluster_ids, num_new_clusters]

    if len(cluster_args) > 0:
        cluster_ret_args = []
        for arg in cluster_args:
            if arg is None:
                cluster_ret_args.append(None)
            else:
                cluster_ret_args.append(arg[valid_mask])
        returns += cluster_ret_args
    
    if len(point_args) > 0:
        point_ret_args = []
        for arg in point_args:
            if arg is None:
                point_ret_args.append(None)
            else:
                point_ret_args.append(arg[point_valid_mask])
        returns += point_ret_args

    return returns

def pairwise_dist(a, b):
    """
        a [N, 3]
        b [M, 3]

    return 
        pairwise dist [N, M]
    """
    return (a[:, :].unsqueeze(1) - b[:, :].unsqueeze(0)).square().sum(-1)
    #sqa = a.square().sum(-1) # [N]
    #sqb = b.square().sum(-1) # [M]
    #abT = (a @ b.T) # [N, M]
    #pdist = sqa[:, None] + sqb[None, :] - 2 * abT

    #return pdist

def prepare_database(cfg, frame_id, points, inst_labels, cls_labels, boxes):
    """
    Args:
        cfg (yaml configurations)
        frame_id (int)
        points [N, 6]
        feat [N, 3+6]
        cls_labels [N]
        inst_labels [N]

        boxes [M, 8]

    Returns:
        database (dict)
    """
    # unpack
    prefix = 'data/waymo_aug_db_val'
    record_save_path = f'{prefix}/frame_{frame_id:06d}.pkl'
    feat = points[:, :3]#data_dict['points'][:, :3]
    points = points[:, 3:6] # data_dict['points'][:, 3:6]
    if boxes is None:
        boxes = np.zeros((0, 8))
    boxes = boxes.reshape(-1, 8)
    boxes = boxes[:, [1,2,3,4,5,6,7,0]] #data_dict.get('bbox', None)[:, [1,2,3,4,5,6,7,0]]
    
    # data copy
    points = to_torch_cuda(points)
    feat = to_torch_cuda(feat)
    boxes = to_torch_cuda(boxes)
    boxes_cpu = to_numpy_cpu(boxes)
    cls_labels = to_torch_cuda(cls_labels)
    if inst_labels is not None:
        inst_labels = to_torch_cuda(inst_labels)

    foreground_class = cfg["FOREGROUND_CLASS"]
    strategies = cfg['STRATEGIES']

    cls_points_dict = {support_class: points[cls_labels == support_class].contiguous() 
                        for support_class in [18, 21, 22]}
    cls_feat_dict = {support_class: feat[cls_labels == support_class].contiguous() 
                     for support_class in [18, 21, 22]}
    for g in foreground_class:
        cls_points_dict[g] = points[cls_labels == g].contiguous()
        cls_feat_dict[g] = feat[cls_labels == g].contiguous()

    instance_dict = {fg_cls: [] for fg_cls in foreground_class}

    for fg_cls, strategy in zip(foreground_class, strategies):
        support = strategy['support']
        attach_box = strategy.get("attach_box", False)
        radius = strategy.get('radius', None)
        group_radius = strategy.get('group_radius', None)
        support_radius = strategy.get('support_radius', None)
        min_num_point = strategy.get('min_num_points', 5)
        use_inst_label = strategy.get('use_inst_label', False)
        group_with = strategy.get('group_with', [])
        
        # clustering
        cls_mask = cls_labels == fg_cls
        if not cls_mask.any():
            continue
        cls_points = points[cls_mask]
        cls_feat = feat[cls_mask]
        cls_cls_labels = cls_labels[cls_mask]
        if inst_labels is not None:
            cls_inst_labels = inst_labels[cls_mask]
        else:
            cls_inst_labels = None
        if use_inst_label:
            unique, cluster_ids = torch.unique(cls_inst_labels, sorted=True, return_inverse=True)
            num_clusters = unique.shape[0]
        else:
            cluster_ids, num_clusters = fps_clustering(cls_points, radius)
        
        # group with other classes
        if len(group_with) > 0:
            cluster_centers = scatter(cls_points, cluster_ids, dim=0, dim_size=num_clusters, reduce='mean')
            offsets = cluster_ids.new_zeros(num_clusters, 1)
            sizes = scatter(torch.ones_like(cluster_ids), cluster_ids, dim=0, dim_size=num_clusters, reduce='sum').unsqueeze(-1)
            for g in group_with:
                g_points = cls_points_dict[g]
                g_feat = cls_feat_dict[g]
                g_dist = pairwise_dist(cluster_centers[:, :2], g_points[:, :2]) ** 0.5 # [C, G]
                g_indices = torch.where(g_dist < group_radius) # [2, N]
                grouped_points = g_points[g_indices[1]] # [N]
                grouped_feat = g_feat[g_indices[1]] # [N]
                grouped_cluster_ids = g_indices[0] # [N]
                num_new_points = scatter(torch.ones_like(grouped_cluster_ids), grouped_cluster_ids,
                                         dim=0, dim_size=num_clusters, reduce='sum').unsqueeze(-1)
                offsets = torch.cat([offsets, offsets[:, -1:] + sizes[:, -1:]], dim=1)
                sizes = torch.cat([sizes, num_new_points[:, -1:]], dim=1)

                cluster_ids = torch.cat([cluster_ids, grouped_cluster_ids], dim=0)
                cls_points = torch.cat([cls_points, grouped_points], dim=0)
                cls_feat = torch.cat([cls_feat, grouped_feat], dim=0)
                cls_cls_labels = torch.cat([cls_cls_labels,
                                            cls_cls_labels.new_zeros(grouped_points.shape[0])+g],
                                           dim=0)
                num_points_by_cluster = scatter(torch.ones_like(cluster_ids), cluster_ids, dim=0, dim_size=num_clusters, reduce='sum')
                assert (num_points_by_cluster == sizes.sum(-1)).all()

        else:
            offsets = None
            sizes = None
        num_points_by_cluster = scatter(torch.ones_like(cluster_ids), cluster_ids, dim=0, dim_size=num_clusters, reduce='sum')

        # filter by min num points
        cls_points, cluster_ids, num_clusters, offsets, sizes, cls_cls_labels, cls_feat = \
                filter_clusters(cls_points, cluster_ids,
                                num_points_by_cluster > min_num_point,
                                num_clusters, point_args=[cls_cls_labels, cls_feat],
                                cluster_args=[offsets, sizes])

        if num_clusters == 0:
            continue
        
        #vis_dict = dict(
        #    points=points,
        #    aug_points=cls_points,
        #    aug_seg_cls_labels=cls_cls_labels,
        #    batch_idx=torch.zeros(points.shape[0], 1).long(),
        #    batch_idx_aug=torch.zeros(cls_points.shape[0], 1).long(),
        #    batch_size=1,
        #    seg_cls_labels=cls_labels,
        #)
        #vis(vis_dict)
        
        # find support of each cluster
        cluster_centers = scatter(cls_points, cluster_ids, dim=0, dim_size=num_clusters, reduce='mean')
        dist_to_supports = cls_points.new_zeros(num_clusters) + 1e10 # [C]
        support_cls = cluster_ids.new_zeros(num_clusters)
        trans_z = cls_points.new_zeros(num_clusters) # [C]

        for support_idx, support_class in enumerate(support):
            support_points = cls_points_dict[support_class]
            if support_points.shape[0] == 0:
                continue
            dist_to_support = pairwise_dist(cluster_centers[:, :2], support_points[:, :2]) ** 0.5 # [C, S]
            dist_to_support_point, support_point_idx = dist_to_support.min(-1) # [C], [C]
            trans_z_this = cluster_centers[:, 2] - support_points[support_point_idx, 2] # dist in z axis

            mask = dist_to_support_point < dist_to_supports
            support_cls[mask] = support_class
            dist_to_supports[mask] = dist_to_support_point[mask]
            trans_z[mask] = trans_z_this[mask]

        cls_points, cluster_ids, num_clusters, trans_z, cluster_centers, offsets, sizes, \
              cls_cls_labels, cls_feat = filter_clusters(cls_points, cluster_ids,
                                               dist_to_supports < support_radius,
                                               num_clusters,
                                               cluster_args=[trans_z, cluster_centers, offsets, sizes],
                                               point_args=[cls_cls_labels, cls_feat])

        if (num_clusters > 0) and attach_box and (boxes.shape[0] > 0):
            box_index_by_point = points_in_boxes_gpu(cls_points.unsqueeze(0), boxes[:, :7].unsqueeze(0))[0] + 1 # [0, num_box]
            mix_index = cluster_ids * (boxes.shape[0]+1) + box_index_by_point
            sorted_index, inverse = torch.sort(mix_index, dim=0)
            num_points_by_cluster = scatter(torch.ones_like(cluster_ids), cluster_ids, dim=0, dim_size=num_clusters, reduce='sum')
            offset = num_points_by_cluster.cumsum(0) - num_points_by_cluster
            size = num_points_by_cluster // 2
            box_index = sorted_index[offset+size] % (boxes.shape[0] + 1) - 1
            box_index = to_numpy_cpu(box_index)

        #vis_dict = dict(
        #    points=points,
        #    aug_points=cls_points,
        #    aug_seg_cls_labels=cls_cls_labels,
        #    batch_idx=torch.zeros(points.shape[0], 1).long(),
        #    batch_idx_aug=torch.zeros(cls_points.shape[0], 1).long(),
        #    batch_size=1,
        #    seg_cls_labels=cls_labels,
        #)
        #vis(vis_dict)
                
        cls_points = to_numpy_cpu(cls_points)
        cls_feat = to_numpy_cpu(cls_feat)
        trans_z = to_numpy_cpu(trans_z)
        cluster_ids = to_numpy_cpu(cluster_ids)
        offsets = to_numpy_cpu(offsets)
        support_cls = to_numpy_cpu(support_cls)
        sizes = to_numpy_cpu(sizes)
        cls_cls_labels = to_numpy_cpu(cls_cls_labels)
        for inst_id in range(num_clusters):
            save_path = f'{prefix}/frame_{frame_id:06d}_class_{fg_cls:02d}_inst_{inst_id:07d}.npy'
            mask = cluster_ids == inst_id
            np.save(save_path, np.concatenate([cls_points[mask], cls_feat[mask], cls_cls_labels[mask].reshape(-1, 1)], axis=1))
            record = dict(
                path=save_path,
                trans_z=trans_z[inst_id],
                support_cls=support_cls[inst_id],
                num_points=cls_points.shape[0],
            )
            if attach_box and (boxes.shape[0] > 0):
                if box_index[inst_id] == -1:
                    box_this = np.zeros(8)
                else:
                    box_this = boxes_cpu[box_index[inst_id]]
                record['box3d'] = box_this
            else:
                record['box3d'] = np.zeros(8)
            if offsets is not None:
                record['offsets']=offsets[inst_id],
            if sizes is not None:
                record['sizes']=sizes[inst_id],
                assert sum(sizes[inst_id]) == cls_points[mask].shape[0]
            instance_dict[fg_cls].append(record)

    with open(record_save_path, 'wb') as fout:
        pickle.dump(instance_dict, fout)
                
if __name__ == '__main__':
    import joblib
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=int)
    #size = 23691 // 8
    size = 747 # 5976 // 8
    args = parser.parse_args()
    offset = size * args.split
    args.size = size

    #prefix = 'data/WaymoV2/semseg/waymo_v132_with_r2_top/waymo_with_r2_top_training' 
    prefix = 'validation'
    #prefix = 'training' #'data/WaymoV2/semseg/waymo_v132_with_r2_top/waymo_with_r2_top_training' 
    points = joblib.load(f'{prefix}.point{args.split}')
    inst_labels = joblib.load(f'{prefix}.instance{args.split}')
    cls_labels = joblib.load(f'{prefix}.label{args.split}')
    bboxs = joblib.load(f'{prefix}.bbox{args.split}')
    
    with open('prepare_db.yaml', 'r') as fin:
        cfg = yaml.safe_load(fin)

    #import time
    #t0 = time.time()
    #from functools import partial
    #import multiprocessing
    from tqdm import tqdm
    #import os
    prefix = 'data/waymo_aug_db_val'
    os.makedirs(prefix, exist_ok=True)
    #torch.multiprocessing.set_start_method('spawn')
    frame_ids = [i for i in range(offset, offset+args.size)]
    for i in tqdm(range(args.size)):
        prepare_database(cfg, frame_ids[i], points[i], inst_labels[i], cls_labels[i], bboxs[i])

    #foreground_class = cfg["FOREGROUND_CLASS"]
    #db_infos = {g: [] for g in foreground_class}
    #print('merging infos from all frames')
    #for frame_id in tqdm(range()):
    #    with open(f'{prefix}/frame_{frame_id:06d}.pkl', 'rb') as fin:
    #        instance_dict = pickle.load(fin)
    #        for key, val in instance_dict.items():
    #            db_infos[key] += val

    #with open('data/waymo_aug_db.pkl', 'wb') as fout:
    #    pickle.dump(db_infos, fout)
