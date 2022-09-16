import os
import pickle
import copy
import numpy as np
import torch
import multiprocessing
import SharedArray
import torch.distributed as dist
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import scipy.io as sio

from sklearn.neighbors import NearestNeighbors as NN

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils, polar_utils
from ..dataset import DatasetTemplate
from .smpl_utils import SMPLModel
from torch_scatter import scatter
from torch_cluster import knn

def pca_fitting(point_xyz, stride, k, dist_thresh, count_gain):
    point_xyz = torch.from_numpy(point_xyz)
    sampled_xyz = point_xyz[::stride]
    num_planes = sampled_xyz.shape[0]
    e_point, e_plane = knn(sampled_xyz, point_xyz, k=1)
    plane_xyz = scatter(point_xyz[e_point], e_plane, dim=0, dim_size=num_planes, reduce='mean')

    point_d = point_xyz[e_point] - plane_xyz[e_plane]
    point_ddT = point_d[:, None, :] * point_d[:, :, None]
    plane_ddT = scatter(point_ddT, e_plane, dim=0, dim_size=num_planes, reduce='mean')
    eigvals, eigvecs = torch.linalg.eigh(plane_ddT)
    plane_normal = eigvecs[:, 0]
    p2plane_dist = (point_d[e_point] * plane_normal[e_plane]).sum(-1).abs()
    count = (p2plane_dist < dist_thresh).float()
    plane_count = scatter(count, e_plane, dim=0, dim_size=num_planes, reduce='sum')
    plane_fitness = plane_count * count_gain
    plane_mask = plane_fitness > 0.6

    plane_max0 = scatter((point_d * eigvecs[e_plane, 0]).sum(-1).abs(), e_plane, dim=0, dim_size=num_planes, reduce='max')
    plane_min0 = scatter((point_d * eigvecs[e_plane, 0]).sum(-1).abs(), e_plane, dim=0, dim_size=num_planes, reduce='min')
    plane_max1 = scatter((point_d * eigvecs[e_plane, 1]).sum(-1).abs(), e_plane, dim=0, dim_size=num_planes, reduce='max')
    plane_min1 = scatter((point_d * eigvecs[e_plane, 1]).sum(-1).abs(), e_plane, dim=0, dim_size=num_planes, reduce='min')
    plane_max2 = scatter((point_d * eigvecs[e_plane, 2]).sum(-1).abs(), e_plane, dim=0, dim_size=num_planes, reduce='max')
    plane_min2 = scatter((point_d * eigvecs[e_plane, 2]).sum(-1).abs(), e_plane, dim=0, dim_size=num_planes, reduce='min')

    plane_coords = torch.stack([plane_min0, plane_max0, plane_min1, plane_max1, plane_min2, plane_max2], dim=-1)
    import ipdb; ipdb.set_trace()

    plane_wise_dict = dict(
        plane_xyz=plane_xyz,
        plane_coords=plane_coords,
        plane_eigvecs=eigvecs,
        plane_eigvals=eigvals,
        plane_fitness=plane_fitness,
    )

    point_mask = scatter(plane_fitness[e_plane], e_point, dim=0, dim_size=point_xyz.shape[0], reduce='max') > 0.6
    plane_wise_dict = common_utils.apply_to_dict(plane_wise_dict, lambda x: x.numpy())

    return plane_wise_dict, plane_mask, ~point_mask

class SurrealDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, training=training, root_path=root_path, logger=logger
        )

        self.num_sweeps = 1 # single frame dataset
        self.test_mode = dataset_cfg.get("TEST_MODE", None)
        plane_cfg = dataset_cfg.get("PLANE", None)
        if plane_cfg is not None:
            self.use_plane = True
            self.plane_cfg = plane_cfg
        else:
            self.use_plane = False
        
        #self.data_path = self.root_path / self.dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        #split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        #self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        #self.num_sweeps = self.dataset_cfg.get('NUM_SWEEPS', 1)
        #self._merge_all_iters_to_one_epoch = dataset_cfg.get("MERGE_ALL_ITERS_TO_ONE_EPOCH", False)
        #self.more_cls5 = self.segmentation_cfg.get('MORE_CLS5', False)
        #self.use_spherical_resampling = self.dataset_cfg.get("SPHERICAL_RESAMPLING", False)
        params = sio.loadmat(f'{self.root_path}/surreal_smpl_params.mat')['params'].reshape(-1, 2, 83)
        #params = params[:, :, :]
        params = params.reshape(-1, 10, 2, 83)
        self._params = dict(
            train=params[:, 1:].reshape(-1, 83),
            test=params[:, :1].reshape(-1, 83),
        )
        self.params = self._params[self.mode]
        if dataset_cfg.SAMPLED_INTERVAL[self.mode] > 1:
            self.params = self.params[::dataset_cfg.SAMPLED_INTERVAL[self.mode]]

        self.embedding = np.load(f'{self.root_path}/laplacian.npy')
        self.smpl_model = {
          0: SMPLModel(f'{self.root_path}/smpl_female_model.mat'),
          1: SMPLModel(f'{self.root_path}/smpl_male_model.mat'),
        }
        self.smpl_model[0].set_params(pose=np.zeros(72), beta=np.zeros(10), trans=np.zeros(3))
        self.smpl_model[1].set_params(pose=np.zeros(72), beta=np.zeros(10), trans=np.zeros(3))
        self.rest_pose_xyz = np.copy(self.smpl_model[0].verts)
        self.rest_pose_xyz = self.rest_pose_xyz[:, [0, 2, 1]]
        logger.info(f'template median.0 = {np.median(self.rest_pose_xyz[:, 0])}')
        logger.info(f'template median.1 = {np.median(self.rest_pose_xyz[:, 1])}')
        logger.info(f'template median.2 = {np.median(self.rest_pose_xyz[:, 2])}')

    def __getitem__(self, index):
        gender = int(self.params[index, 0])
        smpl_model = self.smpl_model[gender]
        smpl_model.set_params(
            trans=np.zeros(3),
            beta=self.params[index, 1:11],
            pose=self.params[index, 11:],
        )
        vertices = smpl_model.verts

        data_dict = dict(
            point_wise=dict(
                point_xyz=vertices.astype(np.float32),
                point_feat=vertices.astype(np.float32),
                segmentation_label=np.arange(vertices.shape[0]),
            ),
            object_wise=dict(),
            scene_wise=dict(
                template_xyz=self.rest_pose_xyz.astype(np.float32),
                template_embedding=self.embedding.astype(np.float32),
                smpl_params=self.params[index, 1:],
                frame_id=index,
            ),
        )
        if self.use_plane:
            plane_wise_dict, plane_mask, point_mask = pca_fitting(vertices, **self.plane_cfg)
            data_dict['object_wise'] = common_utils.filter_dict(plane_wise_dict, plane_mask)
            data_dict['point_wise'] = common_utils.filter_dict(data_dict['point_wise'], point_mask)

        if self.test_mode == 'Hard':
            point_xyz = data_dict['point_wise']['point_xyz']
            pc_range_min = point_xyz.min(0)
            pc_range_max = point_xyz.max(0)

            pc_range_max[0] = pc_range_min[0] - 0.05
            pc_range_min[0] = pc_range_min[0] - 0.05
            num_disturbs = 200
            ratio = np.random.uniform(size=[num_disturbs, 3])
            disturb_points = pc_range_min * ratio + pc_range_max * (1-ratio)
            data_dict['point_wise']['point_xyz'] = np.concatenate([point_xyz, disturb_points], axis=0).astype(np.float32)
            data_dict['point_wise']['point_feat'] = data_dict['point_wise']['point_xyz']
            data_dict['point_wise']['segmentation_label'] = np.concatenate([data_dict['point_wise']['segmentation_label'],
                                                                            np.full(num_disturbs, -1, dtype=np.int32)],
                                                                           axis=0)

        return data_dict

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        annos = []

        if (output_path is not None) and (not os.path.exists(output_path)):
            os.makedirs(output_path, exist_ok=True)
        template_xyz = batch_dict['template_xyz']
        for index, cur_dict in enumerate(pred_dicts):
            pred_dict = dict(
                object_wise=dict(),
                point_wise=dict(),
                scene_wise=dict(),
            )
            dist = (template_xyz[cur_dict['point_wise']['gt_corres']] - template_xyz[cur_dict['point_wise']['corres']]).norm(p=2, dim=-1)
            mean_dist = dist.mean()
            for threshold in [2, 5, 10]:
                error_rate = (dist > threshold*0.01).float().mean()
                pred_dict['scene_wise'][f'error_rate_{threshold}cm'] = error_rate.detach().cpu().item()
            pred_dict['scene_wise']['mean_dist'] = dist.mean().detach().cpu().item()
            pred_dict['scene_wise']['median_dist'] = dist.median().detach().cpu().item()

            annos.append(pred_dict)

        return annos
    
    def evaluation(self, pred_dicts, box_class_names, **kwargs):
        result_dict = defaultdict(list)
        
        for pred_dict in pred_dicts:
            for key in pred_dict['scene_wise'].keys():
                result_dict[key].append(pred_dict['scene_wise'][key])
        result_str = ''
        for key in result_dict.keys():
            res = np.mean(result_dict[key])
            result_dict[key] = res
            result_str += f'{key}={res:.6f} \n'

        return result_str, result_dict

    def __len__(self):
        return len(self.params)
