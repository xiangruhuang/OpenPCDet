import os
import copy
import numpy as np
import torch
import torch.distributed as dist
from collections import defaultdict
import scipy.io as sio
import open3d as o3d
from easydict import EasyDict

from sklearn.neighbors import NearestNeighbors as NN

from ...utils import common_utils
from ..dataset import DatasetTemplate
from .smpl_utils import SMPLModel
from .surreal_utils import pca_fitting

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
        
        self.repeat = dataset_cfg.get("REPEAT", 1)
        if isinstance(self.repeat, dict):
            self.repeat = self.repeat[self.mode]

        params = sio.loadmat(f'{self.root_path}/surreal_smpl_params.mat')['params'].reshape(-1, 2, 83)
        #params = params[:, :, :]
        params = params.reshape(-1, 10, 2, 83)
        self._params = dict(
            train=params[:, 1:].reshape(-1, 83),
            val=params[:, :1].reshape(-1, 83),
        )
        self.params = self._params[self.split]
        if dataset_cfg.SAMPLED_INTERVAL[self.mode] > 1:
            self.params = self.params[::dataset_cfg.SAMPLED_INTERVAL[self.mode]]
        self.params = self.params.repeat(self.repeat, axis=0)

        self.embedding = np.load(f'{self.root_path}/laplacian.npy')
        self.smpl_model = {
          0: SMPLModel(f'{self.root_path}/smpl_female_model.mat'),
          1: SMPLModel(f'{self.root_path}/smpl_male_model.mat'),
        }
        self.smpl_model[0].set_params(pose=np.zeros(72), beta=np.zeros(10), trans=np.zeros(3))
        self.smpl_model[1].set_params(pose=np.zeros(72), beta=np.zeros(10), trans=np.zeros(3))
        self.template_xyz = np.copy(self.smpl_model[0].verts)
        self.template_xyz = self.template_xyz[:, [0, 2, 1]]
        logger.info(f'template median.0 = {np.median(self.template_xyz[:, 0])}')
        logger.info(f'template median.1 = {np.median(self.template_xyz[:, 1])}')
        logger.info(f'template median.2 = {np.median(self.template_xyz[:, 2])}')

    def __getitem__(self, index):
        gender = int(self.params[index, 0])
        smpl_model = self.smpl_model[gender]
        smpl_model.set_params(
            trans=np.zeros(3),
            beta=self.params[index, 1:11],
            pose=self.params[index, 11:],
        )
        template_vertices = smpl_model.verts
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(template_vertices)
        mesh.triangles = o3d.utility.Vector3iVector(smpl_model.faces)
        pcd = mesh.sample_points_uniformly(6890*3)
        vertices = np.array(pcd.points)
        tree = NN(n_neighbors=1).fit(template_vertices)
        dists, indices = tree.kneighbors(vertices)

        data_dict = EasyDict(
            point_wise=EasyDict(
                point_xyz=vertices.astype(np.float32),
                point_feat=vertices.astype(np.float32), #np.zeros((vertices.shape[0], 0), dtype=np.float32),
                segmentation_label=indices[:, 0],
                point_template_xyz=self.template_xyz[indices[:, 0]],
            ),
            object_wise=EasyDict(),
            scene_wise=EasyDict(
                template_xyz=self.template_xyz.astype(np.float32),
                template_embedding=self.embedding.astype(np.float32),
                smpl_params=self.params[index, 1:],
                frame_id=index,
            ),
        )

        #if self.use_plane:
        #    point_wise_dict, plane_wise_dict = pca_fitting(vertices, **self.plane_cfg)
        #    data_dict['object_wise'] = plane_wise_dict
        #    data_dict['point_wise'].update(point_wise_dict)

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
