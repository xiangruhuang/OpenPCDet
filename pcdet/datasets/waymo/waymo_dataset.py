# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.

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
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate
import glob

from pcdet.models.visualizers import PolyScopeVisualizer
from pcdet.config import cfg_from_yaml_file, cfg
import SharedArray as SA
import gc
import joblib

SIZE = {
    'waymo_seg_with_r2_top_training.point': 23691,
    'waymo_seg_with_r2_top_training.label': 23691,
    'waymo_seg_with_r2_top_training.instance': 23691,
    'waymo_seg_with_r2_top_training.box_label_attr': 23691,
    'waymo_seg_with_r2_top_training.db_point_feat_label': 2863660,
    'waymo_seg_with_r2_top_toy_training.point': 237,
    'waymo_seg_with_r2_top_toy_training.label': 237,
    'waymo_seg_with_r2_top_toy_training.instance': 237,
    'waymo_seg_with_r2_top_toy_training.box_label_attr': 237,
    'waymo_seg_with_r2_top_toy_training.top_lidar_origin': 237,
    'waymo_seg_with_r2_top_toy_training.db_point_feat_label': 28637,
    'waymo_seg_with_r2_top_toy_validation.point': 60,
    'waymo_seg_with_r2_top_toy_validation.label': 60,
    'waymo_seg_with_r2_top_toy_validation.instance': 60,
    'waymo_seg_with_r2_top_toy_validation.box_label_attr': 60,
}

class WaymoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, training=training,
            root_path=root_path, logger=logger
        )
        self.data_path = self.root_path
        self.data_tag = self.dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.repeat = self.dataset_cfg.REPEAT[self.mode]
        self._index_list = np.arange(self.dataset_cfg.TOTAL_NUM_SAMPLES)
        if self.dataset_cfg.SAMPLE_INTERVAL[self.mode] > 1:
            self.logger.info(f"Sample Interval: {self.dataset_cfg.SAMPLE_INTERVAL[self.mode]}")
            self._index_list = self._index_list[::self.dataset_cfg.SAMPLE_INTERVAL[self.mode]]
        if self.repeat > 1:
            self.logger.info(f"Repeating: {self.repeat} times")
            self._index_list = self._index_list[np.newaxis, :].repeat(self.repeat, axis=0).reshape(-1)

        # class translation
        num_all_seg_classes = self.dataset_cfg.NUM_ALL_SEG_CLASSES
        self.seg_cls_label_translation = np.zeros(num_all_seg_classes, dtype=np.int32) - 1
        for i, cls in enumerate(self.dataset_cfg.SEG_CLASSES):
            self.seg_cls_label_translation[cls] = i
        self.num_seg_class = len(self.dataset_cfg.SEG_CLASSES)
        self.logger.info(f"Number of Segmentation Class: {self.num_seg_class}")
        
        # shared memory allocation
        for data_type in self.dataset_cfg.SHARED_MEMORY_ALLOCATION[self.mode]:
            self._allocate_data(self.data_tag, self.split, data_type, self.root_path)
            self.logger.info(f"Allocated {self.data_template.format('*', data_type)} into Shared Memory")
        
        self.logger.info(f"Mode: {self.mode}, on {len(self._index_list)} LiDAR scenes.")
        #seg_label_translation = dataset_cfg.get('SEG_LABEL_TRANSLATION', None)
        #self.strategies = dataset_cfg.get("STRATEGIES", None)
        #if seg_label_translation is not None:
        #    self.seg_label_translation = np.array(seg_label_translation).astype(np.int32)
        #    self.num_seg_class = self.seg_label_translation.max()+1
        #else:
        #    self.seg_label_translation = None
        #self.evaluation_list = dataset_cfg.get('EVALUATION_LIST', [])

        #self.infos = []
        #self.include_waymo_data(self.mode)
        #self.with_seg = self.dataset_cfg.get('WITH_SEG', False)
        #if 'SEG_ONLY' in self.dataset_cfg: 
        #    seg_only = self.dataset_cfg.get('SEG_ONLY', {}).get(self.split, False)
        #    if seg_only:
        #        self.infos = [info for info in self.infos \
        #                      if info['annos']['seg_label_path'] is not None]
        #        self.logger.info(f"Mode: {self.mode}, on {len(self.infos)} LiDAR scenes with segmentation labels.")

        #self.use_shared_memory = self.dataset_cfg.get('USE_SHARED_MEMORY', False) and (self.mode == 'train')
        #if self.use_shared_memory:
        #    self.shared_memory_file_limit = self.dataset_cfg.get('SHARED_MEMORY_FILE_LIMIT', 0x7FFFFFFF)
        #    self.load_data_to_shared_memory()

    @property
    def data_template(self):
        return f'{self.data_tag}_{self.split}_{{}}.{{}}'

    def _allocate_data(self, data_tag, split, data_type, root_path):
        data_name = f'{data_tag}_{split}.{data_type}'
        num_samples = SIZE[data_name]
        # allocate data to shm:///
        path_template = f'{data_tag}_{split}_{{}}.{data_type}'
        #if len(glob.glob(f'/dev/shm/{data_tag}_{split}_*.{data_type}')) < num_samples:
        if not os.path.exists(f'/dev/shm/{data_tag}_{split}_{num_samples-1}.{data_type}'):
            filename = root_path / data_name
            data_list = joblib.load(filename)
            for idx, data in enumerate(data_list):
                if not os.path.exists('/dev/shm/'+path_template.format(idx)):
                    x = SA.create("shm://"+path_template.format(idx), data.shape, dtype=data.dtype)
                    x[...] = data[...]
                    x.flags.writeable = False
            del data_list
            gc.collect()

    def get_data(self, idx, dtype):
        data = SA.attach("shm://"+self.data_template.format(self._index_list[idx], dtype)).copy()
        return data

    def get_lidar(self, idx):
        points_all = self.get_data(idx, 'point').astype(np.float32) # [x, y, z, intensity, elongation, NLZ_flag]
        points_all = points_all[:, [3,4,5,0,1,2]]
        #points_all[:, 3] = np.tanh(points_all[:, 3])
        return points_all
    
    def get_box3d(self, idx):
        box_label_attr = self.get_data(idx, 'box_label_attr').astype(np.float32).reshape(-1, 8) # [N, 8]
        box_label = box_label_attr[:, 0].round().astype(np.int32)
        box_attr = box_label_attr[:, 1:] # [x, y, z, l, h, w, heading]
        return box_attr, box_label

    def get_seg_cls_label(self, idx):
        seg_cls_labels = self.get_data(idx, 'label').astype(np.int64)

        if self.seg_cls_label_translation is not None:
            valid_mask = seg_cls_labels >= 0
            seg_cls_labels[valid_mask] = self.seg_cls_label_translation[seg_cls_labels[valid_mask]]

        return seg_cls_labels
    
    def get_seg_inst_label(self, idx):
        seg_inst_labels = self.get_data(idx, 'instance').astype(np.int64) + 1

        return seg_inst_labels

    def get_top_lidar_origin(self, idx):
        top_lidar_origin = self.get_data(idx, 'top_lidar_origin').astype(np.float32)
        return top_lidar_origin

    def __len__(self):
        return len(self._index_list)

    def __getitem__(self, index):
        seg_cls_labels = self.get_seg_cls_label(index)
        seg_inst_labels = self.get_seg_inst_label(index)
        points = self.get_lidar(index)
        box_attr, box_cls_label = self.get_box3d(index)
        top_lidar_origin = self.get_top_lidar_origin(index)

        input_dict = dict(
            point_wise=dict(
                points=points,
                seg_cls_labels=seg_cls_labels,
                seg_inst_labels=seg_inst_labels,
            ),
            object_wise=dict(
                gt_box_attr=box_attr,
                gt_box_cls_label=box_cls_label,
                num_points_in_box=np.zeros_like(box_cls_label),
            ),
            scene_wise=dict(
                top_lidar_origin=top_lidar_origin
            )
        )

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, box_class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(cur_dict):
            if 'pred_scores' in cur_dict:
                pred_scores = cur_dict['pred_scores'].cpu().numpy()
                pred_boxes = cur_dict['pred_boxes'].cpu().numpy()
                pred_labels = cur_dict['pred_labels'].cpu().numpy()

                pred_dict = get_template_prediction(pred_scores.shape[0])
                if pred_scores.shape[0] > 0:
                    pred_dict['score'] = pred_scores
                    pred_dict['boxes_lidar'] = pred_boxes
                    pred_dict['name'] = np.array(box_class_names)[pred_labels - 1]
            else:
                pred_dict = {}

            if 'ups' in cur_dict:
                pred_dict['ups'] = cur_dict['ups'].detach().cpu()
                pred_dict['downs'] = cur_dict['downs'].detach().cpu()

            return pred_dict

        annos = []
        for index, cur_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(cur_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, box_class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            map_name_to_kitti = {
                'Vehicle': 'Car',
                'Pedestrian': 'Pedestrian',
                'Cyclist': 'Cyclist',
                'Sign': 'Sign',
                'Car': 'Car'
            }
            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        def waymo_eval(eval_det_annos, eval_gt_annos):
            from .waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=class_names,
                distance_thresh=1000,
                fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]

        if 'box' in self.evaluation_list:
            if kwargs['eval_metric'] == 'kitti':
                ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)
            elif kwargs['eval_metric'] == 'waymo':
                ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
            else:
                raise NotImplementedError

            return ap_result_str, ap_dict
        elif 'seg' in self.evaluation_list:
            total_ups, total_downs = None, None
            for eval_gt_anno, eval_det_anno in zip(eval_det_annos, eval_gt_annos):
                ups, downs = eval_gt_anno['ups'], eval_gt_anno['downs']
                if total_ups is None:
                    total_ups = ups.clone()
                    total_downs = downs.clone()
                else:
                    total_ups += ups
                    total_downs += downs
            seg_result_str = '\n'
            iou_dict = {}
            for cls in range(total_ups.shape[0]):
                print(cls, total_ups[cls], total_downs[cls])
                iou = total_ups[cls]/np.clip(total_downs[cls], 1, None)
                seg_result_str += f'IoU for class {cls} {iou:.4f} \n'
                iou_dict[f'{cls}'] = iou
            return seg_result_str, iou_dict
        else:
            raise NotImplementedError

    def create_groundtruth_seg_database(self, info_path, save_path, used_classes=None, split='train', sampled_interval=10,
                                    processed_data_tag=None, seg_only=False, top_lidar_only=False):
        if seg_only:
            suffix = '_partial_db'
        else:
            suffix = ''
        database_save_path = save_path / ('%s_gt_seg_database_%s_sampled_%d%s' % (processed_data_tag, split, sampled_interval, suffix))
        db_info_save_path = save_path / ('%s_waymo_seg_dbinfos_%s_sampled_%d%s.pkl' % (processed_data_tag, split, sampled_interval, suffix))
        db_data_save_path = save_path / ('%s_gt_seg_database_%s_sampled_%d%s_global.npy' % (processed_data_tag, split, sampled_interval, suffix))
        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        if seg_only:
            infos = [info for info in infos if info['annos']['seg_label_path'] is not None]
        
        #cfg_from_yaml_file('tools/cfgs/visualizers/seg_db_visualizer.yaml', cfg)
        #vis = PolyScopeVisualizer(cfg['VISUALIZER'])

        # process foreground classes
        
        from functools import partial
        from . import waymo_utils

        process_frame = partial(
            waymo_utils.extract_foreground_pointcloud, self,
            top_lidar_only,
            database_save_path,
        )
        foreground_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15]
        instance_dict = {i: [] for i in foreground_class}
        for info in tqdm(infos):
            frame_id = info['frame_id']
            db_info_save_path_i = database_save_path / f'{frame_id}.pkl'
            mod_info = process_frame(info, db_info_save_path_i)
            for i in foreground_class:
                instance_dict[i] += mod_info[i]

        #with multiprocessing.Pool(16) as p:
        #    modified_infos = list(tqdm(p.imap(process_frame, infos),
        #                               total=len(infos)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(instance_dict, f)

        ## it will be used if you choose to use shared memory for gt sampling
        #stacked_gt_points = np.concatenate(stacked_gt_points, axis=0)
        #np.save(db_data_save_path, stacked_gt_points)

    def create_groundtruth_database(self, info_path, save_path, used_classes=None, split='train', sampled_interval=10,
                                    processed_data_tag=None, seg_only=False):
        if seg_only:
            suffix = '_partial_db'
        else:
            suffix = ''
        database_save_path = save_path / ('%s_gt_database_%s_sampled_%d%s' % (processed_data_tag, split, sampled_interval, suffix))
        db_info_save_path = save_path / ('%s_waymo_dbinfos_%s_sampled_%d%s.pkl' % (processed_data_tag, split, sampled_interval, suffix))
        db_data_save_path = save_path / ('%s_gt_database_%s_sampled_%d%s_global.npy' % (processed_data_tag, split, sampled_interval, suffix))
        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        point_offset_cnt = 0
        stacked_gt_points = []
        for k in range(0, len(infos), sampled_interval):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]

            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']
            points = self.get_lidar(sequence_name, sample_idx)

            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            gt_boxes = annos['gt_boxes_lidar']

            if k % 4 != 0 and len(names) > 0:
                mask = (names == 'Vehicle')
                names = names[~mask]
                difficulty = difficulty[~mask]
                gt_boxes = gt_boxes[~mask]

            if k % 2 != 0 and len(names) > 0:
                mask = (names == 'Pedestrian')
                names = names[~mask]
                difficulty = difficulty[~mask]
                gt_boxes = gt_boxes[~mask]

            num_obj = gt_boxes.shape[0]
            if num_obj == 0:
                continue

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            for i in range(num_obj):
                filename = '%s_%04d_%s_%d.bin' % (sequence_name, sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]
                gt_points[:, :3] -= gt_boxes[i, :3]

                if (used_classes is None) or names[i] in used_classes:
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'sequence_name': sequence_name,
                               'sample_idx': sample_idx, 'gt_idx': i, 'box3d_lidar': gt_boxes[i],
                               'num_points_in_gt': gt_points.shape[0], 'difficulty': difficulty[i]}

                    # it will be used if you choose to use shared memory for gt sampling
                    stacked_gt_points.append(gt_points)
                    db_info['global_data_offset'] = [point_offset_cnt, point_offset_cnt + gt_points.shape[0]]
                    point_offset_cnt += gt_points.shape[0]

                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

        # it will be used if you choose to use shared memory for gt sampling
        stacked_gt_points = np.concatenate(stacked_gt_points, axis=0)
        np.save(db_data_save_path, stacked_gt_points)

    def parse_segments(self, tag, max_num_points=100000):
        from torch_scatter import scatter
        infos = [info for info in self.infos if info['annos']['seg_label_path'] is not None]
        info_groups = {}
        for idx, info in enumerate(infos):
            seg_label_path = info['annos']['seg_label_path']
            sequence_uid = info['point_cloud']['lidar_sequence']
            group = info_groups.get(sequence_uid, None)
            if group is None:
                info_groups[sequence_uid] = [idx]
            else:
                info_groups[sequence_uid].append(idx)

        support_dict = {}
        for seq_uid, indices in tqdm(info_groups.items()):
            seq_infos = [infos[idx] for idx in indices]
            poses = [info['pose'] for info in seq_infos]
            points, seg_labels = [], []
            for info in seq_infos:
                seg_label = self.get_seg_label(seq_uid, info['point_cloud']['sample_idx'])
                pose = info['pose'].astype(np.float64)
                point = self.get_lidar(seq_uid, info['point_cloud']['sample_idx'])[:seg_label.shape[0], :3]
                mask = seg_label[:, 1] >= 17
                point, seg_label = point[mask], seg_label[mask]
                point = point.astype(np.float64) @ pose[:3, :3].T + pose[:3, 3]
                
                points.append(point)
                seg_labels.append(seg_label)
            points_all = np.concatenate(points, axis=0)
            seg_labels_all = np.concatenate(seg_labels, axis=0)

            voxel_size = [0.2, 0.2]
            pc_range = points_all[:, :2].min(0)
            grid_size = np.floor((points_all[:, :2].max(0) - pc_range) // voxel_size).astype(np.int32)+1

            coors = np.floor((points_all[:, :2] - pc_range) // voxel_size).astype(np.int32)
            coor1d = coors[:, 1] * grid_size[0] + coors[:, 0]
            voxel_min_z = scatter(torch.tensor(points_all[:, 2]), torch.tensor(coor1d).long(),
                                  reduce='min', dim=0, dim_size=grid_size[0]*grid_size[1]).numpy()
            
            valid_mask = voxel_min_z[coor1d] + 0.3 > points_all[:, 2]
            points_all = points_all[valid_mask]
            seg_labels_all = seg_labels_all[valid_mask]
            coor1d = coor1d[valid_mask]

            if points_all.shape[0] > max_num_points:
                indices = np.random.choice(np.arange(points_all.shape[0]), max_num_points)
                points_all = points_all[indices]
                seg_labels_all = seg_labels_all[indices]

            road_segment_mask = (seg_labels_all[:, 1] >= 18) & (seg_labels_all[:, 1] <= 19)
            road_points = points_all[road_segment_mask]
            walkable_points = points_all[road_segment_mask == False]
            support_dict[seq_uid] = dict(
                road=road_points.astype(np.float32),
                walkable=walkable_points.astype(np.float32),
            )
        save_path = self.root_path / f'{tag}_aug_support.pkl'
        with open(save_path, 'wb') as fout:
            pickle.dump(support_dict, fout)
        print(f'saving support to {save_path}')


    def compute_interaction_index(self, radius_list,
                                  num_workers=multiprocessing.cpu_count()):
        from functools import partial
        from . import waymo_utils

        process_frame = partial(
            waymo_utils.compute_interaction_index_for_frame, self,
            radius_list=radius_list,
        )

        with multiprocessing.Pool(num_workers) as p:
            modified_infos = list(tqdm(p.imap(process_frame, self.infos),
                                       total=len(self.infos)))

        return modified_infos


def create_waymo_infos(dataset_cfg, box_class_names, data_path, save_path,
                       raw_data_tag='raw_data', processed_data_tag='waymo_processed_data',
                       workers=min(16, multiprocessing.cpu_count()),
                       seg_only=False, top_lidar_only=False):
    dataset = WaymoDataset(
        dataset_cfg=dataset_cfg, box_class_names=box_class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, train_split))
    val_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, val_split))

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    #waymo_infos_train = dataset.get_infos(
    #    raw_data_path=data_path / raw_data_tag,
    #    save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
    #    sampled_interval=1, seg_only=seg_only
    #)
    #if seg_only:
    #    waymo_infos_train = [info for info in waymo_infos_train \
    #                         if info['annos']['seg_label_path'] is not None]
    #with open(train_filename, 'wb') as f:
    #    pickle.dump(waymo_infos_train, f)
    print('----------------Waymo info train file is saved to %s----------------' % train_filename)

    dataset.set_split(val_split)
    #waymo_infos_val = dataset.get_infos(
    #    raw_data_path=data_path / raw_data_tag,
    #    save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
    #    sampled_interval=1
    #)
    #with open(val_filename, 'wb') as f:
    #    pickle.dump(waymo_infos_val, f)
    print('----------------Waymo info val file is saved to %s----------------' % val_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    dataset.set_split(train_split)
    #dataset.create_groundtruth_database(
    #    info_path=train_filename, save_path=save_path, split='train', sampled_interval=1,
    #    used_classes=['Vehicle', 'Pedestrian', 'Cyclist'], processed_data_tag=processed_data_tag,
    #    seg_only=seg_only
    #)
    dataset.create_groundtruth_seg_database(
        info_path=train_filename, save_path=save_path, split='train', sampled_interval=1,
        used_classes=['Vehicle', 'Pedestrian', 'Cyclist'], processed_data_tag=processed_data_tag,
        seg_only=seg_only, top_lidar_only=top_lidar_only,
    )
    print('---------------Data preparation Done---------------')


def parse_walkable_and_road_segments(dataset_cfg, box_class_names, data_path,
                                     save_path, raw_data_tag='raw_data',
                                     processed_data_tag='waymo_processed_data',
                                     workers=min(16, multiprocessing.cpu_count()),
                                     seg_only=False):
    dataset = WaymoDataset(
        dataset_cfg=dataset_cfg, box_class_names=box_class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )

    train_split = 'train'

    train_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, train_split))
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print('---------------Start to extract semantic segments---------------')
    
    dataset.set_split(train_split)
    waymo_infos_train = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1, seg_only=seg_only
    )

    dataset.parse_segments(tag=processed_data_tag)

    
def compute_interaction_index(dataset_cfg, box_class_names, data_path,
                              save_path, raw_data_tag='raw_data',
                              processed_data_tag='waymo_processed_data',
                              workers=min(16, multiprocessing.cpu_count()),
                              seg_only=False):

    dataset = WaymoDataset(
        dataset_cfg=dataset_cfg, box_class_names=box_class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )

    val_split = 'val'

    val_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, val_split))
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print('---------------Start to compute interaction index---------------')
    
    dataset.set_split(val_split)
    dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1, seg_only=seg_only
    )
    if seg_only:
        dataset.infos = [info for info in dataset.infos \
                         if info['annos']['seg_label_path'] is not None]

    modified_infos = dataset.compute_interaction_index(
                         radius_list=[0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
                                      4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
                         num_workers=workers
                     )
    val_ii_filename = save_path / ('%s_infos_%s_with_ii.pkl' % (processed_data_tag, val_split))

    with open(val_ii_filename, 'wb') as fout:
        pickle.dump(modified_infos, fout)
    print(f'saving to {val_ii_filename}')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_waymo_infos', help='')
    parser.add_argument('--processed_data_tag', type=str, default='waymo_processed_data_v0_5_0', help='')
    parser.add_argument('--seg_only', action='store_true', help='only parse seg data')
    parser.add_argument('--top_lidar_only', action='store_true', help='only use top lidar point cloud')
    parser.add_argument('--num_workers', type=int, help='number of parallel workers', default=16)
    args = parser.parse_args()
    print(multiprocessing.cpu_count())
    args.num_workers=min(args.num_workers, multiprocessing.cpu_count())

    if args.func == 'create_waymo_infos':
        import yaml
        from easydict import EasyDict
        try:
            yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
        except:
            yaml_config = yaml.safe_load(open(args.cfg_file))
        dataset_cfg = EasyDict(yaml_config)
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg.PROCESSED_DATA_TAG = args.processed_data_tag
        create_waymo_infos(
            dataset_cfg=dataset_cfg,
            box_class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'waymo',
            save_path=ROOT_DIR / 'data' / 'waymo',
            raw_data_tag='raw_data',
            processed_data_tag=args.processed_data_tag,
            workers=args.num_workers,
            seg_only=args.seg_only,
            top_lidar_only=args.top_lidar_only,
        )
    
    if args.func == 'parse_walkable_and_road_segments':
        import yaml
        from easydict import EasyDict
        try:
            yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
        except:
            yaml_config = yaml.safe_load(open(args.cfg_file))
        dataset_cfg = EasyDict(yaml_config)
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg.PROCESSED_DATA_TAG = args.processed_data_tag
        parse_walkable_and_road_segments(
            dataset_cfg=dataset_cfg,
            box_class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'waymo',
            save_path=ROOT_DIR / 'data' / 'waymo',
            raw_data_tag='raw_data',
            processed_data_tag=args.processed_data_tag,
            seg_only=args.seg_only
        )
    
    if args.func == 'compute_interaction_index':
        import yaml
        from easydict import EasyDict
        try:
            yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
        except:
            yaml_config = yaml.safe_load(open(args.cfg_file))
        dataset_cfg = EasyDict(yaml_config)
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg.PROCESSED_DATA_TAG = args.processed_data_tag
        compute_interaction_index(
            dataset_cfg=dataset_cfg,
            box_class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'waymo',
            save_path=ROOT_DIR / 'data' / 'waymo',
            raw_data_tag='raw_data',
            processed_data_tag=args.processed_data_tag,
            seg_only=args.seg_only
        )
