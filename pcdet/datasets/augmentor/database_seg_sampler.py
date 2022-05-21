import pickle

import os
import copy
import numpy as np
import torch
import SharedArray
import torch.distributed as dist

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils, common_utils
#from ...models.visualizers import PolyScopeVisualizer
from ...config import cfg_from_yaml_file, cfg
from ...ops.roiaware_pool3d.roiaware_pool3d_utils import (
    points_in_boxes_cpu
)

class SegDataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, aug_classes=None, logger=None):
        self.root_path = root_path
        self.sampler_cfg = sampler_cfg
        self.logger = logger
        self.db_infos = {}
        self.aug_classes = aug_classes
        self.use_shared_memory = False
        
        for db_info_path in sampler_cfg.DB_INFO_PATH:
            db_info_path = self.root_path.resolve() / db_info_path
            with open(str(db_info_path), 'rb') as f:
                self.db_infos = pickle.load(f)
                #[self.db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]

        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)
        
        self.gt_database_data_key = self.load_db_to_shared_memory() if self.use_shared_memory else None

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)

        self.sample_class_num = sampler_cfg.SAMPLE_GROUPS
        for sample_group in sampler_cfg.SAMPLE_GROUPS:
            cls = sample_group['cls']
            sample_num = sample_group['num']
            self.sample_groups[cls] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[cls]),
                'indices': np.arange(len(self.db_infos[cls])),
                'num_trial': sample_group['num_trial'],
                'scene_limit': sample_group['scene_limit']
            }

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __del__(self):
        if self.use_shared_memory:
            self.logger.info('Deleting GT database from shared memory')
            cur_rank, num_gpus = common_utils.get_dist_info()
            sa_key = self.sampler_cfg.DB_DATA_PATH[0]
            if cur_rank % num_gpus == 0 and os.path.exists(f"/dev/shm/{sa_key}"):
                SharedArray.delete(f"shm://{sa_key}")

            if num_gpus > 1:
                dist.barrier()
            self.logger.info('GT database has been removed from shared memory')

    def load_db_to_shared_memory(self):
        self.logger.info('Loading GT database to shared memory')
        cur_rank, world_size, num_gpus = common_utils.get_dist_info(return_gpu_per_machine=True)

        assert self.sampler_cfg.DB_DATA_PATH.__len__() == 1, 'Current only support single DB_DATA'
        db_data_path = self.root_path.resolve() / self.sampler_cfg.DB_DATA_PATH[0]
        sa_key = self.sampler_cfg.DB_DATA_PATH[0]

        if cur_rank % num_gpus == 0 and not os.path.exists(f"/dev/shm/{sa_key}"):
            gt_database_data = np.load(db_data_path)
            common_utils.sa_create(f"shm://{sa_key}", gt_database_data)
            
        if num_gpus > 1:
            dist.barrier()
        self.logger.info('GT database has been saved to shared memory')
        return sa_key

    def filter_by_min_points(self, db_infos, min_num_points):
        for key in db_infos.keys():
            filtered_infos = []
            for info in db_infos[key]:
                if info['num_points'] >= min_num_points:
                    filtered_infos.append(info)
            if self.logger is not None:
                self.logger.info('Database filter by min points class %s: %d => %d' %
                                 (key, len(db_infos[key]), len(filtered_infos)))
            db_infos[key] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = sample_group['num_trial'], sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib:

        Returns:
        """
        a, b, c, d = road_planes
        center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3])
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict):
        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        points = data_dict['points']
        seg_inst_labels = data_dict.get('seg_inst_labels', None)
        seg_cls_labels = data_dict.get('seg_cls_labels', None)
        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )
            data_dict.pop('calib')
            data_dict.pop('road_plane')

        obj_points_list = []
        if self.use_shared_memory:
            gt_database_data = SharedArray.attach(f"shm://{self.gt_database_data_key}")
            gt_database_data.setflags(write=0)
        else:
            gt_database_data = None 
        
        if seg_inst_labels is not None:
            obj_seg_labels_list = []
            max_instance_label = seg_inst_labels.max()
        for idx, info in enumerate(total_valid_sampled_dict):
            if self.use_shared_memory:
                start_offset, end_offset = info['global_data_offset']
                obj_points = copy.deepcopy(gt_database_data[start_offset:end_offset])
            else:
                file_path = self.root_path / info['path']
                obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                    [-1, self.sampler_cfg.NUM_POINT_FEATURES])

            obj_points[:, :3] += info['box3d_lidar'][:3]
            if seg_cls_labels is not None:
                obj_cls_labels = np.full(obj_points.shape[0],
                                         self.seg_label_map[info['name']],
                                         dtype=seg_cls_labels.dtype)
                obj_instance_labels = np.full(obj_points.shape[0],
                                              max_instance_label+1,
                                              dtype=seg_inst_labels.dtype)
                max_instance_label += 1
                obj_seg_labels = np.stack([obj_instance_labels, obj_cls_labels], axis=-1)
                obj_seg_labels_list.append(obj_seg_labels)

            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            obj_points_list.append(obj_points)

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        if seg_inst_labels is not None:
            points, seg_cls_labels, seg_inst_labels = \
                    box_utils.remove_points_in_boxes3d(
                        points, large_sampled_gt_boxes,
                        seg_cls_labels, seg_inst_labels)
            obj_seg_labels = np.concatenate(obj_seg_labels_list, axis=0)
            obj_seg_cls_labels = obj_seg_labels[:, 1]
            obj_seg_inst_labels = obj_seg_labels[:, 0]
            seg_inst_labels = np.concatenate([obj_seg_inst_labels, seg_inst_labels], axis=0)
            seg_cls_labels = np.concatenate([obj_seg_cls_labels, seg_cls_labels], axis=0)
            data_dict['seg_inst_labels'] = seg_inst_labels
            data_dict['seg_cls_labels'] = seg_cls_labels
        else:
            points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)

        points = np.concatenate([obj_points, points], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points
        return data_dict

    def sample_candidate_locations(self, points, support_indices, support_classes):
        """
        Args:
            points [N, 3+C]
            seg_cls_labels [N]
            support_class [M]
        Returns:
            valid [M1]
            locations [M1, 3]
        """
        valid = []
        locations = []
        for i, support_class in enumerate(support_classes):
            indices = support_indices[support_class]
            if indices.shape[0] == 0:
                continue
            index = indices[np.random.randint(0, indices.shape[0])]
            locations.append(points[index, :3])
            valid.append(i)
        return valid, np.array(locations)

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        points = data_dict['points']
        seg_inst_labels = data_dict['seg_inst_labels']
        seg_cls_labels = data_dict['seg_cls_labels']
        max_inst_label = seg_inst_labels.max()

        foreground_mask = np.zeros_like(seg_cls_labels).astype(bool)
        for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
            foreground_mask = foreground_mask | (seg_cls_labels == i) 
        foreground_points = points[np.where(foreground_mask)[0]]
        existed_boxes = np.zeros((0, 7))

        # for faster computation
        support_indices = {}
        for support_class in [17, 20, 21]:
            support_indices[support_class] = np.where(seg_cls_labels == support_class)[0]
            
        for fg_cls, sample_group in self.sample_groups.items():
            #print(f'augmenting class {fg_cls}')
            aug_point_list = []
            aug_seg_cls_label_list = []
            aug_seg_inst_label_list = []
            aug_box_list = []
            
            if sample_group['scene_limit'] > 0:
                num_instance = np.unique(seg_inst_labels[seg_cls_labels == fg_cls]).shape[0]
                sample_group['sample_num'] = sample_group['scene_limit'] - num_instance

            if sample_group['sample_num'] > 0:
                #print(f"\tsample num = {sample_group['sample_num']}, num_trial={sample_group['num_trial']}")
                sampled_dict = self.sample_with_fixed_number(fg_cls, sample_group)

                # sample locations
                support_classes = [d['support'] for d in sampled_dict]
                valid, candidate_locations = self.sample_candidate_locations(points, support_indices, support_classes)
                if len(valid) == 0:
                    continue
                sampled_dict = [sampled_dict[i] for i in valid]
                for sampled_d, loc in zip(sampled_dict, candidate_locations):
                    path = sampled_d['path']
                    trans_z = sampled_d['trans_z']
                    if trans_z > 0:
                        trans_z = 0.0
                    aug_points = np.load(path)
                    low = aug_points[aug_points[:, 2].argmin()]
                    trans = loc - low[:3]
                    trans[2] -= trans_z
                    aug_points[:, :3] += trans
                    # estimate or reuse bounding boxes
                    if sampled_d['box3d'] is not None:
                        box = sampled_d['box3d']
                        box[:3] += trans
                        aug_box_list.append(box)
                    else:
                        box = np.zeros(7)
                        box[:3] = (aug_points.max(0)[:3] + aug_points.min(0)[:3])/2
                        box[3:6] = (aug_points.max(0) - aug_points.min(0))[:3] + 0.05
                        aug_box_list.append(box)
                    # low + trans = loc - trans_z
                    if sampled_d['grouping'] is not None:
                        grouping = sampled_d['grouping']
                        cls = grouping['cls']
                        offsets = grouping['offsets']
                        sizes = grouping['sizes']
                        aug_seg_cls_labels = torch.zeros(sum(sizes)).long()
                        aug_seg_inst_labels = torch.zeros(sum(sizes)).long()
                        for c, o, s in zip(cls, offsets, sizes):
                            aug_seg_cls_labels[o:(o+s)] = c
                            aug_seg_inst_labels[o:(o+s)] = max_inst_label + 1
                            max_inst_label += 1
                    else:
                        aug_seg_cls_labels = torch.zeros(aug_points.shape[0]).long() + fg_cls
                        aug_seg_inst_labels = torch.zeros(aug_points.shape[0]).long() + max_inst_label + 1
                        max_inst_label += 1
                    aug_point_list.append(aug_points)
                    aug_seg_cls_label_list.append(aug_seg_cls_labels)
                    aug_seg_inst_label_list.append(aug_seg_inst_labels)
                # estimate bounding boxes
                aug_boxes = torch.from_numpy(np.stack(aug_box_list, axis=0)).view(-1, 7).float().numpy()

                # reject by collision
                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(aug_boxes[:, 0:7], existed_boxes[:, 0:7]).astype(np.float32)
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(aug_boxes[:, 0:7], aug_boxes[:, 0:7])
                iou2[range(aug_boxes.shape[0]), range(aug_boxes.shape[0])] = 0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2
                box_valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0)

                point_mask = points_in_boxes_cpu(foreground_points[::5, :3], aug_boxes).any(-1) # [num_boxes]
                box_valid_mask = (box_valid_mask & (point_mask == False)).nonzero()[0]

                aug_boxes = aug_boxes[box_valid_mask]
                aug_point_list = [aug_point_list[i] for i in box_valid_mask]
                aug_seg_cls_label_list = [aug_seg_cls_label_list[i] for i in box_valid_mask]
                aug_seg_inst_label_list = [aug_seg_inst_label_list[i] for i in box_valid_mask]
                
                if len(aug_point_list) > 0:
                    if len(aug_point_list) > sample_group['sample_num']:
                        aug_point_list = aug_point_list[:sample_group['sample_num']]
                        aug_seg_cls_label_list = aug_seg_cls_label_list[:sample_group['sample_num']]
                        aug_seg_inst_label_list = aug_seg_inst_label_list[:sample_group['sample_num']]
                        aug_boxes = aug_boxes[:sample_group['sample_num']]
                    aug_points = torch.from_numpy(np.concatenate(aug_point_list, axis=0))
                    aug_seg_cls_labels = torch.from_numpy(np.concatenate(aug_seg_cls_label_list, axis=0))
                    aug_seg_inst_labels = torch.from_numpy(np.concatenate(aug_seg_inst_label_list, axis=0))
                    # update
                    foreground_points = np.concatenate([foreground_points, aug_points], axis=0)
                    points = np.concatenate([points, aug_points], axis=0)
                    seg_cls_labels = np.concatenate([seg_cls_labels, aug_seg_cls_labels], axis=0)
                    seg_inst_labels = np.concatenate([seg_inst_labels, aug_seg_inst_labels], axis=0)
                    existed_boxes = np.concatenate([existed_boxes, aug_boxes], axis=0)
        
        data_dict['points'] = points
        data_dict['seg_inst_labels'] = seg_inst_labels
        data_dict['seg_cls_labels'] = seg_cls_labels
        
        return data_dict
