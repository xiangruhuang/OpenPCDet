from functools import partial

import numpy as np
from skimage import transform

from ...utils import box_utils, common_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxel_points'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self._voxel_generator = None

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        points = data_dict['point_wise']['points']
        mask = common_utils.mask_points_by_range(points, self.point_cloud_range)
        data_dict['point_wise'] = common_utils.filter_dict(data_dict['point_wise'], mask)

        if config.REMOVE_OUTSIDE_BOXES and self.training:
            gt_boxes = data_dict['object_wise']['gt_box_attr']
            mask = box_utils.mask_boxes_outside_range_numpy(
                gt_boxes, self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['object_wise'] = common_utils.filter_dict(data_dict['object_wise'], mask)
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['point_wise']['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            data_dict['point_wise'] = common_utils.filter_dict(
                                          data_dict['point_wise'],
                                          shuffle_idx
                                      )

        return data_dict
    
    def limit_num_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.limit_num_points, config=config)

        max_num_points = config["MAX_NUM_POINTS"]

        points = data_dict['point_wise']['points']
        if points.shape[0] > max_num_points:
            shuffle_idx = np.random.permutation(points.shape[0])[:max_num_points]
            data_dict['point_wise'] = common_utils.filter_dict(
                                          data_dict['point_wise'],
                                          shuffle_idx
                                      )

        return data_dict

    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):
        # just calculate grid size
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)
        
        return data_dict
        
    def transform_points_to_voxels(self, data_dict=None, config=None):
        point_cloud_range = np.array(config.POINT_CLOUD_RANGE)
        if data_dict is None:
            grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        pointvecs = []
        num_points = data_dict['point_wise']['points'].shape[0]
        for key in data_dict['point_wise'].keys():
            pointvecs.append(data_dict['point_wise'][key])
        pointvecs = [p.reshape(num_points, -1) for p in pointvecs]
        dims = [p.shape[-1] for p in pointvecs]
        dtypes = [p.dtype for p in pointvecs]
        num_point_features = sum(dims)

        if self._voxel_generator is None:
            self._voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=point_cloud_range,
                num_point_features=num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        pointvecs = np.concatenate(pointvecs, axis=-1)

        point_feat_dim = pointvecs.shape[1]
        voxel_output = self._voxel_generator.generate(pointvecs)
        voxels, coordinates, num_points_in_voxel = voxel_output
        voxel_wise = dict(
            voxel_coords = coordinates,
            voxel_num_points = num_points_in_voxel
        )
        offset = 0
        for key, dim, dtype in zip(data_dict['point_wise'].keys(), dims, dtypes):
            voxel_wise['voxel_'+key] = voxels[..., offset:offset+dim].astype(dtype)
            offset += dim
            
        data_dict['voxel_wise'] = voxel_wise

        return data_dict

    #def sample_points(self, data_dict=None, config=None):
    #    if data_dict is None:
    #        return partial(self.sample_points, config=config)

    #    num_points = config.NUM_POINTS[self.mode]
    #    if num_points == -1:
    #        return data_dict

    #    points = data_dict['point_wise']['points']
    #    if num_points < len(points):
    #        pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
    #        pts_near_flag = pts_depth < 40.0
    #        far_idxs_choice = np.where(pts_near_flag == 0)[0]
    #        near_idxs = np.where(pts_near_flag == 1)[0]
    #        choice = []
    #        if num_points > len(far_idxs_choice):
    #            near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
    #            choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
    #                if len(far_idxs_choice) > 0 else near_idxs_choice
    #        else: 
    #            choice = np.arange(0, len(points), dtype=np.int32)
    #            choice = np.random.choice(choice, num_points, replace=False)
    #        np.random.shuffle(choice)
    #    else:
    #        choice = np.arange(0, len(points), dtype=np.int32)
    #        if num_points > len(points):
    #            extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
    #            choice = np.concatenate((choice, extra_choice), axis=0)
    #        np.random.shuffle(choice)
    #    data_dict['point_wise']['points'] = points[choice]
    #    return data_dict

    #def calculate_grid_size(self, data_dict=None, config=None):
    #    if data_dict is None:
    #        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
    #        self.grid_size = np.round(grid_size).astype(np.int64)
    #        self.voxel_size = config.VOXEL_SIZE
    #        return partial(self.calculate_grid_size, config=config)
    #    return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict

    def propagate_box_label_to_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.propagate_box_label_to_points, config=config)
        points = data_dict['points'][:, :3]
        seg_label_map = config['SEG_LABEL_MAP']
        labels = np.array([seg_label_map[n] for n in data_dict['gt_names']]).astype(np.int64)
        boxes = np.copy(data_dict['gt_boxes'])
        boxes = boxes[:, :7]
        boxes[:, 3:6] *= 0.95
        seg_inst_labels = data_dict['seg_inst_labels']
        inst_labels = seg_inst_labels.max() + 1 + np.arange(boxes.shape[0])
        seg_cls_labels = data_dict['seg_cls_labels']

        mask = roiaware_pool3d_utils.points_in_boxes_cpu(points, boxes)
        in_box_points = mask.any(0)
        if in_box_points.any():
            box_indices = mask[:, in_box_points].argmax(0)
            seg_cls_labels[in_box_points] = labels[box_indices]
            data_dict['seg_cls_labels'] = seg_cls_labels
        
            seg_inst_labels[in_box_points] = inst_labels[box_indices]
            data_dict['seg_inst_labels'] = seg_inst_labels

        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
