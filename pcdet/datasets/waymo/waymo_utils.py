# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.


import os
import pickle
import numpy as np
import torch
from ...utils import common_utils
import tensorflow as tf
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import (
    points_in_boxes_cpu
)

try:
    tf.enable_eager_execution()
except:
    pass

WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']

def convert_range_image_to_point_cloud_labels(frame,
        range_images,
        segmentation_labels,
        ri_index=0):
    """Convert segmentation labels from range images to point clouds.

    Args:
      frame: open dataset frame
      range_images: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
      segmentation_labels: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
      ri_index: 0 for the first return, 1 for the second return.

    Returns:
      point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
        points that are not labeled.
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:
      range_image = range_images[c.name][ri_index]
      range_image_tensor = tf.reshape(
          tf.convert_to_tensor(range_image.data), range_image.shape.dims)
      range_image_mask = range_image_tensor[..., 0] > 0

      if c.name in segmentation_labels:
        assert c.name == dataset_pb2.LaserName.TOP
        sl = segmentation_labels[c.name][ri_index]
        sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
        sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
        point_labels.append(sl_points_tensor.numpy())

    return point_labels

def generate_labels(frame):
    obj_name, difficulty, dimensions, locations, heading_angles = [], [], [], [], []
    tracking_difficulty, speeds, accelerations, obj_ids = [], [], [], []
    num_points_in_gt = []
    laser_labels = frame.laser_labels

    for i in range(len(laser_labels)):
        box = laser_labels[i].box
        class_ind = laser_labels[i].type
        loc = [box.center_x, box.center_y, box.center_z]
        heading_angles.append(box.heading)
        obj_name.append(WAYMO_CLASSES[class_ind])
        difficulty.append(laser_labels[i].detection_difficulty_level)
        tracking_difficulty.append(laser_labels[i].tracking_difficulty_level)
        dimensions.append([box.length, box.width, box.height])  # lwh in unified coordinate of OpenPCDet
        locations.append(loc)
        obj_ids.append(laser_labels[i].id)
        num_points_in_gt.append(laser_labels[i].num_lidar_points_in_box)


    annotations = {}
    annotations['name'] = np.array(obj_name)
    annotations['difficulty'] = np.array(difficulty)
    annotations['dimensions'] = np.array(dimensions)
    annotations['location'] = np.array(locations)
    annotations['heading_angles'] = np.array(heading_angles)

    annotations['obj_ids'] = np.array(obj_ids)
    annotations['tracking_difficulty'] = np.array(tracking_difficulty)
    annotations['num_points_in_gt'] = np.array(num_points_in_gt)

    annotations = common_utils.drop_info_with_name(annotations, name='unknown')
    if annotations['name'].__len__() > 0:
        gt_boxes_lidar = np.concatenate([
            annotations['location'], annotations['dimensions'], annotations['heading_angles'][..., np.newaxis]],
            axis=1
        )
    else:
        gt_boxes_lidar = np.zeros((0, 7))
    annotations['gt_boxes_lidar'] = gt_boxes_lidar
    return annotations


def convert_range_image_to_point_cloud(frame,
                                       range_images,
                                       camera_projections,
                                       range_image_top_pose,
                                       ri_index=0,
                                       keep_polar_features=False):
    """Convert range images to point cloud.
    
    Args:
      frame: open dataset frame
      range_images: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
      camera_projections: A dict of {laser_name,
        [camera_projection_from_first_return,
        camera_projection_from_second_return]}.
      range_image_top_pose: range image pixel pose for top lidar.
      ri_index: 0 for the first return, 1 for the second return.
      keep_polar_features: If true, keep the features from the polar range image
        (i.e. range, intensity, and elongation) as the first features in the
        output range image.
      
      Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
          (NOTE: Will be {[N, 6]} if keep_polar_features is true.
        cp_points: {[N, 6]} list of camera projections of length 5
          (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []

    cartesian_range_images = frame_utils.convert_range_image_to_cartesian(
        frame, range_images, range_image_top_pose, ri_index, keep_polar_features)

    for c in calibrations:
      range_image = range_images[c.name][ri_index]
      range_image_tensor = tf.reshape(
          tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
      range_image_mask = range_image_tensor[..., 0] > 0
      x, y = tf.meshgrid(tf.range(range_image_mask.shape[1], dtype=tf.float32),
                         tf.range(range_image_mask.shape[0], dtype=tf.float32))
      x = x / range_image_mask.shape[1]
      y = y / range_image_mask.shape[0]
      xy = tf.stack([x, y], axis=-1)

      range_image_cartesian = cartesian_range_images[c.name]
      range_image_cartesian = tf.concat([range_image_cartesian, xy], axis=-1)
      points_tensor = tf.gather_nd(range_image_cartesian,
                                   tf.compat.v1.where(range_image_mask))

      cp = camera_projections[c.name][ri_index]
      cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
      cp_points_tensor = tf.gather_nd(cp_tensor,
                                      tf.compat.v1.where(range_image_mask))
      points.append(points_tensor.numpy())
      cp_points.append(cp_points_tensor.numpy())

    return points, cp_points


def save_lidar_points(frame, cur_save_path, use_two_returns=True, seg_labels=True):
    if seg_labels:
        (range_images, camera_projections, segmentation_labels,
         range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
            frame)
    else:
        (range_images, camera_projections, range_image_top_pose) = \
            frame_utils.parse_range_image_and_camera_projection(frame)

    points, cp_points = convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose,
        keep_polar_features=True)
    if use_two_returns:
        points_ri2, cp_points_ri2 = convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose,
            ri_index=1, keep_polar_features=True)
        points = [np.concatenate([p1, p2], axis=0) \
                    for p1, p2 in zip(points, points_ri2)]
    num_points_of_each_lidar = [point.shape[0] for point in points]
    
    points = np.concatenate(points, axis=0)

    points = points[:, [3,4,5,1,2,0,6,7]] # [x, y, z, intensity, elongation, range, w, h]

    points = points.astype(np.float32)

    np.save(cur_save_path, points)
    
    if seg_labels:
        # load segmentation labels
        if frame.lasers[0].ri_return1.segmentation_label_compressed:
            assert frame.lasers[0].ri_return2.segmentation_label_compressed
            point_labels = convert_range_image_to_point_cloud_labels(
                frame, range_images, segmentation_labels)
            point_labels = np.concatenate(point_labels, axis=0)
            if use_two_returns:
                point_labels_ri2 = convert_range_image_to_point_cloud_labels(
                    frame, range_images, segmentation_labels, ri_index=1)
                point_labels_ri2 = np.concatenate(point_labels_ri2, axis=0)
                point_labels = np.concatenate([point_labels, point_labels_ri2],
                                              axis=0)
            seg_label_path = str(cur_save_path).replace('.npy', '_seg.npy')
            np.save(seg_label_path, point_labels)
        else:
            seg_label_path = None
        
        return num_points_of_each_lidar, seg_label_path
    else:
        return num_points_of_each_lidar



def process_single_sequence(sequence_file, save_path, sampled_interval,
                            has_label=True, use_two_returns=True,
                            seg_only=False):
    sequence_name = os.path.splitext(os.path.basename(sequence_file))[0]
    cur_save_dir = save_path / sequence_name
    pkl_file = cur_save_dir / ('%s.pkl' % sequence_name)
    sequence_infos = []
    if pkl_file.exists():
        sequence_infos = pickle.load(open(pkl_file, 'rb'))
        print('Skip sequence since it has been processed before: %s' % pkl_file)
        return sequence_infos

    # print('Load record (sampled_interval=%d): %s' % (sampled_interval, sequence_name))
    if not sequence_file.exists():
        print('NotFoundError: %s' % sequence_file)
        return []

    dataset = tf.data.TFRecordDataset(str(sequence_file), compression_type='')
    cur_save_dir.mkdir(parents=True, exist_ok=True)

    for cnt, data in enumerate(dataset):
        if cnt % sampled_interval != 0:
            continue
        # print(sequence_name, cnt)
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        if seg_only:
            if not frame.lasers[0].ri_return1.segmentation_label_compressed:
                continue

        info = {}
        pc_info = {'num_features': 8, 'lidar_sequence': sequence_name, 'sample_idx': cnt}
        info['point_cloud'] = pc_info
        top_lidar_pose = []
        for calibration in frame.context.laser_calibrations:
            top_lidar_pose.append(
                np.array(calibration.extrinsic.transform).astype(np.float32).reshape(-1)
            )

        info['frame_id'] = sequence_name + ('_%03d' % cnt)
        info['metadata'] = {
            'context_name': frame.context.name,
            'timestamp_micros': frame.timestamp_micros,
            'top_lidar_pose': top_lidar_pose
        }
        image_info = {}
        for j in range(5):
            width = frame.context.camera_calibrations[j].width
            height = frame.context.camera_calibrations[j].height
            image_info.update({'image_shape_%d' % j: (height, width)})
        info['image'] = image_info

        pose = np.array(frame.pose.transform, dtype=np.float32).reshape(4, 4)
        info['pose'] = pose
        
        if has_label:
            annotations = generate_labels(frame)

        num_points_of_each_lidar, seg_label_path = save_lidar_points(
            frame, cur_save_dir / ('%04d.npy' % cnt), use_two_returns=use_two_returns,
            seg_labels=True
        )
        
        if has_label:
            annotations['seg_label_path'] = seg_label_path
            info['annos'] = annotations

        info['num_points_of_each_lidar'] = num_points_of_each_lidar
        
        if has_label:
            annotations['seg_label_path'] = seg_label_path
            info['annos'] = annotations

        sequence_infos.append(info)

    with open(pkl_file, 'wb') as f:
        pickle.dump(sequence_infos, f)

    print('Infos are saved to (sampled_interval=%d): %s' % (sampled_interval, pkl_file))
    return sequence_infos

def split_by_seg_label(points, labels):
    """split the point cloud into semantic segments
    

    Args:
        points [N, 3]
        labels [N_top, 2] only top lidar points are labeled,
                          channels are [instance, segment]

    Returns:
        three segments in shape [N_i, 3]:
            road: segment class {18 (road), 19 (lane marker)}
            sidewalk: segment class {17 (curb), 20 (other ground),
                             21 (walkable), 22 (sidewalk)}
            other_obj: any segment class except road and walkable

        labels [N_other_obj, 2]
    """
    
    # drop points from other lidar sensor (no seg label)
    points = points[:labels.shape[0]]

    seg_labels = labels[:, 1]
    
    road_mask = seg_labels == 10
    sidewalk_mask = seg_labels == 11
    other_obj_mask = (road_mask == False) & (sidewalk_mask == False)
    
    road = points[road_mask, :3]
    sidewalk = points[sidewalk_mask, :3]
    other_obj = points[other_obj_mask, :3]
    labels = labels[other_obj_mask, :]
    
    return road, sidewalk, other_obj, labels 

def find_box_instance_label(overlap, instance_labels):
    num_boxes = overlap.shape[0]

    box_instance_labels = np.zeros(num_boxes, dtype=np.int32)
    for i in range(num_boxes):
        mask = overlap[i, :]
        box_instance_labels[i] = np.median(instance_labels[mask])
    return box_instance_labels

def check_box_interaction(boxes, radius, other_obj, seg_labels):
    expected_overlap = points_in_boxes_cpu(other_obj, boxes)

    box_instance_labels = find_box_instance_label(expected_overlap,
                                                  seg_labels[:, 0])
    
    boxes_as_boundary = np.copy(boxes)
    boxes_as_boundary[:, 3:6] += radius
    
    # compute point-box interaction
    interaction = points_in_boxes_cpu(other_obj,
                                      boxes_as_boundary).astype(bool)
    
    # box interacting points with it is allowed
    interaction[np.where(expected_overlap)] = False
    box_index, point_index = np.where(interaction)

    # box interacting with points within the same instance is allowed
    mask = box_instance_labels[box_index] == seg_labels[point_index, 0]
    interaction[(box_index[mask], point_index[mask])] = False

    # others are not allowed
    box_is_interacting = interaction.any(1)
    return box_is_interacting 

def compute_interaction_index_for_frame(dataset, info, radius_list):
    points = dataset.get_lidar(info['point_cloud']['lidar_sequence'],
                            info['point_cloud']['sample_idx'])
    annos = info['annos']
    boxes = annos['gt_boxes_lidar']
    if boxes.shape[0] > 0:
        seg_labels = dataset.get_seg_label(info['point_cloud']['lidar_sequence'],
                                           info['point_cloud']['sample_idx'])

        road, walkable, other_obj, seg_labels = split_by_seg_label(points, seg_labels)

        box_interaction = {}
        for radius in radius_list:
            box_is_interacting = check_box_interaction(
                                     boxes, radius,
                                     other_obj, seg_labels)
            box_interaction[f'{radius}'] = box_is_interacting
        
        info['annos']['interaction_index'] = box_interaction

    return info

def extract_foreground_pointcloud(dataset, top_lidar_only, database_save_path, info, db_info_save_path):
    frame_id = info['frame_id']
    sample_idx = int(frame_id[-3:])
    sequence_name = frame_id[:-4]
    seg_labels = dataset.get_seg_label(sequence_name, sample_idx)

    annos = info['annos']
    gt_boxes = annos['gt_boxes_lidar']
    pc_info = info['point_cloud']
    sequence_name = pc_info['lidar_sequence']
    sample_idx = pc_info['sample_idx']
    points = dataset.get_lidar(sequence_name, sample_idx)
    if top_lidar_only:
        points = points[:seg_labels.shape[0]]
    seg_inst_labels, seg_cls_labels = seg_labels.T
    
    #vis.clear()
    #vis_dict = dict(
    #    points=torch.from_numpy(points),
    #    seg_inst_labels=torch.from_numpy(seg_inst_labels),
    #    seg_cls_labels=torch.from_numpy(seg_cls_labels),
    #    batch_idx=torch.zeros(points.shape[0], 1).long(),
    #    batch_size=1,
    #)
    #vis(vis_dict)
    foreground_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15]
    instance_dict = {i: [] for i in foreground_class}
    instance_count = {i: 0 for i in foreground_class}

    points = torch.from_numpy(points).cuda()
    seg_cls_labels = torch.from_numpy(seg_cls_labels).cuda()
    seg_inst_labels = torch.from_numpy(seg_inst_labels).cuda()
    for fg_idx, fg_cls in enumerate(foreground_class):
        #print('foreground class', fg_cls)
        strategy = dataset.strategies[fg_idx]
        support = strategy['support']
        radius = strategy.get('radius', None)
        group_radius = strategy.get('group_radius', None)
        min_num_point = strategy.get('min_num_points', 5)
        use_inst_label = strategy.get('use_inst_label', False)
        attach_box = strategy.get('attach_box', False)
        group_with = strategy.get('group_with', [])
        cls_mask = seg_cls_labels == fg_cls
        cls_points = points[cls_mask]
        inst_labels = seg_inst_labels[cls_mask]
        while cls_points.shape[0] > min_num_point:
            #print(f'cls={fg_cls}, #points', cls_points.shape[0])
            if use_inst_label:
                inst_label = inst_labels.unique()[0]
                instance_pc = cls_points[inst_labels == inst_label]
                cls_points = cls_points[inst_labels != inst_label]
                inst_labels = inst_labels[inst_labels != inst_label]
            else:
                center = cls_points[0]
                dist = (cls_points - center)[:, :2].norm(p=2, dim=-1)
                inst_mask = dist < radius
                instance_pc = cls_points[inst_mask]
                cls_points = cls_points[inst_mask == False]
            if instance_pc.shape[0] > min_num_point:
                # find box that covers it
                if attach_box:
                    point_masks = points_in_boxes_cpu(instance_pc[:, :3].cpu().numpy(), gt_boxes)
                    average = point_masks.mean(1)
                    if average.max() > 0.9:
                        box_index = average.argmax()
                        attaching_box = gt_boxes[box_index]
                    else:
                        attaching_box = None
                else:
                    attaching_box = None

                
                # group with other classes
                if len(group_with) > 0:
                    center = instance_pc.mean(0)
                    offsets = [0]
                    sizes = [instance_pc.shape[0]]
                    classes = [fg_cls]
                    success = False
                    for g in group_with:
                        g_mask = seg_cls_labels == g
                        if not g_mask.any():
                            continue
                        g_points = points[g_mask]
                        g_dist = (g_points - center)[:, :2].norm(p=2, dim=-1)
                        if not (g_dist < radius).any():
                            continue
                        success = True
                        grouped_points = g_points[g_dist < radius]
                        classes.append(g)
                        offsets.append(offsets[-1]+sizes[-1])
                        sizes.append(grouped_points.shape[0])
                        instance_pc = torch.cat([instance_pc, grouped_points], dim=0)
                    if success:
                        grouping = dict(
                            cls=classes,
                            offsets=offsets,
                            sizes=sizes,
                        )
                    else:
                        grouping = None
                else:
                    grouping = None

                low = instance_pc[instance_pc[:, 2].argmin()]
                # find support of this
                for support_cls in support:
                    support_mask = seg_cls_labels == support_cls
                    if not support_mask.any():
                        continue
                    support_points = points[support_mask]
                    support_dist = (support_points - low)[:, :3].norm(p=2, dim=-1)
                    if not use_inst_label and (support_dist.min() > radius):
                        continue
                    trans = (support_points[support_dist.argmin()] - low)[2]
                    inst_count = instance_count[fg_cls]
                    instance_count[fg_cls] += 1
                    if (fg_cls == 0) and (inst_count % 4 != 0):
                        break
                    if (fg_cls == 6) and (inst_count % 2 != 0):
                        break
                    if (fg_cls == 14) and (inst_count % 2 != 0):
                        break
                    if (fg_cls == 15) and (inst_count % 2 != 0):
                        break
                    #if support_cls in group_with:
                    #    grouped_points = support_points[support_dist < radius]
                    #    grouping = dict(
                    #        cls=[fg_cls, support_cls],
                    #        offsets=[0, instance_pc.shape[0]],
                    #        sizes=[instance_pc.shape[0], grouped_points.shape[0]]
                    #    )
                    #    instance_pc = np.concatenate([instance_pc, grouped_points], axis=0)
                    #else:
                    #    grouping = None
                    inst_save_path = database_save_path / f'{frame_id}_class_{fg_cls:02d}_inst_{inst_count:06d}.npy'
                    np.save(inst_save_path, instance_pc.detach().cpu().numpy())
                    #vis.pointcloud(f'cls-{fg_cls}-inst-{inst_count}-support-{support_cls}',
                    #               torch.from_numpy(instance_pc[:, :3]), None, None, radius=3e-4,
                    #               color=vis._shared_color['seg-class-color'][fg_cls])
                    record = dict(
                        trans_z=trans.detach().cpu().numpy(),
                        grouping=grouping,
                        support=support_cls,
                        path=inst_save_path,
                        obj_class=fg_cls,
                        sample_idx=sample_idx,
                        sequence_name=sequence_name,
                        num_points=instance_pc.shape[0],
                        box3d=attaching_box,
                    )
                    instance_dict[fg_cls].append(record)
                    break
    
    with open(db_info_save_path, 'wb') as f:
        pickle.dump(instance_dict, f)
    return instance_dict
