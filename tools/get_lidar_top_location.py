import numpy as np
import waymo_open_dataset
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.camera.ops import py_camera_model_ops
from waymo_open_dataset.utils import frame_utils
import tensorflow as tf
import glob
import joblib
import sys
from tqdm import tqdm

import polyscope as ps; ps.init(); ps.set_up_dir('z_up')
import matplotlib.pyplot as plt

def project_vehicle_to_image(vehicle_pose, calibration, points):
  """Projects from vehicle coordinate system to image with global shutter.

  Arguments:
    vehicle_pose: Vehicle pose transform from vehicle into world coordinate
      system.
    calibration: Camera calibration details (including intrinsics/extrinsics).
    points: Points to project of shape [N, 3] in vehicle coordinate system.

  Returns:
    Array of shape [N, 3], with the latter dimension composed of (u, v, ok).
  """
  # Transform points from vehicle to world coordinate system (can be
  # vectorized).
  pose_matrix = np.array(vehicle_pose.transform).reshape(4, 4)
  world_points = np.zeros_like(points)
  for i, point in enumerate(points):
    cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
    world_points[i] = (cx, cy, cz)

  # Populate camera image metadata. Velocity and latency stats are filled with
  # zeroes.
  extrinsic = tf.reshape(
      tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32),
      [4, 4])
  intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
  metadata = tf.constant([
      calibration.width,
      calibration.height,
      open_dataset.CameraCalibration.GLOBAL_SHUTTER,
  ], dtype=tf.int32)
  camera_image_metadata = list(vehicle_pose.transform) + [0.0] * 10

  # Perform projection and return projected image coordinates (u, v, ok).
  return py_camera_model_ops.world_to_image(extrinsic, intrinsic, metadata,
                                            camera_image_metadata,
                                            world_points).numpy()

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
      sl = segmentation_labels[c.name][ri_index]
      sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
      sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
    else:
      num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
      sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)
      
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
        obj_name.append(class_ind)
        loc = [box.center_x, box.center_y, box.center_z]
        heading_angles.append(box.heading)
        difficulty.append(laser_labels[i].detection_difficulty_level)
        tracking_difficulty.append(laser_labels[i].tracking_difficulty_level)
        dimensions.append([box.length, box.width, box.height])  # lwh in unified coordinate of OpenPCDet
        locations.append(loc)
        obj_ids.append(laser_labels[i].id)
        num_points_in_gt.append(laser_labels[i].num_lidar_points_in_box)

    annotations = {}
    mask = np.array(obj_name, dtype=np.int32) > 0
    annotations['name'] = np.array(obj_name)[mask]
    annotations['difficulty'] = np.array(difficulty)[mask]
    annotations['dimensions'] = np.array(dimensions)[mask]
    annotations['location'] = np.array(locations)[mask]
    annotations['heading_angles'] = np.array(heading_angles)[mask]

    annotations['obj_ids'] = np.array(obj_ids)[mask]
    annotations['tracking_difficulty'] = np.array(tracking_difficulty)[mask]
    annotations['num_points_in_gt'] = np.array(num_points_in_gt)[mask]

    #annotations = common_utils.drop_info_with_name(annotations, name='unknown')
    if annotations['name'].__len__() > 0:
        gt_boxes_lidar = np.concatenate([
            annotations['location'], annotations['dimensions'], annotations['heading_angles'][..., np.newaxis]],
            axis=1
        )
    else:
        gt_boxes_lidar = np.zeros((0, 7))
    annotations['gt_boxes_lidar'] = gt_boxes_lidar
    return annotations

tfrecord_files = sorted(list(glob.glob(f"{sys.argv[1]}/*.tfrecord")))
top_lidar_origin = []
box_data = []
rgb_data = []
for tfrecord_file in tqdm(tfrecord_files):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    T_dict = {}
    for i, data in enumerate(tqdm(dataset)):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        #if not frame.lasers[0].ri_return1.segmentation_label_compressed:
        #    continue
        annotations = generate_labels(frame)
        box_vec = np.concatenate([
                      annotations['name'].reshape(-1, 1), annotations['gt_boxes_lidar'],
                      annotations['difficulty'].reshape(-1, 1), annotations['num_points_in_gt'].reshape(-1, 1)
                    ], axis=-1)
        (range_images, camera_projections, segmentation_labels,
         range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
            frame)
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose)
        points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose, ri_index=1)
        points = np.concatenate([points[0], points_ri2[0]], axis=0)
        if frame.lasers[0].ri_return1.segmentation_label_compressed:
            point_labels = convert_range_image_to_point_cloud_labels(
                frame, range_images, segmentation_labels)
            point_labels_ri2 = convert_range_image_to_point_cloud_labels(
                frame, range_images, segmentation_labels, ri_index=1)
            point_labels = np.concatenate([point_labels[0], point_labels_ri2[0]], axis=0)
            #mask = (point_labels[:, 1] > 7)
            #point_labels = point_labels[mask]
            #points = points[mask]
        images = sorted(frame.images, key=lambda img: img.name)
        calibrations = sorted(frame.context.camera_calibrations, key=lambda cali: cali.name)
        assert frame.context.laser_calibrations[4].name == 1 # TOP
        T = np.array(frame.context.laser_calibrations[4].extrinsic.transform, dtype=np.float32).reshape(4, 4)
        T_global = np.array(frame.pose.transform).reshape(4, 4)
        if T_dict.get(0, None) is None:
            T_dict[0] = np.linalg.inv(T_global)
            points_world = points
        else:
            T_this = T_dict[0] @ T_global
            points_world = points @ T_this[:3, :3].T + T_this[:3, 3]
        #T = T.astype(np.float64)
        #points = points.astype(np.float64)
        #points_world = points @ T_global[:3, :3].T + T_global[:3, 3] # + T[:3, 3] #(points - T[:3, 3]) @ T[:3, :3].T

        #top_lidar_origin.append(origin)
        ps_p = ps.register_point_cloud(f'points-{i}', points_world, radius=2e-4)
        #ps.register_point_cloud('origin', origin[None, :3], radius=5e-3)
        colors = np.zeros((points.shape[0], 3))
        count = np.zeros((points.shape[0], 3))
        for image_id, (image, calibration) in enumerate(zip(images, calibrations)):
            projected = project_vehicle_to_image(frame.pose, calibration, points)
            in_range = projected[:, -1].astype(np.bool)
            x, y = projected[:, :2].T
            x = x.round().astype(np.int32)
            y = y.round().astype(np.int32)
            in_range &= (0 <= x) & (x < calibration.width)
            in_range &= (0 <= y) & (y < calibration.height)
            x_in_range, y_in_range = x[in_range], y[in_range]
            
            image = tf.image.decode_jpeg(image.image).numpy()
            #plt.imshow(image)
            #plt.show()
            temp = image[(y_in_range, x_in_range)]
            colors[in_range] += temp
            count[in_range] += 1
            #ps.register_point_cloud('camera_origin', np.array(list(calibration.extrinsic.transform)).reshape(4, 4)[:3, 3].reshape(-1, 3), radius=5e-3)
            #ps.look_at((0, 0, 0), (10, 10, 0))
            #ps.look_at_dir(origin, origin*2, (0,0,1))
            #import ipdb; ipdb.set_trace()
        colors[count > 0] = colors[count > 0] / count[count > 0]
        colors[count == 0] = -1

        if frame.lasers[0].ri_return1.segmentation_label_compressed:
            ps_p.add_scalar_quantity('labels', point_labels[:, 1])
        ps_p.add_color_quantity('rgb', colors / 255.0, enabled=True)
        #ps_p.add_scalar_quantity('x', x)
        #ps_p.add_scalar_quantity('y', y)
        rgb_data.append(colors)
        #box_data.append(box_vec)
        if (i + 1) % 5 == 0:
            ps.show()

print(len(rgb_data))
assert False
#joblib.dump(top_lidar_origin, sys.argv[2])
#joblib.dump(rgb_data, sys.argv[2])
