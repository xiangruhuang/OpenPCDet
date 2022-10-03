import numpy as np
import torch
from pcdet.models.visualizers import PolyScopeVisualizer
import yaml, pickle
import glob
from pcdet.utils import common_utils

def get_visualizer(yaml_file):
    from pcdet.models.visualizers import PolyScopeVisualizer as psv
    import yaml
    with open(yaml_file, 'r') as fin:
        cfg = yaml.safe_load(fin)
    vis = psv(cfg['VISUALIZER'])
    return vis

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('sequence_id', type=int)
    parser.add_argument('result_dir', type=str)
    parser.add_argument('--sample_idx', type=int)
    parser.add_argument('--cls', default=int)
    parser.add_argument('--vis_cfg', type=str, default='cfgs/visualizers/semantic_visualizer.yaml')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    vis = get_visualizer(args.vis_cfg)
    
    with open('../data/waymo/ImageSets/val.txt', 'r') as fin:
        sequences = [line.strip().split('.')[0] for line in fin.readlines()]
        args.sequence_id = sequences[args.sequence_id]

    sequence_path = f'../data/waymo/waymo_processed_data_val_seg_only/{args.sequence_id}'
    seg_files = glob.glob(f'{sequence_path}/*_seg.npy')
    frame_ids = [int(seg_file.split('/')[-1].split('.')[0].split('_')[0]) for seg_file in seg_files]
    pred_files = [f'{args.result_dir}/{args.sequence_id}/{frame_id:03d}_pred.npy' for frame_id in frame_ids]
    pc_files = [seg_file.replace('_seg.npy', '.npy') for seg_file in seg_files]
    with open(f'{sequence_path}/{args.sequence_id}.pkl', 'rb') as fin:
        infos = pickle.load(fin)
        infos = [infos[frame_id] for frame_id in frame_ids]

    T0 = infos[0]['pose'].reshape(4, 4)
    T0_inv = np.linalg.inv(T0)
    batch_dicts = []
    for info, frame_id, seg_file, pc_file, pred_file in zip(infos, frame_ids, seg_files, pc_files, pred_files):
        pred_labels = torch.from_numpy(np.load(pred_file)).long()
        seg_labels = torch.from_numpy(np.load(seg_file))[:, 1].long()
        pred_labels[seg_labels == 0] = 0
        error_mask = (seg_labels != pred_labels).long()
        ignore_mask = pred_labels == 0
        type1_mask = (seg_labels <= 7) & (pred_labels > 7)
        type2_mask = (seg_labels > 7) & (pred_labels <= 7)
        type3_mask = (seg_labels <= 7) & (pred_labels <= 7)
        type4_mask = (seg_labels > 7) & (pred_labels > 7)
        error_mask[type1_mask] = 1
        error_mask[type2_mask] = 1
        error_mask[type3_mask] = 2
        error_mask[type4_mask] = 2
        error_mask[ignore_mask] = 0
        error_mask[seg_labels == pred_labels] = 0

        point_xyz = np.load(pc_file)[:, :3]
        T = T0_inv @ info['pose'].reshape(4, 4)
        point_xyz = point_xyz @ T[:3, :3].T + T[:3, 3]
        point_bxyz = torch.from_numpy(
                         np.concatenate([np.zeros_like(point_xyz[:, :1]),
                                         point_xyz], axis=-1),
                         )[:seg_labels.shape[0]]
        gt_bg_labels = (seg_labels > 13).long()
        pred_bg_labels = (pred_labels > 13).long()
        bg_error_mask = (gt_bg_labels != pred_bg_labels).long()
        
        batch_dict = dict(
            point_bxyz=point_bxyz,
            pred_segmentation_label=pred_labels,
            segmentation_label=seg_labels,
            error_mask=error_mask,
            mask_5=(seg_labels == 5).long(),
            bg_error_mask=bg_error_mask,
            fg_bxyz=point_bxyz[pred_bg_labels == 0],
            batch_size=1,
            suffix=frame_id
        )
        vis(batch_dict)
    import polyscope as ps; ps.init()
    ps.show()
