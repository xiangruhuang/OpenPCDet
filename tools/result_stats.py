import numpy as np
import torch
from pcdet.models.visualizers import PolyScopeVisualizer
import yaml, pickle
import glob
from pcdet.utils import common_utils
from tqdm import tqdm
import os

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
    parser.add_argument('result_dir', type=str)
    parser.add_argument('--sample_idx', type=int)
    parser.add_argument('--cls', default=int)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    with open('../data/waymo/ImageSets/val.txt', 'r') as fin:
        sequences = [line.strip().split('.')[0] for line in fin.readlines()]

    foreground_coverage = {i: [0, 0] for i in range(8)}
    foreground_precision = {i: [0, 0] for i in range(8)}
    ups = np.array([0 for i in range(23)])
    downs = np.array([0 for i in range(23)])
    for sequence_id in tqdm(sequences):
        sequence_path = f'../data/waymo/waymo_processed_data_val_seg_only/{sequence_id}'
        pred_files = sorted(glob.glob(f'{args.result_dir}/{sequence_id}/*_pred.npy'))
        frame_ids = [int(pred_file.split('/')[-1].split('.')[0].split('_')[0]) for pred_file in pred_files]
        seg_files = [f'{sequence_path}/{frame_id:04d}_seg.npy' for frame_id in frame_ids]
        pc_files = [seg_file.replace('_seg.npy', '.npy') for seg_file in seg_files]
        with open(f'{sequence_path}/{sequence_id}.pkl', 'rb') as fin:
            infos = pickle.load(fin)
            infos = [infos[frame_id] for frame_id in frame_ids]

        for info, frame_id, seg_file, pc_file, pred_file in zip(infos, frame_ids, seg_files, pc_files, pred_files):
            if not os.path.exists(pred_file):
                continue
            if not os.path.exists(seg_file):
                continue

            #if sequence_id != 'segment-10203656353524179475_7625_000_7645_000_with_camera_labels':
            #    continue
            pred_labels = torch.from_numpy(np.load(pred_file)).long()
            seg_labels = torch.from_numpy(np.load(seg_file))[:, 1].long()
            pred_labels[seg_labels == 0] = 0
            for i in range(23):
                ups[i] += ((pred_labels == i) & (seg_labels == i)).sum()
                downs[i] += ((pred_labels == i) | (seg_labels == i)).sum()
            for i in range(1, 8):
                foreground_coverage[i][0] += ((pred_labels <= 7) & (pred_labels > 0) & (seg_labels == i)).sum()
                foreground_precision[i][0] += ((seg_labels <= 7) & (seg_labels > 0) & (pred_labels == i)).sum()
                foreground_coverage[i][1] += (seg_labels == i).sum()
                foreground_precision[i][1] += (pred_labels == i).sum()
        
            coverage = sum([foreground_coverage[i][0] for i in range(1, 8)]) / sum([foreground_coverage[i][1] for i in range(1, 8)])
            #print(f'coverage={coverage:.4f}, sequence_id={sequence_id}, frame_id={frame_id}')
        #for i in range(1, 8):
        #    print(f'coverage{i}={foreground_coverage[i][0] / foreground_coverage[i][1]}, precision{i}={foreground_precision[i][0] / foreground_precision[i][1]}')
        ious = (ups/(downs+1e-6))
        for i in range(23):
            print(f'i={i}, iou={ious[i]:.4f}')
        print(f'mIoU={ious[1:].mean()}')

