import numpy as np
import torch
import yaml, pickle
import glob
from pcdet.utils import common_utils
from tqdm import tqdm
import itertools

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', type=str)
    parser.add_argument('--sample_idx', type=int)
    parser.add_argument('--cls', default=int)
    parser.add_argument('--vis_cfg', type=str, default='cfgs/visualizers/semantic_visualizer.yaml')
    args = parser.parse_args()

    return args

args = parse_args()

with open('../data/waymo/ImageSets/val.txt', 'r') as fin:
    sequences = [line.strip().split('.')[0] for line in fin.readlines()]

#edge_count = {(i, j): 0 for j in range(23) for i in range(23)}
edge_matrix = np.zeros((23, 23))

for i in tqdm(range(202)):
    sequence_id = sequences[i]
    sequence_path = f'../data/waymo/waymo_processed_data_val_seg_only/{sequence_id}'

    seg_files = glob.glob(f'{sequence_path}/*_seg.npy')[:1]
    frame_ids = [int(seg_file.split('/')[-1].split('.')[0].split('_')[0]) for seg_file in seg_files]
    pred_files = [f'{args.result_dir}/{sequence_id}/{frame_id:03d}_pred.npy' for frame_id in frame_ids]
    pc_files = [seg_file.replace('_seg.npy', '.npy') for seg_file in seg_files]
    with open(f'{sequence_path}/{sequence_id}.pkl', 'rb') as fin:
        infos = pickle.load(fin)
        infos = [infos[frame_id] for frame_id in frame_ids]
    
    for info, frame_id, seg_file, pc_file, pred_file in zip(infos, frame_ids, seg_files, pc_files, pred_files):
        pred_labels = torch.from_numpy(np.load(pred_file)).long()
        seg_labels = torch.from_numpy(np.load(seg_file))[:, 1].long()
        pred_labels[seg_labels == 0] = 0

        for cls1 in range(1, 23):
            for cls2 in range(1, 23):
                mask = (pred_labels == cls1) & (seg_labels == cls2)
                mask.sum()

                edge_matrix[cls1, cls2] += mask.sum().long().item()

print(edge_matrix)
edge_matrix = edge_matrix[5:, 5:]
for k in range(1, 13):
    print(f'source part size={k}')
    min_cut = 3
    best_sol = None
    for xx in tqdm(itertools.combinations(range(edge_matrix.shape[0]), k)):
        yy = [i for i in range(edge_matrix.shape[0]) if i not in xx]
        xx = list(xx)
        cxx = edge_matrix[xx, :].sum()
        cyy = edge_matrix[yy, :].sum()
        cxy = edge_matrix[xx, :][:, yy].sum() + edge_matrix[yy, :][:, xx].sum()
        cut = cxy / cxx + cxy / cyy
        if cut < min_cut:
            min_cut = cut
            best_sol = xx
    print(f'min_cut={min_cut}, sol={best_sol}')

pass
