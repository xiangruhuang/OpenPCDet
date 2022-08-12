import numpy as np
import glob
from tqdm import tqdm

with open('data/waymo/ImageSets/train.txt', 'r') as fin:
    lines = [line.strip().split('.')[0] for line in fin.readlines()]

with open('data/waymo/ImageSets/val.txt', 'r') as fin:
    lines2 = [line.strip().split('.')[0] for line in fin.readlines()]
    lines += lines2

with open('hey', 'w') as fout:
    for i, line in tqdm(enumerate(lines)):
        files = glob.glob(f'data/waymo/waymo_processed_data_v0_5_0/{line}/*_seg.npy')
        count = 0
        for f in files:
            seg_labels = np.load(f)[:, 1]
            count_f = (seg_labels == 5).sum()
            if count_f > 10:
                fout.write(f'{f} {count_f}\n')
