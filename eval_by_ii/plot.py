import matplotlib.pyplot as plt
import glob
import numpy as np
import sys

files = glob.glob(f'{sys.argv[1]}/log_*.txt')
radius_by_level = list(reversed([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 2.0, 4.0, 6.0, 8.0]))
for f in files:
    dist = int(f.split('.')[0].split('_')[1].split('d')[-1])
    if f.endswith('l1.txt'):
        l1 = True
    else:
        l1 = False
    curve = {'VEHICLE': [], 'PEDESTRIAN': [], 'CYCLIST': []}
    with open(f, 'r') as fin:
        lines = fin.readlines()
        lines = [line.strip() for line in lines]
    for line in lines:
        if line.startswith('Level'):
            level = int(line.split('Level ')[-1].split(':')[0])
            amount = int(line.split(':')[-1])
            continue
        if line.startswith('Number'):
            continue
        key, val = line.split(' ')
        if key.endswith('APH'):
            continue
        key = key.split('_')[3]
        val = float(val.split('[')[-1].split(']')[0])
        
        radius = radius_by_level[level]
        curve[key].append([radius, amount, val])
        
    color = {'VEHICLE': 'r', 'PEDESTRIAN': 'g', 'CYCLIST': 'b'}
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10))
    for cls, cls_curve in curve.items():
        cls_curve = np.array(cls_curve)
        ax1.plot(cls_curve[:, 0], cls_curve[:, 2], label=cls, color=color[cls], marker='+')

    l1_str = 'L1' if l1 else 'All'
    ax1.set_title(f'Statics by inter. idx, Range: {dist}m, difficulty: {l1_str}', size=25)
    ax1.set_ylabel('AP', size=20)
    ax1.legend(fontsize=20)
    
    for cls, cls_curve in curve.items():
        cls_curve = np.array(cls_curve)
        ax2.plot(cls_curve[:, 0], cls_curve[:, 1], label=cls, color='k', marker='+')

    ax2.set_xlabel('Radius in [x, +inf)', size=20)
    ax2.set_ylabel('All class population', size=20)
    plt.savefig(f.replace('log_', '').replace('.txt', '.png'))
