import torch

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
    parser.add_argument('data', default=None)
    parser.add_argument('--yaml', default='visualizer.yaml')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    vis = get_visualizer(args.yaml)
    
    vis_dict = torch.load(args.data)
    #if 'batch_idx' not in vis_dict.keys():
    #    import ipdb; ipdb.set_trace()
    #    vis_dict['batch_idx'] = vis_dict['points'][:, 0:1].long()
    #    #vis_dict['batch_idx'] = torch.zeros(vis_dict['points'].shape[0], 1).long()
    if 'batch_size' not in vis_dict.keys():
        vis_dict['batch_size'] = 1

    vis(vis_dict)

    #import ipdb; ipdb.set_trace()
    #import polyscope as ps
    #import numpy as np
    #data=np.load('23691_after.point.npy')
    #ps.register_point_cloud('after', data[:, :3], radius=2e-4)
    #ps.show()
