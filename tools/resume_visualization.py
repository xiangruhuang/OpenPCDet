import torch

def get_visualizer(yaml_file='cfgs/visualizers/volume_visualizer.yaml'):
    from pcdet.models.visualizers import PolyScopeVisualizer as psv
    import yaml
    with open(yaml_file, 'r') as fin:
        cfg = yaml.safe_load(fin)
    vis = psv(cfg['VISUALIZER'])
    return vis

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str)
    parser.add_argument('--vis_cfg', type=str, default='cfgs/visualizers/volume_visualizer.yaml')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    vis = get_visualizer(args.vis_cfg)
    
    data = torch.load(args.data, map_location='cpu')
    
    vis(data)
