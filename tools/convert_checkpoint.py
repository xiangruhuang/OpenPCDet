import torch
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('f1', type=str)
    parser.add_argument('f2', type=str)
    parser.add_argument('fout', type=str)

    args = parser.parse_args()

    assert not os.path.exists(args.fout)

    return args

def parse_skip_module(tokens, num, val):
    table = {
        'conv.weight': 'message_passing.kernel_weights',
        'bn.weight': 'norm.weight',
        'bn.bias': 'norm.bias',
        'bn.running_mean': 'norm.running_mean',
        'bn.running_var': 'norm.running_var',
        'bn.num_batches_tracked': 'norm.num_batches_tracked',
    }

    if tokens[0].startswith('conv'):
        # conv layer
        num2 = int(tokens[0].split('conv')[-1]) - 1
        typ = '.'.join(['conv'] + tokens[1:])
    elif tokens[0].startswith('bn'):
        num2 = int(tokens[0].split('bn')[-1]) - 1
        typ = '.'.join(['bn'] + tokens[1:])
    suffix = table[typ]
    new_key = f'skip_modules.{5-num}.{num2}.{suffix}'
    if 'kernel_weights' in new_key:
        new_val = val.reshape(val.shape[0], -1, val.shape[-1]).transpose(0, 1).transpose(1, 2)
    else:
        new_val = val
    return new_key, new_val

def parse_merge_module(tokens, num, val):
    table = {
        '0.weight': 'message_passing.kernel_weights',
        '1.weight': 'norm.weight',
        '1.bias': 'norm.bias',
        '1.running_mean': 'norm.running_mean',
        '1.running_var': 'norm.running_var',
        '1.num_batches_tracked': 'norm.num_batches_tracked',
    }
    typ = '.'.join(tokens[0:])
    
    suffix = table[typ]
    new_key = f'merge_modules.{5-num}.{suffix}'
    if 'kernel_weights' in new_key:
        new_val = val.reshape(val.shape[0], -1, val.shape[-1]).transpose(0, 1).transpose(1, 2)
    else:
        new_val = val
    return new_key, new_val

def parse_conv_module(tokens, num, val):
    num2 = int(tokens[0])
    typ = '.'.join(tokens[1:])
    table = {
        '0.weight': 'message_passing.kernel_weights',
        '1.weight': 'norm.weight',
        '1.bias': 'norm.bias',
        '1.running_mean': 'norm.running_mean',
        '1.running_var': 'norm.running_var',
        '1.num_batches_tracked': 'norm.num_batches_tracked',
    }
    suffix = table[typ]
    new_key = f'down_modules.{num-1}.{num2}.{suffix}'
    if 'kernel_weights' in new_key:
        new_val = val.reshape(val.shape[0], -1, val.shape[-1]).transpose(0, 1).transpose(1, 2)
    else:
        new_val = val
    return new_key, new_val

def parse_up_module(tokens, num, val):
    typ = '.'.join(tokens[0:])
    table = {
        '0.weight': 'message_passing.kernel_weights',
        '1.weight': 'norm.weight',
        '1.bias': 'norm.bias',
        '1.running_mean': 'norm.running_mean',
        '1.running_var': 'norm.running_var',
        '1.num_batches_tracked': 'norm.num_batches_tracked',
    }
    suffix = table[typ]
    new_key = f'up_modules.{5-num}.{suffix}'
    if 'kernel_weights' in new_key:
        new_val = val.reshape(val.shape[0], -1, val.shape[-1]).transpose(0, 1).transpose(1, 2)
    else:
        new_val = val
    return new_key, new_val

def parse_modules(tokens, val):
    if tokens[0].startswith('conv_up_t'):
        conv_n = int(tokens[0].split('conv_up_t')[-1])
        return parse_skip_module(tokens[1:], conv_n, val)

    if tokens[0].startswith('conv_up_m'):
        conv_n = int(tokens[0].split('conv_up_m')[-1])
        return parse_merge_module(tokens[1:], conv_n, val)

    if tokens[0].startswith('conv'):
        conv_n = int(tokens[0].split('conv')[-1])
        return parse_conv_module(tokens[1:], conv_n, val)
    
    if tokens[0].startswith('inv_conv'):
        conv_n = int(tokens[0].split('inv_conv')[-1])
        return parse_up_module(tokens[1:], conv_n, val)

def parse_key(key, val):
    if not key.startswith('backbone_3d'):
        return key, val
    tokens = key.split('.')
    import ipdb; ipdb.set_trace()
    new_key, new_val = parse_modules(tokens[1:], val)
    new_key = 'backbone_3d.' + new_key
    return new_key, new_val

def convert_lidarmultinet_ckpt_to_pointconvnet(path_l, path_p, output_path):
    ckpt_l = torch.load(path_l)['model_state']
    old_ckpt = torch.load(path_p)
    ckpt_p = old_ckpt['model_state']
    new_ckpt = {}
    new_ckpt.update(ckpt_p)
    empty_ckpt = {}
    for key, val in ckpt_l.items():
        if key in ['global_step']:
            continue
        if isinstance(val, torch.Tensor):
            new_key, new_val = parse_key(key, val)
            assert val.numel() == new_val.numel()
            assert (new_val.shape == ckpt_p[new_key].shape)
            new_ckpt[new_key] = new_val
            assert new_key not in empty_ckpt
            empty_ckpt[new_key] = new_val
            print(key, '------>', new_key)
        else:
            print(key)

    old_ckpt['model_state'].update(new_ckpt)
    torch.save(old_ckpt, output_path)
    print('convertion done')

if __name__ == '__main__':
    args = parse_args()

    #f1 = '../output/waymo_models/segmentation/lidarmultinet_ohem_nogcp/waymo_segmentation_on_full_default_voxel_aug/onecycle_bs2_verify_repeat2/ckpt/checkpoint_epoch_20.pth'
    #f1 = '../checkpoints/lidarmultinet_ohem_remove_outside.pth'
    #f2 = '../output/waymo_models/segmentation/pointconvnet/waymo_segmentation_on_full_default_voxel_aug/onecycle_bs2_verify2/ckpt/checkpoint_epoch_1.pth'
    #fout = '../output/waymo_models/segmentation/pointconvnet/waymo_segmentation_on_full_default_voxel_aug/onecycle_bs2_verify2/ckpt/checkpoint_epoch_3.pth'
    #fout = '../checkpoints/pointconvnet_remove_outside.pth'
    convert_lidarmultinet_ckpt_to_pointconvnet(args.f1, args.f2, args.fout)
