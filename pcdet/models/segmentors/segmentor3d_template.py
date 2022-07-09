import os

import torch
import torch.nn as nn

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils.spconv_utils import find_all_spconv_keys
from .. import backbones_2d, backbones_3d, dense_heads, roi_heads, visualizers
from ..backbones_2d import map_to_bev
from ..backbones_3d import pfe, vfe
from ..model_utils import model_nms_utils

class Segmentor3DTemplate(nn.Module):
    def __init__(self, model_cfg, runtime_cfg, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.runtime_cfg = runtime_cfg
        self.dataset = dataset
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'vfe', 'backbone_3d', 'seg_head', 'visualizer'
        ]

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
        }
        model_info_dict.update(self.dataset.runtime_cfg)
        #    'input_channels': self.dataset.num_point_features,
        #    'grid_size': self.dataset.grid_size,
        #    #'point_cloud_range': self.dataset.point_cloud_range,
        #    #'voxel_size': self.dataset.voxel_size,
        #    #'depth_downsample_factor': self.dataset.depth_downsample_factor
        #}
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_visualizer(self, model_info_dict):
        if self.model_cfg.get('VISUALIZER', None) is None:
            return None, model_info_dict

        visualizer_module = visualizers.__all__[self.model_cfg.VISUALIZER.NAME](
            model_cfg=self.model_cfg.VISUALIZER,
            runtime_cfg=model_info_dict,
            #point_cloud_range=model_info_dict['point_cloud_range'],
            #voxel_size=model_info_dict['voxel_size'],
            #grid_size=model_info_dict['grid_size'],
            #depth_downsample_factor=model_info_dict['depth_downsample_factor']
        )
        model_info_dict['module_list'].append(visualizer_module)
        return visualizer_module, model_info_dict

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict

        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            runtime_cfg=model_info_dict,
            model_cfg=self.model_cfg.VFE,
            #num_point_features=model_info_dict['num_point_features'],
            #point_cloud_range=model_info_dict['point_cloud_range'],
            #voxel_size=model_info_dict['voxel_size'],
            #grid_size=model_info_dict['grid_size'],
            #depth_downsample_factor=model_info_dict['depth_downsample_factor'],
            #num_class=self.dataset.num_seg_class
        )
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        if hasattr(vfe_module, 'output_key'):
            model_info_dict['input_key'] = vfe_module.output_key
        return vfe_module, model_info_dict

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict

        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            runtime_cfg=model_info_dict,
        )
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        model_info_dict['backbone_channels'] = backbone_3d_module.backbone_channels \
            if hasattr(backbone_3d_module, 'backbone_channels') else None
        if hasattr(backbone_3d_module, 'output_key'):
            model_info_dict['input_key'] = backbone_3d_module.output_key
        return backbone_3d_module, model_info_dict

    def build_seg_head(self, model_info_dict):
        if self.model_cfg.get('SEG_HEAD', None) is None:
            return None, model_info_dict

        num_point_features = model_info_dict['num_point_features']

        model_info_dict['input_channels'] = num_point_features
        point_head_module = dense_heads.__all__[self.model_cfg.SEG_HEAD.NAME](
            model_cfg=self.model_cfg.SEG_HEAD,
            runtime_cfg=model_info_dict,
            #input_channels=num_point_features,
            #num_class=self.dataset.num_seg_class,
            #predict_boxes_when_training=False,
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))
            else:
                logger.info('Updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch
