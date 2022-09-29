import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict

from pcdet.utils import common_utils
from pcdet.models.model_utils import graph_utils
from pcdet.models.model_utils.sampler_utils import build_sampler
from pcdet.models.model_utils.graph_utils import build_graph
from pcdet.models.blocks.assigners import build_assigner

from .post_processors import build_post_processor

def guess_value(model_cfg, runtime_cfg, keys, default=None):
    for key in keys:
        if key in runtime_cfg:
            return runtime_cfg[key]
        if key in model_cfg:
            return model_cfg[key]
    return default

class UNetTemplate(nn.Module):
    def __init__(self, runtime_cfg, model_cfg):
        super(UNetTemplate, self).__init__()
        self.model_cfg = model_cfg

        # attributes
        self.input_channels = guess_value(model_cfg, runtime_cfg, ["num_point_features"], 0)
        self.scale = guess_value(model_cfg, runtime_cfg, ["scale"], 1)
        self.output_key = guess_value(model_cfg, runtime_cfg, ["OUTPUT_KEY", "output_key"], None)
        input_cfg = guess_value(model_cfg, runtime_cfg, ["POINT", "VOXEL"], None)
        if input_cfg is not None:
            self.input_key = input_cfg.get("KEY", 'point')
            self.input_attributes = input_cfg["ATTRIBUTES"]
        else:
            self.input_key = guess_value(model_cfg, runtime_cfg, 
                                         ["INPUT_KEY", "input_key"], 'point')
            self.input_attributes = ['bxyz', 'feat']
        self.sa_channels = model_cfg.get("SA_CHANNELS", None)
        self.fp_channels = model_cfg.get("FP_CHANNELS", None)
        self.keys = model_cfg.get("KEYS", None)
        self.norm_cfg = model_cfg.get("NORM_CFG", None)
        self.activation = model_cfg.get("ACTIVATION", None)
        self.num_down_modules = len(self.sa_channels)
        self.num_up_modules = len(self.fp_channels)

        # modules
        for key in model_cfg.keys():
            if key.endswith('SAMPLERS'):
                name = key.lower()
                sampler_cfgs = model_cfg[key]
                num_samplers = sampler_cfgs.get("NUM", 1)
                samplers = nn.ModuleList()
                for i in range(num_samplers):
                    sampler_cfg = common_utils.indexing_list_elements(sampler_cfgs, i)
                    sampler = build_sampler(sampler_cfg, runtime_cfg)
                    samplers.append(sampler)
                self.add_module(name, samplers)

            if key.endswith('GRAPHS'):
                name = key.lower()
                graph_cfgs = model_cfg[key]
                num_graphs = graph_cfgs.get("NUM", 1)
                graphs = nn.ModuleList()
                for i in range(num_graphs):
                    graph_cfg = graph_utils.select_graph(graph_cfgs, i)
                    graph = build_graph(graph_cfg, runtime_cfg)
                    graphs.append(graph)
                self.add_module(name, graphs)
            if key.endswith('ASSIGNERS'):
                name = key.lower()
                assigner_cfgs = model_cfg[key]
                num_assigners = assigner_cfgs.get("NUM", 1)
                assigners = nn.ModuleList()
                for i in range(num_assigners):
                    assigner_cfg = common_utils.indexing_list_elements(assigner_cfgs, i)
                    assigner = build_assigner(assigner_cfg)
                    assigners.append(assigner)
                self.add_module(name, assigners)

    def build_post_processor(self, model_cfg, runtime_cfg):
        self.post_processor = build_post_processor(model_cfg.get("POST_PROCESSING_CFG", {}),
                                                   runtime_cfg)
        if self.post_processor:
            self.num_point_features = self.post_processor.num_point_features

    def forward(self, batch_dict):
        assert NotImplementedError
