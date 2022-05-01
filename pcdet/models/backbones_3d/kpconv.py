from torch import nn
import torch
from .kpconv_blocks import KPConvBasicBlock

class KPConv(nn.Module):
    def __init__(self,
                 model_cfg,
                 input_channels,
                 grid_size,
                 voxel_size,
                 point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg
        
        input_channels = input_channels
        output_channels = model_cfg.get("OUTPUT_CHANNELS", None)
        kernel_influence_dist = model_cfg.get("KERNEL_INFLUENCE_DIST", None)
        num_kernel_points = model_cfg.get("NUM_KERNEL_POINTS", None)
        fixed = model_cfg.get("FIXED", "center")
        KP_influence = model_cfg.get("KP_INFLUENCE", "linear")
        aggr_mode = model_cfg.get("AGGR_MODE", "sum")
        add_one = model_cfg.get("ADD_ONE", False)
        self.kpconv = KPConvBasicBlock(
                        input_channels,
                        output_channels,
                        kernel_influence_dist,
                        num_kernel_points,
                        fixed,
                        KP_influence,
                        aggr_mode,
                        add_one
                      )

    def forward(self, batch_dict):
        import ipdb; ipdb.set_trace()
        pass
