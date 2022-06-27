import torch
import torch.nn as nn
import torch.nn.functional as F

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils, loss_utils

class ReconstructionHeadTemplate(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.input_channels = input_channels
        self.point_feature_key = self.model_cfg.get("POINT_FEATURE_KEY", "point_features")

    def forward(self, **kwargs):
        raise NotImplementedError
