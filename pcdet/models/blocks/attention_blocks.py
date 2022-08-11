import torch
from torch import nn
import torch.nn.functional as F

class SelfAttentionBlock(nn.Module):
    def __init__(self, block_cfg):
        super().__init__()
        in_channel = block_cfg["in_channel"]
        output_channel = block_cfg['out_channel']
        num_heads = block_cfg['num_heads']
        self.self_attn = nn.MultiheadAttention(output_channel, num_heads)

    def forward(self, point_feat):
        point_feat_u = point_feat.unsqueeze(-2)
        output_point_feat = self.self_attn(point_feat_u, point_feat_u, 
                                           value=point_feat_u,
                                           need_weights=False)[0].squeeze(1)

        return output_point_feat

