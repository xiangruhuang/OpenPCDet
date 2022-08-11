import torch
from torch import nn
import torch.nn.functional as F

class SelfAttentionBlock(nn.Module):
    def __init__(self, block_cfg):
        super().__init__()
        in_channel = block_cfg["in_channel"]
        output_channel = block_cfg['out_channel']
        num_heads = block_cfg['num_heads']
        dropout = block_cfg.get('dropout', 0.0)
        self.self_attn = nn.MultiheadAttention(output_channel, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(output_channel, output_channel)
        self.linear2 = nn.Linear(output_channel, output_channel)
        self.norm1 = nn.BatchNorm1d(output_channel)
        self.norm2 = nn.BatchNorm1d(output_channel)
        self.activation = nn.LeakyReLU(0.2)
        self.mlp = nn.Sequential(
                       self.linear1,
                       self.activation,
                       self.dropout,
                       self.linear2,
                   )

    def forward(self, point_feat):
        point_feat_u = point_feat.unsqueeze(-2)
        output_point_feat = self.self_attn(point_feat_u, point_feat_u, 
                                           value=point_feat_u,
                                           need_weights=False)[0].squeeze(1)
        output_point_feat = point_feat + self.dropout1(output_point_feat)
        output_point_feat = self.norm1(output_point_feat)

        output_point_feat2 = self.mlp(output_point_feat)
        output_point_feat = output_point_feat + self.dropout2(output_point_feat2)
        output_point_feat = self.norm2(output_point_feat)

        return output_point_feat

