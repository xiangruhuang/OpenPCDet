from torch import nn
import torch
from .kpconv_blocks import SimpleBlock, KPDualBlock, FPBlockUp
from pcdet.ops.torch_hash.torch_hash_modules import RadiusGraph

class KPConv(nn.Module):
    def __init__(self,
                 model_cfg,
                 input_channels,
                 grid_size,
                 voxel_size,
                 point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg
        down_conv_cfg = model_cfg["down_conv"]
        max_num_points = model_cfg.get("MAX_NUM_POINTS", 200000)
        max_num_neighbors = model_cfg.get("MAX_NUM_NEIGHBORS", 38)
        self.neighbor_finder = RadiusGraph(
                                   max_num_neighbors,
                                   max_num_points=max_num_points
                               )

        self.build_down_conv(down_conv_cfg)
        
        #input_channels = input_channels
        #output_channels = model_cfg.get("OUTPUT_CHANNELS", None)
        #kernel_influence_dist = model_cfg.get("KERNEL_INFLUENCE_DIST", None)
        #num_kernel_points = model_cfg.get("NUM_KERNEL_POINTS", None)
        #fixed = model_cfg.get("FIXED", "center")
        #KP_influence = model_cfg.get("KP_INFLUENCE", "linear")
        #aggr_mode = model_cfg.get("AGGR_MODE", "sum")
        #add_one = model_cfg.get("ADD_ONE", False)
        #max_num_points = model_cfg.get("MAX_NUM_POINTS", 200000)
        #max_num_neighbors = model_cfg.get("MAX_NUM_NEIGHBORS", 32)
        #blocks = []
        #prev_grid_size = -1
        #self.blocks = nn.Sequential()
        #for i, (channel, grid_size) in enumerate(zip(channels, grid_sizes)):
        #    if prev_grid_size == -1:
        #        prev_grid_size = grid_size
        #    block = SimpleBlock(channel, grid_size, prev_grid_size)
        #    self.blocks.add_module(i, block)
        #    
        #self.radius = model_cfg.get("RADIUS", None)
        #self.radius_graph = RadiusGraph(
        #                        max_num_neighbors,
        #                        ndim=3,
        #                        max_num_points=max_num_points,
        #                    )

        self.num_point_features = 128
        self.backbone_channels = {}

    def build_down_conv(self, cfg):
        max_num_neighbors = cfg["max_num_neighbors"]
        channels = cfg["channels"]
        grid_size = cfg["grid_size"]
        prev_grid_size = cfg["prev_grid_size"]
        block_names = cfg["block_names"]
        has_bottleneck = cfg["has_bottleneck"]
        bn_momentum = cfg["bn_momentum"]
        num_down_modules = len(channels)
        down_modules = nn.ModuleList()
        for i in range(num_down_modules):
            block = KPDualBlock(
                        block_names[i],
                        channels[i],
                        grid_size[i],
                        prev_grid_size[i],
                        has_bottleneck[i],
                        max_num_neighbors[i],
                        neighbor_finder=self.neighbor_finder,
                        bn_momentum=bn_momentum[i]
                    )
            down_modules.append(block)
        self.down_modules = down_modules

    def build_up_conv(self, cfg):
        channels = cfg["channels"]
        up_k = cfg["up_k"]
        bn_momentum = cfg["bn_momentum"]
        num_up_modules = len(channels)
        up_modules = nn.ModuleList()
        for i in range(num_up_modules):
            block = FPBlockUp(
                        channels[i],
                        neighbor_finder=self.neighbor_finder,
                        up_k=up_k[i],
                        bn_momentum=bn_momentum[i],
                    )
            up_modules.append(block)
        self.up_modules = up_modules


    def forward(self, batch_dict):
        points = batch_dict["points"][:, :4].contiguous()
        point_features = batch_dict['points'][:, 1:].contiguous()
        batch_dict = dict(
            points = [points],
            point_features = [point_features]
        )
        for i in range(len(self.down_modules)):
            batch_dict = self.down_modules[i](batch_dict)

        for i in range(len(self.up_modules)):
            batch_dict = self.up_modules[i](batch_dict, stack_down.pop())

        pos = batch_dict['points'][:, :4].contiguous()
        edge_indices = self.radius_graph(pos, pos, self.radius)
        batch_dict['edge_indices'] = edge_indices
        batch_dict['pos'] = pos
        batch_dict['pos0'] = pos0

        return batch_dict
