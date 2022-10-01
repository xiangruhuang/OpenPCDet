import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict

from pcdet.utils import common_utils
from pcdet.models.model_utils import graph_utils
#from pcdet.models.model_utils.sampler_utils import build_sampler
#from pcdet.models.model_utils.graph_utils import build_graph
#from pcdet.models.blocks.assigners import build_assigner

from .unet_template import UNetTemplate

def guess_size(ref):
    for key in ['bxyz', 'bcoords', 'bcenter']:
        if key in ref:
            return ref[key].shape[0]
    raise ValueError("ref has no valid coordinates")

def reverse_graphs(graphs):
    new_graphs = {}
    for key, graph in graphs.items():
        e_ref, e_query, e_weight = graph
        new_graphs[key] = e_query, e_ref, e_weight
    return new_graphs


class ANetV1(UNetTemplate):
    """A U-Net architecture, but alternating between points and grid centers
    
    """
    def __init__(self, runtime_cfg, model_cfg):
        super(ANetV1, self).__init__(runtime_cfg, model_cfg)
        
        #cur_channel = self.input_channels

        # EdgeConv Ops
        #self.point2grid_convs = nn.ModuleList()
        #self.grid2point_convs = nn.ModuleList()
        #self.grid2grid_convs = nn.ModuleList()
        #channel_stack = []
        #for i in range(self.num_down_modules):
        #    sc = [int(c*self.scale) for c in self.sa_channels[i]]

        #    # grid to point
        #    conv = self.build_edge_conv(cur_channel, sc)
        #    self.grid2point_convs.append(conv)

        #    # point to grid
        #    conv = build_edge_conv(cur_channel, sc)
        #    self.point2grid_convs.append(conv)

        #    # grid to grid
        #    self.build_grid_conv(cur_channel, sc, self.keys[i])
        #    self.grid2grid_convs.append(grid_convs)
        #    cur_channel = sc[-1]
        #    channel_stack.append(cur_channel)

        #self.up_modules = nn.ModuleList()
        #self.merge_modules = nn.ModuleList()
        #self.skip_modules = nn.ModuleList()
        #for i in range(self.num_up_modules):
        #    fc = [int(c*self.scale) for c in self.fp_channels[i]]
        #    fc0, fc1, fc2 = fc[0], fc[1], fc[-1]
        #    key0, key1, key2 = self.keys[-i-1][:3][::-1]
        #    skip_channel = channel_stack.pop()

        #    self.skip_modules.append(
        #        nn.ModuleList([
        #            GridConv(
        #                self.assigners[0],
        #                dict(
        #                    INPUT_CHANNEL=skip_channel,
        #                    OUTPUT_CHANNEL=fc0,
        #                    KEY=key0,
        #                    NORM_CFG=self.norm_cfg,
        #                    ACTIVATION=self.activation,
        #                ),
        #            ),
        #            *[GridConv(
        #                self.assigners[0],
        #                dict(
        #                    INPUT_CHANNEL=fc0,
        #                    OUTPUT_CHANNEL=fc0,
        #                    KEY=key0,
        #                    RELU=False,
        #                    NORM_CFG=self.norm_cfg,
        #                    ACTIVATION=self.activation,
        #                ),
        #            ) for _ in range(len(fc)-2)]
        #            ]
        #        ))

        #    self.merge_modules.append(
        #        GridConv(
        #            self.assigners[0],
        #            dict(
        #                INPUT_CHANNEL=fc0*2,
        #                OUTPUT_CHANNEL=fc1,
        #                NORM_CFG=self.norm_cfg,
        #                ACTIVATION=self.activation,
        #                KEY=key1,
        #            ),
        #        ))

        #    self.up_modules.append(
        #        GridConv(
        #            self.assigners[0],
        #            dict(
        #                INPUT_CHANNEL=fc1,
        #                OUTPUT_CHANNEL=fc2,
        #                NORM_CFG=self.norm_cfg,
        #                ACTIVATION=self.activation,
        #                KEY=key2,
        #            ),
        #        ))

        #    cur_channel = fc2

        self.num_point_features = self.up_convs[4].output_channel

    def build_grid_conv(self, input_channel, channels, keys):
        grid_convs = nn.ModuleList()
        for j, c in enumerate(channels):
            conv_cfg = dict(
                INPUT_CHANNEL=cur_channel,
                OUTPUT_CHANNEL=c,
                NORM_CFG=self.norm_cfg,
                ACTIVATION=self.activation,
                KEY=keys[j],
            )
            grid_conv = GridConv(
                self.assigners[0],
                conv_cfg
            )
            grid_convs.append(grid_conv)
            cur_channel = c
        return GridConv(conv_cfg)

    def conv_up(self, skip, i, graph, conv_dict):
        up_feat = skip.up_feat
        identity = skip.feat
        for skip_conv in self.skip_convs[i]:
            skip.feat, conv_dict = skip_conv(skip, skip, graph, conv_dict)
        skip.feat = F.relu(skip.feat + identity)

        concat = EasyDict(skip.copy())
        concat.feat = torch.cat([up_feat, skip.feat], dim=-1)
        merge_feat, conv_dict = self.merge_convs[i](concat, concat, graph, conv_dict)
        num_ref_points = guess_size(concat)
        concat.feat = merge_feat + concat.feat.view(num_ref_points, -1, 2).sum(dim=2)
        return concat, conv_dict

    def forward(self, batch_dict):
        points0 = EasyDict(dict())
        for attr in self.input_attributes:
            points0[attr] = batch_dict[f'{self.input_key}_{attr}']
        points0.bcenter = points0.bxyz

        grids1 = EasyDict(dict())
        for attr in self.input_attributes:
            grids1[attr] = batch_dict[f'voxel_{attr}']
        grids1.bcenter = batch_dict['voxel_bcenter']
        grids1.bcoords = batch_dict['voxel_bcoords']
        grids1.bxyz = grids1.bcenter

        runtime_dict = EasyDict(dict())

        # grid sampling
        grids2 = self.grid_samplers[0](grids1, runtime_dict)
        grids3 = self.grid_samplers[1](grids2, runtime_dict)
        grids4 = self.grid_samplers[2](grids3, runtime_dict)
        grids5 = self.grid_samplers[3](grids4, runtime_dict)
        grids6 = self.grid_samplers[4](grids5, runtime_dict)
        grids2.bxyz = grids2.bcenter
        grids3.bxyz = grids3.bcenter
        grids4.bxyz = grids4.bcenter
        grids5.bxyz = grids5.bcenter
        grids6.bxyz = grids6.bcenter

        # point sampling
        points1 = self.point_samplers[0](points0, runtime_dict)
        points2 = self.point_samplers[1](points1, runtime_dict)
        points3 = self.point_samplers[2](points2, runtime_dict)
        points4 = self.point_samplers[3](points3, runtime_dict)
        points5 = self.point_samplers[4](points4, runtime_dict)

        # graph building
        graphs = {}
        graphs['point2grid2'] = self.point2grid_graphs[0](points1, grids2)
        graphs['point2grid3'] = self.point2grid_graphs[1](points2, grids3)
        graphs['point2grid4'] = self.point2grid_graphs[2](points3, grids4)
        graphs['point2grid5'] = self.point2grid_graphs[3](points4, grids5)
        graphs['point2grid6'] = self.point2grid_graphs[4](points5, grids6)

        graphs['grid2point1'] = self.grid2point_graphs[0](grids1, points1)
        graphs['grid2point2'] = self.grid2point_graphs[1](grids2, points2)
        graphs['grid2point3'] = self.grid2point_graphs[2](grids3, points3)
        graphs['grid2point4'] = self.grid2point_graphs[3](grids4, points4)
        graphs['grid2point5'] = self.grid2point_graphs[4](grids5, points5)

        graphs['grid11'] = self.grid2grid_graphs[0](grids1, grids1)
        graphs['grid12'] = self.grid2grid_graphs[0](grids1, grids2)
        graphs['grid22'] = self.grid2grid_graphs[0](grids2, grids2)
        graphs['grid23'] = self.grid2grid_graphs[0](grids2, grids3)
        graphs['grid33'] = self.grid2grid_graphs[1](grids3, grids3)
        graphs['grid34'] = self.grid2grid_graphs[1](grids3, grids4)
        graphs['grid44'] = self.grid2grid_graphs[2](grids4, grids4)
        graphs['grid45'] = self.grid2grid_graphs[2](grids4, grids5)
        graphs['grid55'] = self.grid2grid_graphs[3](grids5, grids5)
        graphs['grid56'] = self.grid2grid_graphs[3](grids5, grids6)
        graphs['grid66'] = self.grid2grid_graphs[4](grids6, grids6)

        # computation
        points1.feat = self.grid2point_convs[0](grids1, points1, graphs['grid2point1'])
        
        grids2.feat = self.point2grid_convs[0](points1, grids2,  graphs['point2grid2'])
        points2.feat = self.grid2point_convs[1](grids2, points2, graphs['grid2point2'])
        
        grids3.feat = self.point2grid_convs[1](points2, grids3,  graphs['point2grid3'])
        points3.feat = self.grid2point_convs[2](grids3, points3, graphs['grid2point3'])

        grids4.feat = self.point2grid_convs[2](points3, grids4,  graphs['point2grid4'])
        points4.feat = self.grid2point_convs[3](grids4, points4, graphs['grid2point4'])

        grids5.feat = self.point2grid_convs[3](points4, grids5,  graphs['point2grid5'])
        points5.feat = self.grid2point_convs[4](grids5, points5, graphs['grid2point5'])
        
        grids6.feat = self.point2grid_convs[4](points5, grids6,  graphs['point2grid6'])

        reversed_graphs = reverse_graphs(graphs)
        grids6.up_feat = grids6.feat
        
        merge6, runtime_dict = self.conv_up(grids6, 0, reversed_graphs['grid66'], runtime_dict)
        grids5.up_feat, runtime_dict = self.up_convs[0](merge6, grids5, reversed_graphs['grid56'], runtime_dict)

        merge5, runtime_dict = self.conv_up(grids5, 1, reversed_graphs['grid55'], runtime_dict)
        grids4.up_feat, runtime_dict = self.up_convs[1](merge5, grids4, reversed_graphs['grid45'], runtime_dict)
        
        merge4, runtime_dict = self.conv_up(grids4, 2, reversed_graphs['grid44'], runtime_dict)
        grids3.up_feat, runtime_dict = self.up_convs[2](merge4, grids3, reversed_graphs['grid34'], runtime_dict)

        merge3, runtime_dict = self.conv_up(grids3, 3, reversed_graphs['grid33'], runtime_dict)
        grids2.up_feat, runtime_dict = self.up_convs[3](merge3, grids2, reversed_graphs['grid23'], runtime_dict)
        
        merge2, runtime_dict = self.conv_up(grids2, 4, reversed_graphs['grid22'], runtime_dict)
        grids1.up_feat, runtime_dict = self.up_convs[4](merge2, grids1, reversed_graphs['grid12'], runtime_dict)

        if self.output_key is not None:
            batch_dict[f'{self.output_key}_feat'] = grids1.up_feat
            batch_dict[f'{self.output_key}_bxyz'] = grids1.bcenter

        # update batch_dict
        for i, grids in enumerate([grids1, grids2, grids3, grids4, grids5, grids6]):
            key = f'grids{i+1}'
            for attr in grids.keys():
                batch_dict[f'{key}_{attr}'] = grids[attr]
        
        for i, points in enumerate([points0, points1, points2, points3, points4, points5]):
            key = f'points{i}'
            for attr in points.keys():
                batch_dict[f'{key}_{attr}'] = points[attr]

        for key, graph in graphs.items():
            e_ref, e_query, e_weight = graph
            batch_dict[f'{key}_edges'] = torch.stack([e_ref, e_query], dim=0)
            batch_dict[f'{key}_weight'] = e_weight
        
        return batch_dict
