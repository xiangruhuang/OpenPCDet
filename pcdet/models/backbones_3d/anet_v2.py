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


class ANetV2(UNetTemplate):
    """A U-Net architecture, but alternating between points and grid centers
    
    """
    def __init__(self, runtime_cfg, model_cfg):
        super(ANetV2, self).__init__(runtime_cfg, model_cfg)
        
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

    def grid_conv_down(self, grids, graphs, i, conv_dict):
        grids[i+2].feat, conv_dict = self.grid2grid_convs[i][0](grids[i+1], grids[i+2], graphs[f'grid{i+1}{i+2}'], conv_dict)
        grids[i+2].feat, conv_dict = self.grid2grid_convs[i][1](grids[i+2], grids[i+2], graphs[f'grid{i+2}{i+2}'], conv_dict)
        grids[i+2].feat, conv_dict = self.grid2grid_convs[i][2](grids[i+2], grids[i+2], graphs[f'grid{i+2}{i+2}'], conv_dict)
        return grids[i+2].feat, conv_dict

    def conv_down_merge(self, points, grids, graphs, i, conv_dict):
        point2grid_feat = self.point2grid_convs[i](points[i+1], grids[i+2], graphs[f'point2grid{i+2}'])
        grid2grid_feat, conv_dict = self.grid_conv_down(grids, graphs, i, conv_dict)
        grids[i+2].feat = torch.cat([grid2grid_feat, point2grid_feat], dim=-1)
        grids[i+2].feat, conv_dict = self.down_merge_convs[i](grids[i+2], grids[i+2], graphs[f'grid{i+2}{i+2}'], conv_dict)
        return grids[i+2].feat, conv_dict

    def forward(self, batch_dict):
        points0 = EasyDict(dict())
        for attr in self.input_attributes:
            points0[attr] = batch_dict[f'{self.input_key}_{attr}']
        points0.bcenter = points0.bxyz
        points = {0: points0}


        grids1 = EasyDict(dict())
        for attr in self.input_attributes:
            grids1[attr] = batch_dict[f'voxel_{attr}']
        grids1.bcenter = batch_dict['voxel_bcenter']
        grids1.bcoords = batch_dict['voxel_bcoords']
        grids1.bxyz = grids1.bcenter
        grids = {1: grids1}

        runtime_dict = EasyDict(dict())

        # grid sampling
        for i in range(2, 7):
            grids[i] = self.grid_samplers[i-2](grids[i-1], runtime_dict)
            grids[i].bxyz = grids[i].bcenter
        #grids2 = self.grid_samplers[0](grids1, runtime_dict)
        #grids3 = self.grid_samplers[1](grids2, runtime_dict)
        #grids4 = self.grid_samplers[2](grids3, runtime_dict)
        #grids5 = self.grid_samplers[3](grids4, runtime_dict)
        #grids6 = self.grid_samplers[4](grids5, runtime_dict)
        #grids2.bxyz = grids2.bcenter
        #grids3.bxyz = grids3.bcenter
        #grids4.bxyz = grids4.bcenter
        #grids5.bxyz = grids5.bcenter
        #grids6.bxyz = grids6.bcenter

        # point sampling
        for i in range(1, 6):
            points[i] = self.point_samplers[i-1](points[i-1], runtime_dict)
        #points1 = self.point_samplers[0](points0, runtime_dict)
        #points2 = self.point_samplers[1](points1, runtime_dict)
        #points3 = self.point_samplers[2](points2, runtime_dict)
        #points4 = self.point_samplers[3](points3, runtime_dict)
        #points5 = self.point_samplers[4](points4, runtime_dict)

        # graph building
        graphs = {}
        #for i in range(2, 7):
        #    graphs[f'point2grid{i}'] = self.point2grid_graphs[i-2](points[i-1], grids[i])
        #graphs['point2grid2'] = self.point2grid_graphs[0](points1, grids2)
        #graphs['point2grid3'] = self.point2grid_graphs[1](points2, grids3)
        #graphs['point2grid4'] = self.point2grid_graphs[2](points3, grids4)
        #graphs['point2grid5'] = self.point2grid_graphs[3](points4, grids5)
        #graphs['point2grid6'] = self.point2grid_graphs[4](points5, grids6)

        #for i in range(1, 6):
        #    graphs[f'grid2point{i}'] = self.grid2point_graphs[i-1](grids[i], points[i])
        #graphs['grid2point1'] = self.grid2point_graphs[0](grids1, points1)
        #graphs['grid2point2'] = self.grid2point_graphs[1](grids2, points2)
        #graphs['grid2point3'] = self.grid2point_graphs[2](grids3, points3)
        #graphs['grid2point4'] = self.grid2point_graphs[3](grids4, points4)
        #graphs['grid2point5'] = self.grid2point_graphs[4](grids5, points5)

        graphs['grid11'] = self.grid2grid_graphs[0](grids[1], grids[1])
        graphs['grid12'] = self.grid2grid_graphs[0](grids[1], grids[2])
        graphs['grid22'] = self.grid2grid_graphs[0](grids[2], grids[2])
        graphs['grid23'] = self.grid2grid_graphs[0](grids[2], grids[3])
        graphs['grid33'] = self.grid2grid_graphs[1](grids[3], grids[3])
        graphs['grid34'] = self.grid2grid_graphs[1](grids[3], grids[4])
        graphs['grid44'] = self.grid2grid_graphs[2](grids[4], grids[4])
        graphs['grid45'] = self.grid2grid_graphs[2](grids[4], grids[5])
        graphs['grid55'] = self.grid2grid_graphs[3](grids[5], grids[5])
        graphs['grid56'] = self.grid2grid_graphs[3](grids[5], grids[6])
        graphs['grid66'] = self.grid2grid_graphs[4](grids[6], grids[6])

        # computation
        for i in range(2, 7):
            grids[i].feat, runtime_dict = self.grid2grid_convs[i-2][0](grids[i-1], grids[i], graphs[f'grid{i-1}{i}'], runtime_dict)
            grids[i].feat, runtime_dict = self.grid2grid_convs[i-2][1](grids[i], grids[i], graphs[f'grid{i}{i}'], runtime_dict)
            grids[i].feat, runtime_dict = self.grid2grid_convs[i-2][2](grids[i], grids[i], graphs[f'grid{i}{i}'], runtime_dict)

        reversed_graphs = reverse_graphs(graphs)
        grids[6].up_feat = grids[6].feat
        
        merge6, runtime_dict = self.conv_up(grids[6], 0, reversed_graphs['grid66'], runtime_dict)
        grids[5].up_feat, runtime_dict = self.up_convs[0](merge6, grids[5], reversed_graphs['grid56'], runtime_dict)

        merge5, runtime_dict = self.conv_up(grids[5], 1, reversed_graphs['grid55'], runtime_dict)
        grids[4].up_feat, runtime_dict = self.up_convs[1](merge5, grids[4], reversed_graphs['grid45'], runtime_dict)
        
        merge4, runtime_dict = self.conv_up(grids[4], 2, reversed_graphs['grid44'], runtime_dict)
        grids[3].up_feat, runtime_dict = self.up_convs[2](merge4, grids[3], reversed_graphs['grid34'], runtime_dict)

        merge3, runtime_dict = self.conv_up(grids[3], 3, reversed_graphs['grid33'], runtime_dict)
        grids[2].up_feat, runtime_dict = self.up_convs[3](merge3, grids[2], reversed_graphs['grid23'], runtime_dict)
        
        merge2, runtime_dict = self.conv_up(grids[2], 4, reversed_graphs['grid22'], runtime_dict)
        grids[1].up_feat, runtime_dict = self.up_convs[4](merge2, grids[1], reversed_graphs['grid12'], runtime_dict)

        if self.output_key is not None:
            batch_dict[f'{self.output_key}_feat'] = grids[1].up_feat
            batch_dict[f'{self.output_key}_bxyz'] = grids[1].bcenter

        # update batch_dict
        for i, grids in grids.items():
            key = f'grids{i}'
            for attr in grids.keys():
                batch_dict[f'{key}_{attr}'] = grids[attr]
        
        for i, points in points.items():
            key = f'points{i}'
            for attr in points.keys():
                batch_dict[f'{key}_{attr}'] = points[attr]

        for key, graph in graphs.items():
            e_ref, e_query, e_weight = graph
            batch_dict[f'{key}_edges'] = torch.stack([e_ref, e_query], dim=0)
            batch_dict[f'{key}_weight'] = e_weight
        
        return batch_dict
