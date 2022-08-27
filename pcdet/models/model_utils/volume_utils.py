import torch
from torch import nn
import numpy as np

from torch_scatter import scatter
from torch_cluster import grid_cluster
from easydict import EasyDict

from pcdet.ops.pointops.functions.pointops import (
    furthestsampling,
    sectorized_fps,
)
from pcdet.ops.voxel.voxel_modules import VoxelAggregation
from .graph_utils import VoxelGraph
from .misc_utils import bxyz_to_xyz_index_offset
from pcdet.utils import common_utils

class VolumeTemplate(nn.Module):
    def __init__(self, runtime_cfg, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg

    def forward(self, ref, runtime_dict=None):
        raise NotImplementedError
    

class PCAVolume(VolumeTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        super(PCAVolume, self).__init__(runtime_cfg, model_cfg)
        model_cfg_cp = {}
        model_cfg_cp.update(model_cfg)
        self.voxel_graph = VoxelGraph(model_cfg=model_cfg_cp, runtime_cfg=runtime_cfg)

    def forward(self, ref, runtime_dict=None):
        if runtime_dict is not None:
            if 'base_bxyz' in runtime_dict:
                # computing volumes
                base_bxyz = runtime_dict['base_bxyz']
                
                e_base, e_voxel = self.voxel_graph(base_bxyz, ref.bcenter)

                num_voxels = ref.bcenter.shape[0]

                ref.bxyz = scatter(base_bxyz[e_base], e_voxel, dim=0,
                                   dim_size=num_voxels, reduce='sum')
                ref.volume = scatter(torch.ones_like(e_base), e_voxel, dim=0,
                                     dim_size=num_voxels, reduce='sum')
                mask = (ref.volume > 0.5)
                ref.volume_mask = mask

                ref.bxyz[mask] = ref.bxyz[mask] / ref.volume[mask, None]
                ref.bxyz[~mask] = ref.bcenter[~mask]
                point_d = base_bxyz[e_base, 1:] - ref.bxyz[e_voxel, 1:]
                point_ddT = point_d.unsqueeze(-1) * point_d.unsqueeze(-2)
                voxel_ddT = scatter(point_ddT, e_voxel, dim=0,
                                    dim_size=num_voxels, reduce='mean')

                voxel_eigvals, voxel_eigvecs = np.linalg.eigh(voxel_ddT.detach().cpu().numpy())
                voxel_eigvals = torch.from_numpy(voxel_eigvals).to(voxel_ddT)
                voxel_eigvecs = torch.from_numpy(voxel_eigvecs).to(voxel_ddT)
                ref.eigvals = voxel_eigvals
                ref.eigvecs = voxel_eigvecs
                #import polyscope as ps; ps.init(); ps.set_up_dir('z_up')

                #ps.register_point_cloud('voxel_centers', ref.bcenter[:, 1:].detach().cpu().numpy(), radius=2e-4)
                #ps.register_point_cloud('base', base_bxyz[:, 1:].detach().cpu().numpy(), radius=2e-4)
                #centers = torch.cat([ref.bcenter, base_bxyz], dim=0)[:, 1:].detach().cpu().numpy()
                #edges = torch.stack([e_base+num_voxels, e_voxel], dim=-1).detach().cpu().numpy()
                #ps.register_curve_network('edges', centers, edges, radius=2e-5)
                #ps.show()
                #import ipdb; ipdb.set_trace()
        
        return ref

        
VOLUMES = {
    'PCAVolume': PCAVolume,
}
