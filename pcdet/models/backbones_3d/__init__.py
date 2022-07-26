from .pointnet2repsurf_backbone import PointNet2MSG, PointNet2RepSurf
from .graphconv_backbone import GraphConv
from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x
from .spconv_unet import UNetV2
from .kpconv import KPConv
#from .hybrid_gnn_backbone import HybridGNN
from .sst_backbone import SST

__all__ = {
    'VoxelBackBone8x': VoxelBackBone8x,
    'UNetV2': UNetV2,
    'PointNet2RepSurf': PointNet2RepSurf,
    'GraphConv': GraphConv,
    'PointNet2MSG': PointNet2MSG,
    'VoxelResBackBone8x': VoxelResBackBone8x,
    'KPConv': KPConv,
#    'HybridGNN': HybridGNN,
    'SST': SST,
}
