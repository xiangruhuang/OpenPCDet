from .pointnet2repsurf_backbone import PointNet2RepSurf
from .pointnet2 import PointNet2
from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x
from .spconv_unet import UNetV2
from .kpconv import KPConv
#from .hybrid_gnn_backbone import HybridGNN
from .sst_backbone import SST

__all__ = {
    'VoxelBackBone8x': VoxelBackBone8x,
    'UNetV2': UNetV2,
    'PointNet2RepSurf': PointNet2RepSurf,
    'PointNet2': PointNet2,
    'VoxelResBackBone8x': VoxelResBackBone8x,
    'KPConv': KPConv,
#    'HybridGNN': HybridGNN,
    'SST': SST,
}
