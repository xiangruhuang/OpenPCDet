from .segmentor3d_template import Segmentor3DTemplate
from .spconv_unetv2_seg import SpconvUNetV2Seg
from .kpconv_seg import KPConvSeg 
from .hkconv_seg import HKConvSeg
from .simple_seg import SimpleSeg
from .influence import Influence
from .hierarchical_seg import HierarchicalSeg
from .sequence_seg import SequenceSeg
from .hybrid_gnn_seg import HybridGNNSeg

__all__ = {
    'Segmentor3DTemplate': Segmentor3DTemplate,
    'SpconvUNetV2Seg': SpconvUNetV2Seg,
    'KPConvSeg': KPConvSeg,
    'HKConvSeg': HKConvSeg,
    'SimpleSeg': SimpleSeg,
    'HierarchicalSeg': HierarchicalSeg,
    'SequenceSeg': SequenceSeg,
    'PointNet2Seg': SimpleSeg,
    'HybridGNNSeg': HybridGNNSeg,
    "Influence": Influence
}


def build_segmentor(model_cfg, runtime_cfg, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, runtime_cfg=runtime_cfg, dataset=dataset
    )

    return model
