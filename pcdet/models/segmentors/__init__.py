from .segmentor3d_template import Segmentor3DTemplate
from .spconv_unetv2_seg import SpconvUNetV2Seg
from .kpconv_seg import KPConvSeg 
from .hkconv_seg import HKConvSeg
from .simple_seg import SimpleSeg
from .hybrid_gnn_seg import HybridGNNSeg

__all__ = {
    'Segmentor3DTemplate': Segmentor3DTemplate,
    'SpconvUNetV2Seg': SpconvUNetV2Seg,
    'KPConvSeg': KPConvSeg,
    'HKConvSeg': HKConvSeg,
    'SimpleSeg': SimpleSeg,
    'PointNet2Seg': SimpleSeg,
    'HybridGNNSeg': HybridGNNSeg,
}


def build_segmentor(model_cfg, cfg, dataset):
    num_class = len(cfg.CLASS_NAMES)
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, runtime_cfg=cfg, dataset=dataset
    )

    return model
