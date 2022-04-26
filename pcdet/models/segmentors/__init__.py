from .segmentor3d_template import Segmentor3DTemplate
from .spconv_unetv2_seg import SpconvUNetV2Seg

__all__ = {
    'Segmentor3DTemplate': Segmentor3DTemplate,
    'SpconvUNetV2Seg': SpconvUNetV2Seg
}


def build_segmentor(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
