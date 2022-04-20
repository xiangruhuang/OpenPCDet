from .detector3d_template import Detector3DTemplate
from .pv_rcnn_plusplus import PVRCNNPlusPlus

__all__ = {
    'Segmentor3DTemplate': Segmentor3DTemplate,
    'PVRCNNPlusPlus': PVRCNNPlusPlus
}


def build_segmentor(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
