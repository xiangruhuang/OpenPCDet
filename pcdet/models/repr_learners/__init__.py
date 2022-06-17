from .repr_learner_template import ReprLearnerTemplate
from .implicit_reconstructor import ImplicitReconstructor

__all__ = {
    'ReprLearnerTemplate': ReprLearnerTemplate,
    'ImplicitReconstructor': ImplicitReconstructor,
}


def build_repr_learner(model_cfg, cfg, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, cfg=cfg, dataset=dataset
    )

    return model
