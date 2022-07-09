from .repr_learner_template import ReprLearnerTemplate
from .simple_repr_learner import SimpleReprLearner
from .sequence_repr_learner import SequenceReprLearner

__all__ = {
    'ReprLearnerTemplate': ReprLearnerTemplate,
    'SimpleReprLearner': SimpleReprLearner,
    'SequenceReprLearner': SequenceReprLearner,

}


def build_repr_learner(model_cfg, runtime_cfg, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, runtime_cfg=runtime_cfg, dataset=dataset
    )

    return model
