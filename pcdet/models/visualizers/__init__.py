from .polyscope_visualizer import PolyScopeVisualizer

__all__ = {
    'PolyScopeVisualizer': PolyScopeVisualizer,
}

def build_visualizer(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
