import torch
from torch import nn
from pcdet.models.model_utils.sampler_utils import SAMPLERS
from pcdet.models.model_utils.graph_utils import GRAPHS
from pcdet.models.model_utils.grouper_utils import GROUPERS
from pcdet.models.model_utils.fusion_utils import FUSIONS

class DownBlockTemplate(nn.Module):
    def __init__(self, block_cfg, sampler_cfg, graph_cfg, grouper_cfg=None, fusion_cfg=None):
        super().__init__()
        if sampler_cfg is not None:
            sampler = SAMPLERS[sampler_cfg.pop("TYPE")]
            self.sampler = sampler(
                               runtime_cfg=None,
                               model_cfg=sampler_cfg,
                           )
        
        if graph_cfg is not None:
            graph = GRAPHS[graph_cfg["TYPE"]]
            self.graph = graph(
                             runtime_cfg=None,
                             model_cfg=graph_cfg,
                         )
        
        if grouper_cfg is not None:
            grouper = GROUPERS[grouper_cfg.pop("TYPE")]
            self.grouper = grouper(
                               runtime_cfg=None,
                               model_cfg=grouper_cfg,
                           )
        
        if fusion_cfg is not None:
            fusion = FUSIONS[fusion_cfg.pop("TYPE")]
            self.fusion = fusion(
                              runtime_cfg=None,
                              model_cfg=fusion_cfg,
                          )

    def forward(self, ref_bxyz, ref_feat):
        assert NotImplementedError

class UpBlockTemplate(nn.Module):
    def __init__(self,
                 block_cfg,
                 graph_cfg=dict(
                     TYPE="KNNGraph",
                     NUM_NEIGHBORS=3,
                 ),
                ):
        super().__init__()
        
        graph = GRAPHS[graph_cfg["TYPE"]]
        self.graph = graph(
                         runtime_cfg=None,
                         model_cfg=graph_cfg,
                     )
    
    def forward(self, ref_bxyz, ref_feat,
                query_bxyz, query_feat,
                e_ref, e_query):
        assert NotImplementedError
