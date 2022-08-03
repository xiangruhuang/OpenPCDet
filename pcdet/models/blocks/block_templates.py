import torch
from torch import nn
from pcdet.models.model_utils.sampler_utils import SAMPLERS
from pcdet.models.model_utils.grouper_utils import GROUPERS

class DownBlockTemplate(nn.Module):
    def __init__(self, block_cfg, sampler_cfg, grouper_cfg):
        super().__init__()
        if sampler_cfg is not None:
            sampler = SAMPLERS[sampler_cfg.pop("TYPE")]
            self.sampler = sampler(
                               runtime_cfg=None,
                               model_cfg=sampler_cfg,
                           )
        
        if grouper_cfg is not None:
            grouper = GROUPERS[grouper_cfg.pop("TYPE")]
            self.grouper = grouper(
                               runtime_cfg=None,
                               model_cfg=grouper_cfg,
                           )

    def forward(self, ref_bxyz, ref_feat):
        assert NotImplementedError

class UpBlockTemplate(nn.Module):
    def __init__(self,
                 block_cfg,
                 grouper_cfg=dict(
                     TYPE="KNNGrouper",
                     NUM_NEIGHBORS=3,
                 ),
                ):
        super().__init__()
        
        grouper = GROUPERS[grouper_cfg["TYPE"]]
        self.grouper = grouper(
                           runtime_cfg=None,
                           model_cfg=grouper_cfg,
                       )
    
    def forward(self, ref_bxyz, ref_feat,
                query_bxyz, query_feat,
                e_ref, e_query):
        assert NotImplementedError
