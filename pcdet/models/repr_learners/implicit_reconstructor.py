from .repr_learner_template import ReprLearnerTemplate
from pcdet.ops.pointnet2.pointnet2_batch.pointnet2_utils import (
    three_interpolate, three_nn
)

class ImplicitReconstructor(ReprLearnerTemplate):
    def __init__(self, model_cfg, cfg, dataset):
        super().__init__(model_cfg=model_cfg, cfg=cfg, dataset=dataset)
        self.module_list = self.build_networks()
        self.num_pos = 0

    def forward(self, batch_dict):
        if self.vfe:
            batch_dict = self.vfe(batch_dict)

        #batch_dict = self.backbone_3d(batch_dict)

        if self.head:
            batch_dict = self.head(batch_dict)

        if self.visualizer:
            self.visualizer(batch_dict)
        import ipdb; ipdb.set_trace()

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            raise NotImplementedError

    def get_training_loss(self):
        disp_dict = {}
        loss_seg, tb_dict = self.head.get_loss()

        loss = loss_seg
        return loss, tb_dict, disp_dict
