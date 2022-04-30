from .segmentor3d_template import Segmentor3DTemplate
from pcdet.ops.pointnet2.pointnet2_batch.pointnet2_utils import (
    three_interpolate,
    three_nn
)

class SpconvUNetV2Seg(Segmentor3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.num_pos = 0
        self.visualize = model_cfg.get('VISUALIZE', False)
        if self.visualize:
            from pcdet.utils import Visualizer
            self.vis = Visualizer()

    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.seg_head(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ious = self.seg_head.get_per_class_iou(batch_dict)

            disp_dict.update({'num_pos': (batch_dict['gt_boxes'][:, :, 3] > 0.5).sum() / batch_dict['batch_size']})
            for i in range(self.seg_head.num_class):
                tb_dict.update({f'IoU_{i}': ious[i]})

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts = self.seg_head.get_evaluation_results(batch_dict)
                
            return pred_dicts, None

    def get_training_loss(self):
        disp_dict = {}
        loss_seg, tb_dict = self.seg_head.get_loss()

        loss = loss_seg
        return loss, tb_dict, disp_dict
