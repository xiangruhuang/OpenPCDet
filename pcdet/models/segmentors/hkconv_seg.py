from .segmentor3d_template import Segmentor3DTemplate

class HKConvSeg(Segmentor3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.num_pos = 0

    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        if self.visualizer:
            self.visualizer(batch_dict)
        if self.backbone_3d:
            batch_dict = self.backbone_3d(batch_dict)
        if self.seg_head:
            batch_dict = self.seg_head(batch_dict)

        import ipdb; ipdb.set_trace()
        if self.visualizer:
            self.visualizer(batch_dict)
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            disp_dict.update({'num_pos': (batch_dict['gt_boxes'][:, :, 3] > 0.5).sum() / batch_dict['batch_size']})

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            if self.seg_head:
                iou_stats = self.seg_head.get_iou_statistics()
                return iou_stats, None
            else:
                return None, None

    def get_training_loss(self):
        disp_dict = {}
        loss = None
        tb_dict = {}
        if self.vfe:
            loss_vfe, tb_dict = self.vfe.get_loss(tb_dict)
            loss = loss+loss_vfe if loss else loss_vfe
        if self.seg_head:
            loss_seg, tb_dict = self.seg_head.get_loss(tb_dict)
            loss = loss+loss_seg if loss else loss_seg

        return loss, tb_dict, disp_dict
