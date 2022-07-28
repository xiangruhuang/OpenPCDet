from .segmentor3d_template import Segmentor3DTemplate
import torch
from time import time

class SimpleSeg(Segmentor3DTemplate):
    def __init__(self, model_cfg, runtime_cfg, dataset):
        super().__init__(model_cfg=model_cfg, runtime_cfg=runtime_cfg, dataset=dataset)
        self.module_list = self.build_networks()
        self.num_pos = 0

    def forward(self, batch_dict):
        if self.vfe:
            batch_dict = self.vfe(batch_dict)

        if self.backbone_3d:
            batch_dict = self.backbone_3d(batch_dict)

        if self.seg_head:
            batch_dict = self.seg_head(batch_dict)
            
        if self.visualizer:
            self.visualizer(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            disp_dict.update({'num_pos': (batch_dict['gt_boxes'][:, :, 3] > 0.5).sum().item() / batch_dict['batch_size']})

            tb_dict['metadata/max_memory_allocated_in_GB'] = torch.cuda.max_memory_allocated() / 2**30
            disp_dict['mem'] = torch.cuda.max_memory_allocated() / 2**30
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            iou_stats = self.seg_head.get_iou_statistics()
                
            return iou_stats, None

    def get_training_loss(self):
        disp_dict, tb_dict = {}, {}

        loss = 0.0
        if self.seg_head:
            loss_seg, tb_dict = self.seg_head.get_loss(tb_dict)
            loss = loss_seg
            

        return loss, tb_dict, disp_dict
