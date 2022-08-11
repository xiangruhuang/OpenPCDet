from .segmentor3d_template import Segmentor3DTemplate
import torch
from time import time

class HierarchicalSeg(Segmentor3DTemplate):
    def __init__(self, model_cfg, runtime_cfg, dataset):
        super().__init__(model_cfg=model_cfg, runtime_cfg=runtime_cfg, dataset=dataset)
        self.module_list = self.build_networks()
        self.num_pos = 0
        self.phase = 2

    def forward(self, batch_dict):
        if self.vfe:
            batch_dict = self.vfe(batch_dict)

        if self.backbone_3d:
            batch_dict = self.backbone_3d(batch_dict)

        if self.seg_head:
            batch_dict = self.seg_head(batch_dict)

        if self.phase == 2:
            if self.group_backbones:
                merged_group_feat = torch.zeros_like(batch_dict[f'pointnet2_out_feat'])
                for idx, (group, group_backbone) in enumerate(zip(self.groups, self.group_backbones)):
                    pred_labels = batch_dict['pred_seg_cls_labels']
                    group_mask = torch.zeros_like(pred_labels).bool()
                    for cls in group:
                        group_mask |= (pred_labels == cls)
                    if not group_mask.any():
                        continue
                    point_bxyz = batch_dict['point_bxyz']
                    point_feat = batch_dict['pointnet2_out_feat']
                    batch_dict[f'point_group{idx}_bxyz'] = point_bxyz[group_mask]
                    batch_dict[f'point_group{idx}_feat'] = point_feat[group_mask]
                    batch_dict = group_backbone(batch_dict)
                    merged_group_feat[group_mask] = batch_dict[f'point_group_out{idx}_feat']
                batch_dict['point_group_out_feat'] = \
                        torch.cat([
                            batch_dict['pointnet2_out_feat'],
                            merged_group_feat
                        ], dim=-1)

            if self.post_seg_head:
                self.post_seg_head(batch_dict)

        if self.visualizer:
            self.visualizer(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            tb_dict['metadata/max_memory_allocated_in_GB'] = torch.cuda.max_memory_allocated() / 2**30
            disp_dict['mem'] = torch.cuda.max_memory_allocated() / 2**30
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            if self.phase == 2:
                iou_stats, ret_dict = self.post_seg_head.get_iou_statistics()
                pred_dicts = self.post_seg_head.get_evaluation_results()
                for pred_dict, iou_stat, frame_id in zip(pred_dicts, iou_stats, batch_dict['frame_id']):
                    pred_dict.update(iou_stat)
                    pred_dict['frame_id'] = frame_id
            else:
                iou_stats, ret_dict = self.seg_head.get_iou_statistics()
                pred_dicts = self.seg_head.get_evaluation_results()
                for pred_dict, iou_stat, frame_id in zip(pred_dicts, iou_stats, batch_dict['frame_id']):
                    pred_dict.update(iou_stat)
                    pred_dict['frame_id'] = frame_id

                
            return pred_dicts, None

    def get_training_loss(self):
        disp_dict, tb_dict = {}, {}

        loss = 0.0
        if self.seg_head:
            loss_seg, tb_dict = self.seg_head.get_loss(tb_dict, prefix='seg_head')
            loss = loss_seg
        
        if self.phase == 2:
            if self.post_seg_head:
                loss_post_seg, tb_dict = self.post_seg_head.get_loss(tb_dict, prefix='post_seg_head')
                loss += loss_post_seg

        return loss, tb_dict, disp_dict
