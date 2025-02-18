import torch

from .detector3d_template import Detector3DTemplate

class CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, runtime_cfg, dataset):
        super().__init__(model_cfg=model_cfg, runtime_cfg=runtime_cfg, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        batch_dict['points'] = torch.cat([batch_dict['point_bxyz'][:, 1:], batch_dict['point_feat']], dim=-1)
        batch_dict['voxel_points'] = torch.cat([batch_dict['voxel_point_xyz'], batch_dict['voxel_point_feat']], dim=-1)
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            for pred_dict, frame_id in zip(pred_dicts, batch_dict['frame_id'].reshape(-1)):
                pred_dict['scene_wise']['frame_id'] = frame_id
            pred_dicts
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dicts = batch_dict['final_box_dicts']
        pred_dicts = []
        for i, final_pred_dict in enumerate(final_pred_dicts):
            pred_dict = dict(
                point_wise=dict(),
                object_wise=dict(
                    pred_box_attr=final_pred_dict['pred_boxes'],
                    pred_box_scores=final_pred_dict['pred_scores'],
                    pred_box_cls_label=final_pred_dict['pred_labels'],
                ),
                scene_wise=dict(
                    frame_id=batch_dict['frame_id'].reshape(-1)[i],
                )
            )
            pred_dicts.append(pred_dict)
                
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dicts[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return pred_dicts, recall_dict
