from .detector3d_template import Detector3DTemplate


class PVRCNNPlusPlusCoTrain(Detector3DTemplate):
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
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)

        batch_dict = self.roi_head.proposal_layer(
            batch_dict, nms_config=self.roi_head.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.roi_head.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_targets_dict'] = targets_dict
            num_rois_per_scene = targets_dict['rois'].shape[1]
            if 'roi_valid_num' in batch_dict:
                batch_dict['roi_valid_num'] = [num_rois_per_scene for _ in range(batch_dict['batch_size'])]

        batch_dict = self.pfe_seg(batch_dict)
        batch_dict = self.pfe(batch_dict)

        if self.visualize:
            # visualize keypoints
            keypoints = batch_dict['point_coords']
            keypoint_labels = batch_dict['point_seg_labels'].long()
            labels = keypoint_labels.long().unique().detach().cpu()
            import numpy as np
            colors = np.random.randn(labels.max().item()-labels.min().item()+1, 3)

            for i in range(batch_dict['batch_size']):
                bs_keypoint_mask = keypoints[:, 0] == i
                keypoint = keypoints[bs_keypoint_mask, 1:4]
                keypoint_label = keypoint_labels[bs_keypoint_mask].detach().cpu()
                bs_point_mask = batch_dict['points'][:, 0] == i
                point = batch_dict['points'][bs_point_mask, 1:4]
                self.vis.pointcloud('points', point.detach().cpu())
                ps_kp = self.vis.pointcloud('keypoints', keypoint.detach().cpu())
                ps_kp.add_scalar_quantity('seg_labels', keypoint_label)
                ps_kp.add_color_quantity('segmentation', colors[keypoint_label-labels.min().item()])
                self.vis.show()
        batch_dict = self.point_head(batch_dict)
        batch_dict = self.seg_head(batch_dict)
        batch_dict = self.roi_head(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            disp_dict.update({'num_pos': (batch_dict['gt_boxes'][:, :, 3] > 0.5).sum() / batch_dict['batch_size']})

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        if self.point_head is not None:
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        else:
            loss_point = 0
        if self.seg_head is not None:
            loss_seg, tb_dict = self.seg_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn + loss_seg
        return loss, tb_dict, disp_dict
