import torch

from ...utils import box_utils
from .point_head_simple import PointHeadSimple


class PointSegHead(PointHeadSimple):
    """
    A simple point-based segmentation head, which are used for PV-RCNN keypoint segmentaion.
    Reference Paper: https://arxiv.org/abs/1912.13192
    PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection
    """
    def __init__(self, num_class, input_channels, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg,
                         input_channels=input_channels,
                         num_class=num_class)

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coords_for_seg']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)
        import ipdb; ipdb.set_trace()

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False
        )

        return targets_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion_for_seg']
        else:
            point_features = batch_dict['point_features_for_seg']
        point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)

        ret_dict = {
            'point_seg_cls_preds': point_cls_preds,
        }

        point_cls_scores = torch.sigmoid(point_cls_preds)
        batch_dict['point_seg_cls_scores'], _ = point_cls_scores.max(dim=-1)

        if self.training:
            ret_dict['point_seg_cls_labels'] = batch_dict['point_seg_labels']
        self.forward_ret_dict = ret_dict

        return batch_dict
