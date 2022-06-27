from .repr_learner_template import ReprLearnerTemplate
from pcdet.ops.pointnet2.pointnet2_batch.pointnet2_utils import (
    three_interpolate, three_nn
)
import torch
from pcdet.utils import polar_utils

class ImplicitReconstructor(ReprLearnerTemplate):
    def __init__(self, model_cfg, cfg, dataset):
        super().__init__(model_cfg=model_cfg, cfg=cfg, dataset=dataset)
        self.module_list = self.build_networks()
        self.num_pos = 0

    def forward(self, batch_dict):
        if self.vfe:
            batch_dict = self.vfe(batch_dict)

        batch_dict = self.backbone_3d(batch_dict)

        if self.head:
            batch_dict = self.head(batch_dict)
        
        #bxyz = batch_dict['point_bxyz']
        #top_lidar_origin = batch_dict['top_lidar_origin'][0]
        #_, polar, azimuth = polar_utils.cartesian2spherical(bxyz[:, 1:4]-top_lidar_origin[1:])
        #spherical = torch.stack([polar.new_zeros(polar.shape[0]), polar, azimuth], dim=-1)

        #bucket_mask = (polar > 1.6) & (polar <= 1.61) & \
        #              (azimuth <= 0.9) & (azimuth >= 0.85)
        #point_dist = (bxyz[:, 1:4] - top_lidar_origin[1:]).norm(p=2, dim=-1)
        #point_dir = (bxyz[:, 1:4] - top_lidar_origin[1:]) / point_dist[:, None]
        #batch_dict['polar'] = polar
        #batch_dict['azimuth'] = azimuth

        #rad_dist = torch.linspace(0, 70, 1000).to(bxyz)

        #implicit_points = []
        #for rad_d in rad_dist:
        #    mask = (rad_d < point_dist) & (rad_d > point_dist - 0.5)
        #    points_on_path = point_dir[mask] * rad_d + top_lidar_origin[1:]
        #    implicit_points.append(points_on_path)
        #implicit_points = torch.cat(implicit_points, dim=0)
        #implicit_points = torch.cat([
        #                             torch.zeros(implicit_points.shape[0], 1).to(implicit_points),
        #                             implicit_points,
        #                            ],
        #                            dim=-1)

        #import ipdb; ipdb.set_trace()
        #batch_dict['implicit_points'] = implicit_points
        #batch_dict['points_in_bucket'] = bxyz[bucket_mask, :4]
        #batch_dict['lidar_origin_to_bucket'] = torch.stack(
        #                                           [
        #                                            torch.arange(bucket_mask.long().sum().item()),
        #                                            torch.zeros(bucket_mask.long().sum().item()),
        #                                           ],
        #                                           dim=0).long()
        if self.visualizer:
            self.visualizer(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss,
                'occupancy_acc': tb_dict['occupancy_acc']
            }
            disp_dict['occupancy_acc'] = tb_dict['occupancy_acc']
            return ret_dict, tb_dict, disp_dict
        else:
            raise NotImplementedError

    def get_training_loss(self):
        disp_dict = {}
        loss_seg, tb_dict = self.head.get_loss()

        loss = loss_seg
        return loss, tb_dict, disp_dict
