from .segmentor3d_template import Segmentor3DTemplate
import torch
from time import time
from torch_scatter import scatter

class Influence(Segmentor3DTemplate):
    def __init__(self, model_cfg, runtime_cfg, dataset):
        super().__init__(model_cfg=model_cfg, runtime_cfg=runtime_cfg, dataset=dataset)
        self.module_list = self.build_networks()
        self.num_pos = 0

    def forward(self, batch_dict):
        if self.vfe:
            batch_dict = self.vfe(batch_dict)

        if self.visualizer:
            self.visualizer(batch_dict)

        while True:
            x = int(input('Next point index:'))

            if self.backbone_3d:
                batch_dict = self.backbone_3d(batch_dict)
        
            point_prob = batch_dict['voxel_bxyz'].new_zeros(batch_dict['voxel_bcenter'].shape[0])
            point_prob[x] = 1.0
            
            point = batch_dict['points'][0]
            batch_dict[f'{point}_prob'] = point_prob
            batch_dict[f'{point}_mask'] = (point_prob > 0).long()
            for graph, point in zip(batch_dict['graphs'], batch_dict['points'][1:]):
                if graph.startswith('-'):
                    e_query, e_ref, _ = batch_dict[f'{graph[1:]}_graph']
                else:
                    e_ref, e_query, _ = batch_dict[f'{graph}_graph']

                point_prob = scatter(point_prob[e_ref], e_query, dim=0,
                                     dim_size=e_query.max().item()+1, reduce='sum')
                point_prob = point_prob.clamp(max=1e10)
                print(graph, point)
                if point is not None:
                    batch_dict[f'{point}_prob'] = point_prob
                    batch_dict[f'{point}_mask'] = (point_prob > 0).long()

            if self.visualizer:
                self.visualizer(batch_dict)

        if self.seg_head:
            batch_dict = self.seg_head(batch_dict)
            
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
            iou_stats, ret_dict = self.seg_head.get_iou_statistics()
            pred_dicts = self.seg_head.get_evaluation_results()
            for batch_idx, (pred_dict, iou_stat, frame_id) in enumerate(zip(pred_dicts, iou_stats, batch_dict['frame_id'])):
                pred_dict.update(iou_stat)
                pred_dict['frame_id'] = frame_id
                
            return pred_dicts, None

    def get_training_loss(self):
        disp_dict, tb_dict = {}, {}

        loss = 0.0
        if self.seg_head:
            loss_seg, tb_dict = self.seg_head.get_loss(tb_dict)
            loss = loss_seg
            

        return loss, tb_dict, disp_dict
