import torch
from torch import nn

from pcdet.ops.pointops.functions.pointops import (
    furthestsampling,
    sectorized_fps,
)

@torch.no_grad()
def bxyz_to_xyz_index_offset(point_bxyz):
    num_points = []
    batch_size = point_bxyz[:, 0].max().round().long().item() + 1
    for i in range(batch_size):
        num_points.append((point_bxyz[:, 0].round().long() == i).int().sum())
    num_points = torch.stack(num_points, dim=0).reshape(-1).int()
    _, indices = torch.sort(point_bxyz[:, 0])
    offset = num_points.cumsum(dim=0).int()
    point_xyz = point_bxyz[indices, 1:4].contiguous()
    return point_xyz, indices.long(), offset

class SamplerTemplate(nn.Module):
    def __init__(self, runtime_cfg, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg

    def forward(self, point_bxyz):
        assert NotImplementedError

        
class FPSSampler(SamplerTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        super(FPSSampler, self).__init__(
                              runtime_cfg=runtime_cfg,
                              model_cfg=model_cfg,
                          )
        self.stride = model_cfg.get("STRIDE", 1)
        self.num_sectors = model_cfg.get("NUM_SECTORS", 1)
        
    def forward(self, point_bxyz):
        """
        Args:
            point_bxyz [N, 4]: (b,x,y,z), first dimension is batch index
        Returns:
            new_bxyz: [M, 4] sampled points, M roughly equals (N // self.stride)
        """
        if self.stride == 1:
            return point_bxyz

        point_xyz, point_indices, offset = bxyz_to_xyz_index_offset(point_bxyz)

        # sample
        new_offset = [offset[0].item() // self.stride]
        sample_idx = offset[0].item() // self.stride
        for i in range(1, offset.shape[0]):
            sample_idx += (offset[i].item() - offset[i - 1].item()) // self.stride
            new_offset.append(sample_idx)
        new_offset = torch.cuda.IntTensor(new_offset)
        if self.num_sectors > 1:
            fps_idx = sectorized_fps(point_xyz, offset, new_offset, self.num_sectors) # [M]
        else:
            fps_idx = furthestsampling(point_xyz, offset, new_offset) # [M]
        fps_idx = point_indices[fps_idx.long()]

        return point_bxyz[fps_idx]

    def __repr__(self):
        return f"FPSSampler(stride={self.stride})"


SAMPLERS = {
    'FPSSampler': FPSSampler,
}

