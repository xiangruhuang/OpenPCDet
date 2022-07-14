import torch
import numpy as np
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcdet.utils import common_utils

from .dataset import DatasetTemplate
from .kitti.kitti_dataset import KittiDataset
from .nuscenes.nuscenes_dataset import NuScenesDataset
from .waymo.waymo_dataset import WaymoDataset
from .waymo.waymo_seg_dataset import WaymoSegDataset
from .waymo import waymo_utils
from .pandaset.pandaset_dataset import PandasetDataset
from .lyft.lyft_dataset import LyftDataset

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'NuScenesDataset': NuScenesDataset,
    'WaymoDataset': WaymoDataset,
    'WaymoSegDataset': WaymoSegDataset,
    'PandasetDataset': PandasetDataset,
    'LyftDataset': LyftDataset
}

class SequenceSampler(_DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.index_matrix = dataset.index_matrix
        if self.index_matrix.shape[0] % self.num_replicas != 0:
            residual = self.num_replicas - self.index_matrix.shape[0] % self.num_replicas
            self.index_matrix = np.concatenate([self.index_matrix, self.index_matrix[:residual]], axis=0)

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(self.index_matrix.shape[0], generator=g).tolist()
        else:
            indices = torch.arange(self.index_matrix.shape[0]).tolist()

        #indices += indices[:(self.total_size - len(indices))]
        #assert len(indices) == self.total_size

        indices = self.index_matrix[self.rank::self.num_replicas]
        indices = indices.reshape(-1)
        assert len(indices) == self.num_samples, f"indices={len(indices)}, num_samples={self.num_samples}"

        return iter(indices)

class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset_cfg, batch_size, dist, root_path=None, workers=4, seed=None,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            if dataset.num_sweeps > 1:
                rank, world_size = common_utils.get_dist_info()
                sampler = SequenceSampler(dataset, world_size, rank)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            if dataset.num_sweeps > 1:
                sampler = SequenceSampler(dataset, world_size, rank, shuffle=False)
            else:
                sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
    )

    return dataset, dataloader, sampler
