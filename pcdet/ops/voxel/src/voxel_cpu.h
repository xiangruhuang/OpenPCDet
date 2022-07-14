#ifndef PRIMITIVES_CPU_H
#define PRIMITIVES_CPU_H

#include <torch/serialize/tensor.h>

std::vector<torch::Tensor> voxelization(
    at::Tensor point_tensor, at::Tensor voxel_size,
    bool ambient);

torch::Tensor voxel_graph(
    at::Tensor point_tensor, at::Tensor query_tensor, 
    at::Tensor voxel_size, int temporal_offset,
    int max_num_neighbors);

#endif
