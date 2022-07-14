#include <stdio.h>
#include <math.h>
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include "primitives_cpu.h"
#include <unordered_map>
#include "primitives_hash.h"

using namespace std;

vector<torch::Tensor> voxelization(
  at::Tensor point_tensor, at::Tensor voxel_size, bool ambient=false) {
  VoxelHashTable h(point_tensor);
  h.hash(voxel_size);
  vector<int> qmin = {0, 0, 0, 0}, qmax = {0, 0, 0, 0};
  torch::Tensor edges_tensor = h.query_points_in_voxel(qmin, qmax);
  if (ambient) {
    qmin = {-1,-1,-1,0};
    qmax = { 1, 1, 1,0};
    torch::Tensor amb_edges_tensor = h.query_points_in_voxel(qmin, qmax);
    return {edges_tensor, amb_edges_tensor};
  } else {
    return {edges_tensor};
  }
}

torch::Tensor voxel_graph(
  at::Tensor point_tensor, at::Tensor query_tensor,
  at::Tensor voxel_size, int temporal_offset, int max_num_neighbors) {
  VoxelHashTable h(point_tensor);
  h.hash(voxel_size);
  vector<int> qmin = {-1,-1,-1,temporal_offset};
  vector<int> qmax = { 1, 1, 1,temporal_offset};
  torch::Tensor edges_tensor = h.query_point_edges(
                                   query_tensor, qmin, qmax,
                                   max_num_neighbors);
  return edges_tensor;
}
