#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>

#include "voxel_cpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("voxelization", &voxelization, "voxelize points with ambient edges");
  m.def("voxel_graph", &voxel_graph, "build graph of points/voxels");
}
