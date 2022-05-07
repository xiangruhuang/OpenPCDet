#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "hybrid_geop.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("svd3_gpu", &svd3_gpu, "compute svd 3x3");
}
