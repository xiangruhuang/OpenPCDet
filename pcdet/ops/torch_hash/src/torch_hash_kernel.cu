#include <stdio.h>
#include "stdint.h"
#include "torch_hash.h"
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

using namespace std;

#define CHECK_CUDA(x) do { \
    if (!x.type().is_cuda()) { \
          fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
          exit(-1); \
        } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
    if (!x.is_contiguous()) { \
          fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
          exit(-1); \
        } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

//__device__ Key EMPTY = 0xffffffffffffffff;
__device__ Key EMPTY = -1;
__device__ Key rp0 = 999269;
__device__ Key rp1 = 999437;
__device__ Key rp2 = 1999377;

__device__ index_t map2key(const Key* keys, const Key* dims, int num_dim) {
  index_t ans = 0;
  Key key;
  for (int i = 0; i < num_dim; i++) {
    key = keys[i];
    if (key >= dims[i]) {
      key = dims[i]-1;
    }
    if (key < 0) {
      key = 0;
    }
    ans = ans * dims[i] + key;
  }
  return ans;
}

__device__ index_t hashkey(const Key key, index_t ht_size) {
  return ((key % ht_size) * rp0 + rp1) % ht_size;
}

__global__ void hash_insert_gpu_kernel(
                  Key* ht_keys,
                  Float* ht_values,
                  Key* reverse_indices,
                  index_t ht_size,
                  const Key* dims,
                  int num_dim,
                  const Key* insert_keys, // keys in [N, D]
                  const Float* insert_values,
                  uint32 num_inserts
                  ) {
  unsigned int threadid = blockIdx.x*blockDim.x + threadIdx.x;
  if (threadid < num_inserts) {
    const Key* insert_key_ptr = &insert_keys[threadid*num_dim];
    Key insert_key = map2key(insert_key_ptr, dims, num_dim);
    index_t hash_idx = hashkey(insert_key, ht_size);
    
    const Float* insert_value = &insert_values[threadid*num_dim];
    Key prev = atomicCAS((unsigned long long int*)(&ht_keys[hash_idx]),
                         (unsigned long long int)EMPTY,
                         (unsigned long long int)insert_key);
    while (prev != EMPTY) {
      hash_idx = (hash_idx + 1) % ht_size;
      prev = atomicCAS((unsigned long long int*)(&ht_keys[hash_idx]),
                       (unsigned long long int)EMPTY,
                       (unsigned long long int)insert_key);
    }
    if (prev == EMPTY) {
      // hit
      ht_keys[hash_idx] = insert_key;
      Float* ht_value = &ht_values[hash_idx*num_dim];
      for (int i = 0; i < num_dim; i++) {
        ht_value[i] = insert_value[i];
      }
      reverse_indices[hash_idx] = threadid;
    }
  }
}

__global__ void correspondence_kernel(
                  Key* ht_keys, // hash table keys, values
                  Float* ht_values,
                  Key* reverse_indices, // indices to original hashed array
                  index_t ht_size, // hashtable size
                  const Key* dims, // maximum size of each dimension
                  int num_dim, // number of dimensions
                  Key* query_keys, // query keys in shape [N, D]
                  const Float* query_values, // query values
                  uint32 num_queries, //
                  const int* qmin, const int* qmax, // query range in each dimension
                  Key* corres_indices // correspondence results
                  ) {
  unsigned int threadid = blockIdx.x*blockDim.x + threadIdx.x;
  if (threadid < num_queries) {
    Key* query_key_ptr = &query_keys[threadid*num_dim];
    
    // number of voxels to query
    int num_combination = 1;
    for (int i = 0; i < num_dim; i++) {
      num_combination *= (qmax[i] - qmin[i] + 1);
    }
    Float min_dist = 1e10;
    corres_indices[threadid] = -1;

    // enumerate all directions
    Float dist, di; 
    for (int c = 0; c < num_combination; c++) {
      int temp = c;
      for (int i = 0; i < num_dim; i++) {
        query_key_ptr[i] += temp % (qmax[i] - qmin[i] + 1) + qmin[i];
        temp /= (qmax[i] - qmin[i] + 1);
      }
      Key query_key = map2key(query_key_ptr, dims, num_dim);
      index_t hash_idx = hashkey(query_key, ht_size);
      const Float* query_value = &query_values[threadid*num_dim];
      while (ht_keys[hash_idx] != -1) {
        if (ht_keys[hash_idx] == query_key) {
          const Float* ht_value = &ht_values[hash_idx*num_dim];
          // calculate distance
          dist = 0.0;
          for (int i = 0; i < num_dim; i++) {
            di = ht_value[i] - query_value[i];
            dist = dist + di*di;
          }
          if (dist < min_dist) {
            min_dist = dist;
            corres_indices[threadid] = reverse_indices[hash_idx];
          }
        }
        hash_idx = (hash_idx + 1) % ht_size;
      }
      temp = c;
      for (int i = 0; i < num_dim; i++) {
        query_key_ptr[i] -= temp % (qmax[i] - qmin[i] + 1) + qmin[i];
        temp /= (qmax[i] - qmin[i] + 1);
      }
    }
  }
}

__global__ void points_in_radius_kernel(
                  Key* ht_keys, // hash table keys, values
                  Float* ht_values,
                  Key* reverse_indices, // indices to original hashed array
                  index_t ht_size, // hashtable size
                  const Key* dims, // maximum size of each dimension
                  int num_dim, // number of dimensions
                  Key* query_keys, // query keys in shape [N, D]
                  const Float* query_values, // query values
                  uint32 num_queries, //
                  const int* qmin, const int* qmax, // query range in each dimension
                  const Float radius,
                  Key* visited // correspondence results
                  ) {
  unsigned int threadid = blockIdx.x*blockDim.x + threadIdx.x;
  if (threadid < num_queries) {
    Key* query_key_ptr = &query_keys[threadid*num_dim];
    
    int num_neighbors = 0;
    // number of voxels to query
    int num_combination = 1;
    for (int i = 0; i < num_dim; i++) {
      num_combination *= (qmax[i] - qmin[i] + 1);
    }

    // enumerate all directions
    Float dist, di;
    Float radius2 = radius*radius;
    for (int c = 0; c < num_combination; c++) {
      int temp = c;
      for (int i = 0; i < num_dim; i++) {
        query_key_ptr[i] += temp % (qmax[i] - qmin[i] + 1) + qmin[i];
        temp /= (qmax[i] - qmin[i] + 1);
      }
      Key query_key = map2key(query_key_ptr, dims, num_dim);
      index_t hash_idx = hashkey(query_key, ht_size);
      const Float* query_value = &query_values[threadid*num_dim];
      while (ht_keys[hash_idx] != -1) {
        if (ht_keys[hash_idx] == query_key) {
          const Float* ht_value = &ht_values[hash_idx*num_dim];
          // calculate distance
          dist = 0.0;
          for (int i = 0; i < num_dim; i++) {
            di = ht_value[i] - query_value[i];
            dist = dist + di*di;
          }

          if (dist < radius2) {
            int reverse_idx = reverse_indices[hash_idx];
            auto prev = atomicCAS((unsigned long long int*)(&visited[reverse_idx]),
                                  (unsigned long long int)0,
                                  (unsigned long long int)1);
          }
        }
        hash_idx = (hash_idx + 1) % ht_size;
      }
      temp = c;
      for (int i = 0; i < num_dim; i++) {
        query_key_ptr[i] -= temp % (qmax[i] - qmin[i] + 1) + qmin[i];
        temp /= (qmax[i] - qmin[i] + 1);
      }
    }
  }
}

__global__ void voxel_graph_kernel(
                  Key* ht_keys, // hash table keys, values
                  Float* ht_values,
                  Key* reverse_indices, // indices to original hashed array
                  index_t ht_size, // hashtable size
                  const Key* dims, // maximum size of each dimension
                  int num_dim, // number of dimensions
                  Key* query_keys, // query keys in shape [N, D]
                  const Float* query_values, // query values
                  uint32 num_queries, //
                  const int* qmin, const int* qmax, // query range in each dimension
                  const int max_num_neighbors, const Float radius,
                  bool sort_by_dist,
                  Key* corres_indices // correspondence results
                  ) {
  unsigned int threadid = blockIdx.x*blockDim.x + threadIdx.x;
  extern __shared__ Float dists[];
  if (threadid < num_queries) {
    Key* query_key_ptr = &query_keys[threadid*num_dim];
    
    int num_neighbors = 0;
    // number of voxels to query
    int num_combination = 1;
    for (int i = 0; i < num_dim; i++) {
      num_combination *= (qmax[i] - qmin[i] + 1);
    }
    for (int i = 0; i < max_num_neighbors; i++) {
      corres_indices[threadid*max_num_neighbors+i] = -1;
    }

    // enumerate all directions
    Float dist, di;
    Float radius2 = radius*radius;
    int nn_offset = threadid*max_num_neighbors;
    int dist_offset = threadIdx.x*max_num_neighbors;
    for (int c = 0; c < num_combination; c++) {
      int temp = c;
      for (int i = 0; i < num_dim; i++) {
        query_key_ptr[i] += temp % (qmax[i] - qmin[i] + 1) + qmin[i];
        temp /= (qmax[i] - qmin[i] + 1);
      }
      Key query_key = map2key(query_key_ptr, dims, num_dim);
      index_t hash_idx = hashkey(query_key, ht_size);
      const Float* query_value = &query_values[threadid*num_dim];
      while (ht_keys[hash_idx] != -1) {
        if (ht_keys[hash_idx] == query_key) {
          const Float* ht_value = &ht_values[hash_idx*num_dim];
          // calculate distance
          dist = 0.0;
          for (int i = 0; i < num_dim; i++) {
            di = ht_value[i] - query_value[i];
            dist = dist + di*di;
          }
          if ((dist <= radius2) && (num_neighbors <= max_num_neighbors)) {
            int nid = num_neighbors;
            if (sort_by_dist) {
                // insertion sort
                while ((nid > 0) && (dist < dists[dist_offset+nid-1])) {
                    // move element (nid-1) to place (nid)
                    if (nid < max_num_neighbors) {
                        dists[dist_offset+nid] = dists[dist_offset+nid-1];
                        corres_indices[nn_offset+nid] = corres_indices[nn_offset+nid-1];
                    }
                    nid--;
                }
            }
            if (nid < max_num_neighbors) {
                dists[dist_offset+nid] = dist;
                num_neighbors++;
                corres_indices[nn_offset+nid] = reverse_indices[hash_idx];
            }
          }
        }
        hash_idx = (hash_idx + 1) % ht_size;
      }
      temp = c;
      for (int i = 0; i < num_dim; i++) {
        query_key_ptr[i] -= temp % (qmax[i] - qmin[i] + 1) + qmin[i];
        temp /= (qmax[i] - qmin[i] + 1);
      }
    }
  }
}

void hash_insert_gpu(at::Tensor keys, at::Tensor values,
                     at::Tensor reverse_indices, at::Tensor dims,
                     at::Tensor insert_keys, at::Tensor insert_values
                     ) {
  CHECK_INPUT(keys);
  CHECK_INPUT(values);
  CHECK_INPUT(reverse_indices);
  CHECK_INPUT(dims);
  CHECK_INPUT(insert_keys);
  CHECK_INPUT(insert_values);

  Key* key_data = keys.data<Key>();
  Key* reverse_indices_data = reverse_indices.data<Key>();
  int num_dim = insert_values.size(1);
  const Key* insert_key_data = insert_keys.data<Key>();
  Float* value_data = values.data<Float>();
  const Float* insert_value_data = insert_values.data<Float>();
  index_t ht_size = keys.size(0);
  uint32 num_inserts = insert_keys.size(0);
  const Key* dims_data = dims.data<Key>();
  int mingridsize, threadblocksize;
  cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize,
     hash_insert_gpu_kernel, 0, 0);

  uint32 gridsize = (num_inserts + threadblocksize - 1) / threadblocksize;
  hash_insert_gpu_kernel<<<gridsize, threadblocksize>>>(
    key_data, value_data, reverse_indices_data, 
    ht_size, dims_data, num_dim,
    insert_key_data, insert_value_data,
    num_inserts
  );
}

void correspondence(at::Tensor keys, at::Tensor values, at::Tensor reverse_indices,
                    at::Tensor dims, at::Tensor query_keys, at::Tensor query_values,
                    at::Tensor qmin, at::Tensor qmax,
                    at::Tensor corres_indices) {
  CHECK_INPUT(keys);
  CHECK_INPUT(values);
  CHECK_INPUT(dims);
  CHECK_INPUT(query_keys);
  CHECK_INPUT(query_values);
  CHECK_INPUT(qmin);
  CHECK_INPUT(qmax);
  CHECK_INPUT(corres_indices);

  Key* key_data = keys.data<Key>();
  Key* reverse_index_data = reverse_indices.data<Key>();
  int num_dim = query_values.size(1);
  Key* query_key_data = query_keys.data<Key>();
  Float* value_data = values.data<Float>();
  const Float* query_value_data = query_values.data<Float>();
  index_t ht_size = keys.size(0);
  uint32 num_queries = query_keys.size(0);
  const int* qmin_data = qmin.data<int>();
  const int* qmax_data = qmax.data<int>();
  const Key* dims_data = dims.data<Key>();

  Key* corres_index_data = corres_indices.data<Key>();

  int mingridsize, threadblocksize;
  cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize,
     correspondence_kernel, 0, 0);

  uint32 gridsize = (num_queries + threadblocksize - 1) / threadblocksize;
  correspondence_kernel<<<gridsize, threadblocksize>>>(
    key_data, value_data, reverse_index_data, 
    ht_size, dims_data, num_dim, 
    query_key_data, query_value_data,
    num_queries,
    qmin_data, qmax_data,
    corres_index_data
  );
  
}

void voxel_graph_gpu(at::Tensor keys, at::Tensor values, at::Tensor reverse_indices,
                     at::Tensor dims, at::Tensor query_keys, at::Tensor query_values,
                     at::Tensor qmin, at::Tensor qmax,
                     int max_num_neighbors, Float radius, bool sort_by_dist,
                     at::Tensor corres_indices) {
  CHECK_INPUT(keys);
  CHECK_INPUT(values);
  CHECK_INPUT(dims);
  CHECK_INPUT(query_keys);
  CHECK_INPUT(query_values);
  CHECK_INPUT(qmin);
  CHECK_INPUT(qmax);
  CHECK_INPUT(corres_indices);

  Key* key_data = keys.data<Key>();
  Key* reverse_index_data = reverse_indices.data<Key>();
  int num_dim = query_values.size(1);
  Key* query_key_data = query_keys.data<Key>();
  Float* value_data = values.data<Float>();
  const Float* query_value_data = query_values.data<Float>();
  index_t ht_size = keys.size(0);
  uint32 num_queries = query_keys.size(0);
  const int* qmin_data = qmin.data<int>();
  const int* qmax_data = qmax.data<int>();
  const Key* dims_data = dims.data<Key>();

  Key* corres_index_data = corres_indices.data<Key>();

  int mingridsize, threadblocksize;
  cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize,
     correspondence_kernel, 0, 0);

  uint32 gridsize = (num_queries + threadblocksize - 1) / threadblocksize;
  voxel_graph_kernel<<<gridsize, threadblocksize, threadblocksize*max_num_neighbors*sizeof(Float)>>>(
    key_data, value_data, reverse_index_data, 
    ht_size, dims_data, num_dim, 
    query_key_data, query_value_data,
    num_queries,
    qmin_data, qmax_data,
    max_num_neighbors, radius, sort_by_dist,
    corres_index_data
  );
}

void points_in_radius_gpu(at::Tensor keys, at::Tensor values, at::Tensor reverse_indices,
                          at::Tensor dims, at::Tensor query_keys, at::Tensor query_values,
                          at::Tensor qmin, at::Tensor qmax,
                          Float radius, at::Tensor visited) {
  CHECK_INPUT(keys);
  CHECK_INPUT(values);
  CHECK_INPUT(dims);
  CHECK_INPUT(query_keys);
  CHECK_INPUT(query_values);
  CHECK_INPUT(qmin);
  CHECK_INPUT(qmax);
  CHECK_INPUT(visited);

  Key* key_data = keys.data<Key>();
  Key* reverse_index_data = reverse_indices.data<Key>();
  int num_dim = query_values.size(1);
  Key* query_key_data = query_keys.data<Key>();
  Float* value_data = values.data<Float>();
  const Float* query_value_data = query_values.data<Float>();
  index_t ht_size = keys.size(0);
  uint32 num_queries = query_keys.size(0);
  const int* qmin_data = qmin.data<int>();
  const int* qmax_data = qmax.data<int>();
  const Key* dims_data = dims.data<Key>();

  Key* visited_data = visited.data<Key>();

  int mingridsize, threadblocksize;
  cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize,
     correspondence_kernel, 0, 0);

  uint32 gridsize = (num_queries + threadblocksize - 1) / threadblocksize;
  points_in_radius_kernel<<<gridsize, threadblocksize>>>(
    key_data, value_data, reverse_index_data,
    ht_size, dims_data, num_dim,
    query_key_data, query_value_data,
    num_queries,
    qmin_data, qmax_data,
    radius,
    visited_data
  );
  
}
