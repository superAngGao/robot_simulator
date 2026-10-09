"""Native CUDA/CUB source for the caller-owned optical LBVH rebuild."""

CPP_SOURCE = r"""
#include <torch/extension.h>

#include <tuple>

std::tuple<int64_t, int64_t, int64_t> cuda_lbvh_workspace_requirements(
    int64_t primitive_count,
    int64_t device_ordinal);

void cuda_lbvh_rebuild_out(
    torch::Tensor primitive_aabb_min,
    torch::Tensor primitive_aabb_max,
    torch::Tensor workspace,
    torch::Tensor scene_bounds_min,
    torch::Tensor scene_bounds_max,
    torch::Tensor morton_keys,
    torch::Tensor primitive_ids,
    torch::Tensor sorted_morton_keys,
    torch::Tensor sorted_primitive_ids,
    torch::Tensor left,
    torch::Tensor right,
    torch::Tensor parent,
    torch::Tensor start,
    torch::Tensor count,
    torch::Tensor range_start,
    torch::Tensor range_end,
    torch::Tensor split,
    torch::Tensor node_bounds_min,
    torch::Tensor node_bounds_max,
    torch::Tensor child_counter);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "cuda_lbvh_workspace_requirements",
      &cuda_lbvh_workspace_requirements);
  m.def("cuda_lbvh_rebuild_out", &cuda_lbvh_rebuild_out);
}
"""


CUDA_SOURCE = r"""
#include <torch/types.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <tuple>

namespace {

constexpr int kThreads = 256;
constexpr int64_t kWorkspaceAlignment = 256;
constexpr int64_t kMaxPrimitiveCount = 357913941;
static_assert((kMaxPrimitiveCount * 2 - 1) * 3 <= INT32_MAX);
static_assert(((kMaxPrimitiveCount + 1) * 2 - 1) * 3 > INT32_MAX);

constexpr int ceil_div_positive_int(int value, int divisor) {
  return value / divisor + (value % divisor != 0);
}

constexpr int kMaxBoundValueCount =
    static_cast<int>((kMaxPrimitiveCount * 2 - 1) * 3);
static_assert(
    ceil_div_positive_int(kMaxBoundValueCount, kThreads) == 8388608);

struct AxisValue {
  const float* values;
  int axis;

  __host__ __device__ float operator()(int index) const {
    return values[index * 3 + axis];
  }
};

using CountingIterator = cub::CountingInputIterator<int>;
using AxisIterator = cub::TransformInputIterator<float, AxisValue, CountingIterator>;

void check_cuda_tensor(
    const torch::Tensor& value,
    const torch::Device& device,
    torch::ScalarType dtype,
    const char* name) {
  TORCH_CHECK(value.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(value.device() == device, name, " must be on the anchor device");
  TORCH_CHECK(value.scalar_type() == dtype, name, " has the wrong dtype");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

uintptr_t span_start(const torch::Tensor& value) {
  return reinterpret_cast<uintptr_t>(value.data_ptr());
}

uintptr_t span_end(const torch::Tensor& value) {
  return span_start(value) +
      static_cast<uintptr_t>(value.numel() * value.element_size());
}

bool spans_overlap(const torch::Tensor& lhs, const torch::Tensor& rhs) {
  if (lhs.numel() == 0 || rhs.numel() == 0) {
    return false;
  }
  return span_start(lhs) < span_end(rhs) && span_start(rhs) < span_end(lhs);
}

template <int InputCount, int OutputCount>
void check_output_disjointness(
    const torch::Tensor* const (&inputs)[InputCount],
    const torch::Tensor* const (&outputs)[OutputCount]) {
  for (int output_index = 0; output_index < OutputCount; ++output_index) {
    for (int input_index = 0; input_index < InputCount; ++input_index) {
      TORCH_CHECK(
          !spans_overlap(*outputs[output_index], *inputs[input_index]),
          "CUDA LBVH output overlaps an input/workspace tensor");
    }
    for (int prior = 0; prior < output_index; ++prior) {
      TORCH_CHECK(
          !spans_overlap(*outputs[output_index], *outputs[prior]),
          "CUDA LBVH output tensors overlap");
    }
  }
}

__device__ __forceinline__ uint32_t expand_bits_10(uint32_t value) {
  value = (value * 0x00010001u) & 0xFF0000FFu;
  value = (value * 0x00000101u) & 0x0F00F00Fu;
  value = (value * 0x00000011u) & 0xC30C30C3u;
  value = (value * 0x00000005u) & 0x49249249u;
  return value;
}

__device__ __forceinline__ uint32_t morton3d_30(float x, float y, float z) {
  x = fminf(fmaxf(x, 0.0f), 0.999999f);
  y = fminf(fmaxf(y, 0.0f), 0.999999f);
  z = fminf(fmaxf(z, 0.0f), 0.999999f);
  const uint32_t xx = static_cast<uint32_t>(x * 1024.0f);
  const uint32_t yy = static_cast<uint32_t>(y * 1024.0f);
  const uint32_t zz = static_cast<uint32_t>(z * 1024.0f);
  return (expand_bits_10(xx) << 2) |
      (expand_bits_10(yy) << 1) |
      expand_bits_10(zz);
}

__global__ void compute_morton_keys_kernel(
    const float* aabb_min,
    const float* aabb_max,
    const float* scene_min,
    const float* scene_max,
    int64_t* keys,
    int32_t* ids,
    int primitive_count) {
  const int primitive = blockIdx.x * blockDim.x + threadIdx.x;
  if (primitive >= primitive_count) {
    return;
  }
  const float cx = 0.5f *
      (aabb_min[primitive * 3 + 0] + aabb_max[primitive * 3 + 0]);
  const float cy = 0.5f *
      (aabb_min[primitive * 3 + 1] + aabb_max[primitive * 3 + 1]);
  const float cz = 0.5f *
      (aabb_min[primitive * 3 + 2] + aabb_max[primitive * 3 + 2]);
  const float ex = fmaxf(scene_max[0] - scene_min[0], 1.0e-12f);
  const float ey = fmaxf(scene_max[1] - scene_min[1], 1.0e-12f);
  const float ez = fmaxf(scene_max[2] - scene_min[2], 1.0e-12f);
  const uint32_t morton = morton3d_30(
      (cx - scene_min[0]) / ex,
      (cy - scene_min[1]) / ey,
      (cz - scene_min[2]) / ez);
  const uint64_t composite =
      (static_cast<uint64_t>(morton) << 32) |
      static_cast<uint32_t>(primitive);
  keys[primitive] = static_cast<int64_t>(composite);
  ids[primitive] = primitive;
}

__device__ __forceinline__ int common_prefix_bits(
    const int64_t* keys,
    int primitive_count,
    int first,
    int second) {
  if (second < 0 || second >= primitive_count) {
    return -1;
  }
  const uint64_t lhs = static_cast<uint64_t>(keys[first]);
  const uint64_t rhs = static_cast<uint64_t>(keys[second]);
  return __clzll(lhs ^ rhs);
}

__device__ int find_split(const int64_t* keys, int first, int last) {
  const uint64_t first_key = static_cast<uint64_t>(keys[first]);
  const uint64_t last_key = static_cast<uint64_t>(keys[last]);
  const int common_prefix = __clzll(first_key ^ last_key);
  int split = first;
  int step = last - first;
  do {
    step = (step + 1) >> 1;
    const int candidate = split + step;
    if (candidate < last) {
      const uint64_t candidate_key = static_cast<uint64_t>(keys[candidate]);
      const int candidate_prefix = __clzll(first_key ^ candidate_key);
      if (candidate_prefix > common_prefix) {
        split = candidate;
      }
    }
  } while (step > 1);
  return split;
}

__global__ void initialize_leaf_metadata_kernel(
    int32_t* start,
    int32_t* count,
    int primitive_count) {
  const int leaf_rank = blockIdx.x * blockDim.x + threadIdx.x;
  if (leaf_rank >= primitive_count) {
    return;
  }
  const int leaf = primitive_count - 1 + leaf_rank;
  start[leaf] = leaf_rank;
  count[leaf] = 1;
}

__global__ void build_lbvh_internal_topology_kernel(
    const int64_t* sorted_keys,
    int32_t* left,
    int32_t* right,
    int32_t* parent,
    int32_t* range_start,
    int32_t* range_end,
    int32_t* split_out,
    int primitive_count) {
  const int internal = blockIdx.x * blockDim.x + threadIdx.x;
  if (internal >= primitive_count - 1) {
    return;
  }
  const int prefix_next =
      common_prefix_bits(sorted_keys, primitive_count, internal, internal + 1);
  const int prefix_previous =
      common_prefix_bits(sorted_keys, primitive_count, internal, internal - 1);
  const int direction = (prefix_next - prefix_previous) >= 0 ? 1 : -1;
  const int prefix_min = common_prefix_bits(
      sorted_keys,
      primitive_count,
      internal,
      internal - direction);
  int maximum_length = 2;
  while (common_prefix_bits(
             sorted_keys,
             primitive_count,
             internal,
             internal + maximum_length * direction) > prefix_min) {
    maximum_length <<= 1;
  }
  int length = 0;
  for (int step = maximum_length >> 1; step >= 1; step >>= 1) {
    const int candidate = length + step;
    if (common_prefix_bits(
            sorted_keys,
            primitive_count,
            internal,
            internal + candidate * direction) > prefix_min) {
      length = candidate;
    }
  }
  const int other = internal + length * direction;
  const int first = min(internal, other);
  const int last = max(internal, other);
  const int split = find_split(sorted_keys, first, last);
  const int leaf_base = primitive_count - 1;
  const int left_child = split == first ? leaf_base + split : split;
  const int right_child =
      split + 1 == last ? leaf_base + split + 1 : split + 1;
  left[internal] = left_child;
  right[internal] = right_child;
  parent[left_child] = internal;
  parent[right_child] = internal;
  range_start[internal] = first;
  range_end[internal] = last;
  split_out[internal] = split;
}

__global__ void initialize_node_bounds_kernel(
    float* node_bounds_min,
    float* node_bounds_max,
    int value_count) {
  const int value_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (value_index >= value_count) {
    return;
  }
  node_bounds_min[value_index] = __int_as_float(0x7F800000);
  node_bounds_max[value_index] = __int_as_float(0xFF800000);
}

__global__ void initialize_leaf_bounds_kernel(
    const int32_t* sorted_primitive_ids,
    const float* primitive_aabb_min,
    const float* primitive_aabb_max,
    float* node_bounds_min,
    float* node_bounds_max,
    int primitive_count) {
  const int leaf_rank = blockIdx.x * blockDim.x + threadIdx.x;
  if (leaf_rank >= primitive_count) {
    return;
  }
  const int node = primitive_count - 1 + leaf_rank;
  const int primitive = sorted_primitive_ids[leaf_rank];
  #pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    node_bounds_min[node * 3 + axis] =
        primitive_aabb_min[primitive * 3 + axis];
    node_bounds_max[node * 3 + axis] =
        primitive_aabb_max[primitive * 3 + axis];
  }
}

__device__ __forceinline__ float atomic_min_float(float* address, float value) {
  int* address_as_int = reinterpret_cast<int*>(address);
  int old = *address_as_int;
  int assumed;
  do {
    assumed = old;
    old = atomicCAS(
        address_as_int,
        assumed,
        __float_as_int(fminf(value, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ __forceinline__ float atomic_max_float(float* address, float value) {
  int* address_as_int = reinterpret_cast<int*>(address);
  int old = *address_as_int;
  int assumed;
  do {
    assumed = old;
    old = atomicCAS(
        address_as_int,
        assumed,
        __float_as_int(fmaxf(value, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__global__ void reduce_internal_bounds_kernel(
    const int32_t* parent,
    float* node_bounds_min,
    float* node_bounds_max,
    int32_t* child_counter,
    int primitive_count) {
  const int leaf_rank = blockIdx.x * blockDim.x + threadIdx.x;
  if (leaf_rank >= primitive_count) {
    return;
  }
  int node = primitive_count - 1 + leaf_rank;
  int ancestor = parent[node];
  while (ancestor >= 0) {
    #pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
      atomic_min_float(
          &node_bounds_min[ancestor * 3 + axis],
          node_bounds_min[node * 3 + axis]);
      atomic_max_float(
          &node_bounds_max[ancestor * 3 + axis],
          node_bounds_max[node * 3 + axis]);
    }
    const int previous = atomicAdd(&child_counter[ancestor], 1);
    if (previous == 0) {
      return;
    }
    node = ancestor;
    ancestor = parent[node];
  }
}

size_t reduction_workspace_bytes(int primitive_count) {
  if (primitive_count <= 0) {
    return 0;
  }
  size_t minimum_bytes = 0;
  size_t maximum_bytes = 0;
  const CountingIterator counting(0);
  const AxisIterator values(counting, AxisValue{nullptr, 0});
  C10_CUDA_CHECK(cub::DeviceReduce::Min(
      nullptr,
      minimum_bytes,
      values,
      static_cast<float*>(nullptr),
      primitive_count));
  C10_CUDA_CHECK(cub::DeviceReduce::Max(
      nullptr,
      maximum_bytes,
      values,
      static_cast<float*>(nullptr),
      primitive_count));
  return minimum_bytes > maximum_bytes ? minimum_bytes : maximum_bytes;
}

size_t radix_workspace_bytes(int primitive_count) {
  if (primitive_count <= 0) {
    return 0;
  }
  size_t bytes = 0;
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr,
      bytes,
      static_cast<int64_t*>(nullptr),
      static_cast<int64_t*>(nullptr),
      static_cast<int32_t*>(nullptr),
      static_cast<int32_t*>(nullptr),
      primitive_count,
      0,
      64));
  return bytes;
}

}  // namespace

std::tuple<int64_t, int64_t, int64_t> cuda_lbvh_workspace_requirements(
    int64_t primitive_count,
    int64_t device_ordinal) {
  TORCH_CHECK(primitive_count >= 0, "primitive_count must be non-negative");
  TORCH_CHECK(
      primitive_count <= kMaxPrimitiveCount,
      "primitive_count exceeds int32 flattened-node range");
  int device_count = 0;
  C10_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  TORCH_CHECK(
      device_ordinal >= 0 && device_ordinal < device_count,
      "device_ordinal is out of range");
  const c10::cuda::CUDAGuard device_guard(
      torch::Device(torch::kCUDA, device_ordinal));
  const int count = static_cast<int>(primitive_count);
  return std::make_tuple(
      static_cast<int64_t>(reduction_workspace_bytes(count)),
      static_cast<int64_t>(radix_workspace_bytes(count)),
      kWorkspaceAlignment);
}

void cuda_lbvh_rebuild_out(
    torch::Tensor primitive_aabb_min,
    torch::Tensor primitive_aabb_max,
    torch::Tensor workspace,
    torch::Tensor scene_bounds_min,
    torch::Tensor scene_bounds_max,
    torch::Tensor morton_keys,
    torch::Tensor primitive_ids,
    torch::Tensor sorted_morton_keys,
    torch::Tensor sorted_primitive_ids,
    torch::Tensor left,
    torch::Tensor right,
    torch::Tensor parent,
    torch::Tensor start,
    torch::Tensor count,
    torch::Tensor range_start,
    torch::Tensor range_end,
    torch::Tensor split,
    torch::Tensor node_bounds_min,
    torch::Tensor node_bounds_max,
    torch::Tensor child_counter) {
  TORCH_CHECK(node_bounds_min.is_cuda(), "node_bounds_min must be CUDA");
  const c10::cuda::CUDAGuard device_guard(node_bounds_min.device());
  const torch::Device device = node_bounds_min.device();
  check_cuda_tensor(
      primitive_aabb_min, device, torch::kFloat32, "primitive_aabb_min");
  check_cuda_tensor(
      primitive_aabb_max, device, torch::kFloat32, "primitive_aabb_max");
  check_cuda_tensor(workspace, device, torch::kUInt8, "workspace");
  check_cuda_tensor(scene_bounds_min, device, torch::kFloat32, "scene_bounds_min");
  check_cuda_tensor(scene_bounds_max, device, torch::kFloat32, "scene_bounds_max");
  check_cuda_tensor(morton_keys, device, torch::kInt64, "morton_keys");
  check_cuda_tensor(primitive_ids, device, torch::kInt32, "primitive_ids");
  check_cuda_tensor(
      sorted_morton_keys, device, torch::kInt64, "sorted_morton_keys");
  check_cuda_tensor(
      sorted_primitive_ids, device, torch::kInt32, "sorted_primitive_ids");
  check_cuda_tensor(left, device, torch::kInt32, "left");
  check_cuda_tensor(right, device, torch::kInt32, "right");
  check_cuda_tensor(parent, device, torch::kInt32, "parent");
  check_cuda_tensor(start, device, torch::kInt32, "start");
  check_cuda_tensor(count, device, torch::kInt32, "count");
  check_cuda_tensor(range_start, device, torch::kInt32, "range_start");
  check_cuda_tensor(range_end, device, torch::kInt32, "range_end");
  check_cuda_tensor(split, device, torch::kInt32, "split");
  check_cuda_tensor(node_bounds_min, device, torch::kFloat32, "node_bounds_min");
  check_cuda_tensor(node_bounds_max, device, torch::kFloat32, "node_bounds_max");
  check_cuda_tensor(child_counter, device, torch::kInt32, "child_counter");

  TORCH_CHECK(
      primitive_aabb_min.dim() == 2 && primitive_aabb_min.size(1) == 3,
      "primitive_aabb_min must have shape [N, 3]");
  TORCH_CHECK(
      primitive_aabb_max.sizes() == primitive_aabb_min.sizes(),
      "primitive_aabb_max must match primitive_aabb_min");
  const int64_t primitive_count_64 = primitive_aabb_min.size(0);
  TORCH_CHECK(primitive_count_64 > 0, "native LBVH entry requires primitives");
  TORCH_CHECK(
      primitive_count_64 <= kMaxPrimitiveCount,
      "primitive_count exceeds int32 flattened-node range");
  const int primitive_count = static_cast<int>(primitive_count_64);
  const int64_t node_count = primitive_count_64 * 2 - 1;
  const int64_t internal_count = primitive_count_64 - 1;
  TORCH_CHECK(workspace.dim() == 1, "workspace must be rank-1");
  TORCH_CHECK(workspace.numel() > 0, "workspace must not be empty");
  TORCH_CHECK(
      span_start(workspace) % kWorkspaceAlignment == 0,
      "workspace pointer is not sufficiently aligned");
  TORCH_CHECK(
      scene_bounds_min.dim() == 1 && scene_bounds_min.size(0) == 3,
      "scene_bounds_min must have shape [3]");
  TORCH_CHECK(
      scene_bounds_max.dim() == 1 && scene_bounds_max.size(0) == 3,
      "scene_bounds_max must have shape [3]");
  TORCH_CHECK(
      morton_keys.dim() == 1 && morton_keys.size(0) == primitive_count_64,
      "morton_keys shape");
  TORCH_CHECK(
      primitive_ids.dim() == 1 && primitive_ids.size(0) == primitive_count_64,
      "primitive_ids shape");
  TORCH_CHECK(
      sorted_morton_keys.dim() == 1 &&
          sorted_morton_keys.size(0) == primitive_count_64,
      "sorted_morton_keys shape");
  TORCH_CHECK(
      sorted_primitive_ids.dim() == 1 &&
          sorted_primitive_ids.size(0) == primitive_count_64,
      "sorted_primitive_ids shape");
  const torch::Tensor* const node_int_outputs[5] = {
      &left, &right, &parent, &start, &count};
  for (int index = 0; index < 5; ++index) {
    TORCH_CHECK(
        node_int_outputs[index]->dim() == 1 &&
            node_int_outputs[index]->size(0) == node_count,
        "node integer output shape");
  }
  const torch::Tensor* const internal_outputs[4] = {
      &range_start, &range_end, &split, &child_counter};
  for (int index = 0; index < 4; ++index) {
    TORCH_CHECK(
        internal_outputs[index]->dim() == 1 &&
            internal_outputs[index]->size(0) == internal_count,
        "internal output shape");
  }
  TORCH_CHECK(
      node_bounds_min.dim() == 2 && node_bounds_min.size(0) == node_count &&
          node_bounds_min.size(1) == 3,
      "node_bounds_min shape");
  TORCH_CHECK(
      node_bounds_max.sizes() == node_bounds_min.sizes(),
      "node_bounds_max shape");

  TORCH_CHECK(
      !spans_overlap(primitive_aabb_min, primitive_aabb_max),
      "primitive AABB inputs overlap");
  TORCH_CHECK(
      !spans_overlap(primitive_aabb_min, workspace) &&
          !spans_overlap(primitive_aabb_max, workspace),
      "workspace overlaps primitive AABB inputs");
  const torch::Tensor* const inputs[3] = {
      &primitive_aabb_min, &primitive_aabb_max, &workspace};
  const torch::Tensor* const outputs[17] = {
      &scene_bounds_min,
      &scene_bounds_max,
      &morton_keys,
      &primitive_ids,
      &sorted_morton_keys,
      &sorted_primitive_ids,
      &left,
      &right,
      &parent,
      &start,
      &count,
      &range_start,
      &range_end,
      &split,
      &node_bounds_min,
      &node_bounds_max,
      &child_counter};
  check_output_disjointness(inputs, outputs);

  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(node_bounds_min.get_device());
  void* workspace_pointer = workspace.data_ptr<uint8_t>();
  const size_t workspace_bytes = static_cast<size_t>(workspace.numel());
  for (int axis = 0; axis < 3; ++axis) {
    const CountingIterator counting(0);
    const AxisIterator minimum_values(
        counting,
        AxisValue{primitive_aabb_min.data_ptr<float>(), axis});
    const AxisIterator maximum_values(
        counting,
        AxisValue{primitive_aabb_max.data_ptr<float>(), axis});
    size_t minimum_bytes = workspace_bytes;
    size_t maximum_bytes = workspace_bytes;
    C10_CUDA_CHECK(cub::DeviceReduce::Min(
        workspace_pointer,
        minimum_bytes,
        minimum_values,
        scene_bounds_min.data_ptr<float>() + axis,
        primitive_count,
        stream));
    C10_CUDA_CHECK(cub::DeviceReduce::Max(
        workspace_pointer,
        maximum_bytes,
        maximum_values,
        scene_bounds_max.data_ptr<float>() + axis,
        primitive_count,
        stream));
  }

  C10_CUDA_CHECK(cudaMemsetAsync(
      left.data_ptr<int32_t>(), 0xFF, left.numel() * sizeof(int32_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      right.data_ptr<int32_t>(), 0xFF, right.numel() * sizeof(int32_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      parent.data_ptr<int32_t>(), 0xFF, parent.numel() * sizeof(int32_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      start.data_ptr<int32_t>(), 0xFF, start.numel() * sizeof(int32_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      count.data_ptr<int32_t>(), 0, count.numel() * sizeof(int32_t), stream));
  if (internal_count > 0) {
    C10_CUDA_CHECK(cudaMemsetAsync(
        range_start.data_ptr<int32_t>(),
        0xFF,
        range_start.numel() * sizeof(int32_t),
        stream));
    C10_CUDA_CHECK(cudaMemsetAsync(
        range_end.data_ptr<int32_t>(),
        0xFF,
        range_end.numel() * sizeof(int32_t),
        stream));
    C10_CUDA_CHECK(cudaMemsetAsync(
        split.data_ptr<int32_t>(),
        0xFF,
        split.numel() * sizeof(int32_t),
        stream));
    C10_CUDA_CHECK(cudaMemsetAsync(
        child_counter.data_ptr<int32_t>(),
        0,
        child_counter.numel() * sizeof(int32_t),
        stream));
  }

  const int primitive_blocks =
      ceil_div_positive_int(primitive_count, kThreads);
  compute_morton_keys_kernel<<<primitive_blocks, kThreads, 0, stream>>>(
      primitive_aabb_min.data_ptr<float>(),
      primitive_aabb_max.data_ptr<float>(),
      scene_bounds_min.data_ptr<float>(),
      scene_bounds_max.data_ptr<float>(),
      morton_keys.data_ptr<int64_t>(),
      primitive_ids.data_ptr<int32_t>(),
      primitive_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  size_t radix_bytes = workspace_bytes;
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      workspace_pointer,
      radix_bytes,
      morton_keys.data_ptr<int64_t>(),
      sorted_morton_keys.data_ptr<int64_t>(),
      primitive_ids.data_ptr<int32_t>(),
      sorted_primitive_ids.data_ptr<int32_t>(),
      primitive_count,
      0,
      64,
      stream));

  initialize_leaf_metadata_kernel<<<primitive_blocks, kThreads, 0, stream>>>(
      start.data_ptr<int32_t>(),
      count.data_ptr<int32_t>(),
      primitive_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (internal_count > 0) {
    const int internal_blocks = ceil_div_positive_int(
        static_cast<int>(internal_count), kThreads);
    build_lbvh_internal_topology_kernel<<<
        internal_blocks, kThreads, 0, stream>>>(
        sorted_morton_keys.data_ptr<int64_t>(),
        left.data_ptr<int32_t>(),
        right.data_ptr<int32_t>(),
        parent.data_ptr<int32_t>(),
        range_start.data_ptr<int32_t>(),
        range_end.data_ptr<int32_t>(),
        split.data_ptr<int32_t>(),
        primitive_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  const int bound_value_count = static_cast<int>(node_count * 3);
  const int bound_blocks =
      ceil_div_positive_int(bound_value_count, kThreads);
  initialize_node_bounds_kernel<<<bound_blocks, kThreads, 0, stream>>>(
      node_bounds_min.data_ptr<float>(),
      node_bounds_max.data_ptr<float>(),
      bound_value_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  initialize_leaf_bounds_kernel<<<primitive_blocks, kThreads, 0, stream>>>(
      sorted_primitive_ids.data_ptr<int32_t>(),
      primitive_aabb_min.data_ptr<float>(),
      primitive_aabb_max.data_ptr<float>(),
      node_bounds_min.data_ptr<float>(),
      node_bounds_max.data_ptr<float>(),
      primitive_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (internal_count > 0) {
    reduce_internal_bounds_kernel<<<primitive_blocks, kThreads, 0, stream>>>(
        parent.data_ptr<int32_t>(),
        node_bounds_min.data_ptr<float>(),
        node_bounds_max.data_ptr<float>(),
        child_counter.data_ptr<int32_t>(),
        primitive_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
"""
