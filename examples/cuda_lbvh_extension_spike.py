"""Spike CUDA/CUB extension loading for the future optical LBVH builder.

This is not the production LBVH implementation. It validates the riskiest
integration pieces first:

1. `torch.utils.cpp_extension` can JIT compile CUDA code in this environment.
2. CUDA code can include and call CUB.
3. CUDA code can compute 30-bit Morton keys from AABBs and CUB-sort primitive ids.
4. Torch CUDA tensors returned by the extension can be viewed by Warp with
   `wp.from_torch` without device-to-host copies.

Example:

    conda run -n env_tilelang_20260119 python examples/cuda_lbvh_extension_spike.py
"""

from __future__ import annotations

import argparse

CPP_SOURCE = r"""
#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> cub_sort_codes(torch::Tensor codes);
std::vector<torch::Tensor> morton_sort_aabbs(
    torch::Tensor aabb_min,
    torch::Tensor aabb_max,
    torch::Tensor scene_min,
    torch::Tensor scene_max);
std::vector<torch::Tensor> build_lbvh_topology(torch::Tensor sorted_keys);
std::vector<torch::Tensor> build_lbvh_topology_and_bounds(
    torch::Tensor sorted_keys,
    torch::Tensor sorted_prim_ids,
    torch::Tensor aabb_min,
    torch::Tensor aabb_max);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cub_sort_codes", &cub_sort_codes, "Sort int64 Morton-like codes with CUB");
  m.def("morton_sort_aabbs", &morton_sort_aabbs, "Compute 30-bit Morton keys from AABBs and sort ids");
  m.def("build_lbvh_topology", &build_lbvh_topology, "Build minimal Karras LBVH topology");
  m.def(
      "build_lbvh_topology_and_bounds",
      &build_lbvh_topology_and_bounds,
      "Build minimal Karras LBVH topology and node bounds");
}
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>

#include <cstdint>
#include <limits>
#include <vector>

__global__ void fill_ids_kernel(int32_t* ids, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    ids[i] = i;
  }
}

__device__ __forceinline__ uint32_t expand_bits_10(uint32_t v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

__device__ __forceinline__ uint32_t morton3d_30(float x, float y, float z) {
  x = fminf(fmaxf(x, 0.0f), 0.999999f);
  y = fminf(fmaxf(y, 0.0f), 0.999999f);
  z = fminf(fmaxf(z, 0.0f), 0.999999f);
  uint32_t xx = static_cast<uint32_t>(x * 1024.0f);
  uint32_t yy = static_cast<uint32_t>(y * 1024.0f);
  uint32_t zz = static_cast<uint32_t>(z * 1024.0f);
  return (expand_bits_10(xx) << 2) | (expand_bits_10(yy) << 1) | expand_bits_10(zz);
}

__global__ void compute_morton_keys_kernel(
    const float* aabb_min,
    const float* aabb_max,
    const float* scene_min,
    const float* scene_max,
    int64_t* keys,
    int32_t* ids,
    int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  float cx = 0.5f * (aabb_min[i * 3 + 0] + aabb_max[i * 3 + 0]);
  float cy = 0.5f * (aabb_min[i * 3 + 1] + aabb_max[i * 3 + 1]);
  float cz = 0.5f * (aabb_min[i * 3 + 2] + aabb_max[i * 3 + 2]);

  float ex = fmaxf(scene_max[0] - scene_min[0], 1.0e-12f);
  float ey = fmaxf(scene_max[1] - scene_min[1], 1.0e-12f);
  float ez = fmaxf(scene_max[2] - scene_min[2], 1.0e-12f);

  uint32_t morton = morton3d_30(
      (cx - scene_min[0]) / ex,
      (cy - scene_min[1]) / ey,
      (cz - scene_min[2]) / ez);
  uint64_t extended = (static_cast<uint64_t>(morton) << 32) | static_cast<uint32_t>(i);
  keys[i] = static_cast<int64_t>(extended);
  ids[i] = i;
}

__device__ __forceinline__ int common_prefix_bits(const int64_t* keys, int n, int i, int j) {
  if (j < 0 || j >= n) {
    return -1;
  }
  uint64_t a = static_cast<uint64_t>(keys[i]);
  uint64_t b = static_cast<uint64_t>(keys[j]);
  return __clzll(a ^ b);
}

__device__ int find_split(const int64_t* keys, int first, int last) {
  uint64_t first_key = static_cast<uint64_t>(keys[first]);
  uint64_t last_key = static_cast<uint64_t>(keys[last]);
  int common_prefix = __clzll(first_key ^ last_key);
  int split = first;
  int step = last - first;

  do {
    step = (step + 1) >> 1;
    int candidate = split + step;
    if (candidate < last) {
      uint64_t candidate_key = static_cast<uint64_t>(keys[candidate]);
      int candidate_prefix = __clzll(first_key ^ candidate_key);
      if (candidate_prefix > common_prefix) {
        split = candidate;
      }
    }
  } while (step > 1);

  return split;
}

__global__ void build_lbvh_topology_kernel(
    const int64_t* sorted_keys,
    int32_t* left,
    int32_t* right,
    int32_t* parent,
    int32_t* start,
    int32_t* count,
    int32_t* range_start,
    int32_t* range_end,
    int32_t* split_out,
    int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int leaf_base = n - 1;

  if (i < n) {
    int leaf = leaf_base + i;
    start[leaf] = i;
    count[leaf] = 1;
  }

  if (i >= n - 1) {
    return;
  }

  int prefix_next = common_prefix_bits(sorted_keys, n, i, i + 1);
  int prefix_prev = common_prefix_bits(sorted_keys, n, i, i - 1);
  int direction = (prefix_next - prefix_prev) >= 0 ? 1 : -1;
  int prefix_min = common_prefix_bits(sorted_keys, n, i, i - direction);

  int l_max = 2;
  while (common_prefix_bits(sorted_keys, n, i, i + l_max * direction) > prefix_min) {
    l_max <<= 1;
  }

  int length = 0;
  for (int step = l_max >> 1; step >= 1; step >>= 1) {
    int candidate = length + step;
    if (common_prefix_bits(sorted_keys, n, i, i + candidate * direction) > prefix_min) {
      length = candidate;
    }
  }

  int j = i + length * direction;
  int first = min(i, j);
  int last = max(i, j);
  int split = find_split(sorted_keys, first, last);

  int left_child = split == first ? leaf_base + split : split;
  int right_child = split + 1 == last ? leaf_base + split + 1 : split + 1;

  left[i] = left_child;
  right[i] = right_child;
  parent[left_child] = i;
  parent[right_child] = i;
  range_start[i] = first;
  range_end[i] = last;
  split_out[i] = split;
}

__device__ __forceinline__ float atomic_min_float(float* address, float value) {
  int* address_as_i = reinterpret_cast<int*>(address);
  int old = *address_as_i;
  int assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed, __float_as_int(fminf(value, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ __forceinline__ float atomic_max_float(float* address, float value) {
  int* address_as_i = reinterpret_cast<int*>(address);
  int old = *address_as_i;
  int assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(value, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__global__ void initialize_lbvh_leaf_bounds_kernel(
    const int32_t* sorted_prim_ids,
    const float* primitive_aabb_min,
    const float* primitive_aabb_max,
    float* node_bounds_min,
    float* node_bounds_max,
    int n) {
  int leaf_rank = blockIdx.x * blockDim.x + threadIdx.x;
  if (leaf_rank >= n) {
    return;
  }
  int node = n - 1 + leaf_rank;
  int prim = sorted_prim_ids[leaf_rank];
  for (int axis = 0; axis < 3; ++axis) {
    node_bounds_min[node * 3 + axis] = primitive_aabb_min[prim * 3 + axis];
    node_bounds_max[node * 3 + axis] = primitive_aabb_max[prim * 3 + axis];
  }
}

__global__ void reduce_lbvh_internal_bounds_kernel(
    const int32_t* parent,
    float* node_bounds_min,
    float* node_bounds_max,
    int32_t* child_counter,
    int n) {
  int leaf_rank = blockIdx.x * blockDim.x + threadIdx.x;
  if (leaf_rank >= n) {
    return;
  }
  int node = n - 1 + leaf_rank;
  int p = parent[node];

  while (p >= 0) {
    for (int axis = 0; axis < 3; ++axis) {
      float min_v = node_bounds_min[node * 3 + axis];
      float max_v = node_bounds_max[node * 3 + axis];
      atomic_min_float(&node_bounds_min[p * 3 + axis], min_v);
      atomic_max_float(&node_bounds_max[p * 3 + axis], max_v);
    }

    int previous = atomicAdd(&child_counter[p], 1);
    if (previous == 0) {
      return;
    }

    node = p;
    p = parent[node];
  }
}

std::vector<torch::Tensor> cub_sort_codes(torch::Tensor codes) {
  TORCH_CHECK(codes.is_cuda(), "codes must be a CUDA tensor");
  TORCH_CHECK(codes.dtype() == torch::kInt64, "codes must be int64");
  TORCH_CHECK(codes.is_contiguous(), "codes must be contiguous");
  TORCH_CHECK(codes.dim() == 1, "codes must be rank-1");
  TORCH_CHECK(codes.numel() <= static_cast<int64_t>(INT32_MAX), "too many codes for spike");

  int n = static_cast<int>(codes.numel());
  auto id_options = torch::TensorOptions().dtype(torch::kInt32).device(codes.device());
  auto ids_in = torch::empty({n}, id_options);
  auto ids_out = torch::empty({n}, id_options);
  auto sorted_codes = torch::empty_like(codes);

  if (n == 0) {
    return {sorted_codes, ids_out};
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int block = 256;
  int grid = (n + block - 1) / block;
  fill_ids_kernel<<<grid, block, 0, stream>>>(ids_in.data_ptr<int32_t>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
      nullptr,
      temp_bytes,
      codes.data_ptr<int64_t>(),
      sorted_codes.data_ptr<int64_t>(),
      ids_in.data_ptr<int32_t>(),
      ids_out.data_ptr<int32_t>(),
      n,
      0,
      64,
      stream);

  auto temp = torch::empty(
      {static_cast<int64_t>(temp_bytes)},
      torch::TensorOptions().dtype(torch::kUInt8).device(codes.device()));

  cub::DeviceRadixSort::SortPairs(
      temp.data_ptr(),
      temp_bytes,
      codes.data_ptr<int64_t>(),
      sorted_codes.data_ptr<int64_t>(),
      ids_in.data_ptr<int32_t>(),
      ids_out.data_ptr<int32_t>(),
      n,
      0,
      64,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {sorted_codes, ids_out};
}

std::vector<torch::Tensor> morton_sort_aabbs(
    torch::Tensor aabb_min,
    torch::Tensor aabb_max,
    torch::Tensor scene_min,
    torch::Tensor scene_max) {
  TORCH_CHECK(aabb_min.is_cuda(), "aabb_min must be a CUDA tensor");
  TORCH_CHECK(aabb_max.is_cuda(), "aabb_max must be a CUDA tensor");
  TORCH_CHECK(scene_min.is_cuda(), "scene_min must be a CUDA tensor");
  TORCH_CHECK(scene_max.is_cuda(), "scene_max must be a CUDA tensor");
  TORCH_CHECK(aabb_min.dtype() == torch::kFloat32, "aabb_min must be float32");
  TORCH_CHECK(aabb_max.dtype() == torch::kFloat32, "aabb_max must be float32");
  TORCH_CHECK(scene_min.dtype() == torch::kFloat32, "scene_min must be float32");
  TORCH_CHECK(scene_max.dtype() == torch::kFloat32, "scene_max must be float32");
  TORCH_CHECK(aabb_min.is_contiguous(), "aabb_min must be contiguous");
  TORCH_CHECK(aabb_max.is_contiguous(), "aabb_max must be contiguous");
  TORCH_CHECK(scene_min.is_contiguous(), "scene_min must be contiguous");
  TORCH_CHECK(scene_max.is_contiguous(), "scene_max must be contiguous");
  TORCH_CHECK(aabb_min.dim() == 2 && aabb_min.size(1) == 3, "aabb_min must have shape [N, 3]");
  TORCH_CHECK(aabb_max.sizes() == aabb_min.sizes(), "aabb_max must match aabb_min shape");
  TORCH_CHECK(scene_min.numel() == 3, "scene_min must have 3 values");
  TORCH_CHECK(scene_max.numel() == 3, "scene_max must have 3 values");
  TORCH_CHECK(aabb_min.size(0) <= static_cast<int64_t>(INT32_MAX), "too many AABBs for spike");

  int n = static_cast<int>(aabb_min.size(0));
  auto key_options = torch::TensorOptions().dtype(torch::kInt64).device(aabb_min.device());
  auto id_options = torch::TensorOptions().dtype(torch::kInt32).device(aabb_min.device());
  auto keys = torch::empty({n}, key_options);
  auto sorted_keys = torch::empty({n}, key_options);
  auto ids_in = torch::empty({n}, id_options);
  auto ids_out = torch::empty({n}, id_options);

  if (n == 0) {
    return {sorted_keys, ids_out};
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int block = 256;
  int grid = (n + block - 1) / block;
  compute_morton_keys_kernel<<<grid, block, 0, stream>>>(
      aabb_min.data_ptr<float>(),
      aabb_max.data_ptr<float>(),
      scene_min.data_ptr<float>(),
      scene_max.data_ptr<float>(),
      keys.data_ptr<int64_t>(),
      ids_in.data_ptr<int32_t>(),
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
      nullptr,
      temp_bytes,
      keys.data_ptr<int64_t>(),
      sorted_keys.data_ptr<int64_t>(),
      ids_in.data_ptr<int32_t>(),
      ids_out.data_ptr<int32_t>(),
      n,
      0,
      64,
      stream);

  auto temp = torch::empty(
      {static_cast<int64_t>(temp_bytes)},
      torch::TensorOptions().dtype(torch::kUInt8).device(aabb_min.device()));

  cub::DeviceRadixSort::SortPairs(
      temp.data_ptr(),
      temp_bytes,
      keys.data_ptr<int64_t>(),
      sorted_keys.data_ptr<int64_t>(),
      ids_in.data_ptr<int32_t>(),
      ids_out.data_ptr<int32_t>(),
      n,
      0,
      64,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {sorted_keys, ids_out};
}

std::vector<torch::Tensor> build_lbvh_topology(torch::Tensor sorted_keys) {
  TORCH_CHECK(sorted_keys.is_cuda(), "sorted_keys must be a CUDA tensor");
  TORCH_CHECK(sorted_keys.dtype() == torch::kInt64, "sorted_keys must be int64");
  TORCH_CHECK(sorted_keys.is_contiguous(), "sorted_keys must be contiguous");
  TORCH_CHECK(sorted_keys.dim() == 1, "sorted_keys must be rank-1");
  TORCH_CHECK(sorted_keys.numel() >= 1, "sorted_keys must be non-empty");
  TORCH_CHECK(sorted_keys.numel() <= static_cast<int64_t>((INT32_MAX + 1LL) / 2), "too many keys");

  int n = static_cast<int>(sorted_keys.numel());
  int num_nodes = 2 * n - 1;
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(sorted_keys.device());
  auto left = torch::full({num_nodes}, -1, options);
  auto right = torch::full({num_nodes}, -1, options);
  auto parent = torch::full({num_nodes}, -1, options);
  auto start = torch::full({num_nodes}, -1, options);
  auto count = torch::zeros({num_nodes}, options);
  auto range_start = torch::full({std::max(n - 1, 0)}, -1, options);
  auto range_end = torch::full({std::max(n - 1, 0)}, -1, options);
  auto split = torch::full({std::max(n - 1, 0)}, -1, options);

  if (n == 1) {
    start[0] = 0;
    count[0] = 1;
    return {left, right, parent, start, count, range_start, range_end, split};
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int block = 256;
  int grid = (n + block - 1) / block;
  build_lbvh_topology_kernel<<<grid, block, 0, stream>>>(
      sorted_keys.data_ptr<int64_t>(),
      left.data_ptr<int32_t>(),
      right.data_ptr<int32_t>(),
      parent.data_ptr<int32_t>(),
      start.data_ptr<int32_t>(),
      count.data_ptr<int32_t>(),
      range_start.data_ptr<int32_t>(),
      range_end.data_ptr<int32_t>(),
      split.data_ptr<int32_t>(),
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {left, right, parent, start, count, range_start, range_end, split};
}

std::vector<torch::Tensor> build_lbvh_topology_and_bounds(
    torch::Tensor sorted_keys,
    torch::Tensor sorted_prim_ids,
    torch::Tensor aabb_min,
    torch::Tensor aabb_max) {
  TORCH_CHECK(sorted_prim_ids.is_cuda(), "sorted_prim_ids must be a CUDA tensor");
  TORCH_CHECK(aabb_min.is_cuda(), "aabb_min must be a CUDA tensor");
  TORCH_CHECK(aabb_max.is_cuda(), "aabb_max must be a CUDA tensor");
  TORCH_CHECK(sorted_prim_ids.dtype() == torch::kInt32, "sorted_prim_ids must be int32");
  TORCH_CHECK(aabb_min.dtype() == torch::kFloat32, "aabb_min must be float32");
  TORCH_CHECK(aabb_max.dtype() == torch::kFloat32, "aabb_max must be float32");
  TORCH_CHECK(sorted_prim_ids.is_contiguous(), "sorted_prim_ids must be contiguous");
  TORCH_CHECK(aabb_min.is_contiguous(), "aabb_min must be contiguous");
  TORCH_CHECK(aabb_max.is_contiguous(), "aabb_max must be contiguous");
  TORCH_CHECK(aabb_min.dim() == 2 && aabb_min.size(1) == 3, "aabb_min must have shape [N, 3]");
  TORCH_CHECK(aabb_max.sizes() == aabb_min.sizes(), "aabb_max must match aabb_min shape");
  TORCH_CHECK(sorted_keys.numel() == sorted_prim_ids.numel(), "sorted key/id length mismatch");
  TORCH_CHECK(sorted_keys.numel() == aabb_min.size(0), "AABB count must match sorted keys");

  auto topology = build_lbvh_topology(sorted_keys);
  auto left = topology[0];
  auto right = topology[1];
  auto parent = topology[2];
  auto start = topology[3];
  auto count = topology[4];
  auto range_start = topology[5];
  auto range_end = topology[6];
  auto split = topology[7];

  int n = static_cast<int>(sorted_keys.numel());
  int num_nodes = 2 * n - 1;
  auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(sorted_keys.device());
  auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(sorted_keys.device());
  auto bounds_min = torch::full({num_nodes, 3}, std::numeric_limits<float>::infinity(), float_options);
  auto bounds_max = torch::full({num_nodes, 3}, -std::numeric_limits<float>::infinity(), float_options);
  auto child_counter = torch::zeros({num_nodes}, int_options);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int block = 256;
  int grid = (n + block - 1) / block;
  initialize_lbvh_leaf_bounds_kernel<<<grid, block, 0, stream>>>(
      sorted_prim_ids.data_ptr<int32_t>(),
      aabb_min.data_ptr<float>(),
      aabb_max.data_ptr<float>(),
      bounds_min.data_ptr<float>(),
      bounds_max.data_ptr<float>(),
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  reduce_lbvh_internal_bounds_kernel<<<grid, block, 0, stream>>>(
      parent.data_ptr<int32_t>(),
      bounds_min.data_ptr<float>(),
      bounds_max.data_ptr<float>(),
      child_counter.data_ptr<int32_t>(),
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {
      left,
      right,
      parent,
      start,
      count,
      range_start,
      range_end,
      split,
      bounds_min,
      bounds_max};
}
"""


def main() -> None:
    args = _parse_args()
    try:
        import torch
        from torch.utils.cpp_extension import load_inline
    except Exception as exc:  # pragma: no cover - local spike guard.
        raise SystemExit(f"torch extension tooling is unavailable: {exc}") from exc

    try:
        import warp as wp
    except Exception as exc:  # pragma: no cover - local spike guard.
        raise SystemExit(f"warp is unavailable: {exc}") from exc

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to torch")

    module = load_inline(
        name="robot_sim_cuda_lbvh_spike_v2",
        cpp_sources=[CPP_SOURCE],
        cuda_sources=[CUDA_SOURCE],
        with_cuda=True,
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=args.verbose,
    )

    device = torch.device(args.device)
    codes = torch.tensor([17, 3, 42, 3, 9, 31, 0, 9], device=device, dtype=torch.int64)
    sorted_codes, sorted_prim_ids = module.cub_sort_codes(codes)
    torch.cuda.synchronize(device)

    expected_codes, _ = torch.sort(codes)
    if not torch.equal(sorted_codes, expected_codes):
        raise AssertionError(
            f"CUB sort mismatch: got {sorted_codes.cpu().tolist()}, expected {expected_codes.cpu().tolist()}"
        )

    warp_written = _verify_warp_from_torch(wp, sorted_prim_ids)
    morton_keys, morton_prim_ids = _run_morton_aabb_spike(torch, module, device)
    topology = _run_topology_spike(torch, module, morton_keys)
    bounded = _run_bounds_spike(torch, module, morton_keys, morton_prim_ids)

    print("CUDA LBVH extension spike passed:")
    print(f"  torch_device: {device}")
    print(f"  sorted_codes: {sorted_codes.cpu().tolist()}")
    print(f"  sorted_primitive_ids: {sorted_prim_ids.cpu().tolist()}")
    print(f"  warp_from_torch_plus_one: {warp_written.cpu().tolist()}")
    print(f"  morton_sorted_keys: {morton_keys.cpu().tolist()}")
    print(f"  morton_sorted_primitive_ids: {morton_prim_ids.cpu().tolist()}")
    print(f"  topology_left: {topology['left']}")
    print(f"  topology_right: {topology['right']}")
    print(f"  topology_parent: {topology['parent']}")
    print(f"  root_bounds_min: {bounded['bounds_min'][0]}")
    print(f"  root_bounds_max: {bounded['bounds_max'][0]}")


def _run_morton_aabb_spike(torch, module, device):
    aabb_min = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [0.50, 0.00, 0.00],
            [0.00, 0.50, 0.00],
            [0.00, 0.00, 0.50],
            [0.25, 0.25, 0.25],
        ],
        device=device,
        dtype=torch.float32,
    )
    aabb_max = aabb_min + 0.05
    scene_min = torch.min(aabb_min, dim=0).values.contiguous()
    scene_max = torch.max(aabb_max, dim=0).values.contiguous()

    sorted_keys, sorted_prim_ids = module.morton_sort_aabbs(
        aabb_min.contiguous(),
        aabb_max.contiguous(),
        scene_min,
        scene_max,
    )
    torch.cuda.synchronize(device)
    if not torch.all(sorted_keys[:-1] <= sorted_keys[1:]):
        raise AssertionError(f"Morton keys are not sorted: {sorted_keys.cpu().tolist()}")
    expected_ids = torch.arange(aabb_min.shape[0], device=device, dtype=torch.int32)
    if not torch.equal(torch.sort(sorted_prim_ids).values, expected_ids):
        raise AssertionError(
            "Morton primitive ids are not a permutation: "
            f"got {sorted_prim_ids.cpu().tolist()}, expected ids {expected_ids.cpu().tolist()}"
        )
    low_bits = (sorted_keys & 0xFFFFFFFF).to(torch.int32)
    if not torch.equal(low_bits, sorted_prim_ids):
        raise AssertionError(
            f"extended key low bits do not match primitive ids: "
            f"keys={sorted_keys.cpu().tolist()}, ids={sorted_prim_ids.cpu().tolist()}"
        )
    return sorted_keys, sorted_prim_ids


def _sample_aabbs(torch, device):
    aabb_min = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [0.50, 0.00, 0.00],
            [0.00, 0.50, 0.00],
            [0.00, 0.00, 0.50],
            [0.25, 0.25, 0.25],
        ],
        device=device,
        dtype=torch.float32,
    )
    return aabb_min, aabb_min + 0.05


def _run_topology_spike(torch, module, sorted_keys):
    arrays = module.build_lbvh_topology(sorted_keys.contiguous())
    torch.cuda.synchronize(sorted_keys.device)
    names = ("left", "right", "parent", "start", "count", "range_start", "range_end", "split")
    host = {name: tensor.cpu().tolist() for name, tensor in zip(names, arrays, strict=True)}
    _validate_topology(host, n=sorted_keys.numel())
    return host


def _run_bounds_spike(torch, module, sorted_keys, sorted_prim_ids):
    aabb_min, aabb_max = _sample_aabbs(torch, sorted_keys.device)
    arrays = module.build_lbvh_topology_and_bounds(
        sorted_keys.contiguous(),
        sorted_prim_ids.contiguous(),
        aabb_min.contiguous(),
        aabb_max.contiguous(),
    )
    torch.cuda.synchronize(sorted_keys.device)
    names = (
        "left",
        "right",
        "parent",
        "start",
        "count",
        "range_start",
        "range_end",
        "split",
        "bounds_min",
        "bounds_max",
    )
    host = {name: tensor.cpu().tolist() for name, tensor in zip(names, arrays, strict=True)}
    _validate_topology(host, n=sorted_keys.numel())
    expected_min = torch.min(aabb_min, dim=0).values.cpu().tolist()
    expected_max = torch.max(aabb_max, dim=0).values.cpu().tolist()
    _assert_close_vec(host["bounds_min"][0], expected_min, label="root bounds min")
    _assert_close_vec(host["bounds_max"][0], expected_max, label="root bounds max")

    leaf_base = sorted_keys.numel() - 1
    sorted_ids = sorted_prim_ids.cpu().tolist()
    aabb_min_host = aabb_min.cpu().tolist()
    aabb_max_host = aabb_max.cpu().tolist()
    for rank, prim in enumerate(sorted_ids):
        leaf = leaf_base + rank
        _assert_close_vec(host["bounds_min"][leaf], aabb_min_host[prim], label=f"leaf {leaf} min")
        _assert_close_vec(host["bounds_max"][leaf], aabb_max_host[prim], label=f"leaf {leaf} max")
    return host


def _validate_topology(topology: dict[str, list[int]], *, n: int) -> None:
    num_nodes = 2 * n - 1
    leaf_base = n - 1
    left = topology["left"]
    right = topology["right"]
    parent = topology["parent"]
    start = topology["start"]
    count = topology["count"]

    if len(left) != num_nodes or len(right) != num_nodes or len(parent) != num_nodes:
        raise AssertionError("topology node arrays have the wrong length")
    if parent[0] != -1:
        raise AssertionError(f"root parent must be -1, got {parent[0]}")

    for node in range(n - 1):
        if count[node] != 0 or start[node] != -1:
            raise AssertionError(f"internal node {node} has invalid start/count")
        for child in (left[node], right[node]):
            if child < 0 or child >= num_nodes:
                raise AssertionError(f"internal node {node} has invalid child {child}")
            if parent[child] != node:
                raise AssertionError(f"child {child} parent mismatch: got {parent[child]}, expected {node}")

    for leaf_rank in range(n):
        leaf = leaf_base + leaf_rank
        if count[leaf] != 1 or start[leaf] != leaf_rank:
            raise AssertionError(f"leaf {leaf} has invalid start/count")
        if left[leaf] != -1 or right[leaf] != -1:
            raise AssertionError(f"leaf {leaf} should not have children")

    reachable: set[int] = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in reachable:
            raise AssertionError(f"node {node} reached twice")
        reachable.add(node)
        if node < leaf_base:
            stack.append(left[node])
            stack.append(right[node])
    if reachable != set(range(num_nodes)):
        missing = sorted(set(range(num_nodes)) - reachable)
        raise AssertionError(f"topology is not fully reachable, missing={missing}")


def _assert_close_vec(actual: list[float], expected: list[float], *, label: str) -> None:
    for axis, (a, e) in enumerate(zip(actual, expected, strict=True)):
        if abs(a - e) > 1.0e-6:
            raise AssertionError(f"{label}[{axis}] mismatch: got {actual}, expected {expected}")


def _verify_warp_from_torch(wp, sorted_prim_ids):
    import torch

    wp.init()
    sorted_ids_wp = wp.from_torch(sorted_prim_ids, dtype=wp.int32)
    out = torch.empty_like(sorted_prim_ids)
    out_wp = wp.from_torch(out, dtype=wp.int32)
    wp.launch(
        _plus_one_kernel,
        dim=sorted_prim_ids.numel(),
        inputs=[sorted_ids_wp, out_wp],
        device=sorted_ids_wp.device,
    )
    wp.synchronize()
    expected = sorted_prim_ids + 1
    if not torch.equal(out, expected):
        raise AssertionError(
            f"Warp from_torch mismatch: got {out.cpu().tolist()}, expected {expected.cpu().tolist()}"
        )
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verbose", action="store_true", help="Print torch extension build logs.")
    return parser.parse_args()


try:
    import warp as _wp_for_kernel
except Exception:  # pragma: no cover - import guard for help text / CPU envs.
    _wp_for_kernel = None

if _wp_for_kernel is not None:

    @_wp_for_kernel.kernel
    def _plus_one_kernel(
        src: _wp_for_kernel.array(dtype=_wp_for_kernel.int32),
        dst: _wp_for_kernel.array(dtype=_wp_for_kernel.int32),
    ):
        i = _wp_for_kernel.tid()
        dst[i] = src[i] + 1


if __name__ == "__main__":
    main()
