"""Sole CUDA scene kernel source literals; no backend imports."""

CPP_SOURCE = r"""
#include <torch/extension.h>

void update_triangles_world_aabb_out(
    torch::Tensor triangle_vertices_local,
    torch::Tensor triangle_instance_index,
    torch::Tensor instance_body_index,
    torch::Tensor instance_X_body_geometry_R,
    torch::Tensor instance_X_body_geometry_r,
    torch::Tensor frame_x_world_R,
    torch::Tensor frame_x_world_r,
    int64_t env_idx,
    torch::Tensor triangle_v0_world,
    torch::Tensor triangle_e1_world,
    torch::Tensor triangle_e2_world,
    torch::Tensor triangle_normal_world,
    torch::Tensor triangle_aabb_min,
    torch::Tensor triangle_aabb_max);

void update_planes_world_out(
    torch::Tensor plane_normal_local,
    torch::Tensor plane_point_local,
    torch::Tensor plane_instance_index,
    torch::Tensor instance_body_index,
    torch::Tensor instance_X_body_geometry_R,
    torch::Tensor instance_X_body_geometry_r,
    torch::Tensor frame_x_world_R,
    torch::Tensor frame_x_world_r,
    int64_t env_idx,
    torch::Tensor plane_normal_world,
    torch::Tensor plane_point_world);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update_triangles_world_aabb_out", &update_triangles_world_aabb_out);
  m.def("update_planes_world_out", &update_planes_world_out);
}
"""


CUDA_SOURCE = r"""
#include <torch/types.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <climits>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr float kNormalEpsilon = 1.0e-8f;

struct RigidTransform {
  float R[9];
  float r[3];
};

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
  return span_start(value) + static_cast<uintptr_t>(value.numel() * value.element_size());
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
          "CUDA scene output overlaps an input tensor");
    }
    for (int prior = 0; prior < output_index; ++prior) {
      TORCH_CHECK(
          !spans_overlap(*outputs[output_index], *outputs[prior]),
          "CUDA scene output tensors overlap");
    }
  }
}

__device__ __forceinline__ RigidTransform compose_transform(
    int instance_index,
    int env_idx,
    int body_count,
    const int32_t* instance_body_index,
    const float* instance_R,
    const float* instance_r,
    const float* frame_R,
    const float* frame_r) {
  RigidTransform result;
  float Rwb[9] = {
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f};
  float rwb[3] = {0.0f, 0.0f, 0.0f};
  const int body_index = instance_body_index[instance_index];
  if (body_index >= 0 && body_index < body_count) {
    const int body_offset = (env_idx * body_count + body_index) * 9;
    const int translation_offset = (env_idx * body_count + body_index) * 3;
    #pragma unroll
    for (int element = 0; element < 9; ++element) {
      Rwb[element] = frame_R[body_offset + element];
    }
    #pragma unroll
    for (int element = 0; element < 3; ++element) {
      rwb[element] = frame_r[translation_offset + element];
    }
  }

  const float* Rbg = instance_R + instance_index * 9;
  const float* rbg = instance_r + instance_index * 3;
  #pragma unroll
  for (int row = 0; row < 3; ++row) {
    #pragma unroll
    for (int col = 0; col < 3; ++col) {
      result.R[row * 3 + col] =
          Rwb[row * 3 + 0] * Rbg[0 * 3 + col] +
          Rwb[row * 3 + 1] * Rbg[1 * 3 + col] +
          Rwb[row * 3 + 2] * Rbg[2 * 3 + col];
    }
    result.r[row] =
        rwb[row] +
        Rwb[row * 3 + 0] * rbg[0] +
        Rwb[row * 3 + 1] * rbg[1] +
        Rwb[row * 3 + 2] * rbg[2];
  }
  return result;
}

__device__ __forceinline__ void transform_point(
    const RigidTransform& transform,
    const float* point,
    float* output) {
  #pragma unroll
  for (int row = 0; row < 3; ++row) {
    output[row] =
        transform.r[row] +
        transform.R[row * 3 + 0] * point[0] +
        transform.R[row * 3 + 1] * point[1] +
        transform.R[row * 3 + 2] * point[2];
  }
}

__global__ void update_triangles_kernel(
    int triangle_count,
    int env_idx,
    int body_count,
    const float* triangle_vertices_local,
    const int32_t* triangle_instance_index,
    const int32_t* instance_body_index,
    const float* instance_R,
    const float* instance_r,
    const float* frame_R,
    const float* frame_r,
    float* triangle_v0_world,
    float* triangle_e1_world,
    float* triangle_e2_world,
    float* triangle_normal_world,
    float* triangle_aabb_min,
    float* triangle_aabb_max) {
  const int primitive = blockIdx.x * blockDim.x + threadIdx.x;
  if (primitive >= triangle_count) {
    return;
  }
  const int instance_index = triangle_instance_index[primitive];
  const RigidTransform transform = compose_transform(
      instance_index,
      env_idx,
      body_count,
      instance_body_index,
      instance_R,
      instance_r,
      frame_R,
      frame_r);
  const float* local = triangle_vertices_local + primitive * 9;
  float p0[3];
  float p1[3];
  float p2[3];
  transform_point(transform, local + 0, p0);
  transform_point(transform, local + 3, p1);
  transform_point(transform, local + 6, p2);
  float e1[3];
  float e2[3];
  #pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    e1[axis] = p1[axis] - p0[axis];
    e2[axis] = p2[axis] - p0[axis];
    triangle_v0_world[primitive * 3 + axis] = p0[axis];
    triangle_e1_world[primitive * 3 + axis] = e1[axis];
    triangle_e2_world[primitive * 3 + axis] = e2[axis];
    triangle_aabb_min[primitive * 3 + axis] = fminf(p0[axis], fminf(p1[axis], p2[axis]));
    triangle_aabb_max[primitive * 3 + axis] = fmaxf(p0[axis], fmaxf(p1[axis], p2[axis]));
  }
  float normal[3] = {
      e1[1] * e2[2] - e1[2] * e2[1],
      e1[2] * e2[0] - e1[0] * e2[2],
      e1[0] * e2[1] - e1[1] * e2[0]};
  const float length = sqrtf(
      normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
  #pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    triangle_normal_world[primitive * 3 + axis] =
        length > kNormalEpsilon ? normal[axis] / length : 0.0f;
  }
}

__global__ void update_planes_kernel(
    int plane_count,
    int env_idx,
    int body_count,
    const float* plane_normal_local,
    const float* plane_point_local,
    const int32_t* plane_instance_index,
    const int32_t* instance_body_index,
    const float* instance_R,
    const float* instance_r,
    const float* frame_R,
    const float* frame_r,
    float* plane_normal_world,
    float* plane_point_world) {
  const int primitive = blockIdx.x * blockDim.x + threadIdx.x;
  if (primitive >= plane_count) {
    return;
  }
  const int instance_index = plane_instance_index[primitive];
  const RigidTransform transform = compose_transform(
      instance_index,
      env_idx,
      body_count,
      instance_body_index,
      instance_R,
      instance_r,
      frame_R,
      frame_r);
  const float* local_normal = plane_normal_local + primitive * 3;
  const float* local_point = plane_point_local + primitive * 3;
  float point[3];
  transform_point(transform, local_point, point);
  float normal[3];
  #pragma unroll
  for (int row = 0; row < 3; ++row) {
    normal[row] =
        transform.R[row * 3 + 0] * local_normal[0] +
        transform.R[row * 3 + 1] * local_normal[1] +
        transform.R[row * 3 + 2] * local_normal[2];
  }
  const float length = sqrtf(
      normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
  #pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    plane_point_world[primitive * 3 + axis] = point[axis];
    plane_normal_world[primitive * 3 + axis] = length > 0.0f ? normal[axis] / length : 0.0f;
  }
}

}  // namespace

void update_triangles_world_aabb_out(
    torch::Tensor triangle_vertices_local,
    torch::Tensor triangle_instance_index,
    torch::Tensor instance_body_index,
    torch::Tensor instance_X_body_geometry_R,
    torch::Tensor instance_X_body_geometry_r,
    torch::Tensor frame_x_world_R,
    torch::Tensor frame_x_world_r,
    int64_t env_idx,
    torch::Tensor triangle_v0_world,
    torch::Tensor triangle_e1_world,
    torch::Tensor triangle_e2_world,
    torch::Tensor triangle_normal_world,
    torch::Tensor triangle_aabb_min,
    torch::Tensor triangle_aabb_max) {
  TORCH_CHECK(triangle_v0_world.is_cuda(), "triangle_v0_world must be CUDA");
  const c10::cuda::CUDAGuard device_guard(triangle_v0_world.device());
  const torch::Device device = triangle_v0_world.device();
  check_cuda_tensor(triangle_vertices_local, device, torch::kFloat32, "triangle_vertices_local");
  check_cuda_tensor(triangle_instance_index, device, torch::kInt32, "triangle_instance_index");
  check_cuda_tensor(instance_body_index, device, torch::kInt32, "instance_body_index");
  check_cuda_tensor(instance_X_body_geometry_R, device, torch::kFloat32, "instance_X_body_geometry_R");
  check_cuda_tensor(instance_X_body_geometry_r, device, torch::kFloat32, "instance_X_body_geometry_r");
  check_cuda_tensor(frame_x_world_R, device, torch::kFloat32, "frame_x_world_R");
  check_cuda_tensor(frame_x_world_r, device, torch::kFloat32, "frame_x_world_r");
  check_cuda_tensor(triangle_v0_world, device, torch::kFloat32, "triangle_v0_world");
  check_cuda_tensor(triangle_e1_world, device, torch::kFloat32, "triangle_e1_world");
  check_cuda_tensor(triangle_e2_world, device, torch::kFloat32, "triangle_e2_world");
  check_cuda_tensor(triangle_normal_world, device, torch::kFloat32, "triangle_normal_world");
  check_cuda_tensor(triangle_aabb_min, device, torch::kFloat32, "triangle_aabb_min");
  check_cuda_tensor(triangle_aabb_max, device, torch::kFloat32, "triangle_aabb_max");

  const int64_t triangle_count = triangle_vertices_local.size(0);
  const int64_t instance_count = instance_body_index.size(0);
  TORCH_CHECK(instance_body_index.dim() == 1, "instance_body_index shape");
  TORCH_CHECK(
      triangle_vertices_local.dim() == 2 && triangle_vertices_local.size(1) == 9,
      "triangle_vertices_local shape");
  TORCH_CHECK(
      triangle_instance_index.dim() == 1 &&
          triangle_instance_index.size(0) == triangle_count,
      "triangle_instance_index shape");
  TORCH_CHECK(
      instance_X_body_geometry_R.dim() == 2 &&
          instance_X_body_geometry_R.size(0) == instance_count &&
          instance_X_body_geometry_R.size(1) == 9,
      "instance rotation shape");
  TORCH_CHECK(
      instance_X_body_geometry_r.dim() == 2 &&
          instance_X_body_geometry_r.size(0) == instance_count &&
          instance_X_body_geometry_r.size(1) == 3,
      "instance translation shape");
  TORCH_CHECK(
      frame_x_world_R.dim() == 4 && frame_x_world_R.size(2) == 3 &&
          frame_x_world_R.size(3) == 3,
      "frame rotation shape");
  TORCH_CHECK(
      frame_x_world_r.dim() == 3 &&
          frame_x_world_r.size(0) == frame_x_world_R.size(0) &&
          frame_x_world_r.size(1) == frame_x_world_R.size(1) &&
          frame_x_world_r.size(2) == 3,
      "frame translation shape");
  TORCH_CHECK(env_idx >= 0 && env_idx < frame_x_world_R.size(0), "env_idx out of range");
  TORCH_CHECK(
      triangle_count <= INT32_MAX && instance_count <= INT32_MAX &&
          frame_x_world_R.size(0) <= INT32_MAX &&
          frame_x_world_R.size(1) <= INT32_MAX,
      "CUDA scene count exceeds int32");
  const torch::Tensor* const inputs[7] = {
      &triangle_vertices_local,
      &triangle_instance_index,
      &instance_body_index,
      &instance_X_body_geometry_R,
      &instance_X_body_geometry_r,
      &frame_x_world_R,
      &frame_x_world_r};
  const torch::Tensor* const outputs[6] = {
      &triangle_v0_world,
      &triangle_e1_world,
      &triangle_e2_world,
      &triangle_normal_world,
      &triangle_aabb_min,
      &triangle_aabb_max};
  for (int index = 0; index < 6; ++index) {
    TORCH_CHECK(
        outputs[index]->dim() == 2 && outputs[index]->size(0) == triangle_count &&
            outputs[index]->size(1) == 3,
        "triangle output shape");
  }
  check_output_disjointness(inputs, outputs);

  if (triangle_count == 0) {
    return;
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(triangle_v0_world.get_device());
  const int blocks = (static_cast<int>(triangle_count) + kThreads - 1) / kThreads;
  update_triangles_kernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<int>(triangle_count),
      static_cast<int>(env_idx),
      static_cast<int>(frame_x_world_R.size(1)),
      triangle_vertices_local.data_ptr<float>(),
      triangle_instance_index.data_ptr<int32_t>(),
      instance_body_index.data_ptr<int32_t>(),
      instance_X_body_geometry_R.data_ptr<float>(),
      instance_X_body_geometry_r.data_ptr<float>(),
      frame_x_world_R.data_ptr<float>(),
      frame_x_world_r.data_ptr<float>(),
      triangle_v0_world.data_ptr<float>(),
      triangle_e1_world.data_ptr<float>(),
      triangle_e2_world.data_ptr<float>(),
      triangle_normal_world.data_ptr<float>(),
      triangle_aabb_min.data_ptr<float>(),
      triangle_aabb_max.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void update_planes_world_out(
    torch::Tensor plane_normal_local,
    torch::Tensor plane_point_local,
    torch::Tensor plane_instance_index,
    torch::Tensor instance_body_index,
    torch::Tensor instance_X_body_geometry_R,
    torch::Tensor instance_X_body_geometry_r,
    torch::Tensor frame_x_world_R,
    torch::Tensor frame_x_world_r,
    int64_t env_idx,
    torch::Tensor plane_normal_world,
    torch::Tensor plane_point_world) {
  TORCH_CHECK(plane_normal_world.is_cuda(), "plane_normal_world must be CUDA");
  const c10::cuda::CUDAGuard device_guard(plane_normal_world.device());
  const torch::Device device = plane_normal_world.device();
  check_cuda_tensor(plane_normal_local, device, torch::kFloat32, "plane_normal_local");
  check_cuda_tensor(plane_point_local, device, torch::kFloat32, "plane_point_local");
  check_cuda_tensor(plane_instance_index, device, torch::kInt32, "plane_instance_index");
  check_cuda_tensor(instance_body_index, device, torch::kInt32, "instance_body_index");
  check_cuda_tensor(instance_X_body_geometry_R, device, torch::kFloat32, "instance_X_body_geometry_R");
  check_cuda_tensor(instance_X_body_geometry_r, device, torch::kFloat32, "instance_X_body_geometry_r");
  check_cuda_tensor(frame_x_world_R, device, torch::kFloat32, "frame_x_world_R");
  check_cuda_tensor(frame_x_world_r, device, torch::kFloat32, "frame_x_world_r");
  check_cuda_tensor(plane_normal_world, device, torch::kFloat32, "plane_normal_world");
  check_cuda_tensor(plane_point_world, device, torch::kFloat32, "plane_point_world");

  const int64_t plane_count = plane_normal_local.size(0);
  const int64_t instance_count = instance_body_index.size(0);
  TORCH_CHECK(instance_body_index.dim() == 1, "instance_body_index shape");
  TORCH_CHECK(plane_normal_local.dim() == 2 && plane_normal_local.size(1) == 3, "plane_normal_local shape");
  TORCH_CHECK(
      plane_point_local.dim() == 2 && plane_point_local.size(0) == plane_count &&
          plane_point_local.size(1) == 3,
      "plane_point_local shape");
  TORCH_CHECK(
      plane_instance_index.dim() == 1 &&
          plane_instance_index.size(0) == plane_count,
      "plane_instance_index shape");
  TORCH_CHECK(
      instance_X_body_geometry_R.dim() == 2 &&
          instance_X_body_geometry_R.size(0) == instance_count &&
          instance_X_body_geometry_R.size(1) == 9,
      "instance rotation shape");
  TORCH_CHECK(
      instance_X_body_geometry_r.dim() == 2 &&
          instance_X_body_geometry_r.size(0) == instance_count &&
          instance_X_body_geometry_r.size(1) == 3,
      "instance translation shape");
  TORCH_CHECK(
      frame_x_world_R.dim() == 4 && frame_x_world_R.size(2) == 3 &&
          frame_x_world_R.size(3) == 3,
      "frame rotation shape");
  TORCH_CHECK(
      frame_x_world_r.dim() == 3 &&
          frame_x_world_r.size(0) == frame_x_world_R.size(0) &&
          frame_x_world_r.size(1) == frame_x_world_R.size(1) &&
          frame_x_world_r.size(2) == 3,
      "frame translation shape");
  TORCH_CHECK(env_idx >= 0 && env_idx < frame_x_world_R.size(0), "env_idx out of range");
  TORCH_CHECK(
      plane_count <= INT32_MAX && instance_count <= INT32_MAX &&
          frame_x_world_R.size(0) <= INT32_MAX &&
          frame_x_world_R.size(1) <= INT32_MAX,
      "CUDA scene count exceeds int32");
  const torch::Tensor* const inputs[8] = {
      &plane_normal_local,
      &plane_point_local,
      &plane_instance_index,
      &instance_body_index,
      &instance_X_body_geometry_R,
      &instance_X_body_geometry_r,
      &frame_x_world_R,
      &frame_x_world_r};
  const torch::Tensor* const outputs[2] = {
      &plane_normal_world,
      &plane_point_world};
  for (int index = 0; index < 2; ++index) {
    TORCH_CHECK(
        outputs[index]->dim() == 2 && outputs[index]->size(0) == plane_count &&
            outputs[index]->size(1) == 3,
        "plane output shape");
  }
  check_output_disjointness(inputs, outputs);

  if (plane_count == 0) {
    return;
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(plane_normal_world.get_device());
  const int blocks = (static_cast<int>(plane_count) + kThreads - 1) / kThreads;
  update_planes_kernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<int>(plane_count),
      static_cast<int>(env_idx),
      static_cast<int>(frame_x_world_R.size(1)),
      plane_normal_local.data_ptr<float>(),
      plane_point_local.data_ptr<float>(),
      plane_instance_index.data_ptr<int32_t>(),
      instance_body_index.data_ptr<int32_t>(),
      instance_X_body_geometry_R.data_ptr<float>(),
      instance_X_body_geometry_r.data_ptr<float>(),
      frame_x_world_R.data_ptr<float>(),
      frame_x_world_r.data_ptr<float>(),
      plane_normal_world.data_ptr<float>(),
      plane_point_world.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
