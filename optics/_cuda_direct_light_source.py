"""Sole CUDA direct light kernel source literals; no backend imports."""

CPP_SOURCE = r"""
#include <torch/extension.h>

void pinhole_camera_rays_out(
    torch::Tensor origins,
    torch::Tensor directions,
    int64_t width,
    int64_t height,
    double fx,
    double fy,
    double cx,
    double cy,
    double origin_x,
    double origin_y,
    double origin_z,
    double r00,
    double r01,
    double r02,
    double r10,
    double r11,
    double r12,
    double r20,
    double r21,
    double r22);

void first_hit_rays_out(
    torch::Tensor origins,
    torch::Tensor directions,
    double max_distance,
    torch::Tensor plane_normals,
    torch::Tensor plane_points,
    torch::Tensor plane_numeric_ids,
    torch::Tensor plane_source_keys,
    torch::Tensor plane_role_masks,
    torch::Tensor plane_material_indices,
    int64_t num_planes,
    torch::Tensor triangle_v0,
    torch::Tensor triangle_e1,
    torch::Tensor triangle_e2,
    torch::Tensor triangle_normal,
    torch::Tensor triangle_numeric_ids,
    torch::Tensor triangle_source_keys,
    torch::Tensor triangle_role_masks,
    torch::Tensor triangle_material_indices,
    torch::Tensor bvh_bounds_min,
    torch::Tensor bvh_bounds_max,
    torch::Tensor bvh_left,
    torch::Tensor bvh_right,
    torch::Tensor bvh_start,
    torch::Tensor bvh_count,
    torch::Tensor bvh_prim_ids,
    int64_t num_bvh_nodes,
    int64_t sensor_role_mask,
    torch::Tensor hit_mask,
    torch::Tensor range_m,
    torch::Tensor position_world,
    torch::Tensor normal_world,
    torch::Tensor numeric_instance_id,
    torch::Tensor material_index,
    torch::Tensor bvh_stack_overflow_count,
    torch::Tensor bvh_max_stack_depth);

void shade_direct_light_no_shadow_out(
    torch::Tensor hit_mask,
    torch::Tensor position_world,
    torch::Tensor normal_world,
    torch::Tensor material_index,
    torch::Tensor material_albedo_rgb,
    torch::Tensor light_kind,
    torch::Tensor light_position_or_direction_world,
    torch::Tensor light_intensity,
    torch::Tensor light_color_rgb,
    int64_t num_lights,
    double ambient_r,
    double ambient_g,
    double ambient_b,
    double background_r,
    double background_g,
    double background_b,
    torch::Tensor rgb,
    torch::Tensor intensity,
    torch::Tensor shadow_stack_overflow_count,
    torch::Tensor shadow_max_stack_depth);

void shade_direct_light_no_shadow_diagnostics_out(
    torch::Tensor hit_mask,
    torch::Tensor position_world,
    torch::Tensor normal_world,
    torch::Tensor material_index,
    torch::Tensor material_albedo_rgb,
    torch::Tensor light_kind,
    torch::Tensor light_position_or_direction_world,
    torch::Tensor light_intensity,
    torch::Tensor light_color_rgb,
    int64_t num_lights,
    double ambient_r,
    double ambient_g,
    double ambient_b,
    double background_r,
    double background_g,
    double background_b,
    torch::Tensor rgb,
    torch::Tensor intensity,
    torch::Tensor shadow_stack_overflow_count,
    torch::Tensor shadow_max_stack_depth,
    torch::Tensor traversal_counter_0,
    torch::Tensor traversal_counter_1,
    torch::Tensor traversal_counter_2,
    torch::Tensor traversal_counter_3,
    torch::Tensor traversal_counter_4,
    torch::Tensor traversal_counter_5,
    torch::Tensor traversal_counter_6,
    torch::Tensor traversal_counter_7,
    torch::Tensor traversal_counter_8);

void shade_direct_light_with_shadows_out(
    torch::Tensor hit_mask,
    torch::Tensor position_world,
    torch::Tensor normal_world,
    torch::Tensor material_index,
    torch::Tensor material_albedo_rgb,
    torch::Tensor light_kind,
    torch::Tensor light_position_or_direction_world,
    torch::Tensor light_intensity,
    torch::Tensor light_color_rgb,
    int64_t num_lights,
    torch::Tensor triangle_v0,
    torch::Tensor triangle_e1,
    torch::Tensor triangle_e2,
    torch::Tensor triangle_normal,
    torch::Tensor triangle_role_masks,
    torch::Tensor bvh_bounds_min,
    torch::Tensor bvh_bounds_max,
    torch::Tensor bvh_left,
    torch::Tensor bvh_right,
    torch::Tensor bvh_start,
    torch::Tensor bvh_count,
    torch::Tensor bvh_prim_ids,
    int64_t num_bvh_nodes,
    int64_t sensor_role_mask,
    double shadow_bias,
    double ambient_r,
    double ambient_g,
    double ambient_b,
    double background_r,
    double background_g,
    double background_b,
    torch::Tensor rgb,
    torch::Tensor intensity,
    torch::Tensor shadow_stack_overflow_count,
    torch::Tensor shadow_max_stack_depth);

void shade_direct_light_with_shadows_diagnostics_out(
    torch::Tensor hit_mask,
    torch::Tensor position_world,
    torch::Tensor normal_world,
    torch::Tensor material_index,
    torch::Tensor material_albedo_rgb,
    torch::Tensor light_kind,
    torch::Tensor light_position_or_direction_world,
    torch::Tensor light_intensity,
    torch::Tensor light_color_rgb,
    int64_t num_lights,
    torch::Tensor triangle_v0,
    torch::Tensor triangle_e1,
    torch::Tensor triangle_e2,
    torch::Tensor triangle_normal,
    torch::Tensor triangle_role_masks,
    torch::Tensor bvh_bounds_min,
    torch::Tensor bvh_bounds_max,
    torch::Tensor bvh_left,
    torch::Tensor bvh_right,
    torch::Tensor bvh_start,
    torch::Tensor bvh_count,
    torch::Tensor bvh_prim_ids,
    int64_t num_bvh_nodes,
    int64_t sensor_role_mask,
    double shadow_bias,
    double ambient_r,
    double ambient_g,
    double ambient_b,
    double background_r,
    double background_g,
    double background_b,
    torch::Tensor rgb,
    torch::Tensor intensity,
    torch::Tensor shadow_stack_overflow_count,
    torch::Tensor shadow_max_stack_depth,
    torch::Tensor traversal_counter_0,
    torch::Tensor traversal_counter_1,
    torch::Tensor traversal_counter_2,
    torch::Tensor traversal_counter_3,
    torch::Tensor traversal_counter_4,
    torch::Tensor traversal_counter_5,
    torch::Tensor traversal_counter_6,
    torch::Tensor traversal_counter_7,
    torch::Tensor traversal_counter_8);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pinhole_camera_rays_out", torch::wrap_pybind_function(pinhole_camera_rays_out), "pinhole_camera_rays_out");
  m.def("first_hit_rays_out", torch::wrap_pybind_function(first_hit_rays_out), "first_hit_rays_out");
  m.def("shade_direct_light_no_shadow_out", torch::wrap_pybind_function(shade_direct_light_no_shadow_out), "shade_direct_light_no_shadow_out");
  m.def("shade_direct_light_no_shadow_diagnostics_out", torch::wrap_pybind_function(shade_direct_light_no_shadow_diagnostics_out), "shade_direct_light_no_shadow_diagnostics_out");
  m.def("shade_direct_light_with_shadows_out", torch::wrap_pybind_function(shade_direct_light_with_shadows_out), "shade_direct_light_with_shadows_out");
  m.def("shade_direct_light_with_shadows_diagnostics_out", torch::wrap_pybind_function(shade_direct_light_with_shadows_diagnostics_out), "shade_direct_light_with_shadows_diagnostics_out");
}
"""


CUDA_SOURCE = r"""
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <cstdint>
#include <limits>
namespace {

constexpr float kBuildEps = 1.0e-8f;
constexpr float kDirEps = 1.0e-8f;
constexpr float kTEps = 1.0e-5f;
constexpr int kMaxBvhStack = 32;

__global__ void pinhole_camera_rays_kernel(
    int width,
    int height,
    float fx,
    float fy,
    float cx,
    float cy,
    float origin_x,
    float origin_y,
    float origin_z,
    float r00,
    float r01,
    float r02,
    float r10,
    float r11,
    float r12,
    float r20,
    float r21,
    float r22,
    float* origins,
    float* directions) {
  int ray = blockIdx.x * blockDim.x + threadIdx.x;
  int num_rays = width * height;
  if (ray >= num_rays) {
    return;
  }

  int px = ray % width;
  int py = ray / width;
  float camera_x = (static_cast<float>(px) - cx) / fx;
  float camera_y = (static_cast<float>(py) - cy) / fy;
  float inv_norm = rsqrtf(camera_x * camera_x + camera_y * camera_y + 1.0f);
  camera_x *= inv_norm;
  camera_y *= inv_norm;
  float camera_z = inv_norm;

  origins[ray * 3 + 0] = origin_x;
  origins[ray * 3 + 1] = origin_y;
  origins[ray * 3 + 2] = origin_z;
  directions[ray * 3 + 0] = r00 * camera_x + r01 * camera_y + r02 * camera_z;
  directions[ray * 3 + 1] = r10 * camera_x + r11 * camera_y + r12 * camera_z;
  directions[ray * 3 + 2] = r20 * camera_x + r21 * camera_y + r22 * camera_z;
}

__device__ __forceinline__ bool better_hit(float t, int64_t key, float best_t, int64_t best_key) {
  if (t < best_t - kTEps) {
    return true;
  }
  return fabsf(t - best_t) <= kTEps && key < best_key;
}

__device__ __forceinline__ bool intersect_aabb_axis(
    float origin,
    float direction,
    float lower,
    float upper,
    float* t_min,
    float* t_max) {
  if (fabsf(direction) <= kDirEps) {
    return origin >= lower && origin <= upper;
  }
  float inv_d = 1.0f / direction;
  float t0 = (lower - origin) * inv_d;
  float t1 = (upper - origin) * inv_d;
  if (t0 > t1) {
    float tmp = t0;
    t0 = t1;
    t1 = tmp;
  }
  *t_min = fmaxf(*t_min, t0);
  *t_max = fminf(*t_max, t1);
  return *t_min <= *t_max;
}

__device__ __forceinline__ bool intersect_aabb(
    float ox,
    float oy,
    float oz,
    float dx,
    float dy,
    float dz,
    const float* bounds_min,
    const float* bounds_max,
    int node,
    float max_distance,
    float* entry_t) {
  float t_min = 0.0f;
  float t_max = max_distance;
  bool hit = intersect_aabb_axis(ox, dx, bounds_min[node * 3 + 0], bounds_max[node * 3 + 0], &t_min, &t_max);
  if (hit) {
    hit = intersect_aabb_axis(oy, dy, bounds_min[node * 3 + 1], bounds_max[node * 3 + 1], &t_min, &t_max);
  }
  if (hit) {
    hit = intersect_aabb_axis(oz, dz, bounds_min[node * 3 + 2], bounds_max[node * 3 + 2], &t_min, &t_max);
  }
  *entry_t = t_min;
  return hit;
}

__device__ __forceinline__ bool intersect_triangle(
    float ox,
    float oy,
    float oz,
    float dx,
    float dy,
    float dz,
    int tri,
    const float* triangle_v0,
    const float* triangle_e1,
    const float* triangle_e2,
    const float* triangle_normal,
    float max_t,
    float* out_t,
    float* out_nx,
    float* out_ny,
    float* out_nz) {
  float v0x = triangle_v0[tri * 3 + 0];
  float v0y = triangle_v0[tri * 3 + 1];
  float v0z = triangle_v0[tri * 3 + 2];
  float e1x = triangle_e1[tri * 3 + 0];
  float e1y = triangle_e1[tri * 3 + 1];
  float e1z = triangle_e1[tri * 3 + 2];
  float e2x = triangle_e2[tri * 3 + 0];
  float e2y = triangle_e2[tri * 3 + 1];
  float e2z = triangle_e2[tri * 3 + 2];

  float pvec_x = dy * e2z - dz * e2y;
  float pvec_y = dz * e2x - dx * e2z;
  float pvec_z = dx * e2y - dy * e2x;
  float det = pvec_x * e1x + pvec_y * e1y + pvec_z * e1z;
  if (fabsf(det) <= kBuildEps) {
    return false;
  }
  float inv_det = 1.0f / det;
  float tvec_x = ox - v0x;
  float tvec_y = oy - v0y;
  float tvec_z = oz - v0z;
  float u = (pvec_x * tvec_x + pvec_y * tvec_y + pvec_z * tvec_z) * inv_det;
  if (u < 0.0f) {
    return false;
  }
  float qvec_x = tvec_y * e1z - tvec_z * e1y;
  float qvec_y = tvec_z * e1x - tvec_x * e1z;
  float qvec_z = tvec_x * e1y - tvec_y * e1x;
  float v = (qvec_x * dx + qvec_y * dy + qvec_z * dz) * inv_det;
  if (v < 0.0f || u + v > 1.0f) {
    return false;
  }
  float t = (qvec_x * e2x + qvec_y * e2y + qvec_z * e2z) * inv_det;
  if (t < 0.0f || t > max_t) {
    return false;
  }
  float nx = triangle_normal[tri * 3 + 0];
  float ny = triangle_normal[tri * 3 + 1];
  float nz = triangle_normal[tri * 3 + 2];
  float ndotd = nx * dx + ny * dy + nz * dz;
  if (ndotd > 0.0f) {
    nx = -nx;
    ny = -ny;
    nz = -nz;
  }
  *out_t = t;
  *out_nx = nx;
  *out_ny = ny;
  *out_nz = nz;
  return true;
}

__global__ void first_hit_rays_kernel(
    const float* origins,
    const float* directions,
    float max_distance,
    const float* plane_normals,
    const float* plane_points,
    const int32_t* plane_numeric_ids,
    const int64_t* plane_source_keys,
    const int64_t* plane_role_masks,
    const int32_t* plane_material_indices,
    int num_planes,
    const float* triangle_v0,
    const float* triangle_e1,
    const float* triangle_e2,
    const float* triangle_normal,
    const int32_t* triangle_numeric_ids,
    const int64_t* triangle_source_keys,
    const int64_t* triangle_role_masks,
    const int32_t* triangle_material_indices,
    const float* bvh_bounds_min,
    const float* bvh_bounds_max,
    const int32_t* bvh_left,
    const int32_t* bvh_right,
    const int32_t* bvh_start,
    const int32_t* bvh_count,
    const int32_t* bvh_prim_ids,
    int num_bvh_nodes,
    int64_t sensor_role_mask,
    int32_t* hit_mask,
    float* range_m,
    float* position_world,
    float* normal_world,
    int32_t* numeric_instance_id,
    int32_t* material_index,
    int32_t* bvh_stack_overflow_count,
    int32_t* bvh_max_stack_depth,
    int num_rays) {
  int ray = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray >= num_rays) {
    return;
  }

  float ox = origins[ray * 3 + 0];
  float oy = origins[ray * 3 + 1];
  float oz = origins[ray * 3 + 2];
  float dx = directions[ray * 3 + 0];
  float dy = directions[ray * 3 + 1];
  float dz = directions[ray * 3 + 2];

  hit_mask[ray] = 0;
  range_m[ray] = INFINITY;
  position_world[ray * 3 + 0] = NAN;
  position_world[ray * 3 + 1] = NAN;
  position_world[ray * 3 + 2] = NAN;
  normal_world[ray * 3 + 0] = NAN;
  normal_world[ray * 3 + 1] = NAN;
  normal_world[ray * 3 + 2] = NAN;
  numeric_instance_id[ray] = 0;
  material_index[ray] = 0;

  float best_t = max_distance;
  int64_t best_key = std::numeric_limits<int64_t>::max();
  float best_px = 0.0f;
  float best_py = 0.0f;
  float best_pz = 0.0f;
  float best_nx = 0.0f;
  float best_ny = 0.0f;
  float best_nz = 0.0f;
  int32_t best_id = 0;
  int32_t best_material_index = 0;
  bool found = false;

  for (int plane = 0; plane < num_planes; ++plane) {
    if ((plane_role_masks[plane] & sensor_role_mask) == 0) {
      continue;
    }
    float nx = plane_normals[plane * 3 + 0];
    float ny = plane_normals[plane * 3 + 1];
    float nz = plane_normals[plane * 3 + 2];
    float px = plane_points[plane * 3 + 0];
    float py = plane_points[plane * 3 + 1];
    float pz = plane_points[plane * 3 + 2];
    float denom = dx * nx + dy * ny + dz * nz;
    if (fabsf(denom) <= kDirEps) {
      continue;
    }
    float numer = (px - ox) * nx + (py - oy) * ny + (pz - oz) * nz;
    float t = numer / denom;
    if (t < 0.0f || t > max_distance) {
      continue;
    }
    int64_t key = plane_source_keys[plane];
    if (better_hit(t, key, best_t, best_key)) {
      best_t = t;
      best_key = key;
      best_px = ox + dx * t;
      best_py = oy + dy * t;
      best_pz = oz + dz * t;
      best_nx = nx;
      best_ny = ny;
      best_nz = nz;
      best_id = plane_numeric_ids[plane];
      best_material_index = plane_material_indices[plane];
      found = true;
    }
  }

  int stack[kMaxBvhStack];
  float stack_t[kMaxBvhStack];
  int stack_size = 0;
  int local_max_stack = 0;
  if (num_bvh_nodes > 0) {
    float root_t = 0.0f;
    if (intersect_aabb(
            ox, oy, oz, dx, dy, dz, bvh_bounds_min, bvh_bounds_max, 0, best_t, &root_t)) {
      stack[0] = 0;
      stack_t[0] = root_t;
      stack_size = 1;
      local_max_stack = 1;
    }
  }

  while (stack_size > 0) {
    --stack_size;
    int node = stack[stack_size];
    float node_t = stack_t[stack_size];
    if (node_t > best_t) {
      continue;
    }
    int leaf_count = bvh_count[node];
    if (leaf_count > 0) {
      int leaf_start = bvh_start[node];
      for (int offset = 0; offset < leaf_count; ++offset) {
        int tri = bvh_prim_ids[leaf_start + offset];
        if ((triangle_role_masks[tri] & sensor_role_mask) == 0) {
          continue;
        }
        float t = 0.0f;
        float nx = 0.0f;
        float ny = 0.0f;
        float nz = 0.0f;
        if (!intersect_triangle(
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                tri,
                triangle_v0,
                triangle_e1,
                triangle_e2,
                triangle_normal,
                best_t,
                &t,
                &nx,
                &ny,
                &nz)) {
          continue;
        }
        int64_t key = triangle_source_keys[tri];
        if (better_hit(t, key, best_t, best_key)) {
          best_t = t;
          best_key = key;
          best_px = ox + dx * t;
          best_py = oy + dy * t;
          best_pz = oz + dz * t;
          best_nx = nx;
          best_ny = ny;
          best_nz = nz;
          best_id = triangle_numeric_ids[tri];
          best_material_index = triangle_material_indices[tri];
          found = true;
        }
      }
    } else {
      int left = bvh_left[node];
      int right = bvh_right[node];
      bool left_hit = false;
      bool right_hit = false;
      float left_t = 0.0f;
      float right_t = 0.0f;
      if (left >= 0) {
        left_hit = intersect_aabb(
            ox, oy, oz, dx, dy, dz, bvh_bounds_min, bvh_bounds_max, left, best_t, &left_t);
      }
      if (right >= 0) {
        right_hit = intersect_aabb(
            ox, oy, oz, dx, dy, dz, bvh_bounds_min, bvh_bounds_max, right, best_t, &right_t);
      }

      int first = left;
      int second = right;
      bool first_hit = left_hit;
      bool second_hit = right_hit;
      float first_t = left_t;
      float second_t = right_t;
      if (right_hit && (!left_hit || right_t < left_t)) {
        first = right;
        second = left;
        first_hit = right_hit;
        second_hit = left_hit;
        first_t = right_t;
        second_t = left_t;
      }

      if (second_hit) {
        if (stack_size < kMaxBvhStack) {
          stack[stack_size] = second;
          stack_t[stack_size] = second_t;
          ++stack_size;
        } else {
          atomicAdd(bvh_stack_overflow_count, 1);
        }
      }
      if (first_hit) {
        if (stack_size < kMaxBvhStack) {
          stack[stack_size] = first;
          stack_t[stack_size] = first_t;
          ++stack_size;
        } else {
          atomicAdd(bvh_stack_overflow_count, 1);
        }
      }
      if (stack_size > local_max_stack) {
        local_max_stack = stack_size;
      }
    }
  }

  atomicMax(bvh_max_stack_depth, local_max_stack);

  if (found) {
    hit_mask[ray] = 1;
    range_m[ray] = best_t;
    position_world[ray * 3 + 0] = best_px;
    position_world[ray * 3 + 1] = best_py;
    position_world[ray * 3 + 2] = best_pz;
    normal_world[ray * 3 + 0] = best_nx;
    normal_world[ray * 3 + 1] = best_ny;
    normal_world[ray * 3 + 2] = best_nz;
    numeric_instance_id[ray] = best_id;
    material_index[ray] = best_material_index;
  }
}

__global__ void shade_direct_light_no_shadow_kernel(
    const int32_t* hit_mask,
    const float* position_world,
    const float* normal_world,
    const int32_t* material_index,
    const float* material_albedo_rgb,
    const int32_t* light_kind,
    const float* light_position_or_direction_world,
    const float* light_intensity,
    const float* light_color_rgb,
    int num_lights,
    float ambient_r,
    float ambient_g,
    float ambient_b,
    float background_r,
    float background_g,
    float background_b,
    float* rgb,
    float* intensity,
    int num_rays) {
  int ray = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray >= num_rays) {
    return;
  }

  float out_r = background_r;
  float out_g = background_g;
  float out_b = background_b;
  if (hit_mask[ray] != 0) {
    int mat = material_index[ray];
    float albedo_r = material_albedo_rgb[mat * 3 + 0];
    float albedo_g = material_albedo_rgb[mat * 3 + 1];
    float albedo_b = material_albedo_rgb[mat * 3 + 2];
    out_r = ambient_r * albedo_r;
    out_g = ambient_g * albedo_g;
    out_b = ambient_b * albedo_b;

    float px = position_world[ray * 3 + 0];
    float py = position_world[ray * 3 + 1];
    float pz = position_world[ray * 3 + 2];
    float nx = normal_world[ray * 3 + 0];
    float ny = normal_world[ray * 3 + 1];
    float nz = normal_world[ray * 3 + 2];

    for (int light = 0; light < num_lights; ++light) {
      float lx = light_position_or_direction_world[light * 3 + 0];
      float ly = light_position_or_direction_world[light * 3 + 1];
      float lz = light_position_or_direction_world[light * 3 + 2];
      float attenuation = 1.0f;
      if (light_kind[light] == 1) {
        lx = lx - px;
        ly = ly - py;
        lz = lz - pz;
        float distance_sq = lx * lx + ly * ly + lz * lz;
        float distance = sqrtf(distance_sq);
        if (distance > kDirEps) {
          float inv_distance = 1.0f / distance;
          lx = lx * inv_distance;
          ly = ly * inv_distance;
          lz = lz * inv_distance;
          attenuation = 1.0f / fmaxf(distance_sq, 1.0e-12f);
        } else {
          attenuation = 0.0f;
        }
      } else {
        float light_norm = sqrtf(lx * lx + ly * ly + lz * lz);
        if (light_norm > kDirEps) {
          float inv_light_norm = 1.0f / light_norm;
          lx = lx * inv_light_norm;
          ly = ly * inv_light_norm;
          lz = lz * inv_light_norm;
        } else {
          attenuation = 0.0f;
        }
      }

      float n_dot_l = nx * lx + ny * ly + nz * lz;
      if (n_dot_l > 0.0f && attenuation > 0.0f) {
        float scale = light_intensity[light] * attenuation * n_dot_l;
        out_r += albedo_r * light_color_rgb[light * 3 + 0] * scale;
        out_g += albedo_g * light_color_rgb[light * 3 + 1] * scale;
        out_b += albedo_b * light_color_rgb[light * 3 + 2] * scale;
      }
    }
  }

  rgb[ray * 3 + 0] = out_r;
  rgb[ray * 3 + 1] = out_g;
  rgb[ray * 3 + 2] = out_b;
  if (hit_mask[ray] != 0) {
    intensity[ray] = out_r * 0.2126f + out_g * 0.7152f + out_b * 0.0722f;
  } else {
    intensity[ray] = 0.0f;
  }
}

__device__ bool shadow_occluded_by_triangles(
    float ox,
    float oy,
    float oz,
    float dx,
    float dy,
    float dz,
    float max_distance,
    const float* triangle_v0,
    const float* triangle_e1,
    const float* triangle_e2,
    const float* triangle_normal,
    const int64_t* triangle_role_masks,
    const float* bvh_bounds_min,
    const float* bvh_bounds_max,
    const int32_t* bvh_left,
    const int32_t* bvh_right,
    const int32_t* bvh_start,
    const int32_t* bvh_count,
    const int32_t* bvh_prim_ids,
    int num_bvh_nodes,
    int64_t sensor_role_mask,
    int32_t* shadow_stack_overflow_count,
    int32_t* shadow_max_stack_depth,
    int traversal_stats_enabled,
    int32_t* shadow_traversal_node_visit_count,
    int32_t* shadow_traversal_leaf_visit_count,
    int32_t* shadow_traversal_triangle_test_count) {
  if (max_distance <= 0.0f || num_bvh_nodes <= 0) {
    return false;
  }

  int stack[kMaxBvhStack];
  int stack_size = 0;
  int local_max_stack = 0;
  int local_node_visit_count = 0;
  int local_leaf_visit_count = 0;
  int local_triangle_test_count = 0;
  float root_t = 0.0f;
  if (intersect_aabb(
          ox, oy, oz, dx, dy, dz, bvh_bounds_min, bvh_bounds_max, 0, max_distance, &root_t)) {
    stack[0] = 0;
    stack_size = 1;
    local_max_stack = 1;
  }

  bool occluded = false;
  while (stack_size > 0 && !occluded) {
    --stack_size;
    int node = stack[stack_size];
    ++local_node_visit_count;
    int leaf_count = bvh_count[node];
    if (leaf_count > 0) {
      ++local_leaf_visit_count;
      int leaf_start = bvh_start[node];
      for (int offset = 0; offset < leaf_count; ++offset) {
        int tri = bvh_prim_ids[leaf_start + offset];
        if ((triangle_role_masks[tri] & sensor_role_mask) == 0) {
          continue;
        }
        ++local_triangle_test_count;
        float t = 0.0f;
        float nx = 0.0f;
        float ny = 0.0f;
        float nz = 0.0f;
        if (intersect_triangle(
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                tri,
                triangle_v0,
                triangle_e1,
                triangle_e2,
                triangle_normal,
                max_distance,
                &t,
                &nx,
                &ny,
                &nz)) {
          occluded = true;
          break;
        }
      }
    } else {
      int left = bvh_left[node];
      int right = bvh_right[node];
      if (right >= 0) {
        float right_t = 0.0f;
        if (intersect_aabb(
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                bvh_bounds_min,
                bvh_bounds_max,
                right,
                max_distance,
                &right_t)) {
          if (stack_size < kMaxBvhStack) {
            stack[stack_size] = right;
            ++stack_size;
          } else {
            atomicAdd(shadow_stack_overflow_count, 1);
          }
        }
      }
      if (left >= 0) {
        float left_t = 0.0f;
        if (intersect_aabb(
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                bvh_bounds_min,
                bvh_bounds_max,
                left,
                max_distance,
                &left_t)) {
          if (stack_size < kMaxBvhStack) {
            stack[stack_size] = left;
            ++stack_size;
          } else {
            atomicAdd(shadow_stack_overflow_count, 1);
          }
        }
      }
      if (stack_size > local_max_stack) {
        local_max_stack = stack_size;
      }
    }
  }

  atomicMax(shadow_max_stack_depth, local_max_stack);
  if (traversal_stats_enabled != 0) {
    atomicAdd(shadow_traversal_node_visit_count, local_node_visit_count);
    atomicAdd(shadow_traversal_leaf_visit_count, local_leaf_visit_count);
    atomicAdd(shadow_traversal_triangle_test_count, local_triangle_test_count);
  }
  return occluded;
}

__global__ void shade_direct_light_with_shadows_kernel(
    const int32_t* hit_mask,
    const float* position_world,
    const float* normal_world,
    const int32_t* material_index,
    const float* material_albedo_rgb,
    const int32_t* light_kind,
    const float* light_position_or_direction_world,
    const float* light_intensity,
    const float* light_color_rgb,
    int num_lights,
    const float* triangle_v0,
    const float* triangle_e1,
    const float* triangle_e2,
    const float* triangle_normal,
    const int64_t* triangle_role_masks,
    const float* bvh_bounds_min,
    const float* bvh_bounds_max,
    const int32_t* bvh_left,
    const int32_t* bvh_right,
    const int32_t* bvh_start,
    const int32_t* bvh_count,
    const int32_t* bvh_prim_ids,
    int num_bvh_nodes,
    int64_t sensor_role_mask,
    float shadow_bias,
    float ambient_r,
    float ambient_g,
    float ambient_b,
    float background_r,
    float background_g,
    float background_b,
    float* rgb,
    float* intensity,
    int32_t* shadow_stack_overflow_count,
    int32_t* shadow_max_stack_depth,
    int traversal_stats_enabled,
    int32_t* shadow_traversal_ray_count,
    int32_t* shadow_traversal_directional_ray_count,
    int32_t* shadow_traversal_point_ray_count,
    int32_t* shadow_traversal_occluded_count,
    int32_t* shadow_traversal_unoccluded_count,
    int32_t* shadow_traversal_node_visit_count,
    int32_t* shadow_traversal_leaf_visit_count,
    int32_t* shadow_traversal_triangle_test_count,
    int32_t* shadow_traversal_plane_test_count,
    int num_rays) {
  int ray = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray >= num_rays) {
    return;
  }

  float out_r = background_r;
  float out_g = background_g;
  float out_b = background_b;
  if (hit_mask[ray] != 0) {
    int mat = material_index[ray];
    float albedo_r = material_albedo_rgb[mat * 3 + 0];
    float albedo_g = material_albedo_rgb[mat * 3 + 1];
    float albedo_b = material_albedo_rgb[mat * 3 + 2];
    out_r = ambient_r * albedo_r;
    out_g = ambient_g * albedo_g;
    out_b = ambient_b * albedo_b;

    float px = position_world[ray * 3 + 0];
    float py = position_world[ray * 3 + 1];
    float pz = position_world[ray * 3 + 2];
    float nx = normal_world[ray * 3 + 0];
    float ny = normal_world[ray * 3 + 1];
    float nz = normal_world[ray * 3 + 2];

    for (int light = 0; light < num_lights; ++light) {
      float lx = light_position_or_direction_world[light * 3 + 0];
      float ly = light_position_or_direction_world[light * 3 + 1];
      float lz = light_position_or_direction_world[light * 3 + 2];
      float attenuation = 1.0f;
      float shadow_max_distance = std::numeric_limits<float>::max();
      if (light_kind[light] == 1) {
        lx = lx - px;
        ly = ly - py;
        lz = lz - pz;
        float distance_sq = lx * lx + ly * ly + lz * lz;
        float distance = sqrtf(distance_sq);
        if (distance > kDirEps) {
          float inv_distance = 1.0f / distance;
          lx = lx * inv_distance;
          ly = ly * inv_distance;
          lz = lz * inv_distance;
          attenuation = 1.0f / fmaxf(distance_sq, 1.0e-12f);
          shadow_max_distance = distance - shadow_bias;
        } else {
          attenuation = 0.0f;
          shadow_max_distance = 0.0f;
        }
      } else {
        float light_norm = sqrtf(lx * lx + ly * ly + lz * lz);
        if (light_norm > kDirEps) {
          float inv_light_norm = 1.0f / light_norm;
          lx = lx * inv_light_norm;
          ly = ly * inv_light_norm;
          lz = lz * inv_light_norm;
        } else {
          attenuation = 0.0f;
          shadow_max_distance = 0.0f;
        }
      }

      float n_dot_l = nx * lx + ny * ly + nz * lz;
      if (n_dot_l > 0.0f && attenuation > 0.0f && shadow_max_distance > 0.0f) {
        float shadow_ox = px + nx * shadow_bias;
        float shadow_oy = py + ny * shadow_bias;
        float shadow_oz = pz + nz * shadow_bias;
        if (traversal_stats_enabled != 0) {
          atomicAdd(shadow_traversal_ray_count, 1);
          if (light_kind[light] == 1) {
            atomicAdd(shadow_traversal_point_ray_count, 1);
          } else {
            atomicAdd(shadow_traversal_directional_ray_count, 1);
          }
        }
        // P12.3d treats triangle meshes as CUDA shadow occluders. Planes can
        // still be primary hits but are intentionally non-occluders here.
        bool occluded = shadow_occluded_by_triangles(
            shadow_ox,
            shadow_oy,
            shadow_oz,
            lx,
            ly,
            lz,
            shadow_max_distance,
            triangle_v0,
            triangle_e1,
            triangle_e2,
            triangle_normal,
            triangle_role_masks,
            bvh_bounds_min,
            bvh_bounds_max,
            bvh_left,
            bvh_right,
            bvh_start,
            bvh_count,
            bvh_prim_ids,
            num_bvh_nodes,
            sensor_role_mask,
            shadow_stack_overflow_count,
            shadow_max_stack_depth,
            traversal_stats_enabled,
            shadow_traversal_node_visit_count,
            shadow_traversal_leaf_visit_count,
            shadow_traversal_triangle_test_count);
        if (traversal_stats_enabled != 0) {
          if (occluded) {
            atomicAdd(shadow_traversal_occluded_count, 1);
          } else {
            atomicAdd(shadow_traversal_unoccluded_count, 1);
          }
        }
        bool visible_to_light = !occluded;
        if (visible_to_light) {
          float scale = light_intensity[light] * attenuation * n_dot_l;
          out_r += albedo_r * light_color_rgb[light * 3 + 0] * scale;
          out_g += albedo_g * light_color_rgb[light * 3 + 1] * scale;
          out_b += albedo_b * light_color_rgb[light * 3 + 2] * scale;
        }
      }
    }
  }

  rgb[ray * 3 + 0] = out_r;
  rgb[ray * 3 + 1] = out_g;
  rgb[ray * 3 + 2] = out_b;
  if (hit_mask[ray] != 0) {
    intensity[ray] = out_r * 0.2126f + out_g * 0.7152f + out_b * 0.0722f;
  } else {
    intensity[ray] = 0.0f;
  }
}

void check_float_2d_3(torch::Tensor value, const char* name) {
  TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(value.dtype() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(value.dim() == 2 && value.size(1) == 3, name, " must have shape [N, 3]");
}

void check_float_1d(torch::Tensor value, const char* name) {
  TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(value.dtype() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(value.dim() == 1, name, " must be rank-1");
}

void check_i32_1d(torch::Tensor value, const char* name) {
  TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(value.dtype() == torch::kInt32, name, " must be int32");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(value.dim() == 1, name, " must be rank-1");
}

void check_i64_1d(torch::Tensor value, const char* name) {
  TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(value.dtype() == torch::kInt64, name, " must be int64");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(value.dim() == 1, name, " must be rank-1");
}

void check_same_device(torch::Tensor anchor, torch::Tensor value, const char* name) {
  TORCH_CHECK(value.device() == anchor.device(), name, " must be on ", anchor.device());
}

}  // namespace

void pinhole_camera_rays_out(
    torch::Tensor origins,
    torch::Tensor directions,
    int64_t width,
    int64_t height,
    double fx,
    double fy,
    double cx,
    double cy,
    double origin_x,
    double origin_y,
    double origin_z,
    double r00,
    double r01,
    double r02,
    double r10,
    double r11,
    double r12,
    double r20,
    double r21,
    double r22) {
  check_float_2d_3(origins, "origins");
  check_float_2d_3(directions, "directions");
  TORCH_CHECK(directions.sizes() == origins.sizes(), "directions must match origins shape");
  check_same_device(origins, directions, "directions");
  TORCH_CHECK(width > 0 && height > 0, "width and height must be > 0");
  TORCH_CHECK(width <= INT32_MAX && height <= INT32_MAX, "width or height is too large");
  TORCH_CHECK(width <= INT32_MAX / height, "camera ray count exceeds int32 range");
  int64_t num_rays = width * height;
  TORCH_CHECK(origins.size(0) == num_rays, "ray outputs must have width * height rows");
  TORCH_CHECK(std::isfinite(fx) && fx > 0.0, "fx must be finite and > 0");
  TORCH_CHECK(std::isfinite(fy) && fy > 0.0, "fy must be finite and > 0");
  TORCH_CHECK(std::isfinite(cx) && std::isfinite(cy), "cx and cy must be finite");

  const c10::cuda::CUDAGuard device_guard(origins.device());
  if (num_rays > 0) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(origins.get_device());
    int block = 256;
    int grid = (static_cast<int>(num_rays) + block - 1) / block;
    pinhole_camera_rays_kernel<<<grid, block, 0, stream>>>(
        static_cast<int>(width),
        static_cast<int>(height),
        static_cast<float>(fx),
        static_cast<float>(fy),
        static_cast<float>(cx),
        static_cast<float>(cy),
        static_cast<float>(origin_x),
        static_cast<float>(origin_y),
        static_cast<float>(origin_z),
        static_cast<float>(r00),
        static_cast<float>(r01),
        static_cast<float>(r02),
        static_cast<float>(r10),
        static_cast<float>(r11),
        static_cast<float>(r12),
        static_cast<float>(r20),
        static_cast<float>(r21),
        static_cast<float>(r22),
        origins.data_ptr<float>(),
        directions.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void first_hit_rays_out(
    torch::Tensor origins,
    torch::Tensor directions,
    double max_distance,
    torch::Tensor plane_normals,
    torch::Tensor plane_points,
    torch::Tensor plane_numeric_ids,
    torch::Tensor plane_source_keys,
    torch::Tensor plane_role_masks,
    torch::Tensor plane_material_indices,
    int64_t num_planes,
    torch::Tensor triangle_v0,
    torch::Tensor triangle_e1,
    torch::Tensor triangle_e2,
    torch::Tensor triangle_normal,
    torch::Tensor triangle_numeric_ids,
    torch::Tensor triangle_source_keys,
    torch::Tensor triangle_role_masks,
    torch::Tensor triangle_material_indices,
    torch::Tensor bvh_bounds_min,
    torch::Tensor bvh_bounds_max,
    torch::Tensor bvh_left,
    torch::Tensor bvh_right,
    torch::Tensor bvh_start,
    torch::Tensor bvh_count,
    torch::Tensor bvh_prim_ids,
    int64_t num_bvh_nodes,
    int64_t sensor_role_mask,
    torch::Tensor hit_mask,
    torch::Tensor range_m,
    torch::Tensor position_world,
    torch::Tensor normal_world,
    torch::Tensor numeric_instance_id,
    torch::Tensor material_index,
    torch::Tensor bvh_stack_overflow_count,
    torch::Tensor bvh_max_stack_depth) {
  check_float_2d_3(origins, "origins");
  check_float_2d_3(directions, "directions");
  TORCH_CHECK(directions.sizes() == origins.sizes(), "directions must match origins shape");
  TORCH_CHECK(std::isfinite(max_distance), "max_distance must be finite");
  check_float_2d_3(plane_normals, "plane_normals");
  check_float_2d_3(plane_points, "plane_points");
  check_i32_1d(plane_numeric_ids, "plane_numeric_ids");
  check_i64_1d(plane_source_keys, "plane_source_keys");
  check_i64_1d(plane_role_masks, "plane_role_masks");
  check_i32_1d(plane_material_indices, "plane_material_indices");
  check_float_2d_3(triangle_v0, "triangle_v0");
  check_float_2d_3(triangle_e1, "triangle_e1");
  check_float_2d_3(triangle_e2, "triangle_e2");
  check_float_2d_3(triangle_normal, "triangle_normal");
  check_i32_1d(triangle_numeric_ids, "triangle_numeric_ids");
  check_i64_1d(triangle_source_keys, "triangle_source_keys");
  check_i64_1d(triangle_role_masks, "triangle_role_masks");
  check_i32_1d(triangle_material_indices, "triangle_material_indices");
  check_float_2d_3(bvh_bounds_min, "bvh_bounds_min");
  check_float_2d_3(bvh_bounds_max, "bvh_bounds_max");
  check_i32_1d(bvh_left, "bvh_left");
  check_i32_1d(bvh_right, "bvh_right");
  check_i32_1d(bvh_start, "bvh_start");
  check_i32_1d(bvh_count, "bvh_count");
  check_i32_1d(bvh_prim_ids, "bvh_prim_ids");
  check_i32_1d(hit_mask, "hit_mask");
  check_float_1d(range_m, "range_m");
  check_float_2d_3(position_world, "position_world");
  check_float_2d_3(normal_world, "normal_world");
  check_i32_1d(numeric_instance_id, "numeric_instance_id");
  check_i32_1d(material_index, "material_index");
  check_i32_1d(bvh_stack_overflow_count, "bvh_stack_overflow_count");
  check_i32_1d(bvh_max_stack_depth, "bvh_max_stack_depth");
  check_same_device(origins, directions, "directions");
  check_same_device(origins, plane_normals, "plane_normals");
  check_same_device(origins, plane_points, "plane_points");
  check_same_device(origins, plane_numeric_ids, "plane_numeric_ids");
  check_same_device(origins, plane_source_keys, "plane_source_keys");
  check_same_device(origins, plane_role_masks, "plane_role_masks");
  check_same_device(origins, plane_material_indices, "plane_material_indices");
  check_same_device(origins, triangle_v0, "triangle_v0");
  check_same_device(origins, triangle_e1, "triangle_e1");
  check_same_device(origins, triangle_e2, "triangle_e2");
  check_same_device(origins, triangle_normal, "triangle_normal");
  check_same_device(origins, triangle_numeric_ids, "triangle_numeric_ids");
  check_same_device(origins, triangle_source_keys, "triangle_source_keys");
  check_same_device(origins, triangle_role_masks, "triangle_role_masks");
  check_same_device(origins, triangle_material_indices, "triangle_material_indices");
  check_same_device(origins, bvh_bounds_min, "bvh_bounds_min");
  check_same_device(origins, bvh_bounds_max, "bvh_bounds_max");
  check_same_device(origins, bvh_left, "bvh_left");
  check_same_device(origins, bvh_right, "bvh_right");
  check_same_device(origins, bvh_start, "bvh_start");
  check_same_device(origins, bvh_count, "bvh_count");
  check_same_device(origins, bvh_prim_ids, "bvh_prim_ids");
  check_same_device(origins, hit_mask, "hit_mask");
  check_same_device(origins, range_m, "range_m");
  check_same_device(origins, position_world, "position_world");
  check_same_device(origins, normal_world, "normal_world");
  check_same_device(origins, numeric_instance_id, "numeric_instance_id");
  check_same_device(origins, material_index, "material_index");
  check_same_device(origins, bvh_stack_overflow_count, "bvh_stack_overflow_count");
  check_same_device(origins, bvh_max_stack_depth, "bvh_max_stack_depth");

  int64_t num_rays = origins.size(0);
  int64_t num_triangles = triangle_v0.size(0);
  TORCH_CHECK(num_rays <= static_cast<int64_t>(INT32_MAX), "too many rays");
  TORCH_CHECK(num_planes >= 0 && num_planes <= plane_normals.size(0), "num_planes out of range");
  TORCH_CHECK(num_planes <= plane_points.size(0), "num_planes exceeds plane_points");
  TORCH_CHECK(num_planes <= plane_numeric_ids.size(0), "num_planes exceeds plane_numeric_ids");
  TORCH_CHECK(num_planes <= plane_source_keys.size(0), "num_planes exceeds plane_source_keys");
  TORCH_CHECK(num_planes <= plane_role_masks.size(0), "num_planes exceeds plane_role_masks");
  TORCH_CHECK(
      num_planes <= plane_material_indices.size(0), "num_planes exceeds plane_material_indices");
  TORCH_CHECK(triangle_e1.size(0) == num_triangles, "triangle_e1 length mismatch");
  TORCH_CHECK(triangle_e2.size(0) == num_triangles, "triangle_e2 length mismatch");
  TORCH_CHECK(triangle_normal.size(0) == num_triangles, "triangle_normal length mismatch");
  TORCH_CHECK(triangle_numeric_ids.size(0) == num_triangles, "triangle_numeric_ids length mismatch");
  TORCH_CHECK(triangle_source_keys.size(0) == num_triangles, "triangle_source_keys length mismatch");
  TORCH_CHECK(triangle_role_masks.size(0) == num_triangles, "triangle_role_masks length mismatch");
  TORCH_CHECK(
      triangle_material_indices.size(0) == num_triangles,
      "triangle_material_indices length mismatch");
  TORCH_CHECK(num_bvh_nodes >= 0 && num_bvh_nodes <= bvh_bounds_min.size(0), "num_bvh_nodes out of range");
  TORCH_CHECK(num_bvh_nodes <= bvh_bounds_max.size(0), "num_bvh_nodes exceeds bvh_bounds_max");
  TORCH_CHECK(num_bvh_nodes <= bvh_left.size(0), "num_bvh_nodes exceeds bvh_left");
  TORCH_CHECK(num_bvh_nodes <= bvh_right.size(0), "num_bvh_nodes exceeds bvh_right");
  TORCH_CHECK(num_bvh_nodes <= bvh_start.size(0), "num_bvh_nodes exceeds bvh_start");
  TORCH_CHECK(num_bvh_nodes <= bvh_count.size(0), "num_bvh_nodes exceeds bvh_count");
  TORCH_CHECK(bvh_prim_ids.size(0) >= num_triangles, "bvh_prim_ids is shorter than triangles");
  TORCH_CHECK(hit_mask.size(0) == num_rays, "hit_mask length must match origins");
  TORCH_CHECK(range_m.size(0) == num_rays, "range_m length must match origins");
  TORCH_CHECK(position_world.size(0) == num_rays, "position_world length must match origins");
  TORCH_CHECK(normal_world.size(0) == num_rays, "normal_world length must match origins");
  TORCH_CHECK(numeric_instance_id.size(0) == num_rays, "numeric_instance_id length must match origins");
  TORCH_CHECK(material_index.size(0) == num_rays, "material_index length must match origins");
  TORCH_CHECK(bvh_stack_overflow_count.size(0) == 1, "bvh_stack_overflow_count must have shape [1]");
  TORCH_CHECK(bvh_max_stack_depth.size(0) == 1, "bvh_max_stack_depth must have shape [1]");

  const c10::cuda::CUDAGuard device_guard(hit_mask.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(hit_mask.get_device());
  C10_CUDA_CHECK(cudaMemsetAsync(bvh_stack_overflow_count.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(bvh_max_stack_depth.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));

  if (num_rays > 0) {
    int block = 256;
    int grid = (static_cast<int>(num_rays) + block - 1) / block;
    first_hit_rays_kernel<<<grid, block, 0, stream>>>(
        origins.data_ptr<float>(),
        directions.data_ptr<float>(),
        static_cast<float>(max_distance),
        plane_normals.data_ptr<float>(),
        plane_points.data_ptr<float>(),
        plane_numeric_ids.data_ptr<int32_t>(),
        plane_source_keys.data_ptr<int64_t>(),
        plane_role_masks.data_ptr<int64_t>(),
        plane_material_indices.data_ptr<int32_t>(),
        static_cast<int>(num_planes),
        triangle_v0.data_ptr<float>(),
        triangle_e1.data_ptr<float>(),
        triangle_e2.data_ptr<float>(),
        triangle_normal.data_ptr<float>(),
        triangle_numeric_ids.data_ptr<int32_t>(),
        triangle_source_keys.data_ptr<int64_t>(),
        triangle_role_masks.data_ptr<int64_t>(),
        triangle_material_indices.data_ptr<int32_t>(),
        bvh_bounds_min.data_ptr<float>(),
        bvh_bounds_max.data_ptr<float>(),
        bvh_left.data_ptr<int32_t>(),
        bvh_right.data_ptr<int32_t>(),
        bvh_start.data_ptr<int32_t>(),
        bvh_count.data_ptr<int32_t>(),
        bvh_prim_ids.data_ptr<int32_t>(),
        static_cast<int>(num_bvh_nodes),
        sensor_role_mask,
        hit_mask.data_ptr<int32_t>(),
        range_m.data_ptr<float>(),
        position_world.data_ptr<float>(),
        normal_world.data_ptr<float>(),
        numeric_instance_id.data_ptr<int32_t>(),
        material_index.data_ptr<int32_t>(),
        bvh_stack_overflow_count.data_ptr<int32_t>(),
        bvh_max_stack_depth.data_ptr<int32_t>(),
        static_cast<int>(num_rays));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void shade_direct_light_no_shadow_impl(
    torch::Tensor hit_mask,
    torch::Tensor position_world,
    torch::Tensor normal_world,
    torch::Tensor material_index,
    torch::Tensor material_albedo_rgb,
    torch::Tensor light_kind,
    torch::Tensor light_position_or_direction_world,
    torch::Tensor light_intensity,
    torch::Tensor light_color_rgb,
    int64_t num_lights,
    double ambient_r,
    double ambient_g,
    double ambient_b,
    double background_r,
    double background_g,
    double background_b,
    torch::Tensor rgb,
    torch::Tensor intensity,
    torch::Tensor shadow_stack_overflow_count,
    torch::Tensor shadow_max_stack_depth,
    torch::Tensor* traversal_counter_0,
    torch::Tensor* traversal_counter_1,
    torch::Tensor* traversal_counter_2,
    torch::Tensor* traversal_counter_3,
    torch::Tensor* traversal_counter_4,
    torch::Tensor* traversal_counter_5,
    torch::Tensor* traversal_counter_6,
    torch::Tensor* traversal_counter_7,
    torch::Tensor* traversal_counter_8) {
  check_i32_1d(hit_mask, "hit_mask");
  check_float_2d_3(position_world, "position_world");
  check_float_2d_3(normal_world, "normal_world");
  check_i32_1d(material_index, "material_index");
  check_float_2d_3(material_albedo_rgb, "material_albedo_rgb");
  check_i32_1d(light_kind, "light_kind");
  check_float_2d_3(light_position_or_direction_world, "light_position_or_direction_world");
  check_float_2d_3(light_color_rgb, "light_color_rgb");
  TORCH_CHECK(light_intensity.is_cuda(), "light_intensity must be a CUDA tensor");
  TORCH_CHECK(light_intensity.dtype() == torch::kFloat32, "light_intensity must be float32");
  TORCH_CHECK(light_intensity.is_contiguous(), "light_intensity must be contiguous");
  TORCH_CHECK(light_intensity.dim() == 1, "light_intensity must be rank-1");
  check_float_2d_3(rgb, "rgb");
  check_float_1d(intensity, "intensity");
  check_i32_1d(shadow_stack_overflow_count, "shadow_stack_overflow_count");
  check_i32_1d(shadow_max_stack_depth, "shadow_max_stack_depth");
  torch::Tensor* traversal_counters[9] = {
      traversal_counter_0,
      traversal_counter_1,
      traversal_counter_2,
      traversal_counter_3,
      traversal_counter_4,
      traversal_counter_5,
      traversal_counter_6,
      traversal_counter_7,
      traversal_counter_8};
  bool traversal_stats_enabled = traversal_counter_0 != nullptr;
  for (torch::Tensor* counter : traversal_counters) {
    TORCH_CHECK(
        (counter != nullptr) == traversal_stats_enabled,
        "shadow traversal counters must be completely absent or present");
    if (counter != nullptr) {
      check_i32_1d(*counter, "shadow_traversal_counter");
      TORCH_CHECK(counter->size(0) == 1, "shadow traversal counters must have shape [1]");
    }
  }
  TORCH_CHECK(position_world.sizes() == normal_world.sizes(), "position_world/normal_world shape mismatch");
  TORCH_CHECK(hit_mask.size(0) == position_world.size(0), "hit_mask length must match position_world");
  TORCH_CHECK(
      material_index.size(0) == position_world.size(0),
      "material_index length must match position_world");
  TORCH_CHECK(num_lights >= 0 && num_lights <= light_kind.size(0), "num_lights out of range");
  TORCH_CHECK(num_lights <= light_position_or_direction_world.size(0), "num_lights exceeds light vectors");
  TORCH_CHECK(num_lights <= light_intensity.size(0), "num_lights exceeds light intensities");
  TORCH_CHECK(num_lights <= light_color_rgb.size(0), "num_lights exceeds light colors");
  TORCH_CHECK(
      std::isfinite(ambient_r) && std::isfinite(ambient_g) && std::isfinite(ambient_b),
      "ambient RGB must be finite");
  TORCH_CHECK(
      std::isfinite(background_r) && std::isfinite(background_g) && std::isfinite(background_b),
      "background RGB must be finite");
  check_same_device(hit_mask, position_world, "position_world");
  check_same_device(hit_mask, normal_world, "normal_world");
  check_same_device(hit_mask, material_index, "material_index");
  check_same_device(hit_mask, material_albedo_rgb, "material_albedo_rgb");
  check_same_device(hit_mask, light_kind, "light_kind");
  check_same_device(hit_mask, light_position_or_direction_world, "light_position_or_direction_world");
  check_same_device(hit_mask, light_intensity, "light_intensity");
  check_same_device(hit_mask, light_color_rgb, "light_color_rgb");
  check_same_device(hit_mask, rgb, "rgb");
  check_same_device(hit_mask, intensity, "intensity");
  check_same_device(hit_mask, shadow_stack_overflow_count, "shadow_stack_overflow_count");
  check_same_device(hit_mask, shadow_max_stack_depth, "shadow_max_stack_depth");
  for (torch::Tensor* counter : traversal_counters) {
    if (counter != nullptr) {
      check_same_device(hit_mask, *counter, "shadow_traversal_counter");
    }
  }

  int64_t num_rays = hit_mask.size(0);
  TORCH_CHECK(num_rays <= static_cast<int64_t>(INT32_MAX), "too many rays");
  TORCH_CHECK(rgb.size(0) == num_rays, "rgb length must match hit_mask");
  TORCH_CHECK(intensity.size(0) == num_rays, "intensity length must match hit_mask");
  TORCH_CHECK(shadow_stack_overflow_count.size(0) == 1, "shadow_stack_overflow_count must have shape [1]");
  TORCH_CHECK(shadow_max_stack_depth.size(0) == 1, "shadow_max_stack_depth must have shape [1]");

  const c10::cuda::CUDAGuard device_guard(rgb.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(rgb.get_device());
  C10_CUDA_CHECK(cudaMemsetAsync(
      shadow_stack_overflow_count.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(shadow_max_stack_depth.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
  for (torch::Tensor* counter : traversal_counters) {
    if (counter != nullptr) {
      C10_CUDA_CHECK(cudaMemsetAsync(counter->data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
    }
  }

  if (num_rays > 0) {
    int block = 256;
    int grid = (static_cast<int>(num_rays) + block - 1) / block;
    shade_direct_light_no_shadow_kernel<<<grid, block, 0, stream>>>(
        hit_mask.data_ptr<int32_t>(),
        position_world.data_ptr<float>(),
        normal_world.data_ptr<float>(),
        material_index.data_ptr<int32_t>(),
        material_albedo_rgb.data_ptr<float>(),
        light_kind.data_ptr<int32_t>(),
        light_position_or_direction_world.data_ptr<float>(),
        light_intensity.data_ptr<float>(),
        light_color_rgb.data_ptr<float>(),
        static_cast<int>(num_lights),
        static_cast<float>(ambient_r),
        static_cast<float>(ambient_g),
        static_cast<float>(ambient_b),
        static_cast<float>(background_r),
        static_cast<float>(background_g),
        static_cast<float>(background_b),
        rgb.data_ptr<float>(),
        intensity.data_ptr<float>(),
        static_cast<int>(num_rays));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void shade_direct_light_no_shadow_out(
    torch::Tensor hit_mask,
    torch::Tensor position_world,
    torch::Tensor normal_world,
    torch::Tensor material_index,
    torch::Tensor material_albedo_rgb,
    torch::Tensor light_kind,
    torch::Tensor light_position_or_direction_world,
    torch::Tensor light_intensity,
    torch::Tensor light_color_rgb,
    int64_t num_lights,
    double ambient_r,
    double ambient_g,
    double ambient_b,
    double background_r,
    double background_g,
    double background_b,
    torch::Tensor rgb,
    torch::Tensor intensity,
    torch::Tensor shadow_stack_overflow_count,
    torch::Tensor shadow_max_stack_depth) {
  shade_direct_light_no_shadow_impl(
      hit_mask,
      position_world,
      normal_world,
      material_index,
      material_albedo_rgb,
      light_kind,
      light_position_or_direction_world,
      light_intensity,
      light_color_rgb,
      num_lights,
      ambient_r,
      ambient_g,
      ambient_b,
      background_r,
      background_g,
      background_b,
      rgb,
      intensity,
      shadow_stack_overflow_count,
      shadow_max_stack_depth,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr);
}

void shade_direct_light_no_shadow_diagnostics_out(
    torch::Tensor hit_mask,
    torch::Tensor position_world,
    torch::Tensor normal_world,
    torch::Tensor material_index,
    torch::Tensor material_albedo_rgb,
    torch::Tensor light_kind,
    torch::Tensor light_position_or_direction_world,
    torch::Tensor light_intensity,
    torch::Tensor light_color_rgb,
    int64_t num_lights,
    double ambient_r,
    double ambient_g,
    double ambient_b,
    double background_r,
    double background_g,
    double background_b,
    torch::Tensor rgb,
    torch::Tensor intensity,
    torch::Tensor shadow_stack_overflow_count,
    torch::Tensor shadow_max_stack_depth,
    torch::Tensor traversal_counter_0,
    torch::Tensor traversal_counter_1,
    torch::Tensor traversal_counter_2,
    torch::Tensor traversal_counter_3,
    torch::Tensor traversal_counter_4,
    torch::Tensor traversal_counter_5,
    torch::Tensor traversal_counter_6,
    torch::Tensor traversal_counter_7,
    torch::Tensor traversal_counter_8) {
  shade_direct_light_no_shadow_impl(
      hit_mask,
      position_world,
      normal_world,
      material_index,
      material_albedo_rgb,
      light_kind,
      light_position_or_direction_world,
      light_intensity,
      light_color_rgb,
      num_lights,
      ambient_r,
      ambient_g,
      ambient_b,
      background_r,
      background_g,
      background_b,
      rgb,
      intensity,
      shadow_stack_overflow_count,
      shadow_max_stack_depth,
      &traversal_counter_0,
      &traversal_counter_1,
      &traversal_counter_2,
      &traversal_counter_3,
      &traversal_counter_4,
      &traversal_counter_5,
      &traversal_counter_6,
      &traversal_counter_7,
      &traversal_counter_8);
}

void shade_direct_light_with_shadows_impl(
    torch::Tensor hit_mask,
    torch::Tensor position_world,
    torch::Tensor normal_world,
    torch::Tensor material_index,
    torch::Tensor material_albedo_rgb,
    torch::Tensor light_kind,
    torch::Tensor light_position_or_direction_world,
    torch::Tensor light_intensity,
    torch::Tensor light_color_rgb,
    int64_t num_lights,
    torch::Tensor triangle_v0,
    torch::Tensor triangle_e1,
    torch::Tensor triangle_e2,
    torch::Tensor triangle_normal,
    torch::Tensor triangle_role_masks,
    torch::Tensor bvh_bounds_min,
    torch::Tensor bvh_bounds_max,
    torch::Tensor bvh_left,
    torch::Tensor bvh_right,
    torch::Tensor bvh_start,
    torch::Tensor bvh_count,
    torch::Tensor bvh_prim_ids,
    int64_t num_bvh_nodes,
    int64_t sensor_role_mask,
    double shadow_bias,
    double ambient_r,
    double ambient_g,
    double ambient_b,
    double background_r,
    double background_g,
    double background_b,
    torch::Tensor rgb,
    torch::Tensor intensity,
    torch::Tensor shadow_stack_overflow_count,
    torch::Tensor shadow_max_stack_depth,
    torch::Tensor* traversal_counter_0,
    torch::Tensor* traversal_counter_1,
    torch::Tensor* traversal_counter_2,
    torch::Tensor* traversal_counter_3,
    torch::Tensor* traversal_counter_4,
    torch::Tensor* traversal_counter_5,
    torch::Tensor* traversal_counter_6,
    torch::Tensor* traversal_counter_7,
    torch::Tensor* traversal_counter_8) {
  check_i32_1d(hit_mask, "hit_mask");
  check_float_2d_3(position_world, "position_world");
  check_float_2d_3(normal_world, "normal_world");
  check_i32_1d(material_index, "material_index");
  check_float_2d_3(material_albedo_rgb, "material_albedo_rgb");
  check_i32_1d(light_kind, "light_kind");
  check_float_2d_3(light_position_or_direction_world, "light_position_or_direction_world");
  check_float_2d_3(light_color_rgb, "light_color_rgb");
  TORCH_CHECK(light_intensity.is_cuda(), "light_intensity must be a CUDA tensor");
  TORCH_CHECK(light_intensity.dtype() == torch::kFloat32, "light_intensity must be float32");
  TORCH_CHECK(light_intensity.is_contiguous(), "light_intensity must be contiguous");
  TORCH_CHECK(light_intensity.dim() == 1, "light_intensity must be rank-1");
  check_float_2d_3(rgb, "rgb");
  check_float_1d(intensity, "intensity");
  check_i32_1d(shadow_stack_overflow_count, "shadow_stack_overflow_count");
  check_i32_1d(shadow_max_stack_depth, "shadow_max_stack_depth");
  torch::Tensor* traversal_counters[9] = {
      traversal_counter_0,
      traversal_counter_1,
      traversal_counter_2,
      traversal_counter_3,
      traversal_counter_4,
      traversal_counter_5,
      traversal_counter_6,
      traversal_counter_7,
      traversal_counter_8};
  bool traversal_stats_enabled = traversal_counter_0 != nullptr;
  for (torch::Tensor* counter : traversal_counters) {
    TORCH_CHECK(
        (counter != nullptr) == traversal_stats_enabled,
        "shadow traversal counters must be completely absent or present");
    if (counter != nullptr) {
      check_i32_1d(*counter, "shadow_traversal_counter");
      TORCH_CHECK(counter->size(0) == 1, "shadow traversal counters must have shape [1]");
    }
  }
  check_float_2d_3(triangle_v0, "triangle_v0");
  check_float_2d_3(triangle_e1, "triangle_e1");
  check_float_2d_3(triangle_e2, "triangle_e2");
  check_float_2d_3(triangle_normal, "triangle_normal");
  check_i64_1d(triangle_role_masks, "triangle_role_masks");
  check_float_2d_3(bvh_bounds_min, "bvh_bounds_min");
  check_float_2d_3(bvh_bounds_max, "bvh_bounds_max");
  check_i32_1d(bvh_left, "bvh_left");
  check_i32_1d(bvh_right, "bvh_right");
  check_i32_1d(bvh_start, "bvh_start");
  check_i32_1d(bvh_count, "bvh_count");
  check_i32_1d(bvh_prim_ids, "bvh_prim_ids");
  TORCH_CHECK(position_world.sizes() == normal_world.sizes(), "position_world/normal_world shape mismatch");
  TORCH_CHECK(hit_mask.size(0) == position_world.size(0), "hit_mask length must match position_world");
  TORCH_CHECK(
      material_index.size(0) == position_world.size(0),
      "material_index length must match position_world");
  TORCH_CHECK(num_lights >= 0 && num_lights <= light_kind.size(0), "num_lights out of range");
  TORCH_CHECK(num_lights <= light_position_or_direction_world.size(0), "num_lights exceeds light vectors");
  TORCH_CHECK(num_lights <= light_intensity.size(0), "num_lights exceeds light intensities");
  TORCH_CHECK(num_lights <= light_color_rgb.size(0), "num_lights exceeds light colors");
  int64_t num_triangles = triangle_v0.size(0);
  TORCH_CHECK(triangle_e1.size(0) == num_triangles, "triangle_e1 length mismatch");
  TORCH_CHECK(triangle_e2.size(0) == num_triangles, "triangle_e2 length mismatch");
  TORCH_CHECK(triangle_normal.size(0) == num_triangles, "triangle_normal length mismatch");
  TORCH_CHECK(triangle_role_masks.size(0) == num_triangles, "triangle_role_masks length mismatch");
  TORCH_CHECK(num_bvh_nodes >= 0 && num_bvh_nodes <= bvh_bounds_min.size(0), "num_bvh_nodes out of range");
  TORCH_CHECK(num_bvh_nodes <= bvh_bounds_max.size(0), "num_bvh_nodes exceeds bvh bounds");
  TORCH_CHECK(num_bvh_nodes <= bvh_left.size(0), "num_bvh_nodes exceeds bvh left");
  TORCH_CHECK(num_bvh_nodes <= bvh_right.size(0), "num_bvh_nodes exceeds bvh right");
  TORCH_CHECK(num_bvh_nodes <= bvh_start.size(0), "num_bvh_nodes exceeds bvh start");
  TORCH_CHECK(num_bvh_nodes <= bvh_count.size(0), "num_bvh_nodes exceeds bvh count");
  TORCH_CHECK(bvh_prim_ids.size(0) >= num_triangles, "bvh_prim_ids is shorter than triangles");
  TORCH_CHECK(std::isfinite(shadow_bias) && shadow_bias >= 0.0, "shadow_bias must be finite and >= 0");
  TORCH_CHECK(
      std::isfinite(ambient_r) && std::isfinite(ambient_g) && std::isfinite(ambient_b),
      "ambient RGB must be finite");
  TORCH_CHECK(
      std::isfinite(background_r) && std::isfinite(background_g) && std::isfinite(background_b),
      "background RGB must be finite");
  check_same_device(hit_mask, position_world, "position_world");
  check_same_device(hit_mask, normal_world, "normal_world");
  check_same_device(hit_mask, material_index, "material_index");
  check_same_device(hit_mask, material_albedo_rgb, "material_albedo_rgb");
  check_same_device(hit_mask, light_kind, "light_kind");
  check_same_device(hit_mask, light_position_or_direction_world, "light_position_or_direction_world");
  check_same_device(hit_mask, light_intensity, "light_intensity");
  check_same_device(hit_mask, light_color_rgb, "light_color_rgb");
  check_same_device(hit_mask, triangle_v0, "triangle_v0");
  check_same_device(hit_mask, triangle_e1, "triangle_e1");
  check_same_device(hit_mask, triangle_e2, "triangle_e2");
  check_same_device(hit_mask, triangle_normal, "triangle_normal");
  check_same_device(hit_mask, triangle_role_masks, "triangle_role_masks");
  check_same_device(hit_mask, bvh_bounds_min, "bvh_bounds_min");
  check_same_device(hit_mask, bvh_bounds_max, "bvh_bounds_max");
  check_same_device(hit_mask, bvh_left, "bvh_left");
  check_same_device(hit_mask, bvh_right, "bvh_right");
  check_same_device(hit_mask, bvh_start, "bvh_start");
  check_same_device(hit_mask, bvh_count, "bvh_count");
  check_same_device(hit_mask, bvh_prim_ids, "bvh_prim_ids");
  check_same_device(hit_mask, rgb, "rgb");
  check_same_device(hit_mask, intensity, "intensity");
  check_same_device(hit_mask, shadow_stack_overflow_count, "shadow_stack_overflow_count");
  check_same_device(hit_mask, shadow_max_stack_depth, "shadow_max_stack_depth");
  for (torch::Tensor* counter : traversal_counters) {
    if (counter != nullptr) {
      check_same_device(hit_mask, *counter, "shadow_traversal_counter");
    }
  }

  int64_t num_rays = hit_mask.size(0);
  TORCH_CHECK(num_rays <= static_cast<int64_t>(INT32_MAX), "too many rays");
  TORCH_CHECK(rgb.size(0) == num_rays, "rgb length must match hit_mask");
  TORCH_CHECK(intensity.size(0) == num_rays, "intensity length must match hit_mask");
  TORCH_CHECK(shadow_stack_overflow_count.size(0) == 1, "shadow_stack_overflow_count must have shape [1]");
  TORCH_CHECK(shadow_max_stack_depth.size(0) == 1, "shadow_max_stack_depth must have shape [1]");

  const c10::cuda::CUDAGuard device_guard(rgb.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(rgb.get_device());
  C10_CUDA_CHECK(cudaMemsetAsync(
      shadow_stack_overflow_count.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(shadow_max_stack_depth.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
  for (torch::Tensor* counter : traversal_counters) {
    if (counter != nullptr) {
      C10_CUDA_CHECK(cudaMemsetAsync(counter->data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
    }
  }
  int32_t* traversal_counter_ptrs[9];
  for (int counter = 0; counter < 9; ++counter) {
    traversal_counter_ptrs[counter] = traversal_stats_enabled
        ? traversal_counters[counter]->data_ptr<int32_t>()
        : nullptr;
  }

  if (num_rays > 0) {
    int block = 256;
    int grid = (static_cast<int>(num_rays) + block - 1) / block;
    shade_direct_light_with_shadows_kernel<<<grid, block, 0, stream>>>(
        hit_mask.data_ptr<int32_t>(),
        position_world.data_ptr<float>(),
        normal_world.data_ptr<float>(),
        material_index.data_ptr<int32_t>(),
        material_albedo_rgb.data_ptr<float>(),
        light_kind.data_ptr<int32_t>(),
        light_position_or_direction_world.data_ptr<float>(),
        light_intensity.data_ptr<float>(),
        light_color_rgb.data_ptr<float>(),
        static_cast<int>(num_lights),
        triangle_v0.data_ptr<float>(),
        triangle_e1.data_ptr<float>(),
        triangle_e2.data_ptr<float>(),
        triangle_normal.data_ptr<float>(),
        triangle_role_masks.data_ptr<int64_t>(),
        bvh_bounds_min.data_ptr<float>(),
        bvh_bounds_max.data_ptr<float>(),
        bvh_left.data_ptr<int32_t>(),
        bvh_right.data_ptr<int32_t>(),
        bvh_start.data_ptr<int32_t>(),
        bvh_count.data_ptr<int32_t>(),
        bvh_prim_ids.data_ptr<int32_t>(),
        static_cast<int>(num_bvh_nodes),
        sensor_role_mask,
        static_cast<float>(shadow_bias),
        static_cast<float>(ambient_r),
        static_cast<float>(ambient_g),
        static_cast<float>(ambient_b),
        static_cast<float>(background_r),
        static_cast<float>(background_g),
        static_cast<float>(background_b),
        rgb.data_ptr<float>(),
        intensity.data_ptr<float>(),
        shadow_stack_overflow_count.data_ptr<int32_t>(),
        shadow_max_stack_depth.data_ptr<int32_t>(),
        static_cast<int>(traversal_stats_enabled),
        traversal_counter_ptrs[0],
        traversal_counter_ptrs[1],
        traversal_counter_ptrs[2],
        traversal_counter_ptrs[3],
        traversal_counter_ptrs[4],
        traversal_counter_ptrs[5],
        traversal_counter_ptrs[6],
        traversal_counter_ptrs[7],
        traversal_counter_ptrs[8],
        static_cast<int>(num_rays));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void shade_direct_light_with_shadows_out(
    torch::Tensor hit_mask,
    torch::Tensor position_world,
    torch::Tensor normal_world,
    torch::Tensor material_index,
    torch::Tensor material_albedo_rgb,
    torch::Tensor light_kind,
    torch::Tensor light_position_or_direction_world,
    torch::Tensor light_intensity,
    torch::Tensor light_color_rgb,
    int64_t num_lights,
    torch::Tensor triangle_v0,
    torch::Tensor triangle_e1,
    torch::Tensor triangle_e2,
    torch::Tensor triangle_normal,
    torch::Tensor triangle_role_masks,
    torch::Tensor bvh_bounds_min,
    torch::Tensor bvh_bounds_max,
    torch::Tensor bvh_left,
    torch::Tensor bvh_right,
    torch::Tensor bvh_start,
    torch::Tensor bvh_count,
    torch::Tensor bvh_prim_ids,
    int64_t num_bvh_nodes,
    int64_t sensor_role_mask,
    double shadow_bias,
    double ambient_r,
    double ambient_g,
    double ambient_b,
    double background_r,
    double background_g,
    double background_b,
    torch::Tensor rgb,
    torch::Tensor intensity,
    torch::Tensor shadow_stack_overflow_count,
    torch::Tensor shadow_max_stack_depth) {
  shade_direct_light_with_shadows_impl(
      hit_mask,
      position_world,
      normal_world,
      material_index,
      material_albedo_rgb,
      light_kind,
      light_position_or_direction_world,
      light_intensity,
      light_color_rgb,
      num_lights,
      triangle_v0,
      triangle_e1,
      triangle_e2,
      triangle_normal,
      triangle_role_masks,
      bvh_bounds_min,
      bvh_bounds_max,
      bvh_left,
      bvh_right,
      bvh_start,
      bvh_count,
      bvh_prim_ids,
      num_bvh_nodes,
      sensor_role_mask,
      shadow_bias,
      ambient_r,
      ambient_g,
      ambient_b,
      background_r,
      background_g,
      background_b,
      rgb,
      intensity,
      shadow_stack_overflow_count,
      shadow_max_stack_depth,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr);
}

void shade_direct_light_with_shadows_diagnostics_out(
    torch::Tensor hit_mask,
    torch::Tensor position_world,
    torch::Tensor normal_world,
    torch::Tensor material_index,
    torch::Tensor material_albedo_rgb,
    torch::Tensor light_kind,
    torch::Tensor light_position_or_direction_world,
    torch::Tensor light_intensity,
    torch::Tensor light_color_rgb,
    int64_t num_lights,
    torch::Tensor triangle_v0,
    torch::Tensor triangle_e1,
    torch::Tensor triangle_e2,
    torch::Tensor triangle_normal,
    torch::Tensor triangle_role_masks,
    torch::Tensor bvh_bounds_min,
    torch::Tensor bvh_bounds_max,
    torch::Tensor bvh_left,
    torch::Tensor bvh_right,
    torch::Tensor bvh_start,
    torch::Tensor bvh_count,
    torch::Tensor bvh_prim_ids,
    int64_t num_bvh_nodes,
    int64_t sensor_role_mask,
    double shadow_bias,
    double ambient_r,
    double ambient_g,
    double ambient_b,
    double background_r,
    double background_g,
    double background_b,
    torch::Tensor rgb,
    torch::Tensor intensity,
    torch::Tensor shadow_stack_overflow_count,
    torch::Tensor shadow_max_stack_depth,
    torch::Tensor traversal_counter_0,
    torch::Tensor traversal_counter_1,
    torch::Tensor traversal_counter_2,
    torch::Tensor traversal_counter_3,
    torch::Tensor traversal_counter_4,
    torch::Tensor traversal_counter_5,
    torch::Tensor traversal_counter_6,
    torch::Tensor traversal_counter_7,
    torch::Tensor traversal_counter_8) {
  shade_direct_light_with_shadows_impl(
      hit_mask,
      position_world,
      normal_world,
      material_index,
      material_albedo_rgb,
      light_kind,
      light_position_or_direction_world,
      light_intensity,
      light_color_rgb,
      num_lights,
      triangle_v0,
      triangle_e1,
      triangle_e2,
      triangle_normal,
      triangle_role_masks,
      bvh_bounds_min,
      bvh_bounds_max,
      bvh_left,
      bvh_right,
      bvh_start,
      bvh_count,
      bvh_prim_ids,
      num_bvh_nodes,
      sensor_role_mask,
      shadow_bias,
      ambient_r,
      ambient_g,
      ambient_b,
      background_r,
      background_g,
      background_b,
      rgb,
      intensity,
      shadow_stack_overflow_count,
      shadow_max_stack_depth,
      &traversal_counter_0,
      &traversal_counter_1,
      &traversal_counter_2,
      &traversal_counter_3,
      &traversal_counter_4,
      &traversal_counter_5,
      &traversal_counter_6,
      &traversal_counter_7,
      &traversal_counter_8);
}
"""
