Initiative: q54-gpu-optical-executor
Stage: l5c1b-derived-triangle-layout-implementation-note
Author: codex
Version: v1
Date: 2026-05-04
Status: implemented
Related Files: optics/device_scene.py, optics/warp_execution.py, tests/gpu/test_optical_gpu_runtime.py, MANIFEST.md, OPEN_QUESTIONS.md
Owner Summary: Implemented L5C.1b derived triangle layout. `DeviceOpticalSceneSnapshot` now stores `triangle_v0_world`, `triangle_e1_world`, `triangle_e2_world`, and `triangle_normal_world` instead of `triangles_world`. The device-scene executor consumes the derived buffers directly, avoiding per-ray edge and normal reconstruction. Existing L5C.1a/L5A parity tests remain green.

# Q54 L5C.1b Derived Triangle Layout Implementation Note

## 1. Scope

This implements L5C.1b:

```text
DeviceOpticalSceneSnapshot triangle payload:
  triangle_v0_world
  triangle_e1_world
  triangle_e2_world
  triangle_normal_world

GpuDeviceSceneOpticalExecutor:
  consumes v0/e1/e2/normal directly
```

It removes the device-scene `triangles_world[num_triangles, 9]` buffer from
the live L5C path.

## 2. Snapshot Layout

`DeviceOpticalSceneSnapshot` now contains:

```text
triangle_v0_world:      float32[num_triangles, 3]
triangle_e1_world:      float32[num_triangles, 3]
triangle_e2_world:      float32[num_triangles, 3]
triangle_normal_world:  float32[num_triangles, 3]
plane_normal_world:     float32[num_planes, 3]
plane_point_world:      float32[num_planes, 3]
```

The triangle vertices can be reconstructed as:

```text
v0 = triangle_v0_world
v1 = triangle_v0_world + triangle_e1_world
v2 = triangle_v0_world + triangle_e2_world
```

## 3. Update Kernel

The triangle update kernel now computes derived world-space payloads once per
frame:

```text
v0_world = R_world_geometry @ v0_local + r_world_geometry
e1_world = R_world_geometry @ (v1_local - v0_local)
e2_world = R_world_geometry @ (v2_local - v0_local)
normal_world = R_world_geometry @ normalize(cross(e1_local, e2_local))
```

This uses the rigid-transform property:

```text
(R @ e1) x (R @ e2) = det(R) * R @ (e1 x e2)
```

For rotation matrices `det(R) = 1`, so the local normalized normal can be
rotated directly into world space. Build-time filtering already removes
degenerate triangles before device packing.

## 4. Traversal Kernel

`GpuDeviceSceneOpticalExecutor` now reads:

```text
triangle_v0_world
triangle_e1_world
triangle_e2_world
triangle_normal_world
```

The Moller-Trumbore intersection no longer reconstructs `e1/e2` or normal per
ray/triangle pair. On hit, the stored normal is flipped against the ray
direction just like L5A.

Role filtering, source-order tie-break, result channels, and resource lifetime
semantics are unchanged from L5C.1a.

## 5. Tests

Verification:

```text
PYTHONPATH=. pytest tests/unit/optics tests/unit/sensing -q
93 passed

conda run -n env_tilelang_20260119 python -m pytest \
  tests/gpu/test_optical_warp_executor.py tests/gpu/test_optical_gpu_runtime.py -q
16 passed
```

Relevant coverage:

- body-bound triangle update reconstructs vertices via `v0/e1/e2` and checks
  `triangle_normal_world`;
- explicit L5C.1b derived executor traversal parity with L5A for a
  world-static triangle mesh;
- body-bound plane parity, unknown role all-miss, and int64 role bit > 31 tests
  remain green.

## 6. Deferred

Next steps:

```text
L5C.1c:
  add triangle_aabb_min/max and a benchmarkable AABB traversal variant.

L5C.2:
  evaluate GPU BVH / OptiX.
```
