Initiative: q54-gpu-optical-executor
Stage: l5c0-device-scene-cache-implementation-note
Author: codex
Version: v1
Date: 2026-05-03
Status: implemented
Related Files: optics/device_scene.py, optics/__init__.py, tests/unit/optics/test_device_scene.py, tests/gpu/test_optical_gpu_runtime.py, MANIFEST.md, OPEN_QUESTIONS.md
Owner Summary: Implemented L5C.0 device-resident optical scene cache. Registry geometry and metadata are uploaded once to device buffers; per-frame update kernels read `GpuPublishedFrame.x_world_*` and produce fresh world-space primitive buffers. This stabilizes the device scene boundary before L5C.1 executor-over-device-scene and L5C.2 GPU BVH/OptiX work.

# Q54 L5C.0 Device Scene Cache Implementation Note

## 1. Scope

This implements L5C.0 only:

```text
OpticalWorldRegistry
  -> DeviceOpticalSceneCache
  -> DeviceOpticalScene(long-lived local geometry + metadata buffers)

GpuPublishedFrame N
  -> DeviceOpticalSceneSnapshot(fresh world primitive buffers)
```

It does not yet make the ray executor consume device scene buffers. That is
L5C.1.

## 2. New API

Added `optics/device_scene.py`:

- `DeviceOpticalRoleTable`
- `DeviceOpticalScene`
- `DeviceOpticalSceneSnapshot`
- `DeviceOpticalSceneCache`
- `build_device_optical_scene(...)`
- `update_device_optical_scene_from_gpu_frame(...)`

Exports are added through `optics.__init__`.

## 3. Transform Convention

Confirmed before implementation:

```text
OpticalInstanceSpec.X_body_geometry
```

is the pose of the geometry frame expressed in the body frame. It maps
geometry-local points into the body frame.

The update kernel follows the existing host scene composition:

```text
X_world_geometry = X_world_body @ X_body_geometry
R_world_geometry = R_world_body @ R_body_geometry
r_world_geometry = r_world_body + R_world_body @ r_body_geometry
```

World-static instances use `body_index == -1`, so `X_body_geometry` is
interpreted directly as world-from-geometry.

## 4. Device Layout

Long-lived scene buffers:

```text
triangle_vertices_local:           float32[num_triangles, 9]
triangle_instance_index:           int32[num_triangles]
triangle_source_order_key:         int64[num_triangles]
triangle_role_mask:                int32[num_triangles]
triangle_numeric_instance_id:      int32[num_triangles]

plane_normal_local:                float32[num_planes, 3]
plane_point_local:                 float32[num_planes, 3]
plane_instance_index:              int32[num_planes]
plane_source_order_key:            int64[num_planes]
plane_role_mask:                   int32[num_planes]
plane_numeric_instance_id:         int32[num_planes]

instance_body_index:               int32[num_instances]  # -1 for world-static
instance_X_body_geometry_R:        float32[num_instances, 9]
instance_X_body_geometry_r:        float32[num_instances, 3]
```

Per-frame snapshot buffers:

```text
triangles_world:                   float32[num_triangles, 9]
plane_normal_world:                float32[num_planes, 3]
plane_point_world:                 float32[num_planes, 3]
```

## 5. Role Mask

`DeviceOpticalRoleTable` assigns deterministic bit positions from sorted role
strings and uses int32 masks with a 31-role cap.

Unknown role returns mask `0`. This matches host semantics: if no instance has
the requested role, all primitives are invisible and the query should miss.

## 6. Buffer Lifetime

L5C.0 intentionally fresh-allocates world primitive buffers per snapshot.

Reason: the purpose of this pass is correctness and boundary validation. A
ring/pool requires event-based overwrite rules; that should wait until the
device scene executor path is stable.

`DeviceOpticalSceneSnapshot.resources` keeps the borrowed frame object alive
while the update stream may still reference its device arrays.

## 7. Kernel Boundary

The update kernels are scene/cache work:

- transform triangle vertices;
- transform plane normal/point;
- normalize plane normals.

They do not:

- traverse rays;
- shade;
- produce `OpticalComputeResult`;
- apply sensor noise or camera response.

## 8. Tests

CPU:

```text
PYTHONPATH=. pytest tests/unit/optics/test_device_scene.py -q
3 passed
```

GPU:

```text
conda run -n env_tilelang_20260119 python -m pytest tests/gpu/test_optical_gpu_runtime.py -q
7 passed
```

Coverage added:

- deterministic int32 role masks;
- 31-role limit;
- registry-derived role table;
- world-static plane update from GPU frame;
- body-bound triangle update from `GpuPublishedFrame.x_world_*`.

Full optical verification:

```text
PYTHONPATH=. pytest tests/unit/optics tests/unit/sensing -q
93 passed

conda run -n env_tilelang_20260119 python -m pytest tests/gpu/test_optical_warp_executor.py tests/gpu/test_optical_gpu_runtime.py -q
12 passed
```

## 9. Deferred

- `GpuDeviceSceneOpticalExecutor` consuming `DeviceOpticalSceneSnapshot`;
- device-scene parity against L5B.1 first-hit results;
- result buffer pooling;
- GPU BVH / OptiX evaluation;
- multi-env batched optical scene update and per-ray env indexing.
