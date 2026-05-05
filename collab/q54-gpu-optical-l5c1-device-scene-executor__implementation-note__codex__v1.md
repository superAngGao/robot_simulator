Initiative: q54-gpu-optical-executor
Stage: l5c1a-device-scene-executor-implementation-note
Author: codex
Version: v1
Date: 2026-05-04
Status: implemented
Related Files: optics/device_scene.py, optics/warp_execution.py, optics/__init__.py, tests/unit/optics/test_device_scene.py, tests/gpu/test_optical_gpu_runtime.py, MANIFEST.md, OPEN_QUESTIONS.md
Owner Summary: Implemented L5C.1-pre and L5C.1a. Device optical role masks now use int64/63-role buffers. Added `GpuDeviceSceneOpticalExecutor`, a correctness-first ray-major Warp executor that consumes `DeviceOpticalSceneSnapshot` directly, waits on snapshot readiness, filters roles on device, and returns device `OpticalComputeResult` buffers without retaining broad snapshot/frame objects.

# Q54 L5C.1 Device Scene Executor Implementation Note

## 1. Scope

This implements:

```text
L5C.1-pre:
  int64 device role masks

L5C.1a:
  GpuDeviceSceneOpticalExecutor over current triangles_world layout
```

It does not yet implement:

- derived triangle buffers (`v0/e1/e2/normal`);
- AABB early reject;
- GPU BVH / OptiX;
- result buffer pooling;
- multi-env batched optical queries.

## 2. Role Mask ABI

`DeviceOpticalRoleTable` now supports 63 roles:

```text
role mask dtype: int64
unknown role: 0
max role count: 63
```

Device scene buffers changed accordingly:

```text
triangle_role_mask: int64[num_triangles]
plane_role_mask:    int64[num_planes]
```

The GPU runtime test includes a role above bit 31 (`role_40`) to verify Warp
int64 bitwise visibility checks on the target CUDA path.

## 3. New Executor

Added `GpuDeviceSceneOpticalExecutor` in `optics/warp_execution.py`.

Input:

```text
DeviceOpticalSceneSnapshot
OpticalRaySensorSpec
```

Output:

```text
OpticalComputeResult(location="device")
channels:
  hit_mask
  range_m
  position_world
  normal_world
  numeric_instance_id
```

The executor:

- waits on `snapshot.ready_event`;
- uploads per-call ray origins/directions;
- reads world primitive buffers and metadata directly from device scene arrays;
- filters primitives by `primitive_role_mask & sensor_role_mask`;
- preserves L5A distance/source-order tie-break semantics;
- records a result `ready_event`.

## 4. Resource Lifetime

Following review feedback, `OpticalComputeResult.resources` does not hold the
whole `DeviceOpticalSceneSnapshot` or borrowed `GpuPublishedFrame`.

It holds the concrete device arrays required by the traversal kernel:

- uploaded ray arrays;
- snapshot world primitive buffers;
- scene primitive metadata buffers.

The traversal stream waits on `snapshot.ready_event` before launch, so holding
the broader snapshot/frame object is not required for execution safety.

## 5. Tests

CPU/unit:

```text
PYTHONPATH=. pytest tests/unit/optics/test_device_scene.py -q
3 passed
```

GPU runtime:

```text
conda run -n env_tilelang_20260119 python -m pytest tests/gpu/test_optical_gpu_runtime.py -q
11 passed
```

Full focused optical verification:

```text
PYTHONPATH=. pytest tests/unit/optics tests/unit/sensing -q
93 passed

conda run -n env_tilelang_20260119 python -m pytest \
  tests/gpu/test_optical_warp_executor.py tests/gpu/test_optical_gpu_runtime.py -q
16 passed
```

Coverage added:

- L5C.1a device-scene executor parity with L5B.1 for a body-bound plane;
- L5C.1a device-scene executor parity with L5A for a world-static triangle mesh;
- unknown role returns all misses on device;
- int64 role mask above 31 bits works in the Warp visibility check;
- result resources do not retain the snapshot or borrowed frame object.

## 6. Deferred

Next steps:

```text
L5C.1b:
  add triangle_v0/e1/e2/normal derived buffers;
  validate against temporary triangles_world;
  switch traversal to derived buffers.

L5C.1c:
  add triangle_aabb_min/max and benchmarkable AABB traversal variant.

L5C.2:
  evaluate GPU BVH / OptiX.
```
