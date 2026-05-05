Initiative: q54-gpu-optical-executor
Stage: l5a-implementation-note
Author: codex
Version: v1
Date: 2026-05-03
Status: implemented
Related Files: optics/device.py, optics/warp_execution.py, optics/__init__.py, tests/unit/optics/test_device_optical.py, tests/gpu/test_optical_warp_executor.py
Owner Summary: Implemented L5A Warp brute-force first-hit optical executor. The device path consumes host `OpticalSceneSnapshot`, packs visible primitives per query, launches one Warp thread per ray, returns `OpticalComputeResult(location="device")`, and stages back to canonical host dtypes through `optics/device.py`.

# Q54 L5A GPU Optical Executor Implementation Note

## 1. Implemented Scope

This pass implements L5A only:

```text
host OpticalSceneSnapshot
  + OpticalRaySensorSpec
  -> host-packed visible primitive workload
  -> Warp brute-force first-hit kernel
  -> OpticalComputeResult(location="device")
  -> stage_optical_compute_result_to_host(...)
```

It does not integrate with `GpuPublishedFrame` / Q52 yet. That remains L5B.

## 2. New API

Added `optics/device.py`:

- `HostOpticalPrimitiveWorkload`
- `build_host_optical_primitive_workload(snapshot, sensor_role=...)`
- `pack_source_order_key(instance_index, primitive_index_within_instance)`
- `stage_optical_compute_result_to_host(result)`
- `MAX_PRIMITIVES_PER_INSTANCE = 2**32`
- `DEVICE_FLOAT32_RECOMMENDED_SCENE_SCALE_M = 1000.0`

Added `optics/warp_execution.py`:

- `GpuBruteForceOpticalExecutor`

Exported from `optics.__init__`.

## 3. Workload Packing

L5A filters roles on the host before upload:

```text
if spec.sensor_role not in instance.roles:
    skip instance
```

The device kernel therefore receives only visible primitives and does not need
Python strings or role bitmasks.

Packed arrays:

```text
triangles_world: float32[num_triangles, 3, 3]
triangle_numeric_instance_id: int32[num_triangles]
triangle_source_order_key: int64[num_triangles]

plane_normal_world: float32[num_planes, 3]
plane_point_world: float32[num_planes, 3]
plane_numeric_instance_id: int32[num_planes]
plane_source_order_key: int64[num_planes]
```

`source_order_key` follows the Claude-reviewed packed int64 rule:

```text
key = instance_index * 2**32 + primitive_index_within_instance
```

This preserves CPU lexicographic source order with one integer comparison in
Warp.

## 4. Kernel Algorithm

The first kernel uses one Warp thread per ray.

For each ray:

1. initialize `best_t = max_distance`;
2. scan all visible planes;
3. scan all visible triangles with Moller-Trumbore;
4. update winner by:

```text
t < best_t - 1e-5
or abs(t - best_t) <= 1e-5 and source_order_key < best_source_order_key
```

5. write device result buffers.

Miss values are produced by fresh preinitialized result buffers:

```text
hit_mask = 0
range_m = inf
position_world = NaN
normal_world = NaN
numeric_instance_id = 0
```

## 5. Result Contract

Device result:

```text
OpticalComputeResult(location="device")
channels:
  hit_mask              int32[num_rays]
  range_m               float32[num_rays]
  position_world        float32[num_rays, 3]
  normal_world          float32[num_rays, 3]
  numeric_instance_id   int32[num_rays]
```

Host staging normalizes to canonical host dtypes:

```text
hit_mask              bool
range_m               float64
position_world        float64
normal_world          float64
numeric_instance_id   int64
```

## 6. Deliberate Constraints

L5A requires finite `OpticalRaySensorSpec.max_distance`. Future shadow rays
with infinite max distance belong to L5C and should explicitly test Warp
float32 infinity behavior.

`GpuBruteForceOpticalExecutor` documents that float32 device buffers are meant
for robot-scale scenes below roughly 1000 meters. Larger scenes may require
float64 device buffers or origin rebasing.

## 7. Tests

CPU-only tests:

```text
PYTHONPATH=. pytest tests/unit/optics/test_device_optical.py -q
5 passed
```

GPU tests in the Warp conda environment:

```text
conda run -n env_tilelang_20260119 python -m pytest tests/gpu/test_optical_warp_executor.py -q
5 passed
```

GPU coverage:

- plane hit + miss parity;
- triangle hit + miss parity;
- packed source-order tie-break parity;
- role filtering parity;
- staged device result works with pinhole camera postprocess.

## 8. Deferred To L5B

- borrowing `GpuPublishedFrame`;
- Q52 `borrow_device_frame(...)` / `complete_device_consumer(...)`;
- result `ready_event`;
- result buffer pooling;
- device scene cache;
- GPU BVH;
- GPU direct-light / shadows.
