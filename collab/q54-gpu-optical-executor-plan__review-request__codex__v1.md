Initiative: q54-gpu-optical-executor
Stage: review-request
Author: codex
Version: v1
Date: 2026-05-02
Status: ready-for-claude-review
Related Files: optics/execution.py, optics/scene.py, sensing/optical.py, physics/gpu_engine.py, collab/q52-device-consumer-completion__implementation-note__codex__v1.md
Owner Summary: Proposed first GPU optical path. The recommended route is a Warp brute-force first-hit executor before GPU BVH/direct-light: first validate device result buffers and CPU parity from a host OpticalSceneSnapshot, then integrate with GpuPublishedFrame/Q52 device-consumer completion. This is GPU optical ray tracing, not Rerun or raster rendering.

# Q54 GPU Optical Executor Plan

## 0. Claude Review Decisions

Claude review accepted the plan with edits:

- keep one `OpticalComputeResult` type; use `location="device"`;
- put `stage_optical_compute_result_to_host(...)` in `optics/device.py`;
- keep L5A role filtering on the host before upload;
- use float32 device buffers for robot-scale scenes and normalize staged host
  channels to the canonical host dtypes;
- keep L5B Q52 borrow/complete in an explicit helper before folding it into a
  broader executor/cache abstraction;
- allocate fresh device result buffers per call for L5A/L5B;
- pack source-order keys into a single int64 for GPU tie-break.

Review archive:
`collab/q54-gpu-optical-executor-plan__review__claude__v1.md`.

## 1. Scope

This plan covers our first in-repo GPU optical computation path.

It is not:

- Rerun installation;
- a Rerun backend feature;
- a raster framebuffer renderer;
- OptiX;
- GPU BVH;
- GPU direct-light shading;
- a physically based renderer.

It is:

```text
OpticalSceneSnapshot / GpuPublishedFrame
  + OpticalRaySensorSpec
  -> Warp first-hit optical kernels
  -> OpticalComputeResult(location="device")
  -> explicit host staging or downstream device consumer
```

The first GPU goal is to make the optical result contract work on device, not
to chase performance immediately.

## 2. Current State

Already implemented:

- CPU registry / scene / executor pipeline;
- CPU BVH first-hit executor;
- CPU direct-light executor;
- pinhole camera ray generation and projected `depth_m` postprocess;
- host `OpticalCameraReading`;
- Rerun sink for already-computed readings;
- Q52 device-consumer completion API in `GpuEngine`.

Not implemented:

- device optical scene buffers;
- Warp optical kernels;
- device `OpticalComputeResult` staging helpers;
- optical executor integration with `GpuPublishedFrame`;
- GPU result buffer pooling / lifecycle.

## 3. Recommended Sequence

### L5A — Warp Brute-Force Executor From Host Snapshot

Implement a correctness-first `GpuBruteForceOpticalExecutor`.

Input:

```text
OpticalSceneSnapshot(location="host")
OpticalRaySensorSpec(location effectively host)
```

Preparation:

```text
host snapshot
  -> executor-owned per-query primitive packing
  -> Warp arrays on device
  -> one thread per ray
```

Output:

```text
OpticalComputeResult(
    location="device",
    channels={
        "hit_mask": wp.array[int32 or bool],
        "range_m": wp.array[float32],
        "position_world": wp.array[vec3/float32],
        "normal_world": wp.array[vec3/float32],
        "numeric_instance_id": wp.array[int32],
    },
    ready_event=<device event or None>
)
```

Then add:

```text
stage_optical_compute_result_to_host(device_result)
  -> OpticalComputeResult(location="host")
```

The staged host result should normalize to the existing host contract:

- `hit_mask`: bool;
- `range_m`: float64;
- `position_world`: float64 with NaN misses;
- `normal_world`: float64 with NaN misses;
- `numeric_instance_id`: int64 with `0` background id.

Why start here:

- isolates kernel math and result schema;
- avoids Q52 slot lifetime complexity in the first device test;
- lets CPU/GPU parity tests use the same static optical scene;
- keeps the first GPU step small enough to debug.

### L5B — Q52 GpuPublishedFrame Integration

After L5A parity works, integrate with GPU physics frames.

Flow:

```text
GpuEngine publishes frame N
  -> optical runtime borrows frame via engine.borrow_device_frame(consumer_id, stream)
  -> optical kernels read frame-owned x_world_R/x_world_r if needed
  -> optical kernels write result-owned device buffers
  -> engine.complete_device_consumer(consumer_id, frame_id, stream)
  -> OpticalComputeResult.ready_event = done_event
```

Key rule:

The Q52 device consumer is complete when optical kernels have finished reading
the published frame slot, not when every downstream consumer has finished using
the optical result buffers.

Therefore result buffers must be owned by the optical executor/runtime, not by
the `GpuPublishedFrame` slot. In the first version, allocate fresh result
buffers per call. Later, introduce an optical result pool with its own reclaim
rules.

### L5C — GPU Direct-Light Or GPU BVH

Only after L5A/L5B:

- GPU direct-light can reuse first-hit results and add a shading kernel;
- GPU shadow rays need acceleration, so direct-light with shadows should wait
  for GPU BVH or restrict itself to tiny debug scenes;
- GPU BVH can be added once device scene buffers and result lifecycle are
  stable.

## 4. Data Layout For L5A

Use simple flat arrays. Prefer float32 on device for the first version.
`GpuBruteForceOpticalExecutor` should document that this first device path is
intended for robot-scale scenes, roughly below 1000 meters. Larger scenes may
need float64 device buffers or origin rebasing later.

### Triangle Workload

```text
triangles_world: float32[num_triangles, 3, 3]
triangle_normals_world: float32[num_triangles, 3]
triangle_instance_index: int32[num_triangles]
triangle_numeric_instance_id: int32[num_triangles]
triangle_source_order_key: int64[num_triangles]
```

### Plane Workload

```text
plane_normal_world: float32[num_planes, 3]
plane_point_world: float32[num_planes, 3]
plane_instance_index: int32[num_planes]
plane_numeric_instance_id: int32[num_planes]
plane_source_order_key: int64[num_planes]
```

Packed source-order key:

```text
MAX_PRIMITIVES_PER_INSTANCE = 2**32
source_order_key = instance_index * MAX_PRIMITIVES_PER_INSTANCE + primitive_index_within_instance
```

The packer must validate:

```text
primitive_index_within_instance < MAX_PRIMITIVES_PER_INSTANCE
```

This preserves CPU lexicographic order `(instance_index,
primitive_index_within_instance)` while keeping the Warp kernel comparison to a
single int64 comparison.

### Ray Workload

```text
origins_world: float32[num_rays, 3]
directions_world: float32[num_rays, 3]
max_distance: float32 scalar
```

### Result Buffers

```text
hit_mask: int32[num_rays]        # 0/1 on device; staged to bool on host
range_m: float32[num_rays]
position_world: float32[num_rays, 3]
normal_world: float32[num_rays, 3]
numeric_instance_id: int32[num_rays]
```

Miss values:

```text
hit_mask = 0
range_m = inf
position_world = (nan, nan, nan)
normal_world = (nan, nan, nan)
numeric_instance_id = 0
```

## 5. Visibility / Roles

Do not put Python strings in device kernels.

For L5A, the executor can prepare a per-query workload by filtering snapshot
instances on the host using `spec.sensor_role`, then uploading only visible
primitives. This makes the uploaded workload sensor-specific, but it is
executor preparation, not scene-cache state.

Later, cache-owned device scene buffers should carry compact role masks:

```text
rgb          -> bit 0
depth        -> bit 1
lidar        -> bit 2
segmentation -> bit 3
```

Arbitrary role strings can remain host-only until we need them on GPU.

## 6. Kernel Algorithm

First version: one Warp thread per ray.

Pseudo-code:

```text
for ray i:
    best_t = max_distance
    best_source_order = (+inf, +inf)
    miss defaults

    for each plane:
        intersect plane
        if hit and better_by_t_then_source_order:
            update best hit

    for each triangle:
        Moller-Trumbore intersect
        if hit and better_by_t_then_source_order:
            update best hit

    write result buffers
```

Tie-break must match CPU:

```text
if t < best_t - t_eps:
    update
elif abs(t - best_t) <= t_eps and packed_source_order_key < best_source_order_key:
    update
```

Suggested GPU constants:

```text
build_eps = 1e-8   # float32 geometric degeneracy
dir_eps   = 1e-8
t_eps     = 1e-5   # float32 tie-break tolerance
```

These are deliberately looser than CPU `float64` constants. CPU/GPU parity
tests should use tolerances such as `atol=1e-4` or `1e-3` depending on scene
scale.

L5A only uses finite primary-ray `max_distance` values supplied by
`OpticalRaySensorSpec`. Future GPU shadow rays may need `max_distance=inf`;
when L5C adds shadows, explicitly test Warp float32 infinity behavior instead
of relying on an unstated assumption.

## 7. What Not To Implement First

Do not implement these in L5A:

- GPU BVH;
- GPU direct-light / RGB;
- shadow rays;
- material albedo buffers;
- string `instance_id` / `material_id` channels on device;
- semantic maps beyond numeric instance id;
- texture sampling;
- medium / volume;
- result buffer pooling;
- Rerun integration.

Rerun and PNG previews should consume staged host readings/results for now.

## 8. API Sketch

Potential module layout:

```text
optics/device.py
  DeviceOpticalWorkloadBuffers
  DeviceOpticalResultBuffers
  stage_optical_compute_result_to_host(...)

optics/warp_execution.py
  GpuBruteForceOpticalExecutor
```

Executor shape:

```python
class GpuBruteForceOpticalExecutor(OpticalExecutor):
    capabilities = frozenset({
        "hit_mask",
        "range_m",
        "position_world",
        "normal_world",
        "numeric_instance_id",
    })

    def __init__(self, *, device: str | None = None, stream=None) -> None:
        ...

    def execute(self, snapshot, spec) -> OpticalComputeResult:
        ...
```

Host staging:

```python
host_result = stage_optical_compute_result_to_host(device_result)
```

For L5B Q52 integration, avoid hiding borrow/complete inside a generic executor
until the ownership is stable. Prefer an explicit runtime helper first:

```python
execute_optical_on_gpu_published_frame(
    engine,
    consumer_id,
    registry,
    spec,
    *,
    stream=None,
) -> OpticalComputeResult
```

This helper can:

1. borrow the frame;
2. launch kernels;
3. complete the device consumer;
4. return a result with `ready_event`.

## 9. Tests

### L5A Tests

Skip if Warp/GPU is unavailable.

Required tests:

- device executor returns `OpticalComputeResult(location="device")`;
- staged host result has canonical channel dtypes and shapes;
- plane first-hit parity with CPU reference;
- triangle first-hit parity with CPU reference;
- miss values;
- source-order tie-break parity;
- role filtering parity;
- pinhole camera postprocess works after staging.

### L5B Tests

Use existing GPU test environment.

Required tests:

- optical runtime borrows via `GpuEngine.borrow_device_frame(...)`;
- completion calls `GpuEngine.complete_device_consumer(...)`;
- published slot is not released before optical kernel completion;
- result buffers remain valid after frame consumer completion;
- missing completion can still be caught by Q52 lag/stall mechanics.

## 10. Resolved Review Questions

1. L5A host-snapshot Warp parity -> L5B GpuPublishedFrame/Q52 integration is
   the right split.
2. L5A should return the existing `OpticalComputeResult(location="device")`.
   Do not introduce a separate `DeviceOpticalComputeResult` type.
3. L5A should filter roles on the host before upload. Device role bitmasks are
   deferred until longer-lived device scene buffers exist.
4. Float32 device buffers are acceptable for the first GPU executor. Host
   staging normalizes to canonical host dtypes.
5. `stage_optical_compute_result_to_host(...)` belongs in `optics/device.py`.
6. L5B should use an explicit optical runtime helper for Q52 borrow/complete
   before folding that lifecycle into cache/executor abstractions.
7. Fresh per-call device result allocation is acceptable for L5A/L5B; result
   pooling is deferred until correctness and lifecycle are stable.
