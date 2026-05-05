Initiative: q54-gpu-optical-executor
Stage: l5c1-kernel-layout-plan-review-request
Author: codex
Version: v1
Date: 2026-05-04
Status: ready-for-claude-review
Related Files: optics/device_scene.py, optics/warp_execution.py, optics/device.py, optics/gpu_runtime.py, sensing/optical.py
Owner Summary: Proposed L5C.1 kernel/layout path after L5C.0. The recommendation is to first land a correctness-focused `GpuDeviceSceneOpticalExecutor` over the current `triangles_world` layout, then add an intersection-friendly derived triangle layout (`v0/e1/e2/normal`, optionally AABB) as a measured performance step before GPU BVH/OptiX.

# Q54 L5C.1 GPU Device Scene Executor Kernel/Layout Plan

## 1. Current State

L5C.0 has implemented the device-resident scene cache:

```text
OpticalWorldRegistry
  -> DeviceOpticalSceneCache
  -> DeviceOpticalScene(long-lived local geometry + metadata buffers)

GpuPublishedFrame
  -> DeviceOpticalSceneSnapshot(fresh world primitive buffers)
```

The per-frame snapshot currently exposes:

```text
triangles_world:      float32[num_triangles, 9]
plane_normal_world:  float32[num_planes, 3]
plane_point_world:   float32[num_planes, 3]
```

The existing L5A GPU executor remains ray-major brute force:

```text
one thread = one ray
for every plane:
    intersect
for every triangle:
    reconstruct edges/normal and intersect
write best hit
```

L5A is a good correctness baseline, but it repeatedly computes triangle
derived data for every ray and still uploads a host-packed primitive workload
per execute call.

## 2. Goal

L5C.1 should make the GPU ray executor consume
`DeviceOpticalSceneSnapshot` directly:

```text
DeviceOpticalSceneSnapshot
  + OpticalRaySensorSpec
  + sensor_role_mask
  -> GpuDeviceSceneOpticalExecutor
  -> OpticalComputeResult(location="device")
```

The first L5C.1 goal is correctness and lifecycle stability:

- no host body-transform staging;
- no per-call primitive upload;
- device role-mask filtering;
- same `OpticalComputeResult` channel schema as L5A;
- parity against L5B.1 for world-static and body-bound scenes.

## 3. Layout Option A: Current Vertex Layout

Current snapshot triangle layout:

```text
triangles_world:
  [v0.x, v0.y, v0.z,
   v1.x, v1.y, v1.z,
   v2.x, v2.y, v2.z]
```

Benefits:

- simple and geometry-faithful;
- smallest triangle world buffer: 9 float32 values per triangle;
- update kernel is straightforward;
- easy to debug, stage, and compare with CPU/host mesh geometry;
- good for landing the first device-scene executor with minimal moving parts.

Costs:

- traversal recomputes `e1 = v1 - v0`, `e2 = v2 - v0`, and triangle normal for
  every ray/triangle pair;
- the repeated work scales as `num_rays * num_triangles`;
- adding primitive AABB culling later requires either extra buffers or repeated
  bound computation;
- not ideal for camera/LiDAR workloads where ray count is high.

## 4. Layout Option B: Derived Intersection Layout

Proposed performance-oriented snapshot triangle layout:

```text
triangle_v0_world:      float32[num_triangles, 3]
triangle_e1_world:      float32[num_triangles, 3]
triangle_e2_world:      float32[num_triangles, 3]
triangle_normal_world:  float32[num_triangles, 3]
```

Optional coarse-culling extension:

```text
triangle_aabb_min:      float32[num_triangles, 3]
triangle_aabb_max:      float32[num_triangles, 3]
```

Benefits:

- Moller-Trumbore traversal reads the exact values it needs;
- triangle normal is computed once per frame, not once per ray/triangle pair;
- AABB buffers can provide a cheap early reject before full triangle
  intersection;
- this layout is closer to a future BVH leaf primitive payload.

Costs:

- update kernel becomes heavier;
- memory increases:
  - `v0/e1/e2/normal`: 12 float32 values per triangle, about 33% more than
    `triangles_world`;
  - with AABB: 18 float32 values per triangle, 2x the current world vertex
    buffer;
- complete triangle vertices must be reconstructed for debug/export:
  `v1 = v0 + e1`, `v2 = v0 + e2`;
- if ray count is tiny, derived layout may not pay for its extra update work.

## 5. Recommendation

Do not combine the first device-scene executor with the derived layout change.

Recommended sequence:

### L5C.1a — Correctness Executor Over Current Layout

Implement:

```text
GpuDeviceSceneOpticalExecutor
  consumes DeviceOpticalSceneSnapshot.triangles_world / plane_*_world
  uses DeviceOpticalScene role masks/source keys/numeric ids
  returns OpticalComputeResult(location="device")
```

Kernel shape:

```text
one thread = one ray
for every visible plane:
    intersect
for every visible triangle:
    reconstruct e1/e2/normal
    intersect
write best hit
```

This keeps the implementation close to L5A and makes parity failures easier to
diagnose.

Tests:

- world-static plane parity with L5B.1/L5A;
- body-bound plane or triangle parity after `GpuPublishedFrame` transform;
- source-order tie-break parity;
- unknown role returns all misses;
- staged result still satisfies camera postprocess.

### L5C.1b — Derived Triangle Buffers

Extend `DeviceOpticalSceneSnapshot` with:

```text
triangle_v0_world
triangle_e1_world
triangle_e2_world
triangle_normal_world
```

Update the triangle world update kernel to compute these values once per
frame. Then switch `GpuDeviceSceneOpticalExecutor` triangle traversal to the
derived layout.

Keep `triangles_world` temporarily if it reduces churn in tests/debugging.
Remove or demote it only after parity and benchmark coverage are stable.

Tests:

- same parity suite as L5C.1a;
- explicit normal orientation parity for front/back hits;
- degenerate triangle behavior remains consistent with the current build-time
  filtering contract.

### L5C.1c — Optional AABB Early Reject

Add:

```text
triangle_aabb_min
triangle_aabb_max
```

Use ray-AABB as a cheap rejection step before full triangle intersection.

This should be benchmark-gated. It helps when mesh triangles are spatially
spread out, but adds memory traffic and branch work.

### L5C.2 — GPU BVH / OptiX Evaluation

Defer GPU BVH or OptiX until the device-scene executor and derived primitive
payload are stable. BVH/OptiX should consume the same metadata semantics:

- role mask;
- source-order key;
- numeric instance id;
- frame/env timeline;
- result buffer ownership independent from `GpuPublishedFrame` slots.

## 6. Kernel Design Notes

### Ray-Major Baseline

The initial kernel should stay ray-major:

```text
dim = num_rays
tid = ray index
local best_t/key/id/normal/position
single write per ray
```

Reasons:

- no atomics;
- deterministic tie-break is simple;
- output writes are coalesced by ray index;
- easiest to compare against L5A and CPU executors.

Expected limitation:

```text
O(num_rays * num_primitives)
```

This is acceptable for L5C.1a but not the final performance model.

### Plane Handling

Planes should remain simple:

- plane count is expected to be small;
- plane intersection is cheap;
- no need to force planes into triangle-style derived buffers.

If infinite planes become a performance or semantic issue, add per-plane bounds
or convert bounded render/query surfaces to triangles later.

### Role Filtering

L5C keeps one cached scene for all roles. Kernel visibility should use:

```text
if primitive_role_mask & sensor_role_mask == 0:
    skip
```

Unknown sensor role maps to mask `0`, producing all misses.

### Tie-Break

Keep the L5A packed int64 source-order key:

```text
source_order_key = instance_index * 2**32 + primitive_index_within_instance
```

First compare distance with `_T_EPS`, then source-order key for ties.

### Events And Lifetime

`GpuDeviceSceneOpticalExecutor` should:

- wait on `DeviceOpticalSceneSnapshot.ready_event` before traversal;
- keep snapshot and uploaded ray arrays alive in `OpticalComputeResult.resources`;
- record a result `ready_event`;
- keep result buffer ownership separate from `GpuPublishedFrame` slot lifetime.

## 7. Benchmark Questions

Before making AABB or BVH decisions, add small benchmark cases:

- few rays, few primitives;
- camera-like rays, few primitives;
- camera-like rays, many triangles;
- body-bound moving mesh vs world-static mesh;
- role-filtered scene where most primitives are invisible.

Useful counters:

- update kernel time;
- traversal kernel time;
- total optical execute time;
- host staging time when requested;
- primitive count and ray count.

## 8. Questions For Claude Review

1. Is the staged sequence L5C.1a -> L5C.1b -> L5C.1c conservative enough, or
   should derived buffers be introduced before the first device-scene executor?
2. Should `triangles_world` remain in `DeviceOpticalSceneSnapshot` after
   derived buffers land, or should debug/export reconstruct vertices from
   `v0/e1/e2`?
3. Is AABB early reject worth adding before a real GPU BVH, or should it wait
   for benchmark evidence?
4. Should role masks stay int32/31 roles for L5C.1, or should we move to int64
   before more GPU code depends on the mask type?
5. Are there event/lifetime risks in making `OpticalComputeResult.resources`
   hold the `DeviceOpticalSceneSnapshot` and ray arrays until result staging or
   downstream completion?

## 9. Non-Goals

This plan does not implement:

- GPU BVH;
- OptiX;
- direct lighting or shadows;
- material/PBR shading;
- texture sampling;
- result buffer pooling;
- multi-env batched optical queries;
- per-ray env indexing.

The intent is to stabilize the device-scene executor contract first, then
optimize layout and traversal based on parity tests and benchmark data.
