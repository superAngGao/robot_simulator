Initiative: q54-gpu-optical-executor
Stage: l5c-device-scene-cache-plan
Author: codex
Version: v1
Date: 2026-05-03
Status: review-request
Related Files: optics/device.py, optics/gpu_runtime.py, optics/warp_execution.py, optics/scene.py, physics/gpu_engine.py, physics/publish.py
Owner Summary: Proposed L5C path after L5B.1. The next step should not jump directly to GPU BVH. It should first introduce a device-resident optical scene cache: registry geometry and metadata are uploaded once, a per-frame transform/update kernel realizes world-space primitive buffers from `GpuPublishedFrame.x_world_*`, and the GPU ray executor consumes those device scene buffers. GPU BVH becomes L5C.2 after this cache boundary is stable.

# Q54 L5C Device-Resident Optical Scene Cache Plan

## 1. Current State

L5A:

```text
host OpticalSceneSnapshot
  -> host role-filtered primitive workload
  -> upload every execute call
  -> Warp brute-force first-hit kernel
  -> OpticalComputeResult(location="device")
```

L5B.0:

```text
GpuPublishedFrame borrow/complete lifecycle
  + world-static registry
  -> L5A executor
```

L5B.1:

```text
GpuPublishedFrame borrow/complete lifecycle
  + body-bound registry
  -> host-stage selected env body transforms
  -> host OpticalSceneSnapshot
  -> L5A executor
```

L5B.1 is semantically correct but still performs host staging and host primitive
packing every call.

## 2. Goal

L5C should introduce a device-resident scene/cache layer before GPU BVH:

```text
OpticalWorldRegistry
  -> DeviceOpticalSceneCache.build_from_registry(...)
  -> long-lived local geometry + metadata buffers on device

GpuPublishedFrame N
  -> DeviceOpticalSceneCache.snapshot_from_gpu_frame(frame, env_idx, stream)
  -> transform/update kernel realizes world-space primitive buffers
  -> GpuDeviceSceneOpticalExecutor first-hit query
  -> OpticalComputeResult(location="device")
```

This removes the host round-trip for body transforms and removes per-query
geometry upload for static registry data.

## 3. Non-Goals For L5C.0

L5C.0 should not implement:

- GPU BVH;
- GPU direct-light or shadow rays;
- result buffer pooling;
- multi-env batched optical queries;
- deformable/cloth/fluid producers;
- material shading on device;
- texture/PBR/medium data.

The purpose is to stabilize the device scene boundary.

## 4. Proposed New Types

Add a new module:

```text
optics/device_scene.py
```

Proposed dataclasses:

```python
@dataclass(frozen=True)
class DeviceOpticalRoleTable:
    role_to_mask: dict[str, int]

@dataclass(frozen=True)
class DeviceOpticalScene:
    device: object
    role_table: DeviceOpticalRoleTable
    num_triangles: int
    num_planes: int
    # long-lived device buffers
    triangle_vertices_local: object
    triangle_instance_index: object
    triangle_source_order_key: object
    triangle_role_mask: object
    triangle_numeric_instance_id: object
    plane_normal_local: object
    plane_point_local: object
    plane_instance_index: object
    plane_source_order_key: object
    plane_role_mask: object
    plane_numeric_instance_id: object
    instance_body_index: object
    instance_X_body_geometry_R: object
    instance_X_body_geometry_r: object

@dataclass(frozen=True)
class DeviceOpticalSceneSnapshot:
    frame_id: int
    sim_time: float
    env_idx: int
    scene: DeviceOpticalScene
    # per-frame device buffers produced by update kernel
    triangles_world: object
    plane_normal_world: object
    plane_point_world: object
    ready_event: object | None = None
    resources: tuple[object, ...] = ()
```

Rationale: avoid overloading host `OpticalSceneSnapshot` with many optional
device-only fields. The executor can accept a device snapshot explicitly.

## 5. Device Data Layout

Flatten registry geometry once, in registry instance order.

Triangles:

```text
triangle_vertices_local:            float32[num_triangles, 3, 3]
triangle_instance_index:            int32[num_triangles]
triangle_source_order_key:          int64[num_triangles]
triangle_role_mask:                 int32[num_triangles]
triangle_numeric_instance_id:       int32[num_triangles]
```

Planes:

```text
plane_normal_local:                 float32[num_planes, 3]
plane_point_local:                  float32[num_planes, 3]
plane_instance_index:               int32[num_planes]
plane_source_order_key:             int64[num_planes]
plane_role_mask:                    int32[num_planes]
plane_numeric_instance_id:          int32[num_planes]
```

Instances:

```text
instance_body_index:                int32[num_instances]   # -1 for world-static
instance_X_body_geometry_R:         float32[num_instances, 3, 3]
instance_X_body_geometry_r:         float32[num_instances, 3]
```

World buffers produced per frame:

```text
triangles_world:                    float32[num_triangles, 3, 3]
plane_normal_world:                 float32[num_planes, 3]
plane_point_world:                  float32[num_planes, 3]
```

## 6. Role Encoding

L5A/L5B filtered roles on host. L5C should keep one cached scene for all roles.

Build a deterministic role table from registry instances:

```text
sorted(all_roles) -> bit positions
```

Each primitive stores a role bitmask. At execute time:

```text
sensor_role_mask = role_table.mask_for(spec.sensor_role)
```

Kernel visibility check:

```text
if primitive_role_mask & sensor_role_mask == 0:
    skip primitive
```

If `spec.sensor_role` is unknown, the mask is zero and the query returns all
misses. This matches host executor semantics: no instance contains that role.

Open question: should we cap at 31 roles using `int32`, or use `int64` masks
from the beginning?

## 7. Source Order

Keep the L5A packed key:

```text
key = instance_index * 2**32 + primitive_index_within_instance
```

Planes use primitive index `0`. Triangles use their triangle index within the
instance mesh. Geometry kind does not affect source order.

This preserves the CPU lexicographic tie-break:

```text
(instance_index, primitive_index_within_instance)
```

## 8. Per-Frame Transform Update Kernel

L5C.0 should introduce a sensor-independent update kernel:

```text
_update_device_optical_scene_kernel(...)
```

One thread per primitive family:

- triangle thread transforms three local vertices into `triangles_world`;
- plane thread transforms local normal/point into `plane_*_world`.

For each primitive:

```text
instance_index = primitive_instance_index[p]
body_index = instance_body_index[instance_index]

if body_index < 0:
    R_world_body = I
    r_world_body = 0
else:
    R_world_body = frame.x_world_R[env_idx, body_index]
    r_world_body = frame.x_world_r[env_idx, body_index]

R_world_geometry = R_world_body @ X_body_geometry_R[instance_index]
r_world_geometry = r_world_body + R_world_body @ X_body_geometry_r[instance_index]

point_world = R_world_geometry @ point_local + r_world_geometry
normal_world = normalize(R_world_geometry @ normal_local)
```

This belongs to scene/cache, not executor, because it is sensor-independent
geometry preparation. It does not traverse rays or compute optical results.

Transform convention clarification after review:

`OpticalInstanceSpec.X_body_geometry` is the pose of the geometry frame
expressed in the body frame. It maps geometry-local points into the body frame.
The existing host scene path composes:

```text
X_world_geometry = X_world_body @ X_body_geometry
```

and host point transforms use:

```text
p_world = R_world_geometry @ p_geometry + r_world_geometry
```

Therefore the device update kernel must use:

```text
R_world_geometry = R_world_body @ R_body_geometry
r_world_geometry = r_world_body + R_world_body @ r_body_geometry
```

where `R_body_geometry` and `r_body_geometry` come from
`instance_X_body_geometry_*`.

## 9. Ray Query Kernel

After L5C.0, the ray query kernel should consume device scene snapshot buffers:

```text
origins_world
directions_world
sensor_role_mask
triangles_world
triangle_role_mask
triangle_source_order_key
triangle_numeric_instance_id
plane_normal_world
plane_point_world
plane_role_mask
plane_source_order_key
plane_numeric_instance_id
```

The intersection math can be the current L5A brute-force kernel with two
changes:

1. no per-query primitive upload;
2. role filtering inside the kernel via bitmask.

This can be implemented either by:

- refactoring `GpuBruteForceOpticalExecutor` to accept both host-packed and
  device-scene inputs; or
- adding `GpuDeviceSceneOpticalExecutor` and keeping L5A untouched.

Recommendation: add `GpuDeviceSceneOpticalExecutor`. It keeps L5A/L5B
correctness tests stable and makes the new boundary explicit.

## 10. Q52 Lifecycle

For a Q52 GPU frame:

```text
borrow_device_frame(...)
  -> enqueue scene update kernel on optical stream
  -> enqueue ray query kernel on same stream
  -> complete_device_consumer(...)
```

The done event returned by `complete_device_consumer(...)` is attached to
`OpticalComputeResult.ready_event`.

The result should retain:

```text
resources=(device_scene_snapshot, ray_input_arrays, ...)
```

so that update buffers and ray inputs cannot be reclaimed before the optical
stream has finished.

## 11. Buffer Ownership

Long-lived:

- local geometry buffers;
- instance metadata buffers;
- role/source-order/id buffers.

Per-frame fresh in L5C.0:

- `triangles_world`;
- `plane_normal_world`;
- `plane_point_world`.

Reason: fresh per-frame buffers avoid overwrite hazards while the first device
cache is being validated. Pooling can come later when the event dependency
model is proven.

Open question: should the per-frame world buffers live on the snapshot object,
or should the cache own a small ring of world-buffer slots keyed by event?

## 12. Multi-Env Semantics

L5C.0 should remain selected-env only:

```text
snapshot_from_gpu_frame(frame, env_idx=i)
```

This matches current host `OpticalSceneCache` semantics. Multi-env batching
should be designed separately before Phase C:

- one snapshot per env;
- all-env device scene update;
- batched ray specs with env index per ray.

Do not mix that with the first device scene cache.

## 13. Tests

CPU-only:

- role table deterministic assignment;
- unknown role maps to zero mask;
- source-order packing preserved;
- device scene host packer shapes/dtypes without importing Warp when possible.

GPU:

- device scene update for world-static plane matches host snapshot;
- device scene update for body-bound plane matches `GpuPublishedFrame.x_world_*`;
- device scene first-hit parity with L5B.1 for world-static + body-bound scene;
- role filtering happens on device;
- source-order tie-break matches CPU/reference;
- Q52 helper with device scene cache completes consumer and survives slot reuse;
- unknown role returns misses;
- env_idx out of range and body_index out of range raise clear errors.

## 14. Complexity

Build cache:

```text
O(num_primitives + num_instances)
```

Per frame scene update:

```text
O(num_primitives)
```

Per ray query remains brute-force in L5C.0:

```text
O(num_rays * num_primitives)
```

GPU BVH should be introduced only after this boundary is stable.

## 15. Proposed Phasing

L5C.0:

- add `optics/device_scene.py`;
- pack registry into device local geometry + metadata buffers;
- update world primitive buffers from `GpuPublishedFrame.x_world_*`;
- add tests for update buffers.

L5C.1:

- add `GpuDeviceSceneOpticalExecutor`;
- reuse brute-force intersection over device scene buffers;
- add parity tests with L5B.1.

L5C.2:

- add optional GPU acceleration/BVH over world buffers, or choose OptiX/Embree
  adapter direction for GPU acceleration.

## 16. Review Questions

1. Is splitting L5C into device scene cache before GPU BVH the right sequence?
2. Should `DeviceOpticalSceneSnapshot` be a new type, or should we extend
   `OpticalSceneSnapshot(location="device")`?
3. Should role masks be `int32` or `int64`?
4. Should per-frame world buffers be fresh per snapshot in L5C.0, or should the
   cache own a ring from the beginning?
5. Is `GpuDeviceSceneOpticalExecutor` preferable to widening
   `GpuBruteForceOpticalExecutor`?
6. Should unknown `sensor_role` return all misses, or raise?
7. Is selected-env-only acceptable for L5C.0?
8. Should L5C.2 be in-repo GPU BVH, or should this be the point where OptiX is
   evaluated?
