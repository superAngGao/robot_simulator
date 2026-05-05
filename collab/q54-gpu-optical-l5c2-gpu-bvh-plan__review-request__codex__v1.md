Initiative: q54-gpu-optical-l5c2-gpu-bvh
Stage: review-request
Author: codex
Version: v1
Date: 2026-05-05
Status: ready-for-claude-review
Related Files: optics/device_scene.py, optics/warp_execution.py, benchmarks/bench_optical_device_scene.py, benchmarks/robot_optical_scene.py, examples/mujoco_menagerie_robot_preview.py
Owner Summary: Proposed L5C.2 GPU BVH path after L5C.1 derived triangle/AABB work. The recommended path is staged: first land a CPU-built, GPU-traversed BVH over existing device-scene payloads for correctness and benchmark parity; then add GPU refit/build design using Morton/LBVH-compatible buffers; finally evaluate OptiX as the high-performance backend once the payload and lifecycle contracts are stable.

# Q54 L5C.2 GPU BVH Plan

## 1. Current State

L5C.1 has stabilized the GPU optical device scene path:

```text
OpticalWorldRegistry
  -> DeviceOpticalSceneCache
  -> DeviceOpticalScene(long-lived local geometry + metadata buffers)

GpuPublishedFrame
  -> DeviceOpticalSceneSnapshot(per-frame derived world triangle buffers)

DeviceOpticalSceneSnapshot + OpticalRaySensorSpec
  -> GpuDeviceSceneOpticalExecutor
  -> OpticalComputeResult(location="device")
```

The current traversal is still brute-force:

```text
O(num_rays * num_triangles)
```

L5C.1c per-triangle AABB is useful as a measured constant-factor variant, but
it does not change complexity. On H200:

```text
xlarge_mesh_1m_tris:
  no-AABB execute ~1195 ms
  AABB execute    ~ 919 ms

robot_dense_single:
  no-AABB execute ~331 ms
  AABB execute    ~267 ms
```

The MuJoCo Menagerie Unitree Go2 visual mesh imports as:

```text
33 visual geoms
398,432 triangles
```

CPU preview images are good enough for README as a temporary checkpoint, but
CPU direct-light/BVH is too slow for repeated high-resolution rendering. The
next step should be GPU hierarchy traversal, not more brute-force tuning.

## 2. Goal

Introduce GPU triangle acceleration while preserving the L5C.1 semantic
contract:

```text
hit_mask
range_m
position_world
normal_world
numeric_instance_id
role mask filtering
source-order tie-break
plane analytical side pass
frame/env timeline validation
OpticalComputeResult resource lifetime
Q52 GpuPublishedFrame slot lifetime independence
```

The plan must leave room for three increasingly capable backends:

```text
L5C.2a: CPU build + GPU traversal BVH
L5C.2b: GPU refit/build over compatible node/primitive buffers
L5C.2c: OptiX backend evaluation
```

## 3. Why Not Jump Directly To OptiX

OptiX is the likely long-term high-performance NVIDIA ray tracing backend, but
using it as the first BVH step would combine too many unknowns:

- GAS/TLAS build lifecycle;
- shader binding table and payload packing;
- custom metadata lookup for role masks and instance ids;
- source-order tie-break behavior;
- Python/CUDA extension integration;
- Q52 event/resource lifetime coupling;
- direct-light/shadow extension path.

The first GPU BVH should be an in-repo, reviewable bridge that validates
payloads, parity, and benchmark targets. OptiX can then be evaluated against a
known contract rather than defining the contract implicitly.

## 4. L5C.2a — CPU-Built, GPU-Traversed BVH

### 4.1 Build Policy

Build the BVH on CPU from the current per-frame device-scene snapshot data
model, then upload compact node arrays to device.

This is not the final performance model. It is the fastest way to validate:

- node layout;
- GPU stack traversal;
- role/source-order metadata;
- parity against brute-force L5C.1;
- benchmark coverage on large grid, robot-like, and Menagerie Go2 scenes.

For the first implementation, CPU build can use the same deterministic
longest-axis median split as the existing CPU BVH.

L5C.2a benchmarks must always report CPU build time and GPU traversal time as
separate metrics. Do not collapse them into a single `execute_ms` number when
making acceleration decisions; CPU build is a bridge cost, while GPU traversal
is the kernel path being validated.

### 4.2 Node Layout

Do not upload Python node objects. Use device-friendly SoA arrays:

```text
bvh_bounds_min: float32[num_nodes, 3]
bvh_bounds_max: float32[num_nodes, 3]
bvh_left:       int32[num_nodes]
bvh_right:      int32[num_nodes]
bvh_start:      int32[num_nodes]
bvh_count:      int32[num_nodes]
bvh_prim_ids:   int32[num_triangles]
```

Interpretation:

```text
bvh_count[node] > 0:
  leaf over bvh_prim_ids[start:start+count]

bvh_count[node] == 0:
  internal node with left/right children
```

Primitive ids index the existing snapshot arrays:

```text
triangle_v0_world
triangle_e1_world
triangle_e2_world
triangle_normal_world
triangle_aabb_min
triangle_aabb_max
triangle_role_mask
triangle_source_order_key
triangle_numeric_instance_id
```

This keeps BVH as an acceleration layer over existing payloads rather than
creating a second triangle representation.

L5C.2a.0 must also materialize primitive metadata that makes a future
BLAS/TLAS split mechanical:

```text
primitive_global_id:                int32[num_triangles]
primitive_instance_index:           int32[num_triangles]
primitive_index_within_instance:    int32[num_triangles]
primitive_geometry_index_or_id:     int32[num_triangles]
primitive_geometry_primitive_index: int32[num_triangles]
primitive_source_order_key:         int64[num_triangles]
```

The first global BVH traversal may only need `primitive_global_id`, role mask,
numeric id, and source key. The other metadata is still a hard L5C.2a.0
requirement because adding it later would make the BLAS/TLAS migration a
payload-breaking change.

### 4.3 Executor

Add:

```text
GpuDeviceBvhOpticalExecutor
```

Kernel shape:

```text
one thread = one ray
local fixed-size traversal stack
traverse BVH triangle hierarchy
test role mask at leaf primitive
apply _is_better_hit tie-break
run analytical plane side pass
write result channels
```

First stack policy:

```text
MAX_BVH_STACK = 32
```

If a traversal overflows the stack, set a device-side overflow flag and either:

1. fall back to brute-force for that ray inside the same kernel, or
2. mark an executor error after synchronization.

Recommendation for L5C.2a: use an overflow flag and raise after the kernel.
That keeps correctness failures loud while we tune tree depth/leaf size.

Register pressure is the main risk in this kernel. A per-thread fixed stack can
reduce occupancy before traversal math becomes the bottleneck. L5C.2a should:

- start with a node-id-only stack of 32 entries;
- record CPU build max depth for every BVH;
- add traversal instrumentation for observed max stack depth and overflow
  count;
- keep stack overflow rare before increasing stack size;
- consider shared-memory/per-warp stacks, stackless traversal, or wide BVH only
  after measurements show the 32-entry stack is insufficient.

Do not store both node id and `t_near` in the stack in the first version unless
benchmark data proves that it pays for itself. Compute child hit distances
locally when ordering child traversal.

### 4.4 Tie-Break Semantics

BVH traversal order is not source order. The executor must preserve L5A/L5C.1
semantics:

```text
better hit if:
  smaller t by epsilon
  or equal t and smaller source_order_key
```

Use the existing packed int64 source-order key used by device scene buffers.

Planes remain outside the BVH but participate in the same best-hit merge. A
plane instance before a mesh instance must still win an equal-distance tie.

Document the first kernel order explicitly:

```text
for each ray:
  best = analytical plane pass
  mesh_best = BVH triangle traversal
  best = merge(best, mesh_best, _is_better_hit)
  write result
```

This keeps plane/triangle parity simple and makes equal-distance source-order
tests easier to reason about. Running planes first is not a semantic
requirement as long as `_is_better_hit` is used for every candidate, but it is
the recommended L5C.2a implementation.

### 4.5 First Tests

Correctness:

- tiny triangle mesh parity vs L5C.1 brute-force;
- plane + triangle equal-distance source-order tie;
- unknown role all miss;
- body-bound triangle mesh parity after transform;
- AABB/BVH node boundary ray cases;
- stack overflow guard with intentionally pathological tree if feasible.
- CPU build reports max tree depth;
- traversal instrumentation reports observed max stack depth.

Benchmarks:

- `xlarge_mesh_1m_tris`;
- `robot_dense_single`;
- `robot_dense_pack`;
- MuJoCo Menagerie Go2 visual mesh, 398k triangles.

## 5. L5C.2b — GPU Refit/Build Design

L5C.2a must not choose layouts that block GPU build/refit. The target design is
compatible with both refit and LBVH.

### 5.1 Two Acceleration Modes

Use explicit build modes:

```text
mode="cpu_build_gpu_traverse"
mode="gpu_refit"
mode="gpu_lbvh_build"
mode="optix"
```

The executor should not infer the mode from missing buffers.

### 5.2 Refit Path For Fixed Topology

For rigid body-bound meshes whose topology and BVH primitive assignment are
unchanged, the cheapest GPU path is refit:

```text
update triangle_v0/e1/e2/normal/aabb
for leaves:
  leaf bounds = union primitive AABBs
for internal nodes, bottom-up:
  node bounds = union child bounds
```

Required extra build-time data:

```text
node_parent:       int32[num_nodes]
node_level/order:  int32[num_nodes] or level ranges
level_ranges:      int32[num_levels, 2]
leaf_node_ids:     int32[num_leaves]
```

Implementation options:

1. level-by-level kernels from deepest level to root;
2. atomic counters per internal node to trigger parent update when both
   children are complete.

Recommendation: first design for level-by-level refit because it is easier to
review and deterministic. Atomic bottom-up refit can come later. The
`level_ranges` array is required for this path so each refit kernel launch can
process one deterministic node range from deepest level to root.

### 5.3 GPU LBVH Build Path

When topology changes or a world-space rebuild is needed, use a GPU-friendly
LBVH path rather than attempting SAH first.

Inputs:

```text
triangle_aabb_min/max
scene_bounds
primitive centroid
primitive id
```

Steps:

```text
1. compute centroid and Morton code per primitive
2. sort by Morton code, tie-break by source-order key
3. build leaves in sorted order
4. build internal hierarchy from Morton code ranges
5. bottom-up compute node bounds
```

Sorting options:

- first version may use a host/Warp-compatible sort if available;
- after L5C.2a.3, explicitly evaluate Warp utilities and Torch/CUDA sort as a
  temporary sort backend, even if the latter pays D2H/H2D or framework interop
  cost;
- otherwise keep GPU LBVH as design-only until a reliable radix sort utility
  exists in the repo, but record that decision as a L5C.2b blocker;
- do not hand-roll a large radix sort inside L5C.2a.

### 5.4 SAH Policy

Do not implement GPU SAH first.

Policy:

```text
median split CPU build  -> L5C.2a correctness bridge
GPU refit               -> L5C.2b fixed topology performance
GPU LBVH                -> L5C.2b rebuild path
SAH / spatial splits    -> later, only after benchmarks show LBVH/refit is insufficient
```

SAH is valuable for quality, but it increases build complexity and makes GPU
build much harder. For robot meshes with coherent topology and repeated frames,
refit/TLAS-style structure should matter more than perfect per-frame SAH.

## 6. BLAS/TLAS Direction

The current L5C device scene flattens all triangles into one global primitive
array. That is acceptable for L5C.2a, but future GPU build should move toward:

```text
BLAS per geometry mesh:
  local-space topology/primitive BVH or OptiX GAS

TLAS per instance:
  instance transform
  instance role mask / numeric id / source order base
```

Why:

- multiple robot instances can share geometry;
- rigid body-bound meshes can update transforms without rebuilding local
  geometry BVHs;
- OptiX maps naturally to GAS/TLAS;
- collision and optical can eventually share static mesh acceleration assets
  at the geometry level while keeping query-specific payloads separate.

Do not force BLAS/TLAS into L5C.2a. But design node/primitive metadata so that
the transition is mechanical. The following fields are L5C.2a.0 requirements,
not optional future notes:

```text
global_primitive_id
instance_index
primitive_index_within_instance
geometry_id / geometry_primitive_id
source_order_key
```

## 7. Collision Sharing Policy

Do not share the same live BVH object between collision and optical yet.

Share:

- mesh asset loading;
- local triangle/vertex buffers;
- optional build utilities;
- future geometry-level BLAS metadata when semantics align.

Do not share:

- role masks;
- optical material/light payload;
- source-order tie-break metadata;
- collision margins/contact-specific AABB inflation;
- contact manifold / closest-feature data.

Collision BVH and optical BVH have overlapping geometry needs but different
query semantics. Prematurely sharing the executable BVH would couple unrelated
lifecycles.

## 8. OptiX Evaluation Criteria

Evaluate OptiX after L5C.2a has parity and benchmarks.

OptiX backend must answer:

- Can custom per-primitive payload recover `numeric_instance_id` and
  source-order key cheaply?
- Is role filtering done by SBT records, instance visibility masks, or
  closest-hit/any-hit logic?
- How are infinite planes handled? Recommendation: keep analytical side pass
  outside OptiX at first.
- What is the GAS/TLAS rebuild/refit cost for body-bound robot meshes?
- How does result buffer ownership integrate with `OpticalComputeResult` and
  Q52 frame slot completion?
- Can the same backend support future shadow rays/direct-light without
  rewriting the scene contract?

OptiX should be an executor/backend adapter, not a replacement for the optical
scene contract.

## 9. High-Resolution README Rendering

Current README images:

```text
docs/assets/optical/menagerie_go2_front/panel.png
docs/assets/optical/menagerie_go2_side/panel.png
```

These were generated with CPU direct-light/BVH and are intentionally temporary.

Do not spend more time raising CPU preview resolution. Once GPU BVH/OptiX can
render the Menagerie Go2 scene at high resolution, regenerate README images at
1080p/2K using the same importer.

## 10. Performance Alignment With High-Performance Renderers

After L5C.2a correctness lands, performance work must be framed as an
apples-to-apples benchmark program, not a vague "make it fast" effort.

### 10.1 Comparison Targets

Use a ladder of increasingly strong baselines:

```text
internal:
  L5C.1 brute-force derived traversal
  L5C.1c per-triangle AABB traversal
  L5C.2a CPU-built/GPU-traversed BVH
  L5C.2b GPU refit / LBVH when available
  L5C.2c OptiX adapter when available

external:
  TinyBVH CPU/GPU-friendly layouts for first external BVH reference
  Embree CPU BVH for CPU-side build-quality sanity when practical
  pbrt-v4 GPU path / OptiX as aspirational reference where scene conversion is practical
```

Do not compare our first-hit depth/segmentation kernel directly against a full
path tracer's final image time unless the shading work is made comparable. Use
first-hit / visibility / shadow-ray style microbenchmarks for alignment.

### 10.2 Shared Benchmark Scenes

Keep a stable scene set that can be exported or reproduced across backends:

```text
grid_regular:
  deterministic grid triangles, stress scaling and degenerate distribution assumptions

robot_dense:
  in-repo robot-like clustered body-bound meshes

menagerie_go2:
  MuJoCo Menagerie Unitree Go2 visual mesh, 398k triangles

menagerie_go2_pack:
  multiple Go2 instances, shared geometry stress for future BLAS/TLAS

large_static_mesh:
  one public static mesh scene, e.g. Sponza/Bistro-style if licensing is clean
```

Each benchmark scene should define:

```text
triangle count
instance count
material/role count
camera matrices
resolution / ray count
ray role
max distance
whether shadow/occlusion rays are included
```

### 10.3 Metrics

Report separate timings:

```text
scene import / conversion ms
device upload ms
triangle world update ms
BVH build ms
BVH refit ms
ray traversal ms
result staging ms
end-to-end frame ms
```

Report throughput:

```text
primary MRays/s
shadow/visibility MRays/s
effective triangle tests/s when available
node visits/ray
primitive tests/ray
hit rate
miss rate
stack overflow count
GPU memory footprint
BVH SAH quality cost
```

Use CUDA/Warp events for GPU timing where possible. Use host timers only for
host-side phases such as import, CPU build, and staging.

For BVH build quality, report a backend-independent SAH-style diagnostic:

```text
sum over leaves:
  (leaf_surface_area / root_surface_area) * leaf_primitive_count
```

This is not a replacement for traversal benchmarks, but it helps compare
median split, LBVH, SAH, and external BVH quality before ray distribution
effects dominate the discussion.

### 10.4 Correctness Before Speed

For every performance case, keep a correctness companion:

```text
L5C.2 output vs L5C.1 brute force on small/medium scenes
CPU BVH/reference parity on Menagerie Go2 at reduced resolution
source-order tie-break parity
role-mask parity
normal orientation parity
```

Large benchmark cases can validate aggregate sanity rather than exact full
image parity every run, but at least one reduced-resolution version must remain
bit/epsilon comparable.

### 10.5 Performance Targets

Use targets as diagnostics, not hard release gates:

```text
L5C.2a:
  should beat L5C.1c by a large factor on 100k+ triangle camera workloads;
  if not, inspect BVH quality, stack pressure, memory layout, and ray ordering.

L5C.2b:
  refit cost should be much smaller than full rebuild for fixed topology robot scenes;
  traversal should improve or at least not regress from L5C.2a.

L5C.2c:
  OptiX should define the high-performance ceiling on NVIDIA hardware;
  in-repo BVH should be measured as a percentage of OptiX throughput, not assumed competitive.
```

Initial aspirational alignment:

```text
software GPU BVH:
  useful if within a small multiple of OptiX for first-hit primary rays on simple scenes

OptiX adapter:
  target backend for high-resolution README images and real camera workloads
```

Do not tune for one scene only. Robot-like clustered meshes and large static
mesh scenes should both be in the reporting table.

Trigger OptiX work from a concrete comparison when possible: if
`robot_dense_pack` or Menagerie multi-Go2 traversal remains more than roughly
3x slower than an OptiX GAS/TLAS spike on the same hardware and first-hit query
shape, move OptiX from aspirational reference to active backend work.

### 10.6 External Project Alignment Notes

The comparison should respect architecture differences:

- TinyBVH is the first external reference to wire because it is small,
  build/layout-focused, and practical to compare against L5C.2a.
- Embree is CPU-oriented; use it to understand CPU BVH scale, not as a GPU
  throughput target. It is most useful for BVH quality/build sanity, not for
  GPU MRays/s.
- pbrt-v4 GPU is a wavefront path tracer and uses OptiX; keep it as an
  aspirational reference until scene conversion and phase-matched ray-query
  benchmarks are practical.
- OptiX/DXR-style systems manage traversal stack and hardware triangle tests;
  our fixed-stack software BVH is an intermediate design, not the expected
  final ceiling.

### 10.7 Reporting Format

Add a dedicated benchmark report artifact after implementation:

```text
collab/q54-gpu-optical-l5c2-performance-alignment__benchmark-report__codex__v1.md
```

Minimum table:

```text
backend, scene, rays, triangles, build_ms, refit_ms, traverse_ms,
MRays/s, memory_mb, notes
```

Include the exact GPU, driver/CUDA/Warp versions, command lines, warmup/repeat,
and whether timings include shadows or only primary first-hit rays.

L5C.2a reports primary first-hit only. Shadow/visibility rays have any-hit
semantics and early termination behavior, so they should be added after the
first-hit correctness bridge is stable.

## 11. Proposed Implementation Sequence

```text
L5C.2-plan:
  write/review this plan

L5C.2a.0:
  add GPU BVH dataclasses / packed node arrays
  CPU-build BVH from device-scene-compatible primitive metadata
  include instance/primitive-within-instance metadata
  record CPU build max depth and SAH quality cost
  upload node arrays to device

L5C.2a.1:
  implement GpuDeviceBvhOpticalExecutor traversal kernel
  analytical plane side pass before mesh traversal, then merge by _is_better_hit
  overflow flag/error handling
  traversal max-stack-depth instrumentation

L5C.2a.2:
  parity tests vs L5C.1 brute force
  source-order/role/body-bound coverage

L5C.2a.3:
  benchmarks on grid xlarge, robot_dense, Menagerie Go2
  force split timings: cpu_build_ms, gpu_traverse_ms, staging_ms
  first external reference target: TinyBVH

L5C.2b.0:
  add refit-compatible metadata: parent/levels/leaf ranges
  prototype GPU refit kernels

L5C.2b.1:
  evaluate GPU LBVH build requirements and sorting utility
  decide Warp sort vs Torch/CUDA sort vs design-only blocker

L5C.2c:
  OptiX adapter spike

L5C.2-perf:
  run performance alignment suite
  compare against internal baselines and available external backends
  regenerate high-resolution README images from GPU backend
```

## 12. Open Questions For Review

1. Is CPU-build/GPU-traverse the right L5C.2a bridge, or should GPU refit be
   implemented before any BVH traversal lands?
2. Should L5C.2a BVH be built over world-space derived triangles, or should it
   immediately introduce a local-BLAS/global-TLAS split?
3. Is stack overflow best handled by raising after kernel completion, or by
   per-ray brute-force fallback?
4. Should role filtering remain per-primitive in the first BVH, or should we
   build role-specific BVHs sooner?
5. Is level-by-level GPU refit the right first refit design, or should the
   atomic bottom-up refit be preferred despite higher complexity?
6. What exact benchmark threshold should trigger OptiX work over continuing
   in-repo BVH?
7. Which external benchmark backend should be wired first for alignment:
   OptiX, pbrt-v4 GPU, Embree CPU, or TinyBVH? Current recommendation:
   TinyBVH first.
8. Should performance reports use primary first-hit only at first, or include
   one shadow/visibility ray mode immediately? Current recommendation:
   primary first-hit only for L5C.2a.
