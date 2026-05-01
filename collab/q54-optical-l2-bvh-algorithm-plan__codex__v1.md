Initiative: q54-optical-l2-bvh-algorithm
Stage: algorithm-plan
Author: codex
Version: v1
Date: 2026-05-01
Status: draft-for-review
Related Files: optics/scene.py, optics/execution.py, optics/registry.py, sensing/optical.py, collab/q54-optical-l2-acceleration__review-request__codex__v1.md
Owner Summary: This document fixes the proposed L2 BVH path before implementation. The recommended path is an in-repo CPU BVH over world-space packed triangles, built by scene/cache, traversed by a dedicated executor, with infinite planes handled by the existing analytical path and explicit source-order tie-breaks for reference parity.

# Q54 L2 BVH Algorithm Plan

## 1. Decision

Use this path for the first L2 optical acceleration implementation:

```text
in-repo CPU BVH
world-space packed triangles
longest-axis median split
leaf_size = 4
iterative stack traversal
analytical infinite-plane side pass
explicit source-order tie-break
scene/cache builds acceleration
executor consumes acceleration only
MissingAccelerationError for missing acceleration
```

Do not use Open3D as an implementation backend. Do not make Embree a required
dependency yet. Embree remains a future optional adapter behind the same
scene/executor contract.

## 2. Why This Path

The goal of L2 is not maximum rendering performance. The goal is to introduce
ray acceleration without changing Q54 semantics:

```text
OpticalSceneSnapshot + OpticalRaySensorSpec -> OpticalComputeResult
```

The current reference executor is deliberately simple but scales as:

```text
O(num_rays * num_triangles)
```

L2 should reduce triangle traversal cost while preserving:

```text
range_m
hit_mask
position_world
normal_world
material_id
instance_id
numeric_instance_id
miss semantics
role filtering
camera postprocessing compatibility
```

An in-repo BVH is the most reviewable first implementation because no current
project dependency provides Embree/Open3D, and adding a heavy backend before
the data contract is stable would make failures harder to understand.

The main tradeoff is that world-space packing transforms all mesh vertices for
each accelerated snapshot. This is acceptable for L2 correctness-first work, but
it is not the final performance model. Future triggers to move toward
local-space triangles plus per-instance transforms, refit, or backend-specific
BLAS/TLAS structures include:

```text
multi-env batching
many rigid instances sharing mesh topology
deformable mesh producers
BVH refit support
Embree / OptiX adapter requirements
```

## 3. Inputs

BVH construction consumes one frame-aligned `OpticalSceneSnapshot`.

For each `OpticalInstanceSnapshot`:

```text
instance_id
numeric_instance_id
geometry_id
material
geometry
X_world_geometry
roles
source_key
body_index
```

Only `OpticalTriangleMeshGeometry` contributes primitives to the BVH.
`OpticalPlaneGeometry` is excluded and handled analytically by the executor.

## 4. Outputs

The scene/cache should attach an acceleration payload to the snapshot:

```text
OpticalSceneSnapshot.acceleration: OpticalSceneAcceleration | None
```

First CPU payload shape:

```text
OpticalSceneAcceleration
  kind: "cpu_bvh"
  triangles_world: float64[num_primitives, 3, 3]
  triangle_normals_world: float64[num_primitives, 3]
  primitive_instance_indices: int64[num_primitives]
  primitive_source_order_keys: int64[num_primitives, 2]
  primitive_aabb_min: float64[num_primitives, 3]
  primitive_aabb_max: float64[num_primitives, 3]
  primitive_indices: int64[num_primitives]
  nodes: list[CpuBvhNode]
```

`primitive_instance_indices[i]` points back into `snapshot.instances`.

`primitive_source_order_keys[i]` records reference-order identity:

```text
(instance_index, primitive_index_within_instance)
```

This is required because BVH traversal order is not the same as reference
executor scan order.

Planes are not stored in triangle primitive arrays, but they must use the same
global source-order key when the executor merges analytical plane hits:

```text
plane key:    (instance_index, 0)
triangle key: (instance_index, triangle_index_within_instance)
```

Geometry kind must not reorder tie-breaks. A plane instance before a mesh
instance in `snapshot.instances` must win an equal-distance tie against that
mesh, matching the reference executor.

## 5. Node Layout

Use a simple dataclass list first:

```text
CpuBvhNode
  bounds_min: float64[3]
  bounds_max: float64[3]
  left: int
  right: int
  start: int
  count: int
```

Interpretation:

```text
count > 0  -> leaf node; test primitive_indices[start:start+count]
count == 0 -> internal node; traverse left and right
```

This is easier to audit than a compact SoA representation, and a list is the
natural build-time container while still providing O(1) indexed access during
traversal. If later profiling shows Python object overhead matters, the same
schema can be converted to arrays without changing executor semantics.

## 6. Primitive Packing

For each triangle-mesh instance:

```text
vertices_world = transform_points(instance.X_world_geometry, mesh.vertices_local)
for triangle in mesh.triangles:
    tri_world = vertices_world[triangle]
    normal = normalize(cross(v1 - v0, v2 - v0))
    aabb_min = min(tri_world, axis=0)
    aabb_max = max(tri_world, axis=0)
```

Degenerate triangles:

```text
if normal_norm <= build_eps:
    skip the primitive
```

Skipping degenerate triangles matches the current reference executor behavior,
which returns no hit for zero-area triangles.

## 7. Build Algorithm

Use longest-axis median split.

Inputs:

```text
primitive ids for this node
primitive AABBs
primitive centroids
```

Algorithm:

```text
build_node(ids):
    node_bounds = union primitive_aabb[ids]

    if len(ids) <= leaf_size:
        emit leaf(ids)
        return node_index

    centroid_bounds = bounds of primitive centroids[ids]
    axis = argmax(centroid_bounds.max - centroid_bounds.min)

    if centroid extent on axis <= build_eps:
        emit leaf(ids)
        return node_index

    sort ids by centroid[axis], then primitive_source_order_keys
    split = len(ids) // 2
    left = build_node(ids[:split])
    right = build_node(ids[split:])
    emit internal(left, right, node_bounds)
```

Tie-breaking the sort by `primitive_source_order_keys` makes the tree
deterministic.

Recommended first constants:

```text
leaf_size = 4
build_eps = 1e-12
dir_eps = 1e-12
t_eps = 1e-9
```

Rationale:

- median split is simple and deterministic;
- no SAH-specific edge cases in the first implementation;
- `leaf_size=4` keeps leaf brute-force work small and makes parity debugging
  easier.

## 8. Ray-AABB Test

Use slab intersection.

For each axis:

```text
if abs(direction[axis]) <= dir_eps:
    ray is parallel to this slab
    if origin[axis] outside [min, max]:
        miss
else:
    t0 = (min - origin) / direction
    t1 = (max - origin) / direction
    update t_enter / t_exit
```

Node hit condition:

```text
t_exit >= max(t_enter, 0.0)
t_enter <= current_best_t
```

The explicit parallel handling is preferred for the first version because it is
easier to reason about than relying on infinities from reciprocal directions.

Slab boundaries are inclusive. If `origin[axis]` is exactly on `min` or `max`,
the ray is considered inside that slab. This matches the inclusive triangle
edge semantics used by Moller-Trumbore:

```text
u >= 0
v >= 0
u + v <= 1
```

## 9. Traversal Algorithm

Use an iterative stack per ray:

```text
best_t = spec.max_distance
best_primitive = none
best_source_order_key = (+inf, +inf)
stack = [root]

while stack:
    node = stack.pop()
    if ray misses node.bounds before best_t:
        continue

    if node is leaf:
        for primitive in node primitives:
            instance = snapshot.instances[primitive_instance_indices[primitive]]
            if spec.sensor_role not in instance.roles:
                continue
            hit = intersect triangle
            update best hit using source-order-key tie-break
    else:
        compute child AABB hit distances
        push farther child first, nearer child second
```

Near/far ordering is an optimization only. Correctness comes from always
testing hits against the current `best_t`.

## 10. Triangle Hit

Use the existing Moller-Trumbore semantics from the reference executor:

```text
t >= 0
t <= current_best_t
u >= 0
v >= 0
u + v <= 1
degenerate triangle -> miss
```

Normals remain flat face normals:

```text
normal_world = normalize(cross(v1 - v0, v2 - v0))
```

No smooth normals, vertex normals, interpolation, or material sampling in L2.

## 11. Tie-Break Rule

The reference executor updates only when:

```text
new_t < best_t
```

So exact equal-distance hits keep the first primitive encountered by reference
scan order.

BVH traversal has a different visitation order, so L2 must make this explicit.
Use `t_eps = 1e-9` here, not `build_eps` or `dir_eps`. `build_eps` is for
geometric degeneracy, `dir_eps` is for slab-parallel checks, and `t_eps` is for
comparing hit distances.

The concrete rule is:

```text
if t < best_t - t_eps:
    update
elif abs(t - best_t) <= t_eps and source_order_key < best_source_order_key:
    update
```

This rule should apply across:

```text
triangle-vs-triangle hits
triangle-vs-plane hits
plane-vs-plane hits if multiple planes exist
```

Source order is lexicographic and global across `snapshot.instances`; geometry
kind does not change ordering. This preserves deterministic behavior even when
a triangle and plane are coplanar.

## 12. Plane Side Pass

Infinite planes do not enter the BVH.

The BVH executor should:

```text
1. traverse triangle BVH
2. run analytical plane intersections
3. merge nearest result with the same tie-break rule
```

Rationale:

- infinite planes do not have finite AABBs;
- fake large boxes introduce arbitrary scene-scale constants;
- analytical plane intersection is cheap and already tested.

## 13. Scene / Executor Boundary

The build path should be opt-in at scene/cache:

```text
snapshot = cache.snapshot_from_frame_inputs(inputs, acceleration="cpu_bvh")
```

The executor should not silently build BVH data:

```text
CpuBvhOpticalExecutor().execute(snapshot, spec)
```

If `snapshot.acceleration` is missing or not `kind == "cpu_bvh"`:

```text
raise MissingAccelerationError("snapshot does not contain CPU BVH acceleration")
```

This keeps performance costs visible and preserves:

```text
scene/cache prepares executable scene data
executor performs sensor-specific optical query
```

## 14. Complexity

Build:

```text
O(N log N)
```

because each recursive level sorts primitive ids by centroid.

Traversal:

```text
average: near O(log N + leaf primitive tests)
worst case: O(N)
```

Known degeneracies:

- all centroids equal -> leaf fallback;
- many overlapping AABBs -> traversal may approach linear;
- very large/slender triangles -> weak pruning;
- many role-invisible primitives -> still traversed until role-specific BVHs
  exist.

These are acceptable for L2 correctness-first implementation.

## 15. Tests

Required before accepting implementation:

```text
reference vs bvh parity on triangle mesh
reference vs bvh parity on mixed plane + mesh scene
miss semantics: inf / NaN / None / numeric background 0
roles filtering: invisible primitive does not hit
camera L1 postprocess works with bvh result
missing acceleration raises clear error
tie-break: coplanar or equal-distance primitives choose reference source order
near-equal non-coplanar hits validate t_eps does not hide real nearest hits
degenerate triangles are skipped
empty mesh-BVH scene with planes still works through plane side pass
```

No performance threshold in first L2 tests. Add microbenchmarks only after
correctness and schema parity are stable.

## 16. Explicit Non-Goals

L2 first version does not implement:

```text
SAH
BVH refit
Embree binding
Open3D backend
GPU BVH
OptiX
direct-light RGB
shadow rays
reflection/refraction
smooth normals
texture sampling
material BRDF
deformable mesh BVH
fluid/volume acceleration
multi-env BVH batching
```

## 17. Review Questions

1. Is world-space packed triangle BVH acceptable for L2 given the documented
   per-snapshot vertex-transform/rebuild cost?
2. Is longest-axis median split sufficient for L2, with SAH deferred?
3. Is `leaf_size = 4` a good first default?
4. Should degenerate triangles be skipped at pack time, matching reference
   miss behavior?
5. Is explicit lexicographic source-order-key tie-break sufficient to preserve
   reference parity?
6. Is `t_eps = 1e-9` the right first tolerance for hit-distance tie-breaks?
7. Should `CpuBvhOpticalExecutor` raise `MissingAccelerationError` on missing
   acceleration rather than fallback to reference traversal?
8. Should the acceleration opt-in live on
   `snapshot_from_frame_inputs(..., acceleration="cpu_bvh")`, or on
   `OpticalSceneCache` construction?
