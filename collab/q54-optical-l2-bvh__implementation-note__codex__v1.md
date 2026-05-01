Initiative: q54-optical-l2-bvh
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-05-01
Status: implemented
Related Files: optics/scene.py, optics/execution.py, optics/__init__.py, tests/unit/optics/test_bvh_executor.py
Owner Summary: Implemented the accepted L2 in-repo CPU BVH path. Scene/cache can opt into `cpu_bvh` acceleration, snapshots carry packed world-space triangle BVH data, and `CpuBvhOpticalExecutor` consumes that acceleration while preserving the reference result schema and source-order tie-break semantics.

# Q54 L2 CPU BVH Implementation Note

## 1. Implemented Path

Implemented the reviewed L2 path:

```text
in-repo CPU BVH
world-space packed triangles
longest-axis median split
leaf_size = 4
iterative stack traversal
analytical infinite-plane side pass
global lexicographic source-order tie-break
scene/cache builds acceleration
executor consumes acceleration only
```

No Embree/Open3D dependency was added.

## 2. Scene / Cache Changes

Added to `optics.scene`:

```text
CpuBvhNode
OpticalSceneAcceleration
build_cpu_bvh_acceleration(...)
```

`OpticalSceneSnapshot` now has:

```text
acceleration: OpticalSceneAcceleration | None = None
```

`OpticalSceneCache.snapshot_from_frame_inputs(...)` and
`snapshot_from_published_frame(...)` now accept:

```text
acceleration: Literal["none", "cpu_bvh"] = "none"
```

Default remains `"none"`, so existing snapshot behavior is unchanged.

## 3. Packed Primitive Schema

The CPU BVH acceleration payload stores:

```text
triangles_world: float64[N, 3, 3]
triangle_normals_world: float64[N, 3]
primitive_instance_indices: int64[N]
primitive_source_order_keys: int64[N, 2]
primitive_aabb_min: float64[N, 3]
primitive_aabb_max: float64[N, 3]
primitive_indices: int64[N]
nodes: list[CpuBvhNode]
```

Only finite triangle meshes are packed. Degenerate triangles are skipped at pack
time. Infinite planes are not placed in the BVH.

## 4. Executor Changes

Added to `optics.execution`:

```text
MissingAccelerationError
CpuBvhOpticalExecutor
```

`CpuBvhOpticalExecutor` inherits the reference executor public contract and
capabilities:

```text
execute(snapshot, spec) -> OpticalComputeResult
```

If `snapshot.acceleration` is missing or is not `kind == "cpu_bvh"`, the
executor raises `MissingAccelerationError`. It does not silently build a BVH.

## 5. Traversal / Merge Semantics

Traversal:

```text
per-ray iterative stack
inclusive slab ray-AABB test
leaf triangle tests with Moller-Trumbore semantics
near/far child ordering for earlier best_t tightening
```

Plane handling:

```text
1. traverse BVH triangle primitives
2. run analytical plane pass
3. merge hits using the same tie-break rule
```

Tie-break:

```text
source_order_key = (instance_index, primitive_index_within_instance)
```

This is global across `snapshot.instances`, so plane and triangle tie behavior
matches the reference executor instead of depending on geometry kind.

Constants:

```text
build_eps = 1e-12
dir_eps = 1e-12
t_eps = 1e-9
```

## 6. Tests

Added `tests/unit/optics/test_bvh_executor.py` covering:

```text
scene cache packs world-space triangle BVH
plane-only snapshot has empty BVH payload
degenerate triangles are skipped at pack time
reference vs BVH parity on triangle mesh
internal BVH node traversal
plane side pass with empty BVH
plane-before-mesh tie-break
mesh-before-plane tie-break
near-equal non-coplanar hit keeps true nearest outside t_eps
roles filtering
missing acceleration raises MissingAccelerationError
camera L1 postprocess accepts BVH executor result
```

## 7. Verification

```text
PYTHONPATH=. pytest tests/unit/optics tests/unit/sensing -q
66 passed

ruff check optics sensing tests/unit/optics tests/unit/sensing
All checks passed

python -m compileall optics sensing tests/unit/optics tests/unit/sensing
passed

git diff --check
passed
```

## 8. Deferred

Still deferred:

```text
SAH
BVH refit
role-specific BVHs
Embree adapter
GPU/OptiX path
direct-light RGB
shadow rays
multi-env BVH batching
deformable/fluid acceleration
```
