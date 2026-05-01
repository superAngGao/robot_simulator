Initiative: q54-optical-l2-acceleration
Stage: review-request
Author: codex
Version: v1
Date: 2026-04-30
Status: draft-for-review
Related Files: optics/scene.py, optics/execution.py, optics/registry.py, sensing/optical.py
Owner Summary: After L1 camera semantics, the next capability is L2 CPU acceleration. Local dependency inspection shows no Embree/Open3D binding is currently installed or declared, so this proposal recommends an in-repo simple BVH as the first implementation, with Embree kept as a future optional adapter behind the same scene/executor contract.

# Q54 L2 Optical Acceleration Review Request

## 1. Current State

Q54 now has:

```text
OpticalWorldRegistry
OpticalFrameInputs
OpticalSceneCache / OpticalSceneSnapshot
CpuReferenceOpticalExecutor
OpticalPinholeCameraSpec
OpticalCameraImageResult
```

The accepted capability path is:

```text
L0 first-hit range/material/instance             done
L1 image-shaped camera depth/range/segmentation done
L2 CPU acceleration                              next
L3 direct-light/simple RGB                       later
L5 GPU/Q52 device result path                    later
```

Local dependency check:

```text
embree   unavailable
pyembree unavailable
embreex  unavailable
open3d   unavailable
```

`setup.py` currently declares only core `numpy/matplotlib`, plus optional
`mesh`, `rl`, `rerun`, and `dev` extras. There is no existing ray-acceleration
dependency.

## 2. Main Proposal

Implement L2 first as an in-repo CPU BVH over finite triangle primitives.

Do not introduce Open3D as an intermediate backend. Do not make Embree a hard
dependency yet. Keep Embree as a later optional adapter once dependency policy
is settled.

Recommended sequence:

```text
0. Review and explain the BVH algorithm before implementation.
1. Add scene/cache-owned packed triangle acceleration data.
2. Add a simple in-repo BVH builder/traverser.
3. Add CpuBvhOpticalExecutor with the same OpticalComputeResult schema.
4. Add parity tests against CpuReferenceOpticalExecutor.
5. Later: add Embree adapter behind the same packed primitive schema.
```

Important process rule: because L2 is self-authored acceleration code, do not
start implementation until the algorithm is written out and reviewed. Each
self-authored algorithm step should document:

```text
inputs / outputs
data layout
build and traversal algorithm
edge cases and degeneracy behavior
complexity expectations
test evidence
explicit non-goals
```

This is intentionally more detailed than adapter work. The goal is to make sure
we can inspect the algorithm before it becomes hidden inside code.

## 3. Boundary Decision

Acceleration structure belongs to scene/cache, not registry and not sensing.

Reason:

- registry owns persistent identity/material/source mapping;
- frame inputs provide dynamic transforms and future deformable geometry;
- scene/cache creates executable, frame-aligned geometry;
- BVH/BLAS/TLAS is executable scene data;
- executor performs ray traversal/query and resolves result channels.

This keeps the previous boundary:

```text
registry + frame inputs -> scene/cache executable scene -> executor query result
```

## 4. Proposed Scene Data Shape

Add an optional acceleration payload to `OpticalSceneSnapshot`, for example:

```text
OpticalSceneSnapshot.acceleration: OpticalSceneAcceleration | None
```

Initial CPU payload:

```text
OpticalSceneAcceleration
  kind = "cpu_bvh"
  triangles_world: float64[num_triangles, 3, 3]
  triangle_normals_world: float64[num_triangles, 3]
  primitive_instance_indices: int64[num_triangles]
  bvh_nodes: tuple/list/arrays
```

Each packed triangle points back to an `OpticalInstanceSnapshot`, so executor
channel resolution still uses:

```text
material_id
instance_id
numeric_instance_id
roles
source_key
```

from the authoritative instance snapshot.

## 5. Plane Handling

Do not put infinite planes in the BVH.

Reason:

- infinite planes do not have finite AABBs;
- clipping them to a fake large box would introduce arbitrary scene scale;
- current analytical plane intersection is cheap and exact.

Initial `CpuBvhOpticalExecutor` should do:

```text
1. traverse BVH for triangle meshes
2. run analytical plane intersection pass
3. choose nearest hit across both
```

This preserves current behavior and avoids contaminating the BVH with special
world-scale constants.

## 6. Build / Refit Policy

Initial L2 should rebuild the simple BVH per snapshot.

This is acceptable for correctness-first L2 because:

- Phase A/B snapshots are one-env CPU;
- the goal is schema and parity, not final performance;
- rebuild is simpler than exposing invalidation bugs too early.

Later refinements:

```text
rigid-only mesh, same topology    -> refit/update world AABBs
topology changed                  -> rebuild
deformable vertices changed       -> refit or rebuild depending on quality
fluid representation changed      -> rebuild all geometry buffers and BVH
```

## 7. Executor Contract

Add a second executor:

```text
CpuBvhOpticalExecutor
```

with the same public method:

```text
execute(snapshot: OpticalSceneSnapshot, spec: OpticalRaySensorSpec)
  -> OpticalComputeResult
```

The returned channel set must match `CpuReferenceOpticalExecutor.capabilities`.

Recommended behavior if acceleration is missing:

```text
raise ValueError("snapshot does not contain CPU BVH acceleration")
```

instead of silently building inside the executor. Silent building would blur the
scene/executor boundary and hide performance costs.

## 8. Scene Cache API Option

Minimal API:

```text
OpticalSceneCache.snapshot_from_frame_inputs(
    inputs,
    acceleration: Literal["none", "cpu_bvh"] = "none",
)
```

Default remains `"none"` so existing tests and behavior are unchanged.

Alternative:

```text
OpticalSceneCache(..., acceleration_policy="cpu_bvh")
```

The per-call option is more explicit for tests and lets consumers opt into BVH
only where needed.

## 9. Tests

Required tests:

```text
reference vs bvh parity on triangle mesh
reference vs bvh parity on mixed plane + mesh scene
camera image postprocess works with bvh result
roles filtering still applies
missing acceleration raises clear error
schema test: dtype/shape/miss values match reference executor
```

No performance assertion in the first L2 tests. Performance microbenchmarks can
come later once traversal is stable.

## 10. Review Questions

1. Should `OpticalSceneSnapshot.acceleration` be the place for BVH/acceleration
   payloads, or should the acceleration object live outside the snapshot and be
   passed separately to the executor?
2. Should first L2 implementation be in-repo BVH, given no Embree/Open3D binding
   is currently installed or declared?
3. Should `snapshot_from_frame_inputs(..., acceleration="cpu_bvh")` be the public
   opt-in API, or should acceleration policy live on `OpticalSceneCache`?
4. Should `CpuBvhOpticalExecutor` raise if acceleration is missing, or fall back
   to reference traversal?
5. Is keeping infinite planes outside the BVH the right first policy?
6. Should the packed primitive schema store world-space triangles per snapshot,
   or local-space triangles plus per-instance transforms?
7. Should L2 include only triangle-mesh BVH parity, leaving direct-light shadow
   ray acceleration for L3?
