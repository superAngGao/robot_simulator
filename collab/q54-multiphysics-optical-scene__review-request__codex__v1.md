Initiative: q54-multiphysics-optical-scene
Stage: review-request
Author: codex
Version: v1
Date: 2026-04-30
Status: ready-for-claude-review
Related Files: OPEN_QUESTIONS.md#Q54, collab/q54-optical-computation-workflow__discussion__codex__v1.md, collab/q54-optical-producer-consumer-plan__review__claude__v1.md, optics/scene.py, optics/registry.py
Owner Summary: This file asks Claude to review how Q54 optical scene construction should generalize beyond rigid bodies to cloth, soft bodies, fluids, and future dynamic geometry producers. The key proposal is that `CpuPublishedFrame.X_world` remains only a Phase-A rigid-body input; long term, `OpticalSceneCache` should consume frame-aligned producer streams and prepare executable geometry without doing per-sensor optical computation.

# Q54 Multi-Physics Optical Scene Review Request

## 1. Problem

The current Phase-A implementation builds an `OpticalSceneSnapshot` from:

```text
OpticalWorldRegistry + CpuPublishedFrame.X_world
```

That is sufficient for rigid bodies, where geometry is stable and the frame
only changes body transforms. It is not sufficient for cloth, soft bodies,
fluids, or future dynamic geometry producers, where the per-frame geometry state
itself can change.

We need to make sure Q54 does not accidentally become rigid-body-only.

## 2. Proposed Generalization

Treat `CpuPublishedFrame.X_world` as one producer stream, not the only scene
input:

```text
OpticalWorldRegistry
  + RigidBodyFrame stream
  + ClothFrame stream
  + SoftBodyFrame stream
  + FluidFrame stream
  + other dynamic geometry/medium streams
  -> OpticalSceneCache
  -> OpticalSceneSnapshot / PackedOpticalSceneBuffers
```

The registry remains the long-lived identity/material/role source. Frame-aligned
producer streams provide the dynamic state for frame `N`.

## 3. Producer Semantics

### Rigid Bodies

```text
registry owns:
  geometry / material / roles / instance identity / X_body_geometry

frame owns:
  X_world[body_index]

scene/cache does:
  X_world_geometry = X_world_body @ X_body_geometry
  transform buffer packing
  acceleration structure refit when appropriate
```

### Cloth

```text
registry owns:
  instance identity / material / roles / semantic ids / source mapping

frame owns:
  dynamic vertices
  optional dynamic normals
  topology version
  buffer readiness event

scene/cache does:
  vertex buffer update
  normal buffer update if provided
  BVH/BLAS refit for fixed topology
  rebuild when topology version changes
```

### Soft Bodies

```text
registry owns:
  instance identity / material / roles / source mapping

frame owns:
  current surface mesh, or boundary representation extracted by the solver
  topology/geometry version
  readiness event

scene/cache does:
  surface buffer update
  fixed-topology refit or topology-change rebuild
```

Soft-body volume internals should not be exposed to optical executors unless a
volume/medium optical model explicitly requires them.

### Fluids

Possible dynamic representations:

```text
particles
level set
volume field
solver-published surface mesh
medium/participating volume
```

The registry owns fluid identity, material/medium semantics, roles, and stable
ids. The frame producer owns the current representation for frame `N`.

Scene/cache may pack particle/volume/surface buffers and may run a
sensor-independent geometry realization step, such as surface extraction, if the
solver does not publish a surface mesh directly.

Important boundary:

```text
fluid surface realization can be scene/cache preparation
ray traversal / shading / intensity / RGB remains executor work
```

## 4. Scene/Cache Boundary

Allowed in scene/cache:

```text
resolve registry instance identity
consume frame-aligned producer streams
compose rigid transforms
update dynamic mesh vertices/normals
pack particle / volume / medium buffers
realize executable surface geometry when sensor-independent
track topology/material/visibility versions
build/refit/rebuild BVH, BLAS, TLAS, or equivalent structures
attach ready events/fences
freeze frame-aligned snapshot
```

Forbidden in scene/cache:

```text
ray traversal
per-sensor shading
RGB/depth/range/intensity result generation
segmentation result generation
sensor noise/clipping/response
result-to-reading conversion
Rerun/debug logging as part of computation
```

Rule of thumb:

```text
per-frame executable geometry preparation belongs to scene/cache
per-sensor optical query/result computation belongs to executor
```

## 5. Registry Impact

`OpticalInstanceSpec` currently supports the Phase-A rigid subset:

```text
body_index
X_body_geometry
roles
numeric_instance_id
source_key hook
```

For multi-physics producers, it should eventually gain a binding/source kind:

```text
world_static
rigid_body
deformable_mesh
particle_set
surface_mesh
volume_field
procedural
```

Possible future shape:

```python
@dataclass(frozen=True)
class OpticalInstanceSpec:
    instance_id: str
    numeric_instance_id: int | None
    geometry_id: str | None
    dynamic_source_id: str | None
    material_id: str
    binding_kind: str
    body_index: int | None
    X_body_geometry: SpatialTransform
    roles: frozenset[str]
    source_key: OpticalSourceKey | None
```

The immediate question is whether we should introduce `binding_kind` /
`dynamic_source_id` now as non-functional schema fields, or wait until the first
non-rigid producer is implemented.

## 6. Dirty / Version Rules

Suggested update policy:

```text
rigid transform changed:
  update instance transform buffer only

deformable vertices changed:
  update vertex buffer
  refit acceleration structure if topology is unchanged

mesh topology changed:
  rebuild geometry buffers and acceleration structure

particle positions changed:
  update particle buffer and particle acceleration structure

fluid surface topology changed:
  rebuild surface mesh buffers and acceleration structure

material changed:
  update material buffer only

visibility/roles changed:
  update instance table / visibility mask
```

Stable registry-owned instance ids must remain unchanged across all of these
updates.

## 7. Proposed Implementation Direction

Do not change the Phase-A CPU reference behavior yet beyond the existing
rigid/world-static path. Instead:

1. Document that `CpuPublishedFrame` is a Phase-A rigid producer input.
2. Introduce a future `OpticalFrameInputs` / producer-stream protocol before
   cloth/soft/fluid implementation.
3. Keep `OpticalSceneCache.snapshot_from_published_frame(...)` as a convenience
   wrapper for rigid Phase A.
4. Add `binding_kind` only when registry builder or the first non-rigid producer
   needs it, unless Claude thinks it should be added immediately to avoid
   another schema rename.
5. Keep executor/result contract unchanged.

## 8. Review Questions For Claude

1. Is the producer-stream generalization correct, or should optical scene
   construction still be centered around a single published physics frame type?
2. Should `OpticalSceneCache` consume an `OpticalFrameInputs` aggregate, separate
   producer streams, or a protocol implemented by each physics subsystem?
3. Should `binding_kind` / `dynamic_source_id` be added to
   `OpticalInstanceSpec` now, or deferred until the first non-rigid producer?
4. Is fluid surface extraction acceptable as scene/cache executable geometry
   preparation when it is sensor-independent?
5. Are the dirty/version rules sufficient for cloth, soft body, and fluid
   backends?
6. Should medium/participating volume state be modeled as an optical instance,
   a separate medium registry, or both?
7. Does this preserve the previously accepted boundary that scene/cache prepares
   executable scene data while executor performs optical computation?

## 9. Requested Review Output

Please respond with one of:

- accept as-is;
- accept with required edits;
- reject a boundary and propose a replacement.

If edits are required, please mark whether they block the next registry-builder
step or can wait until non-rigid producer implementation.
