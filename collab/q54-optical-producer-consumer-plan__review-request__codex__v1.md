Initiative: q54-optical-producer-consumer-plan
Stage: review-request
Author: codex
Version: v1
Date: 2026-04-30
Status: ready-for-claude-review
Related Files: OPEN_QUESTIONS.md#Q54, collab/q54-optical-computation-workflow__discussion__codex__v1.md, optics/, sensing/optical.py
Owner Summary: This file asks Claude to review the proposed producer-to-consumer plan for Q54 optical computation. The central decision is to build the optical pipeline layer by layer: model/assets produce a registry, physics produces frames, scene/cache produces executable frame snapshots, executor produces optical results, and consumers read results without reconstructing scene identity.

# Q54 Producer-To-Consumer Plan Review Request

## 1. Goal

Build the optical pipeline from producer to consumer without mixing
responsibilities across physics publishing, optical asset binding, executable
scene construction, optical computation, and result consumption.

Target flow:

```text
Asset / Model Producer
  -> Registry Builder
  -> OpticalWorldRegistry
  + PublishedFrame N
  -> OpticalSceneCache / OpticalSceneSnapshot N
  + OpticalSensorSpec
  -> OpticalExecutor
  -> OpticalComputeResult
  -> sensing / RL / Rerun / export / tests
```

## 2. Core Boundary

The proposed boundary rules are:

```text
registry does not read frame
frame does not carry optical assets
scene does not perform optical computation
executor does not perform asset binding
consumer does not reconstruct scene identity
Rerun/debug sinks do not participate in computation
```

More concretely:

- `PublishedFrame` only provides physics timeline state: `frame_id`,
  `sim_time`, `X_world`, `q/qdot`, contacts, and telemetry.
- `OpticalWorldRegistry` owns long-lived optical assets and identity:
  geometry, materials, lights, instances, source mappings, semantic ids,
  visibility/roles, and version/dirty metadata.
- `OpticalSceneCache` synchronizes registry + frame into executable scene data:
  resolve binding, compose transforms, pack buffers, and build/refit
  acceleration structures.
- `OpticalSceneSnapshot` is a frame-aligned read-only executable view. It
  answers what exists at frame `N`, where it is, and how it is encoded.
- `OpticalExecutor.execute(snapshot, spec)` performs the actual optical
  computation: ray traversal, rasterization, shading, segmentation resolve, or
  sensor response.
- `OpticalComputeResult` is the result/readiness contract. It must not force
  all consumers through host-side readings.

## 3. Stage 1: Registry / Binding Builder

The missing next producer is:

```text
RobotModel / visual assets / collision geometry / user optical config
  -> OpticalBindingBuildResult
  -> OpticalWorldRegistry
```

Proposed output:

```python
@dataclass
class OpticalBindingBuildResult:
    registry: OpticalWorldRegistry
    source_to_instance_id: dict[OpticalSourceKey, str]
    instance_to_source: dict[str, OpticalSourceKey]
    diagnostics: list[OpticalBindingDiagnostic]
```

Open design points:

- Should `OpticalInstanceSpec` replace the current narrower
  `OpticalGeometryBinding` before registry-builder work?
- Should `OpticalSourceKey` be introduced now to record model/body/geometry
  role/shape/submesh provenance?
- Should `visible_to` or role metadata be added now to distinguish RGB, LiDAR,
  depth, segmentation, and debug visibility?
- Should stable numeric ids be assigned by the registry, with cache only
  packing them, or should cache assign numeric ids per snapshot?
- Which source policies should be supported first?
  - `explicit_only`
  - `visual_preferred`
  - `collision_only`
  - `visual_preferred_with_collision_fallback`

Codex recommendation: start with an explicit `OpticalBindingBuildResult`,
introduce `OpticalSourceKey`, and make numeric ids registry-owned so downstream
consumers and executors see stable identity.

## 4. Stage 2: Registry Contract

The registry should be more than a geometry table. It should become the common
source of truth for downstream executable scenes, executors, RL observations,
debug visualization, exporters, tests, and domain randomization.

Suggested registry layers:

```text
Assets:
  geometry / material / texture / light / medium

Instances:
  instance_id / numeric_instance_id / geometry_id / material_id
  body or world binding / X_body_geometry / visibility / semantic tags

Semantic Maps:
  instance_id <-> source key
  instance_id -> body index/name
  instance_id -> semantic class/id
  material_id -> material class/id

Versioning / Dirty State:
  geometry_version
  material_version
  light_version
  instance_binding_version
  visibility_version
```

Downstream consumption:

- scene cache uses versions/dirty bits to decide rebuild/refit/update;
- executors use packed ids to output segmentation/material/instance channels;
- RL/Rerun/debug/exporters use source mappings to explain results;
- domain randomization can target material, instance, class, or group without
  changing executor logic.

## 5. Stage 3: Scene / Executable View

The scene layer should not compute sensor results. It organizes executable scene
data for a specific frame.

Allowed in scene/cache:

```text
resolve binding
compose X_world_geometry = frame.X_world[body_index] @ X_body_geometry
pack geometry/material/light buffers
update dirty data
build/refit BVH/BLAS/TLAS
attach dependency events/fences
freeze frame-aligned snapshot
```

Forbidden in scene/cache:

```text
ray traversal
shading
reflection/refraction
segmentation result generation
sensor noise/clipping/response
result-to-reading conversion
```

Recommended representation split:

```python
OpticalSceneSnapshot      # semantic/debug-friendly, CPU reference friendly
PackedOpticalSceneBuffers # executor-friendly, GPU/Embree/OptiX friendly
```

Both should derive from the same registry/cache identity. Different backends
should not invent independent instance/material/semantic id systems.

## 6. Stage 4: Executor Contract

Public contract:

```python
OpticalExecutor.execute(snapshot, spec) -> OpticalComputeResult
```

Executor responsibilities:

```text
validate snapshot/spec frame-env match
prepare workload
perform traversal/rasterization/shading/sensor computation
resolve result channels
return location/readiness-aware result
```

Executor non-responsibilities:

```text
do not modify registry
do not fetch a new frame
do not perform asset binding
do not force host-side readings
do not use Rerun/debug sinks for computation
```

Canonical result channels:

```text
hit_mask
depth_m
range_m
position_world
normal_world
instance_id
numeric_instance_id
material_id
numeric_material_id
semantic_id
rgb
intensity
```

Miss semantics:

```text
hit_mask = False
depth/range = np.inf
position/normal = NaN
human-readable ids = None
numeric ids = documented background id
```

Executor phases:

```text
A. CPU reference: first-hit depth + material/instance id
B. Internal split: validate / prepare_workload / intersect / resolve_channels / build_result
C. CPU acceleration: Embree or simple BVH
D. direct-light / RGB capability
E. device results + Q52 device-consumer completion
F. Mitsuba/offline adapter
```

## 7. Stage 5: Result Consumers

Consumers should consume `OpticalComputeResult` and registry mappings. They
should not reconstruct scene identity.

Consumers:

```text
sensing readings
RL observation builders
Rerun/debug visualization
exporters
tests/regression artifacts
```

Rules:

- sensing builders may convert host channels to readings;
- RL paths may consume device channels directly;
- Rerun logs result/debug artifacts after executor has produced them;
- exporters use registry source mappings for explainable ids;
- consumers do not mutate registry or scene cache.

## 8. Current Implementation State

Already landed:

```text
sensing.OpticalRaySensorSpec
optics.OpticalWorldRegistry
OpticalMaterialSpec
OpticalLightSpec
plane / triangle mesh geometry handles
OpticalGeometryBinding
OpticalSceneCache
OpticalSceneSnapshot
OpticalExecutor
OpticalComputeResult
CpuReferenceOpticalExecutor
tests/unit/optics/test_optics_phase_a.py
```

Current limits:

```text
CPU one-env only
no real packed buffers yet
no BVH/accel dirty state yet
no registry builder yet
no visual asset pipeline yet
no direct-light/RGB
no GPU device-result/Q52 integration
multi-env batching unresolved before Phase C/E
```

## 9. Review Questions For Claude

1. Is the producer-to-consumer boundary split correct?
2. Should `OpticalInstanceSpec` replace `OpticalGeometryBinding` now, before
   registry-builder work?
3. Should registry own stable numeric ids, or should cache assign packed
   numeric ids per snapshot?
4. Is `OpticalSourceKey` necessary in the registry itself, not only in the build
   result?
5. Is `visible_to` / role metadata needed now to prevent visual/collision/debug
   geometry ambiguity?
6. Does the scene/cache boundary correctly allow acceleration-structure
   build/refit while forbidding optical computation?
7. Are canonical result channels and miss semantics sufficient for reference,
   Embree, GPU, RGB, and RL consumers?
8. Is the staged backend plan reasonable: reference executor -> internal
   split/schema tests -> Embree/BVH -> direct-light/RGB -> GPU/Q52 -> Mitsuba?

## 10. Requested Review Output

Please respond with one of:

- accept as-is;
- accept with required edits;
- reject a boundary and propose a replacement;
- identify missing downstream consumers or lifecycle risks.

If changes are required, please call out whether they block the next
implementation step (`OpticalBindingBuildResult` / registry builder) or can wait
until packed scene buffers / accelerated executors.
