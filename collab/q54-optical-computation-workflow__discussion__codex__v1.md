Initiative: q54-optical-computation-workflow
Stage: discussion
Author: codex
Version: v1
Date: 2026-04-29
Status: draft-review-followup-applied
Related Files: OPEN_QUESTIONS.md#Q53, OPEN_QUESTIONS.md#Q54, physics/publish.py, rendering/render_scene.py, sensing/surface_query.py
Owner Summary: Clarifies the real requirement and the current gap for optical computation after a rigid-body frame has been published. The key gap is not one missing field on `PublishedFrame`; it is the absence of an optical scene synchronization pipeline that composes physics transforms, optical geometry, material bindings, lights, media, acceleration structures, and sensor execution/results.

# Q54 Optical Computation Workflow

## 1. Why This Exists

Q53 deliberately kept `SurfaceQuerySpec` narrow:

- it describes already-expanded geometric rays;
- it returns hit distance/position/normal;
- it does not know about materials, lights, spectra, reflection, refraction,
  exposure, camera response, or renderer execution.

That boundary is still correct. The problem is that optical sensing and
render-backed sensing need a larger workflow than surface queries:

```text
rigid-body step complete
  -> published physics frame available
  -> optical scene state updated for the same frame
  -> optical transport / rendering / ray tracing executes
  -> result buffers become available
  -> sensors, RL observation builders, loggers, and Rerun consume results
```

The missing part is the middle:

```text
PublishedFrame + optical world data -> executable optical scene snapshot
```

This is not solved by adding lights or material fields to `PublishedFrame`.
`PublishedFrame` is a frame-state contract. Optical computation needs a
scene-execution contract.

## 2. Real Requirement

For an environment containing multiple rigid bodies, optical materials, and
registered lights, after the physics engine finishes frame `N`, the system must
be able to compute optical outputs for that same frame.

The first-class outputs may include:

- RGB / grayscale camera images
- depth images
- segmentation / instance-id images
- range or LiDAR readings
- irradiance / intensity readings
- debug rays, hit points, and intermediate optical diagnostics

These outputs need to be consumable by:

- `sensing/` readings and sensor-facing APIs
- future RL observation builders, preferably without forced host copies on the
  hot path
- Rerun / debug visualization
- offline exporters and test harnesses

The computation must know more than rigid-body transforms:

- render/query geometry and topology
- per-shape or per-submesh material bindings
- surface normals, winding, optional UVs/tangents
- light sources and emission profiles
- medium / atmosphere parameters when relevant
- sensor intrinsics, extrinsics, exposure, clipping, and sampling pattern
- acceleration structures and dirty/update rules
- result-buffer location and readiness events

## 3. What We Have Today

### PublishedFrame

`CpuPublishedFrame` / `GpuPublishedFrame` provide the physics timeline state:

- frame id / sim time / step index
- `q`, `qdot`
- body/world transforms
- velocities
- contacts / telemetry / contact mask
- GPU ready events and slot lifetime protection

This is enough for state sensing, debug rendering transforms, and basic surface
query builders.

It is not enough for optical transport. It does not own lights, optical
materials, mesh assets, texture resources, media, renderer configuration, or
acceleration structures.

### RenderScene

`RenderScene` is a debug/inspection snapshot:

- positioned shapes
- terrain
- contacts
- skeleton/body names
- narrow numeric sensor overlays

It is backend-neutral and useful, but it is not an optical world model. It does
not carry physically meaningful material/light contracts and should not become
the canonical camera/LiDAR execution scene.

### RerunBackend

`RerunBackend` logs `RenderScene`-like data for visualization. It is a sink:

```text
scene/result -> Rerun log
```

It should not be treated as the optical executor:

```text
scene -> Rerun -> optical result
```

Rerun can display meshes, debug rays, images, depth previews, scalar timelines,
and point clouds. It does not close the gap between physics state and optical
transport.

### SurfaceQuerySpec

`SurfaceQuerySpec` is a geometric ray batch. It is the correct primitive for
range/depth probes that only need the first surface hit. It intentionally does
not model light transport.

## 4. The Actual Gap

The current architecture is missing an explicit optical scene layer with these
responsibilities:

1. **Stable optical geometry identity**
   - map physics body/shape ids to optical instances;
   - distinguish collision geometry from render/query geometry;
   - support mesh/submesh topology where needed.

2. **Material bindings**
   - assign optical materials per shape, mesh, or submesh;
   - represent parameters needed by future executors: color/albedo,
     roughness/specular, opacity, index of refraction, emissive terms, etc.;
   - avoid putting optical material semantics into `physics/geometry.py`.

3. **Light registry**
   - define point/directional/spot/area/emissive sources;
   - snapshot position, orientation, intensity, color/spectrum, cone angle, and
     enabled state at the frame being computed;
   - support lights attached to bodies or fixed in world.

4. **Medium registry**
   - optional for the first implementation, but the design should leave room for
     fog/attenuation/participating media instead of baking "air" assumptions
     into every executor.

5. **Frame-aligned transform composition**
   - combine `PublishedFrame` transforms with optical instance bindings;
   - handle multi-env selection;
   - produce an immutable view for frame `N` while the physics ring advances.

6. **Acceleration structures**
   - CPU BVH for host executors;
   - GPU BLAS/TLAS-like buffers for future CUDA/Warp executors;
   - dirty flags for topology/material/transform/light changes;
   - refit vs rebuild policy.

7. **Sensor execution contract**
   - input specs: camera, LiDAR, irradiance probe, range finder;
   - execution location: host/device/external renderer;
   - output buffers: host NumPy, device arrays, or external handles;
   - readiness: CPU completion, CUDA/Warp event, or external fence.

8. **Consumer contract**
   - decide which outputs become `sensing` readings;
   - decide which outputs are RL-hot-path device tensors;
   - decide which outputs are only debug/logging artifacts;
   - avoid forcing every optical result through a host-side `Reading`.

## 5. Proposed Layer Split

Keep the current Q52/Q53 boundaries and add a separate optical integration
layer:

```text
physics/
  PublishedFrame, body transforms, contact/telemetry truth

sensing/
  Sensor specs/readings, surface-query specs/results, observation-facing schema

rendering/
  RenderScene, debug render backends, Rerun visualization sink

optics/
  OpticalWorldRegistry
  OpticalSceneCache
  OpticalSceneSnapshot
  OpticalExecutor
  OpticalComputeResult
```

Review follow-up decision: use `optics/`.

Rationale:

- `sensor_rendering/` sounds like a narrow integration layer for camera-style
  rendering sensors.
- Q54 is broader: material bindings, lights, medium state, acceleration
  structures, and light-transport execution are optical computation concerns,
  not merely rendering-sensor concerns.
- `optics/` keeps the dependency story clearer: it is the layer that combines
  published physics state, sensing specs, and optical world data. Rendering/Rerun
  remains a sink for results and debug artifacts.

The important dependency direction:

```text
optics -> physics published frames
optics -> sensing specs/readings
optics -> optional rendering/Rerun debug sinks
physics/ does not import optical integration
sensing/ does not import rendering backends
rendering/ does not become the canonical optical executor
```

## 6. Proposed Core Types

Names are draft, but the responsibilities should stay separate.

### OpticalWorldRegistry

Long-lived registry for optical data that is not produced by the physics step:

- optical geometry handles
- material table
- light table
- medium table

It may reference physics body/shape ids, but it should not mutate physics state.

Review follow-up decision: sensor specs stay in `sensing/`, not in
`OpticalWorldRegistry`.

Rationale:

- The optical registry describes persistent world state: geometry bindings,
  materials, lights, and medium parameters.
- Sensor specs describe the question being asked: camera intrinsics, LiDAR scan
  pattern, range-finder pose, clipping, noise policy, and observation-facing
  conventions.
- Their lifecycles differ. World optical state may be stable for many frames;
  sensor specs can vary per frame or per environment under domain
  randomization.
- Keeping `OpticalExecutor.execute(snapshot, spec)` preserves that separation.

Draft minimal material schema:

```python
@dataclass(frozen=True)
class OpticalMaterialSpec:
    material_id: str
    albedo_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    extension: dict[str, object] = field(default_factory=dict)
```

`material_id` is required because material-id segmentation is part of the first
reference executor. `albedo_rgb` is included because it is likely needed by the
first direct-light executor later. PBR fields such as roughness, metallic, IOR,
spectral emission, and texture maps should wait until an executor requires them.

Draft minimal light schema:

```python
@dataclass(frozen=True)
class OpticalLightSpec:
    light_id: str
    kind: Literal["point", "directional"]
    position_or_direction_world: object
    intensity: float = 1.0
    color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    enabled: bool = True
```

Lights are not required for the first reference executor, but the schema should
be small enough to land early without committing to a full PBR model.

### OpticalSceneCache

Runtime cache that makes the registry executable:

- CPU geometry arrays / BVH
- GPU geometry buffers / acceleration structures
- material and light buffers
- mapping from physics shape/body ids to optical instances
- dirty flags and update policy

This is where the large gap lives. It answers:

```text
Given frame N transforms, what exactly will the optical executor trace/rasterize?
```

Review follow-up decision: rigid-body `PublishedFrame.X_world` is only the
Phase-A producer. The scene/cache design must not assume every optical instance
is driven by a rigid body transform.

Longer-term optical scene construction should consume frame-aligned producer
streams:

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

The split by producer type is:

- rigid bodies: registry owns geometry and per-instance local offsets; frame
  owns body transforms; scene/cache composes `X_world_geometry`;
- cloth: registry owns identity/material/roles; frame owns dynamic vertex
  positions and possibly normals; topology is fixed until a topology version
  changes;
- soft bodies: registry owns identity/material/roles; frame owns current
  surface mesh or extracted boundary representation;
- fluids: registry owns fluid identity/material/medium semantics; frame owns
  particles, level-set, volume field, or a solver-published surface mesh.

Scene/cache may perform executable geometry preparation, but not
sensor-specific optical computation:

- allowed: dynamic vertex/particle/volume buffer updates;
- allowed: fixed-topology BVH refit or topology-change rebuild;
- allowed: particle/surface/volume buffer packing;
- allowed: optional geometry realization plugins such as fluid surface
  extraction when they are sensor-independent;
- forbidden: ray traversal, shading, RGB/depth/intensity generation, sensor
  noise/response, or result-to-reading conversion.

This implies `OpticalInstanceSpec` should eventually carry a binding/source kind
rather than only `body_index`:

```text
world_static
rigid_body
deformable_mesh
particle_set
surface_mesh
volume_field
procedural
```

Phase A keeps the current rigid-body/world-static subset. Review follow-up
decision: introduce `OpticalFrameInputs` as the `OpticalSceneCache` aggregate
input before registry-builder work. `snapshot_from_published_frame(...)` remains
a convenience wrapper that builds `OpticalFrameInputs(rigid=frame)`.

### OpticalSceneSnapshot

Frame-aligned, immutable view passed to an executor:

- frame id / sim time / env ids
- references to stable geometry/material/light buffers
- transform buffer for the selected frame
- optional ready event / dependency event

This should be lightweight. It should not own a Python-level
`GpuPublishedFrame` borrow lease. If it consumes `GpuPublishedFrame` buffers,
slot reclaim must be governed by Q52 device-consumer completion semantics.

Review follow-up decision: `OpticalSceneSnapshot` should not hold a Python-level
`BorrowedFrameLease` for `GpuPublishedFrame`.

Instead, the optical runtime should use the Q52 consumer protocol:

```text
1. borrow/observe frame N long enough to capture or alias the needed buffers
2. launch optical work with the appropriate stream/event dependencies
3. complete the optical device consumer when optical work has finished
4. let Q52 reclaim the publish slot from the consumer completion event
```

This keeps slot lifetime under the publish ring's control instead of tying it to
Python object lifetime. It also handles optical workloads that are slower than a
single physics step, which is expected for heavier Embree/OptiX/Mitsuba-style
backends.

### OpticalExecutor

Backend-specific executor. This is the first-class answer to "how do we compute
the optical scene after obtaining an `OpticalSceneSnapshot`?":

```python
class OpticalExecutor(Protocol):
    def execute(self, snapshot: OpticalSceneSnapshot, spec: OpticalSensorSpec) -> OpticalComputeResult:
        ...
```

Concrete executors may include:

- in-repo CPU reference executor;
- Embree-backed CPU ray/intersection executor;
- Warp/CUDA custom executor for simple sensor kernels;
- OptiX-backed NVIDIA GPU ray tracing executor;
- Mitsuba-backed offline/high-fidelity executor;
- external renderer bridge.

The simulator owns the `OpticalExecutor` contract and result lifecycle. Third
party renderers or ray tracing libraries are backend implementations, not the
architectural source of truth.

Executor responsibilities:

- validate that `snapshot` and `spec` refer to the same frame/env selection;
- execute sensor/query-specific optical work such as ray traversal,
  rasterization, shading, segmentation, or sensor response;
- allocate or write result buffers according to the backend policy;
- return an `OpticalComputeResult` with explicit location and readiness;
- preserve registry/snapshot identity in result channels such as material,
  instance, and semantic ids.

Executor non-responsibilities:

- do not mutate `OpticalWorldRegistry`;
- do not fetch a new `PublishedFrame` from an engine;
- do not bind model assets or create canonical instance identity;
- do not turn every output into a host-side sensing reading;
- do not use Rerun or any visualization sink as part of computation.

Clean boundary:

```text
OpticalSceneSnapshot answers:
  what exists at frame N, where it is, and how it is encoded.

OpticalSensorSpec answers:
  what question a sensor/query asks.

OpticalExecutor answers:
  what optical result that question produces against that scene.
```

### OpticalComputeResult

Execution result with explicit location and readiness:

- frame id / sim time / env id
- sensor id
- channels: color, depth, segmentation, intensity, ranges, debug rays, etc.
- result location: `host`, `device`, or `external`
- optional ready event/fence
- optional conversion helper to `sensing` readings

Do not assume every optical result is a host NumPy reading. That would break the
future RL hot path.

Canonical channel names should be stable across executors. An executor does not
need to return every channel, but if it returns one of these names, it must use
the shared semantics:

- `hit_mask`: boolean hit validity. Misses are `False`.
- `depth_m`: metric projected depth along a sensor optical axis, when that
  sensor model defines one. Misses are `np.inf`.
- `range_m`: metric range when a range-sensor convention differs from camera
  depth. For generic ray specs, this is the first-hit ray parameter along the
  normalized query direction. Misses are `np.inf`.
- `position_world`: world-frame hit points. Miss rows are NaN.
- `normal_world`: world-frame surface normals. Miss rows are NaN.
- `instance_id`: stable human-readable registry instance id. Misses are `None`.
- `numeric_instance_id`: stable numeric instance id assigned by the registry or
  registry-derived packing step. Misses use a documented background id.
- `material_id`: stable human-readable material id. Misses are `None`.
- `numeric_material_id`: stable numeric material id assigned by the registry or
  registry-derived packing step. Misses use a documented background id.
- `semantic_id`: stable semantic label/id when configured. Misses use a
  documented background id.
- `rgb`: rendered RGB or grayscale-expanded RGB output after the executor's
  optical/sensor model.
- `intensity`: scalar optical intensity/irradiance/return strength when the
  executor defines that sensor model.

Miss semantics are part of the contract. Reference and accelerated backends must
agree on hit validity, infinity/NaN fill values, and id background handling.

## 7. Workflow After Physics Step

Concrete frame lifecycle:

```text
1. GpuEngine/CpuEngine completes rigid-body step N.
2. Engine publishes PublishedFrame N.
3. Optical runtime borrows or snapshots frame N according to policy.
4. OpticalSceneCache updates dynamic transforms from PublishedFrame N.
5. Cache applies dirty updates for changed materials/lights/geometry.
6. Cache creates OpticalSceneSnapshot N.
7. OpticalExecutor runs sensor workloads for selected envs/sensors.
8. Executor returns OpticalComputeResult with host/device readiness.
9. Consumers convert/log/observe results.
10. Device/host consumers mark completion so publish slots can be reclaimed.
```

For GPU execution, step 7 should use device-timeline waits/events rather than
host synchronization where possible, matching the Q52 device-consumer design.

## 8. Execution Backend Decision

The recommended decision is:

```text
Own the optical execution contract.
Do not own a full renderer yet.
Make execution backends pluggable.
Start with a tiny in-repo reference executor.
```

In other words:

```text
OpticalSceneSnapshot
  -> OpticalExecutor.execute(sensor_spec)
  -> OpticalComputeResult
  -> sensing reading / RL obs / Rerun log / exporter
```

### Why Not Use Rerun As The Executor?

Rerun is the wrong abstraction for computation. It records and visualizes data
such as images, tensors, point clouds, transforms, meshes, and scalar timelines.
That makes it excellent for inspection, training monitor output, and regression
artifacts.

It does not own:

- optical material semantics;
- light transport;
- acceleration structures;
- per-frame device scheduling;
- result-buffer lifecycle;
- RL hot-path tensor ownership.

Using Rerun as the "renderer" would hide the real unsolved problem: how the
simulator turns physics state plus optical assets into an executable scene and a
well-owned result buffer. Rerun should consume `OpticalComputeResult`, not
produce it.

### Why Not Immediately Bind To A Full Renderer?

Binding Q54 directly to a full renderer or simulator scene graph would solve
the first demo but risk locking our core contracts to someone else's lifecycle:

- scene update model;
- material schema;
- sensor output format;
- threading and synchronization;
- host/device ownership;
- multi-env batching assumptions;
- result timing and backpressure.

Those are exactly the contracts Q52/Q53 have been carefully making explicit.
The optical layer should preserve that direction: our simulator defines what a
frame-aligned optical computation means, then adapters map that contract onto
specific backends.

### Why Not Write Full Ray Tracing Ourselves?

Writing a complete physically based renderer is the wrong near-term scope.
Reflection, refraction, spectral transport, denoising, sampling, texture
filtering, MIS, and acceleration-structure maintenance are all deep subsystems.

However, writing a tiny reference executor is useful:

- it proves `OpticalSceneSnapshot -> OpticalComputeResult`;
- it gives deterministic unit tests;
- it validates material/light/sensor schemas without a heavyweight dependency;
- it creates a baseline for future Embree/OptiX/Mitsuba adapters;
- it can stay deliberately limited to first-hit depth and material-id
  segmentation.

The split should be:

```text
In repo:
  execution interface
  result lifecycle
  scene synchronization/cache contract
  minimal reference executor

External or optional:
  high-performance BVH traversal
  GPU ray tracing
  path tracing / spectral / differentiable rendering
```

### Backend Options

#### In-Repo Reference Executor

Purpose:

- prove contracts;
- support CPU unit tests;
- provide simple debug output;
- avoid optional dependencies in the first implementation.

Recommended first capabilities:

- first-hit depth/range against simple planes/triangles;
- material id or instance id segmentation.

Not in scope:

- physically based path tracing;
- recursive reflection/refraction;
- direct lighting / shading in the first executor;
- texture filtering;
- production image quality.

This should be the first executor because it gives us a stable test harness for
the architecture before integrating a serious ray tracing backend.

Review follow-up decision: the first reference executor should return first-hit
depth plus material-id segmentation, and should not do direct-light intensity.

Rationale:

- depth validates geometry, transform update, and acceleration/query plumbing;
- material id validates `OpticalWorldRegistry` material bindings;
- the combination exercises `OpticalSceneSnapshot -> OpticalComputeResult ->
  sensing/debug consumer` without introducing light transport or shading;
- direct-light intensity requires light registry and shading semantics, so it is
  the right next step after the contract is proven.

#### Embree Executor

Embree is a good candidate for CPU ray/intersection acceleration. It is a ray
tracing kernel library rather than a whole simulation stack, so it fits the
adapter model well:

```text
OpticalSceneCache CPU geometry -> Embree scene/BVH
OpticalSensorSpec rays         -> Embree ray queries
Embree hits                    -> OpticalComputeResult
```

It would be a strong phase-B backend for CPU LiDAR/depth/segmentation once mesh
geometry and material-id plumbing are stable.

Tradeoffs:

- optional dependency and platform packaging work;
- CPU path only for our likely RL hot path;
- still requires us to own material/light/result contracts.

#### OptiX Executor

OptiX is the likely high-performance NVIDIA GPU ray tracing path. It fits future
CUDA/Warp migration because it is CUDA-centric and designed for programmable ray
tracing on NVIDIA GPUs.

It should not be the first implementation because it would force early decisions
about:

- CUDA module/build integration;
- device memory ownership;
- acceleration-structure build/refit;
- stream/event interop with Warp and Q52 device consumers;
- shader binding tables and material program layout.

It is a good long-term executor once the CPU contract, result lifecycle, and
device consumer semantics are stable.

#### Mitsuba Executor

Mitsuba is better treated as an offline/high-fidelity/research adapter:

- physically based rendering;
- spectral/differentiable rendering use cases;
- correctness experiments and reference images.

It is probably not the right first backend for realtime RL sensing because its
scene and rendering abstractions are intentionally rich. That richness is useful
for research but heavy for the simulator hot path.

#### Warp/CUDA Custom Executor

A custom Warp/CUDA executor may be appropriate for simple sensors:

- first-hit depth against simplified geometry;
- height-field or plane queries;
- material-id segmentation;
- learned-sensor synthetic channels.

This is not a replacement for OptiX when full triangle-scene ray tracing is
needed. It is useful when the sensor math is simple and tightly coupled to the
existing GPU physics pipeline.

### Recommended Phasing

Phase A:

- own `OpticalExecutor` and `OpticalComputeResult`;
- implement a tiny in-repo CPU reference executor;
- log results to Rerun only after computation;
- keep all output host-side unless a device result is explicitly part of the
  test.
- compute first-hit `range_m`, `hit_mask`, `position_world`, `normal_world`,
  `material_id`, and `instance_id`;
- include registry-owned `numeric_instance_id`;
- filter instances by the sensor spec's minimal role field;
- reject frame/env mismatch between snapshot and spec;
- do not compute direct light intensity or camera-style projected `depth_m`.

Phase B:

- split executor internals into explicit steps:
  `validate(snapshot, spec)`, `prepare_workload(spec)`,
  `intersect(snapshot, workload)`, `resolve_channels(hits, snapshot)`, and
  `build_result(...)`;
- keep the public `execute(snapshot, spec) -> OpticalComputeResult` contract
  unchanged;
- add capability/schema tests so future backends must preserve result semantics.

2026-04-30 status: implemented for `CpuReferenceOpticalExecutor`.
`execute(...)` now delegates to `_validate`, `_prepare_workload`, `_intersect`,
`_resolve_channels`, and `_build_result`. `capabilities` declares returned
channels, and schema tests cover shape, dtype, and miss-value contracts.

Phase C:

- add Embree or a simple accelerated CPU backend for mesh ray queries;
- keep the same executor/result contract;
- add material-id/depth/segmentation coverage.

Phase D:

- add direct-light or RGB executors only after first-hit/result contracts are
  stable;
- consume light/material semantics from the registry/snapshot;
- define whether shadow rays, occlusion, exposure, and sensor response are in
  scope for each executor capability.

Phase E:

- add device result buffers and Q52-style device consumer completion;
- evaluate Warp/CUDA custom kernels for simple sensors;
- evaluate OptiX for real GPU ray tracing.

Phase F:

- add Mitsuba or another offline renderer adapter only if we need
  high-fidelity/reference optical output or differentiable rendering.

This gives us a clean answer: the simulator owns orchestration, frame alignment,
results, and synchronization; specialized libraries own expensive intersection
or rendering kernels when they become worthwhile.

Executor capability declarations should be added before multiple non-reference
backends exist. A backend may support only a subset, for example:

```text
{"range_m", "hit_mask", "material_id", "instance_id"}
{"depth_m", "rgb", "semantic_id"}
{"range_m", "intensity"}
```

Capability names must align with result channel names or documented sensor
families. Capability declarations are for routing and validation; they do not
replace the runtime result contract.

## 9. Rerun Path

Rerun should be treated as visualization and logging:

```text
OpticalComputeResult -> RerunBackend / future optical debug logger
RenderScene          -> RerunBackend
```

Rerun can consume:

- camera images
- depth images or point clouds
- segmentation previews
- light gizmos
- material/mesh debug views
- rays and hit points
- scalar timing/lag statistics

Rerun should not be the source of truth for:

- material semantics
- light transport
- acceleration structures
- sensor execution ordering
- device buffer ownership

If we use Rerun before a real optical executor exists, it should be framed as a
debug display of registered optical objects, not as proof that optical physics is
implemented.

## 10. Own Renderer Path

If we build our own renderer/executor, the minimum viable path is:

1. Define optical geometry/material/light registry.
2. Build CPU `OpticalSceneCache` for simple meshes/planes.
3. Implement one executor:
   - first-hit depth/range plus material-id segmentation.
4. Return `OpticalComputeResult`.
5. Add Rerun logging only after results exist.
6. Add GPU cache/executor once CPU contract is stable.

The renderer path must make these policies explicit:

- static geometry vs dynamic transform updates;
- per-frame transform buffer ownership;
- material/light dirty updates;
- host/device result location;
- result completion events;
- ring backpressure when optical consumers stall.

## 11. Relationship To Other Projects

This architecture should borrow the separation patterns, not blindly copy API
surface:

- Drake-like split: scene graph / roles / renderer are separate from the physics
  integrator. This supports the idea that `PublishedFrame` should not become the
  optical scene.
- MuJoCo-like split: model assets include cameras/lights/materials, renderer
  consumes model + data after stepping. This supports frame-state plus static
  asset composition, but MuJoCo's renderer is not the same as our sensing/RL
  device path.
- Isaac/Omniverse-like split: USD scene + PhysX + RTX sensors. This supports a
  dedicated scene graph and renderer/sensor stack, but is a much larger system
  than this repository should copy wholesale.
- Rerun-like split: logging/visualization archetypes are a sink for data we
  already computed, not an optical transport engine.

The common pattern is:

```text
physics state != render/optical scene != sensor result
```

Our current code has the first and part of the debug scene. It does not yet have
the executable optical scene.

## 12. Design Risks

1. **Stuffing optical state into PublishedFrame**
   - Makes the physics publish ring own materials/lights/render resources.
   - Couples slot lifetime to unrelated optical asset lifetime.
   - Makes non-optical consumers pay complexity.

2. **Using RenderScene as the optical scene**
   - `RenderScene` is intentionally lossy and debug-oriented.
   - It lacks material/light/sensor/execution semantics.
   - It would turn backend visualization data into canonical simulation state.

3. **Returning only host-side readings**
   - Works for debug tests.
   - Fails for RL hot paths that need device tensors.

4. **Skipping cache/update semantics**
   - Rebuilding all geometry every frame is simple but will not scale.
   - Not defining dirty flags now will make future GPU acceleration structures
     awkward to retrofit.

5. **Confusing geometric ray query with optical transport**
   - Surface hit queries are necessary but not sufficient.
   - Reflection/refraction/shading require material/light/medium state and an
     executor with more semantics.

6. **Letting a third-party renderer define simulator contracts**
   - Useful backends have their own scene graph, material model, scheduling, and
     result conventions.
   - If adopted too early, those conventions can leak into `sensing/`, Q52
     publish semantics, or RL observation ownership.
   - Keep third-party integrations behind `OpticalExecutor` adapters.

7. **Overbuilding a renderer before contracts are stable**
   - A full renderer would consume a lot of work before we know the right result
     and synchronization contracts.
   - A tiny reference executor gives stronger architectural feedback sooner.

8. **Leaving multi-env batching semantics undefined**
   - `GpuPublishedFrame` naturally carries batched per-env physics state.
   - Optical workloads are often per-sensor per-env, and sensor specs may differ
     between envs because of domain randomization.
   - Phase A can use one-env CPU snapshots, but Phase C must decide whether an
     `OpticalSceneSnapshot` contains all envs, a selected env subset, or exactly
     one env.
   - Executors also need a policy for per-env sensor-spec variation: batched
     homogeneous specs, grouped specs, or one dispatch per env/spec.
   - This should be resolved before GPU optical execution or RL hot-path image
     observations.

9. **Assuming all optical geometry is rigid**
   - Rigid bodies update instance transforms, but cloth/soft bodies update
     vertex buffers and fluids may update particles, volumes, or generated
     surfaces.
   - If `OpticalSceneCache` only accepts `PublishedFrame.X_world`, future
     deformable/fluid pipelines will either bypass Q54 or overload rigid-body
     concepts.
   - The cache should move toward frame-aligned producer streams and
     binding/source kinds while preserving the same executor/result contract.
   - Scene/cache can prepare executable geometry and acceleration structures,
     but must not perform per-sensor optical computation.

## 13. Suggested Implementation Order

Do not start with photorealistic rendering. Start by closing contracts.

1. **Q54 decision doc**
   - accept layer split and package name;
   - define which layer owns optical registry/cache/results.

2. **Optical registry skeleton**
   - material spec;
   - light spec;
   - geometry binding from physics shape/body ids.

3. **Optical scene snapshot/cache skeleton**
   - CPU-only first;
   - frame-aligned transform update from `PublishedFrame`;
   - no heavy renderer yet.

4. **Minimal executor**
   - direct geometry visibility/depth with material id segmentation;
   - return `OpticalComputeResult`.

5. **Rerun debug sink**
   - log lights/material ids/sensor outputs after results exist.

6. **Device path**
   - device result buffers;
   - Q52-style device consumer completion;
   - GPU acceleration structures and stream/event contracts.

## 14. Review Questions For Claude

1. Does the `optics/` package name and dependency direction look right now that
   Q54 is broader than camera/render-backed sensing?

2. Does keeping optical sensor specs in `sensing/` and passing them into
   `OpticalExecutor.execute(snapshot, spec)` preserve the right lifecycle
   separation?

3. Is first-hit depth + material-id segmentation the right first reference
   executor, with direct-light intensity deferred?

4. Does the Q52 device-consumer event model cover optical snapshot lifetime
   without holding a Python-level `GpuPublishedFrame` borrow lease?

5. Are the draft minimal material/light schemas small enough, or should even
   `albedo_rgb` / point-directional lights wait?

6. Does the backend phasing make sense: in-repo reference executor first, Embree
   for CPU ray acceleration, OptiX for future NVIDIA GPU ray tracing, Mitsuba
   only for offline/high-fidelity reference work?

7. What multi-env batching semantics should Phase C target: one snapshot for all
   envs, selected env subsets, or one snapshot per env?

## 15. Phase A Implementation Status

2026-04-30 update: the first CPU-only contract skeleton has landed.

Implemented:

- `sensing.OpticalRaySensorSpec`
  - sensor-side ray batch spec;
  - normalized world-frame directions;
  - optional `ray_shape` metadata for image-shaped producers;
  - kept in `sensing/`, not `optics/`.
- `sensing.OpticalPinholeCameraSpec`
  - OpenCV-style pinhole camera query;
  - lowers to `OpticalRaySensorSpec` through `build_pinhole_camera_rays(...)`;
  - keeps camera intrinsics/extrinsics/resolution in `sensing/`.
- `sensing.OpticalCameraImageResult`
  - image-shaped camera postprocess result;
  - reshapes flat executor channels;
  - adds projected `depth_m` from `range_m` and camera optical axis.
- `optics.OpticalWorldRegistry`
  - material specs;
  - minimal point/directional light specs;
  - plane and triangle-mesh geometry handles;
  - `OpticalInstanceSpec` records with optional physics `body_index`;
  - registry-owned stable `numeric_instance_id`;
  - minimal instance roles for sensor filtering.
- `optics.OpticalSceneCache`
  - CPU-only `OpticalFrameInputs` snapshot path;
  - `snapshot_from_published_frame(...)` retained as a Phase-A rigid wrapper;
  - composes body-bound geometry from `PublishedFrame.X_world`;
  - explicitly rejects multi-env CPU selection for Phase A.
- `optics.OpticalFrameInputs`
  - aggregate scene input for frame-aligned producer streams;
  - Phase A carries only `rigid: CpuPublishedFrame`;
  - validates rigid frame id / sim time alignment.
- `optics.build_optical_registry_from_robot_model(...)`
  - Phase-A `collision_only` registry builder for `RobotModel.geometries`;
  - returns `OpticalBindingBuildResult`;
  - records `OpticalSourceKey` provenance maps;
  - emits diagnostics for unsupported shapes instead of silently
    approximating them.
- `optics.OpticalSceneSnapshot`
  - frame id / sim time / env idx;
  - immutable tuple of executable optical instances;
  - optional scene/cache-owned `OpticalSceneAcceleration`;
  - no Python-level `GpuPublishedFrame` borrow lease.
- `optics.OpticalSceneAcceleration`
  - first L2 payload is `kind="cpu_bvh"`;
  - world-space packed triangles;
  - source-order keys for reference-parity tie-breaks;
  - `CpuBvhNode` list and compact leaf primitive indices.
- `optics.CpuReferenceOpticalExecutor`
  - first-hit range;
  - material-id segmentation;
  - instance-id segmentation;
  - numeric instance-id segmentation;
  - hit position/normal channels;
  - internal Phase-B split and channel schema tests;
  - no direct-light intensity.
- `optics.CpuBvhOpticalExecutor`
  - consumes snapshot-owned CPU BVH acceleration;
  - preserves the reference executor result schema;
  - runs analytical infinite-plane side pass;
  - raises `MissingAccelerationError` instead of silently building acceleration.
- `optics.OpticalComputeResult`
  - host result channels;
  - location/readiness fields reserved for device/external backends.

Tests:

```text
PYTHONPATH=. pytest tests/unit/optics tests/unit/sensing -q
66 passed
```

Still deferred:

- non-pinhole sensor pose/ray-pattern builders;
- visual-preferred / explicit optical asset registry builders;
- Rerun optical result sink;
- Embree adapter;
- BVH refit / SAH / role-specific BVHs;
- GPU/device result buffers and Q52 device-consumer integration;
- Phase C multi-env batching semantics.

## 16. Current Conclusion

The current architecture is directionally correct and Phase A has started:

- Q52 gives a publish/control-plane that can protect host/device consumers.
- Q53 gives a narrow geometric query boundary.
- Q50/Rerun gives debug visualization.
- Q54 now has a minimal `optics/` contract skeleton and deterministic reference
  executor.

The unresolved gap is Q54:

```text
How do we turn a published physics frame plus optical assets into an executable,
frame-aligned optical scene, and how do optical results flow to sensors, RL, and
debug visualization without forcing host synchronization or polluting physics?
```

That is the next design problem. It should be solved before implementing
reflection/refraction, camera RGB, or physically meaningful light/material
interaction.

The current execution-backend recommendation is:

```text
Define and own the simulator-side optical execution contract.
Start with a tiny in-repo reference executor.
Treat Embree / OptiX / Mitsuba as optional adapters behind that contract.
Treat Rerun as a result/debug sink, not as the optical executor.
```

## 17. Self-Authored Algorithm Review Rule

2026-04-30 end-of-day note: for self-authored optical algorithms, especially
the planned L2 in-repo BVH, implementation should not begin until the algorithm
has been written out and reviewed.

Each such step should describe:

```text
inputs and outputs
data layout
core algorithm
edge cases and degeneracy behavior
complexity expectations
test strategy
explicit non-goals
```

This applies to BVH construction/traversal, future shadow-ray direct lighting,
and any later in-repo GPU/simple optical kernels. Adapter work for Embree,
OptiX, or Mitsuba can be reviewed mainly through interface and lifecycle
contracts because those algorithms are owned by the external backend.
