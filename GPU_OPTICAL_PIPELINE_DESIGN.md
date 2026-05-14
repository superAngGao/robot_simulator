Initiative: q54-gpu-optical-pipeline-architecture
Document Type: repo-design-doc
Stage: design-baseline
Author: codex
Version: v1
Date: 2026-05-08
Last Updated: 2026-05-14
Status: active-design-baseline
Related Files: optics/execution.py, optics/device.py, optics/device_scene.py, optics/device_bvh.py, optics/cuda_lbvh.py, optics/warp_execution.py, optics/render_api.py, tools/optical_pipeline_lab/, examples/mujoco_menagerie_gpu_preview.py, collab/q54-gpu-optical-output-profile-api__review-request__codex__v1.md, collab/q54-gpu-optical-readback-delivery-policy__review-request__codex__v2.md
Owner Summary: This is the repo-level architecture design for the GPU optical/rendering pipeline. It reframes recent performance work around explicit use scenarios, a shared stage pipeline, per-scenario overlap policies, CPU/GPU boundary responsibilities, CUDA-first performance direction, the Optical Pipeline Lab, a canonical lab render source entrypoint, and a staged roadmap. Collab files remain review/discussion artifacts; this document is the durable design baseline.

# GPU Optical Pipeline Design

## 1. Purpose

The GPU optical/rendering work has reached the point where local optimizations
are no longer enough. We have optimized one scenario deeply, but the system now
needs a durable architecture that separates:

```text
algorithm requirements
runtime delivery semantics
benchmark tooling
consumer-facing API
backend implementation choices
```

This document consolidates the current understanding and should guide the next
phase of delivery.

The key shift:

```text
Do not optimize "rendering" as one generic task.
Optimize explicit pipeline scenarios with explicit delivery policy.
```

2026-05-14 update:

```text
Go2 is not the render pipeline.
Go2 is one scene/source that feeds the render pipeline.
```

The active near-term plan is therefore to make the Optical Pipeline Lab render
foundation source-driven:

```text
external world / scene source
  -> OpticalLabRenderSource
  -> OpticalLabRenderPipeline
  -> RenderFrameContext
  -> RenderResult
  -> delivery/video/reporting as separate layers
```

Physics simulation, Menagerie Go2, and synthetic dynamic smokes should converge
on this source vocabulary instead of each requiring a backend-shaped adapter.

## 2. Current State In One Sentence

The most optimized path today is:

```text
VIDEO_ORDERED_EXPORT:
  static Go2 scene
  CUDA LBVH
  GPU camera raygen
  Warp first-hit + direct-light/shadow render
  ordered host RGB readback
  no dropped frames
```

This is not the same scenario as:

```text
real-time latest preview
sensor lossless observations
parity/debug
device-only render benchmark
dynamic-geometry sensor/video
future path tracing
```

Those scenarios should share pipeline mechanisms, but they should not share one
fixed overlap strategy.

## 3. Ground Rules

### 3.1 Shared Mechanism, Per-Scenario Policy

The total pipeline should share a common stage substrate:

```text
snapshot stage
acceleration build/refit/reuse stage
render stage
delivery/readback stage
consume/export stage
```

Each stage should expose:

```text
input event
output event
resource owner
queue/ring capacity
backpressure behavior
drop legality
stats
```

But the scenarios choose different policies:

```text
PARITY_DEBUG:
  sync every stage, no overlap, no drop

VIDEO_ORDERED_EXPORT:
  overlap render/readback, no drop, ordered consumption, backpressure allowed

REALTIME_PREVIEW:
  overlap aggressively, latest completed frame, drop allowed

SENSOR_ORDERED:
  overlap allowed, no drop, frame-aligned ack/backpressure

RENDER_BENCH:
  device-only, explicit timing sync, no host delivery
```

Design principle:

```text
share overlap capability
do not share one overlap policy
```

### 3.2 Consumer API And Development Logic Are Different

User-facing API should be consumer-first:

```text
What is the caller trying to consume?
debug result?
complete video?
latest preview?
lossless sensor observation?
render benchmark?
```

Implementation should be delivery-first:

```text
Where do events wait?
Who owns device buffers?
When can host slots be reused?
What happens when the ring fills?
Can frames be dropped?
```

Consumer modes are a public interface layer. They should not drive low-level
development sequencing. During development, delivery primitives should be
implemented and benchmarked directly.

### 3.3 CUDA-First For Performance-Critical New Work

Performance-critical new GPU work should prefer CUDA when the implementation
surface is clear.

Recommended backend roles:

```text
Warp:
  reference/compatibility backend
  quick correctness iteration
  existing parity path

CUDA:
  performance backend for hot paths
  LBVH build/refit
  async D2H / RGB pack
  future fused raygen/trace/shade kernels

OptiX:
  future ray acceleration backend
  compare after Q54 CUDA/Warp payload and lifecycle contracts stabilize
```

CUDA is not technically the hard part. The hard parts are:

```text
extension build/cache/distribution
stream ownership
device pointer ownership
allocator interaction with Warp/Torch
test fallback behavior
resource lifetime
```

Therefore the direction should be CUDA-first, but not "rewrite everything at
once."

### 3.4 Example Scripts Are Not The Runtime

`examples/mujoco_menagerie_gpu_preview.py` currently carries too much:

```text
demo rendering
benchmarking
GPU raygen
output profile selection
readback mode selection
render profiling
future delivery experiments
```

This is acceptable as an experimental harness, but it must not become the
formal runtime boundary. The architecture should eventually separate:

```text
runtime/session code
benchmark harness
preview/demo examples
test/parity helpers
```

### 3.5 External Renderer Support Must Be Adapter-Based

The core optical pipeline must remain renderer-toolkit neutral.

External renderers and ray tracing toolkits should be added incrementally
through adapters, not by changing core pipeline semantics for one backend.

Hard rule:

```text
core pipeline contracts stay stable
external toolkits plug in through adapters
adapters translate at the boundary
adapters do not redefine source-order, role-mask, frame lifetime, or delivery semantics
```

The intended extension model:

```text
Scene export adapter:
  snapshot/registry/camera/material/light -> external scene format or API

Reference renderer adapter:
  snapshot + camera + requested outputs -> OpticalComputeResult-like host result

Runtime backend adapter:
  snapshot + accel/camera/output profile -> DeviceFrameResult-compatible result
```

The first two are near-term realistic. Runtime backend replacement is a
longer-term integration target and must preserve the same device/result/delivery
contracts as internal renderers.

Adapter compatibility levels:

```text
Level 1: visual export only
Level 2: RGB/depth reference render
Level 3: semantic parity render
Level 4: runtime backend
```

Useful future adapter targets:

```text
PBRT / Mitsuba:
  offline reference rendering and path tracing validation

Blender / Cycles:
  asset inspection, documentation images, high-quality offline previews

OpenUSD / Hydra:
  scene interchange and future render-delegate ecosystem bridge

Embree / TinyBVH:
  acceleration structure quality and traversal benchmark references
```

Adapter support is a mandatory extensibility requirement. It is not required
for the next implementation slice, but the interfaces below must avoid blocking
it.

## 4. Community Naming Principles

Avoid inventing names where well-known ecosystem terms exist.

Useful terminology from adjacent systems:

```text
FIFO / ordered:
  queue-preserving, no dropped frames

Mailbox / latest:
  keep the newest frame, old frames can be replaced or ignored

Reliable:
  delivery is expected, backpressure is preferable to silent loss

Best effort:
  delivery may be skipped or dropped under pressure

Blocking:
  caller waits synchronously

Device-only:
  no host delivery
```

Suggested naming:

```text
Ordered / FIFO:
  video ordered export, sensor ordered delivery

Latest / Mailbox:
  realtime preview

Blocking:
  parity/debug sync path

DeviceOnly:
  render benchmark

Reliable / Lossless:
  sensor semantics only, with documentation that "lossless" means no frame
  loss, not image compression
```

Avoid using "real-time" as a core policy name because it can mean either:

```text
low latency
high throughput
fixed deadline
```

Those are different policies.

References for naming inspiration:

```text
Vulkan present modes:
  FIFO / MAILBOX / IMMEDIATE

ROS 2 QoS:
  reliable / best effort / keep last / keep all

GStreamer queues/appsink:
  leaky queues, max buffers, drop behavior
```

## 5. Total Pipeline Skeleton

All scenarios should be described against the same stage skeleton:

```text
Simulation / Sensor Request
  |
  v
Frame State
  |
  v
Device Scene Snapshot
  |
  v
Acceleration State
  |
  v
Render Execution
  |
  v
Device Result
  |
  v
Delivery Runtime
  |
  v
Consumer Adapter
```

Expanded:

```text
physics frame / static frame / camera request
  |
  | produces frame_id, sim_time, transforms
  v
device scene snapshot
  |
  | world triangle v0/e1/e2/normal
  | optional triangle AABB
  | ready_event
  v
acceleration
  |
  | reuse / refit / rebuild
  | BVH / LBVH / future OptiX handle
  | ready_event
  v
render
  |
  | raygen
  | first-hit
  | direct light / shadow
  | future path tracing
  | ready_event
  v
device result
  |
  | channels
  | output_profile
  | resources
  v
delivery
  |
  | device-only
  | sync blocking readback
  | async ordered
  | async latest
  v
consumer
  |
  | assert/debug
  | encode/write
  | display
  | sensor publish
  | benchmark accounting
```

## 6. Scenario Diagrams

Each scenario gets its own state/dataflow diagram. These diagrams are not
decorative; they define sync, release, drop, and backpressure semantics.

### 6.1 PARITY_DEBUG

Use case:

```text
tests
single-frame debug
CPU/GPU parity
full channel inspection
```

Policy:

```text
sync
no drop
no overlap
full host result
errors surface immediately
```

Diagram:

```text
Frame_i
  -> Snapshot_i
      wait snapshot_ready
  -> Accel_i
      wait accel_ready
  -> Render_i
      wait render_ready
  -> BlockingReadback_i
      wait/copy/materialize host result
  -> DebugConsumer_i
      assert / compare / inspect
```

Release:

```text
device resources can release after blocking readback completes
host arrays are owned by the returned host result
```

Implementation:

```text
stage_optical_compute_result_to_host(...)
stage_optical_channels(...)
CPU reference executor
GPU executor with full output profiles
```

### 6.2 RENDER_BENCH

Use case:

```text
measure device render cost
kernel regression tracking
benchmark render-only path
```

Policy:

```text
device-only
no host readback
explicit event wait for timing
no consumer frame result
```

Diagram:

```text
Frame_i
  -> Snapshot_i
  -> Accel_i
  -> Render_i
      record render_ready
      synchronize only for timing
  -> BenchmarkStats_i
      consumer = benchmark accounting
```

Release:

```text
device result can be released after timing sync
no host slot exists
```

Implementation:

```text
output_profile = render_only
readback_payload = none
delivery = device_only
```

### 6.3 VIDEO_ORDERED_EXPORT

Use case:

```text
offline video generation
video benchmark
complete ordered frame sequence
no frame drop
```

Current focus:

```text
This is the scenario we have optimized most so far.
```

Current measured subcase:

```text
static Go2 scene
960x640
CUDA LBVH
GPU camera raygen
direct light + shadows
selected RGB readback
no dropped frames
```

Policy:

```text
ordered
no drop
readback/export can lag
backpressure when ring is full
render/readback should overlap in steady state
```

Blocking current path:

```text
Render_i
  wait render_i
Readback_i
  wait readback_i
Encode_i / Account_i
Render_i+1
```

Target async ordered path:

```text
render stream:
  Render_i ---------------- Render_i+1 ---------------- Render_i+2

copy stream:
           wait Render_i -> Copy_i
                              wait Render_i+1 -> Copy_i+1

host/export:
                         Consume_i
                                               Consume_i+1
```

Release:

```text
DeviceFrameResult resources retained by ReadbackJob until copy_ready_event
ReadbackSlot retained until CompletedFrame borrow exits
```

Backpressure:

```text
if ring full:
  block on oldest pending ordered frame
  do not drop
```

Current measurements:

```text
Go2 960x640 cuda:1:

shadows, readback=none:
  render_execute_mean ~= 2.58 ms
  frame_total_p50     ~= 2.75 ms

no shadows, readback=none:
  render_execute_mean ~= 1.23 ms
  frame_total_p50     ~= 1.31 ms

shadows, readback=rgb:
  render_execute_mean ~= 2.61 ms
  readback_host_mean  ~= 4.01 ms
  frame_total_p50     ~= 6.64 ms
```

Expected next target:

```text
async ordered RGB:
  frame_total should approach max(render, readback) ~= 4.0 ms

async ordered RGB8:
  frame_total may approach render cost if readback falls below render
```

### 6.4 REALTIME_PREVIEW

Use case:

```text
interactive viewer
operator UI
visual debugging where newest frame matters most
```

Policy:

```text
latest/mailbox
drop allowed
low latency over complete sequence
prefer rgb8
diagnostics optional or sampled
```

Diagram:

```text
Render_i
  -> submit latest readback if slot available
Render_i+1
  -> maybe skip readback if ring full
Render_i+2

Display thread:
  poll newest completed frame
  discard older completed frames
  display latest
```

Backpressure:

```text
do not block render loop under normal pressure
skip submitting readback or ignore older completed frames
```

Implementation status:

```text
define as a future scenario
do not implement until a real viewer exists
```

### 6.5 SENSOR_ORDERED / SENSOR_LOSSLESS

Use case:

```text
robot optical sensors
training observations
frame-aligned simulation data
```

Policy:

```text
ordered
no silent drop
backpressure allowed
ack semantics must be explicit
```

Diagram:

```text
PhysicsFrame_i
  -> SensorSnapshot_i
  -> Accel_i
  -> Render_i
  -> AsyncOrderedStage_i
      copy_ready_i
  -> SensorConsumer_i
      borrow host result
      publish / hand off
      release
  -> Ack_i
```

Ack rule:

```text
copy_ready_event completion is necessary but not always sufficient
for full sensor ack.

For lossless sensor semantics, ack should occur after:
  host staging copy is complete
  consumer handoff/borrow has completed according to the sensor runtime contract
```

This preserves the Q52 principle:

```text
lossless snapshot/data ack means a self-contained staged copy exists and has
been handed to the consumer boundary.
```

Naming note:

```text
SENSOR_LOSSLESS may be ambiguous because "lossless" can sound like image
compression. SENSOR_ORDERED or SENSOR_COMPLETE may be better public names.
```

### 6.6 DYNAMIC_GEOMETRY

Use case:

```text
moving robots
dynamic rigid bodies
future deformables/fluids
```

Policy:

```text
dynamic geometry is orthogonal to delivery policy
combine with VIDEO_ORDERED_EXPORT, SENSOR_ORDERED, or preview delivery
but snapshot and acceleration stages are now per-frame hot path
```

Diagram:

```text
Frame_i:
  UpdateSnapshot_i
  -> RefitOrRebuildAccel_i
  -> Render_i
  -> Delivery_i
  -> Consume_i

Potential overlap:

snapshot/refit/build stream:
  Snapshot_i -> Accel_i ----- Snapshot_i+1 -> Accel_i+1

render stream:
                         Render_i ----- Render_i+1

copy/export stream:
                                    Copy_i ----- Copy_i+1
```

Key design issue:

```text
acceleration state may need double/triple buffering
render must not read an accel object while it is being rebuilt/refit
```

Current status:

```text
CUDA LBVH rebuild is available
Warp/CUDA traversal uses the built accel
full build/render/readback overlap is not implemented
```

For deformable and fluid geometry, the snapshot stage may also need surface
extraction before triangle buffers exist:

```text
fluid particles -> surface extraction -> triangle buffers -> acceleration
soft/cloth mesh -> updated surface mesh -> acceleration
```

Do not assume snapshot always receives a rigid-body transform update over a
fixed triangle set.

## 7. CPU Side Plan

CPU responsibilities split into two roles.

### 7.1 CPU Reference / Backend

CPU optical execution remains important as:

```text
correctness reference
small scene fallback
CPU-only environment support
debug/parity backend
```

CPU backend should share semantic contracts:

```text
frame_id
sim_time
env_idx
channel names
output_profile where practical
host result schema
```

CPU backend should not share GPU delivery runtime:

```text
no CUDA events
no D2H copy
no GPU readback ring
```

CPU pipeline:

```text
Host Snapshot
  -> optional CPU accel
  -> CPU execute
  -> OpticalComputeResult(location="host")
  -> Consumer
```

Priority:

```text
keep CPU correct
keep CPU/GPU result schemas aligned
do not optimize CPU render performance now
```

### 7.2 CPU Host Consumer / Export Runtime

Even GPU rendering eventually reaches CPU consumers:

```text
image build
video encode/write
sensor message publish
logging
debug assertions
viewer display
```

This side also needs policy:

```text
ordered export:
  no drop, encode queue backpressure

latest preview:
  drop old frames, display latest

sensor:
  publish/ack semantics

debug:
  immediate sync
```

Therefore the architecture should distinguish:

```text
CPU as reference backend
CPU as host consumer/export side
```

## 8. GPU Layered Interfaces

GPU pipeline layers:

```text
DeviceScene
  -> Acceleration
  -> Renderer
  -> DeviceResult
  -> Delivery
  -> ConsumerAdapter
```

External renderer adapters may attach at explicit boundaries:

```text
SceneExportAdapter:
  DeviceScene/registry/camera -> external scene/API

RendererBackendAdapter:
  snapshot/accel/camera/output_profile -> DeviceResult-compatible result

ReferenceRenderAdapter:
  snapshot/camera/output request -> host reference result
```

They must not introduce hidden dependencies from acceleration/delivery/consumer
back into scene or renderer semantics.

### 8.1 DeviceScene Layer

Responsibility:

```text
registry / physics frame -> GPU world-space optical geometry
```

Interface shape:

```python
snapshot = scene_cache.snapshot_from_frame(
    frame,
    include_aabb=True,
    workspace=workspace,
)
```

Output:

```text
DeviceOpticalSceneSnapshot
  frame_id
  sim_time
  env_idx
  triangle_v0/e1/e2/normal_world
  triangle_aabb_min/max optional
  plane_world
  ready_event
  resources
```

Cross-layer needs:

```text
Acceleration needs AABB.
Renderer needs world primitive buffers.
Delivery should not know scene internals.
```

### 8.2 Acceleration Layer

Responsibility:

```text
snapshot -> traversal acceleration structure
```

Interface shape:

```python
accel = accel_cache.update(
    snapshot,
    policy="reuse|refit|rebuild",
    workspace=workspace,
)
```

Output:

```text
DeviceAcceleration
  backend: cuda_lbvh | warp_bvh | optix
  ready_event
  traversal buffers or accel handle
  primitive mapping
  stats
  resources
```

Cross-layer needs:

```text
Renderer needs traversal handle/buffers.
Benchmark needs build/refit stats.
Scene should not know traversal details.
Delivery should not know acceleration details.
```

### 8.3 Renderer Layer

Responsibility:

```text
camera/rays + snapshot + acceleration -> device optical result
```

Interface shape:

```python
result = renderer.render_camera(
    snapshot,
    accel,
    camera,
    output_profile=OpticalOutputProfile.RGB_PREVIEW,
    workspace=workspace,
)
```

Output:

```text
DeviceFrameResult / OpticalComputeResult(location="device")
  channels
  output_profile
  ready_event
  diagnostics
  resources
  optional timing/profile stats
```

Cross-layer rule:

```text
Renderer receives output_profile.
Renderer does not receive consumer mode.
Renderer does not perform host readback.
```

### 8.4 DeviceResult Contract Layer

Responsibility:

```text
describe what the device result contains,
when it is ready,
and which resources must remain alive
```

Interface shape:

```python
result.channel("rgb")
result.has_channel("rgb")
result.ready_event
result.resources
result.output_profile
```

Cross-layer needs:

```text
Delivery must retain result/resources until async copy completion.
Debug readback can synchronously stage the same result.
Future GPU-only consumers can consume device result directly.
```

### 8.5 Delivery Layer

Responsibility:

```text
device result -> host completed frame, or no host transfer
```

Interface shape:

```python
job = scheduler.submit(
    result,
    payload=ReadbackPayload.RGB,
)

completed = scheduler.wait_next_ordered()
```

Output:

```text
CompletedFrame
  host channel views
  frame_id
  sim_time
  diagnostics
  borrow_host() / release
```

Cross-layer needs:

```text
needs ready_event
needs channel buffers
needs dtype/shape schema
needs resources retention
must not know triangle/BVH/material internals
```

### 8.6 ConsumerAdapter Layer

Responsibility:

```text
CompletedFrame -> PNG/video/sensor/debug/viewer
```

Interface shape:

```python
video_writer.consume(completed_frame)
sensor_publisher.consume(completed_frame)
debug_checker.consume(completed_frame)
```

Cross-layer needs:

```text
uses host frame metadata and host channels
does not touch BVH/snapshot/internal CUDA buffers
releases/acks according to consumer policy
```

### 8.7 External Renderer Adapter Layer

Responsibility:

```text
connect stable pipeline contracts to external rendering ecosystems
```

Adapter types:

```text
SceneExportAdapter:
  exports registry/snapshot/camera/material/light into an external scene

ReferenceRenderAdapter:
  runs an offline/reference renderer and returns comparable host outputs

PresentationAdapter:
  consumes device or host image channels and displays/records them without
  becoming the renderer

RendererBackendAdapter:
  participates as a render backend and returns DeviceResult-compatible outputs
```

Required preserved semantics:

```text
frame_id / sim_time
camera model and clipping
numeric instance/material identity where supported
range/depth convention
role-mask filtering where supported
source-order tie-break when semantic parity is claimed
resource lifetime and ready/completed semantics for runtime backends
```

Allowed limitations:

```text
visual-only exporters may omit role masks and tie-break parity
offline reference renderers may return host-only results
external runtimes may support only a subset of output profiles
```

Not allowed:

```text
changing core output_profile meanings
changing readback_payload meanings
changing sensor delivery/ack semantics
requiring consumer code to know external renderer internals
```

This layer is an extensibility boundary. Adding PBRT, Mitsuba, Blender/Cycles,
OpenUSD/Hydra, Embree, or TinyBVH support should be possible by adding adapters
and compatibility declarations, not by rewriting the pipeline contract.

Role-specific examples:

```text
Rerun:
  ConsumerAdapter / PresentationAdapter
  consumes CompletedFrame or OpticalCameraReading
  should not know BVH, kernels, or acceleration internals

Vulkan as display:
  PresentationAdapter
  consumes RGB/RGB8 device or host channel
  may later use CUDA/Vulkan external memory interop
  does not replace the renderer

Vulkan as renderer:
  RendererBackendAdapter
  owns its own shader/ray tracing resources
  must return DeviceResult-compatible outputs
  must declare which output profiles and semantic parity levels it supports
```

The same toolkit may appear at different layers depending on role. The adapter
role, not the toolkit name, decides where it attaches.

## 9. Cross-Layer Dependency Rules

### 9.1 Allowed Mapping Point

A controller/session layer maps high-level scenario to lower-level choices:

```text
scenario / consumer intent
  -> acceleration policy
  -> output_profile
  -> delivery policy
  -> readback payload
  -> diagnostics policy
```

This mapping should live in:

```text
PipelineController / RenderSession / benchmark harness
```

not inside kernels or BVH builders.

### 9.2 Dependency Table

```text
consumer mode:
  allowed in controller/config
  not allowed in kernel/accel

delivery policy:
  allowed in delivery runtime/controller
  not allowed in scene/accel/render kernels

output_profile:
  allowed in renderer
  not allowed in scene/accel

readback payload:
  allowed in delivery
  generally not allowed in renderer,
  except future fused payload paths where the controller maps payload intent to
  a renderer-visible output_profile

accel policy:
  allowed in acceleration/controller
  not allowed in delivery/consumer

diagnostics policy:
  controller decides required/optional
  renderer produces diagnostics
  delivery stages diagnostics only if required by scenario
```

Fused payload rule:

```text
readback_payload=rgb8
  -> controller may select output_profile=RGB8_FUSED
  -> renderer sees RGB8_FUSED
  -> renderer does not inspect readback_payload directly
```

This keeps the delivery/render boundary intact even when a fused kernel produces
a channel that exists mainly to reduce readback cost.

### 9.3 Hard Boundaries

```text
Consumer mode does not enter renderer kernels.
Delivery does not understand geometry.
Acceleration does not understand RGB/PNG/sensor publish.
Scene does not understand readback.
Consumer adapters do not touch device internals.
Presentation adapters do not redefine render semantics.
Renderer backend adapters do not redefine delivery semantics.
```

## 10. Output And Readback Representation

Keep output profile as executor-facing representation:

```text
GEOMETRY_FULL
DIRECT_LIGHT_FULL
RGB_PREVIEW
RENDER_ONLY
```

Keep readback payload as delivery-facing representation:

```text
none
full
rgb float32
rgb8 future
sensor-specific groups future
```

Important:

```text
These are not equal public axes.
They are selected by the scenario/controller.
```

Legal but inefficient combinations may exist:

```text
output_profile=RGB_PREVIEW
readback_payload=none
```

This could be useful for a GPU-only downstream consumer, but not for a normal
render benchmark. A formal consumer config should warn or reject contradictory
defaults.

### 10.1 Output Profile Channel Manifest

Each output profile must have a stable channel guarantee. This manifest is the
basis for adapter compatibility, readback validation, and parity expectations.

Initial manifest:

| Output profile | Guaranteed public channels | DType / shape | Notes |
|----------------|----------------------------|---------------|-------|
| `RENDER_ONLY` | diagnostics only | scalar diagnostics | No per-ray public image channels. Used for device timing. |
| `RGB_PREVIEW` | `rgb`, `hit_mask` | `rgb`: float32 `[H, W, 3]`; `hit_mask`: bool/int32 `[H, W]` | Direct-light preview output. No geometry channels guaranteed. |
| `GEOMETRY_FULL` | `hit_mask`, `range_m`, `position_world`, `normal_world`, `numeric_instance_id`, `numeric_material_id` | `range_m`: float32 `[H, W]`; vectors: float32 `[H, W, 3]`; ids: int32 `[H, W]` | First-hit geometry/identity output. |
| `DIRECT_LIGHT_FULL` | all `GEOMETRY_FULL` channels plus `rgb`, `intensity` | `rgb`: float32 `[H, W, 3]`; `intensity`: float32 `[H, W]` | Full direct-light debug/parity output. |

Diagnostics are not image channels. They may include scalar counters such as:

```text
bvh_stack_overflow_count
shadow_stack_overflow_count
kernel timing/profile fields
```

Future profiles may include:

```text
RGB8_FUSED:
  rgb8 uint8 [H, W, 3]
  optional hit_mask

PATH_TRACING_ACCUMULATION:
  radiance
  sample_count
  accumulation_state
```

Adapter compatibility must state which output profiles and channels are
supported. A renderer claiming semantic parity must preserve the relevant
manifest channels and their conventions.

### 10.2 Backend-Neutral Channel View

The current implementation often exposes device channels as Warp arrays. That
is acceptable for current executors, but it should not become the long-term
pipeline contract.

Future-compatible representation:

```text
DeviceChannelView
  name
  dtype
  shape
  device
  backend_kind:
    warp
    cuda
    torch
    vulkan
    external
  pointer_or_handle
  ready_event_or_fence
  ownership/resources
```

`pointer_or_handle` should be represented as an integer raw pointer or opaque
integer handle, interpreted by `backend_kind`. Vulkan-style timeline semaphores
or external-memory fences must be expressible through `ready_event_or_fence`,
not forced into a CUDA-only event type.

This representation is needed so external adapters can be added without
changing interface logic:

```text
Warp/CUDA renderers:
  expose raw device arrays or pointers

Vulkan presentation:
  may expose or consume an external image/memory handle

Rerun:
  usually consumes host-staged channels through CompletedFrame

future GPU-only consumers:
  may consume device channels without host readback
```

Boundary rule:

```text
DeviceResult may expose backend-neutral channel views.
Delivery decides how to stage them.
Presentation adapters decide how to display them.
Renderer backend adapters decide how to produce them.
```

Do not let one backend-specific representation, such as Warp array or Vulkan
image, leak upward as the public channel contract.

## 11. Lifetime Model

### 11.1 Device Scene Snapshot

Owns:

```text
world primitive buffers
optional AABB buffers
ready_event
```

May release after:

```text
all acceleration/render work that reads it has completed
```

### 11.2 Acceleration

Owns:

```text
BVH/LBVH/OptiX buffers or handles
primitive mapping
ready_event
```

May release after:

```text
all render work that traverses it has completed
```

Dynamic double buffering must ensure:

```text
do not refit/rebuild an accel object while a render still reads it
```

### 11.3 Device Result

Owns:

```text
device output channels
ready_event
resources retaining upstream buffers needed by consumers
```

May release after:

```text
no GPU consumer needs it
and no async D2H copy reads it
```

### 11.4 Readback Job

Owns:

```text
copy event
host slot
references to device result/resources until copy complete
```

May release device references after:

```text
copy_ready_event completed
```

### 11.5 Completed Frame

Owns or borrows:

```text
host slot/channel views
frame metadata
diagnostics
```

Host slot may be reused after:

```text
copy is complete
consumer borrow/release is complete
```

Preferred API:

```python
with completed.borrow_host() as host_frame:
    ...
```

## 12. Optical Pipeline Lab

The old term "benchmark" is now too narrow.

The tool we need is a render-pipeline tuning and profiling platform:

```text
Optical Pipeline Lab
```

It should support:

```text
scenario configuration
stage-by-stage timing
profiling
A/B comparisons
readback/render/build decomposition
optimization experiments
result archival
regression comparison
memory sampling
future capability placeholders
```

It is not yet the production optical runtime. It is also more than a benchmark
script.

Recommended package shape:

```text
tools/optical_pipeline_lab/
  scenarios.py
  runner.py
  timing.py
  reports.py
  presets.py
  profiles.py
  memory.py
```

Possible CLI:

```bash
python -m tools.optical_pipeline_lab run \
  --preset go2_video_ordered \
  --width 960 \
  --height 640

python -m tools.optical_pipeline_lab matrix \
  --preset go2_video_ordered \
  --sweep readback=none,rgb shadows=true,false
```

The existing `examples/mujoco_menagerie_gpu_preview.py` can keep acting as the
current harness while this lab is extracted. After extraction, the example
should become a thin demo wrapper, not the place where runtime semantics live.

### 12.1 What The Lab Must Solidify

#### Scenario Config

Use a structured config instead of scattered CLI flags:

```text
scenario_name
consumer_semantics / scenario_family
device
width
height
scene_preset
geometry_mode: static | dynamic_rigid | deformable | fluid
camera_mode: fixed | orbit | sequence
accel_backend: cuda_lbvh | cpu_bvh | future optix
accel_policy: build_once | refit_each_frame | rebuild_each_frame | double_buffered
render_backend: warp_bvh_direct_light | future cuda | future optix
output_profile
readback_payload: none | rgb | full | future rgb8
delivery_policy: sync | device_only | future async_ordered | future async_latest
write_policy: none | png_sequence | future video_encoder
diagnostics_policy
```

#### Unified Frame CSV Schema

Even when a stage does not apply, emit `NaN` rather than changing the schema:

```text
frame_index
scenario_name
device
width
height
geometry_mode
scene_preset
camera_mode
accel_backend
accel_policy
render_backend
output_profile
readback_payload
delivery_policy
write_policy
snapshot_ms
accel_build_ms
accel_refit_ms
accel_rebuild_ms
render_execute_ms
render_overhead_ms
pack_rgb8_ms
readback_submit_ms
readback_wait_ms
readback_host_ms
image_build_ms
encode_write_ms
frame_total_ms
instant_fps
rolling_fps
primary_overflow
shadow_overflow
primary_max_stack
shadow_max_stack
memory_used_mb
```

Future async columns:

```text
readback_lag_frames
readback_ring_depth
readback_ring_block_count
completed_frame_index
overlap_ratio
```

#### Phase Timing API

The lab should own timing:

```python
timings.measure("render")
timings.add("accel_build", elapsed_ms)
frame_recorder.add_phase(...)
```

Examples and experiments should not grow their own incompatible timing recorders.

#### Warmup / Repeat / Device Rules

Solidify:

```text
setup_warmup
render_warmup
video_frames
device
progress_every
timing_csv
frame_timing_csv
```

#### Result Summary

Every run should summarize:

```text
mean / p50 / p90 / min / max
fps_mean
main phase breakdown
configuration digest
output file locations
memory if sampled
```

#### Render Profile Columns

Keep diagnostic render profiling optional:

```text
render_raygen_kernel_ms
render_first_hit_kernel_ms
render_shade_kernel_ms
render_*_alloc_ms
render_overhead_ms
```

These are profiling columns, not always-on throughput metrics, because enabling
them inserts extra synchronization.

As of E1 timing cleanup, `render_execute_ms` means executor render only:
`execute_camera()`/`execute()` plus `wp.synchronize_event(result.ready_event)`.
RGB8 delivery packing is reported separately as `pack_rgb8_ms`. D2 measurements
before this cleanup folded RGB8 pack into `render_execute_ms`, so old
`render+pack_mean` values should not be compared directly against new
`render_execute_ms` rows without adding `pack_rgb8_ms`.

`render_overhead_ms` is useful when profiling is enabled:

```text
render_overhead_ms =
  render_execute_ms - sum(recorded render kernel/profile subphases)
```

It tracks allocation/init/synchronization overhead that is not explained by the
main raygen, first-hit, and shade kernels.

E1 implements this only when `--render-profile` is enabled. The value is not
clamped, so tiny negative values can appear when synchronized profile subphases
slightly exceed the host-side render wall-time envelope.

#### Scene Preset Registry

First preset:

```text
go2_menagerie_static
```

Future presets:

```text
go2_dynamic
multi_robot_pack
synthetic_mesh_scale
large_mesh_static
dynamic_rigid_sequence
```

### 12.2 Interfaces To Reserve

Reserve these in config/schema even if not implemented.

Delivery:

```text
sync
device_only
async_ordered reserved
async_latest reserved
```

Readback payload:

```text
none
rgb
full
rgb8 reserved
diagnostics reserved
custom channel group reserved
```

Geometry mode:

```text
static
dynamic_rigid reserved
deformable reserved
fluid reserved
```

Acceleration policy:

```text
build_once
refit_each_frame reserved
rebuild_each_frame reserved
double_buffered_build reserved
```

Render backend:

```text
warp_bvh_direct_light
cuda_direct_light reserved
cuda_fused_rgb reserved
optix_first_hit reserved
path_tracer reserved
```

Write policy:

```text
none
png_sequence
video_encoder reserved
streaming_preview reserved
sensor_publish reserved
```

Consumer/scenario vocabulary:

```text
render_bench
video_ordered_export
parity_debug
realtime_preview reserved
sensor_ordered reserved
```

Reserved modes must fail loudly:

```python
NotImplementedError("async_ordered delivery is reserved; use sync for now")
```

Do not silently fall back to sync.

### 12.3 What Not To Freeze In The Lab

Do not freeze these yet:

```text
final public OpticalConsumerMode names
formal production RenderSession API
async scheduler class names
pinned memory implementation details
CUDA/OptiX backend API
path tracing channel schema
viewer drop policy
sensor ack final interface
```

The lab should leave room for these but not pretend they are settled.

### 12.4 Why This Lab Is Enough For Video Tuning

For VIDEO_ORDERED_EXPORT, the lab should support systematic decomposition:

```text
render-only baseline
sync RGB readback
full readback
RGB PNG write
shadow vs no-shadow
resolution scaling
memory scaling
future async ordered
future RGB8 pack
future encode pipeline
```

That is the immediate tuning need. The lab becomes the place where we compare
each optimization against a stable baseline.

### 12.5 Overlap Metrics

Async overlap remains future work, but the metric contract should be reserved.

Overlap ratio:

```text
overlap_ratio = 1 - observed_frame_time / (render_time + readback_time)
```

Interpretation:

```text
near 0:
  no useful overlap

positive and significant:
  readback/render overlap is working

negative:
  scheduling overhead or measurement mismatch
```

For future async ordered delivery, overlap proof is mandatory. Without true
async D2H, the scheduler is only structural.

## 13. Current Optimizations And Their Scenario

Most current optimization work belongs to VIDEO_ORDERED_EXPORT.

Completed or in progress:

```text
CPU raygen -> GPU raygen
full readback -> selected RGB readback
canonical float64 readback -> native float32 readback
full output contract -> rgb_preview/render_only output profiles
host materialized output allocation -> fast device allocation for rgb/render-only
CUDA LBVH build
render profiling breakdown
GPU1 measurements for stable baseline
```

Current measurements:

```text
Go2 960x640 cuda:1:

render-only + shadows:
  ~2.58 ms render, ~2.75 ms frame p50

render-only no shadows:
  ~1.23 ms render, ~1.31 ms frame p50

blocking rgb readback + shadows:
  frame p50 ~6.64 ms
  readback ~4.01 ms
```

Implication:

```text
For ordered video export, next bottleneck is delivery/readback, not render.
```

## 14. Architecture Gaps

### 14.1 No Formal Render Session

Current executor calls are mostly stateless. High performance will likely need:

```text
session/workspace
buffer reuse
stream ownership
ring ownership
allocator policy
accel state lifecycle
```

### 14.2 No Async D2H Backend Yet

Current readback uses blocking Warp `.numpy()` materialization.

Before building scheduler:

```text
spike real async D2H copy backend
```

Candidate order:

```text
1. Warp native stream-aware copy, if available
2. Torch pinned host tensor + non_blocking copy
3. custom CUDA extension only if necessary
```

V1 async delivery is not valid unless overlap is measured.

### 14.3 No Readback Scheduler

Need:

```text
OpticalDeliveryPolicy
ReadbackSlot ring
ReadbackJob
CompletedFrame
backpressure counters
borrow/release API
```

### 14.4 Dynamic Build/Render Overlap Not Designed

Need:

```text
accel double/triple buffer
build/refit stream
render stream
copy stream
topology validity
rebuild vs refit policy
```

### 14.5 Consumer API Not Ready

Consumer modes are vocabulary, not yet public API:

```text
PARITY_DEBUG
RENDER_BENCH
VIDEO_ORDERED
REALTIME_PREVIEW
SENSOR_ORDERED / SENSOR_LOSSLESS
```

Do not freeze public enum names until runtime architecture stabilizes.

E0/E1 decision:

```text
internal API first
public API later
```

After the Optical Pipeline Lab introduced an internal render pipeline shape, the
next API work should formalize a source-driven render foundation before exposing
stable simulator-facing names. The near-term internal vocabulary is:

```text
OpticalLabRenderSource:
  external-world adapter output
  owns registry/base-frame identity and optional scene/camera hints
  does not own device streams, BVH, executor, delivery, or reporting

OpticalLabRenderOptions:
  device, acceleration backend, split strategy, shadows, verbose_warp, runtime knobs

OpticalLabRenderWorkspace:
  owns device and render stream now
  future home for copy stream, scratch buffers, and frame-prep events

OpticalLabRenderSession:
  owns source-derived render lifecycle:
    DeviceOpticalSceneCache, base GpuPublishedFrame, static snapshot/BVH,
    executor, acceleration configuration

RenderRequest:
  camera/rays, output_profile, render_backend, diagnostics/profile flags

RenderResult:
  device-side result, ready_event, output_profile, resources

DeliveryRequest:
  readback_payload, delivery_policy, write_policy, ring depth

DeliveryResult:
  host/device delivered payload, timing, lag/drop/backpressure metadata
```

Frame/result timing ownership decision:

```text
OpticalLabRenderFrameContext:
  frame-scoped lifecycle/input handle
  borrows the current frame input
  may hold frame-specific snapshot/BVH resources while rendering

RenderResult:
  owns render-side output and render-side timing summary
  includes OpticalComputeResult plus render execute/profile/overhead timing
  does not own RGB8 pack, host readback, writer, or frame-level timing

DeliveryResult:
  owns delivery-side output and delivery-side timing summary
  includes pack/readback/write timing, lag/drop/backpressure metadata
  may retain device resources until async copy completion

FrameResult:
  frame-bound observation/result summary
  identifies frame_id, sim_time, env_idx
  aggregates prepare + render + delivery summaries
  does not own or mutate GpuPublishedFrame, DeviceOpticalSceneSnapshot, or BVH
```

`FrameResult` should be bound to the simulation/render frame identity, but it
should remain a lightweight observation object. The heavy resources stay inside
the pipeline/session/frame context or the lower-level compute/delivery results.
This keeps physics ownership separate from optical rendering: physics publishes
or lends a frame, `begin_frame(frame_inputs=...)` borrows it, and the optical
pipeline returns a summary of what happened for that frame.

The canonical lab render entrypoint should look like this in shape:

```python
source = OpticalLabRenderSource(
    registry=registry,
    base_frame=current_gpu_frame,
    bounds_min=bounds_min,
    bounds_max=bounds_max,
)

pipeline = OpticalLabRenderPipeline.create_from_source(
    source=source,
    options=OpticalLabRenderOptions(
        device="cuda:0",
        bvh_backend="cuda_lbvh",
        bvh_split_strategy="sort",
        shadows=True,
        verbose_warp=False,
    ),
    timings=timings,
)

frame = pipeline.begin_frame(frame_inputs=sim.publish_gpu_frame())
result = frame.render(request)
```

Menagerie Go2 should be one implementation of the source-builder layer, not the
name or shape of the render pipeline itself. Existing `Go2Render*` names are a
transitional compatibility layer only and should not receive new generic
responsibilities.

For async delivery, `FrameResult` is constructed when delivery completes, not
when render is submitted or when render returns. It represents a completed
consumer-visible frame. The frame identity should therefore match the completed
frame (`completed_frame_index` in the current lab CSV), while the async submit
path may still be working on a newer render frame.

Timing ownership should follow the same boundary:

```text
prepare timing:
  snapshot_ms, accel_refit_ms, accel_rebuild_ms
  owned by the frame context / pipeline begin-frame path

render timing:
  render_execute_ms, render profile phases, render_overhead_ms
  owned by RenderResult

delivery timing:
  pack_rgb8_ms, readback_submit_ms, readback_wait_ms, readback_host_ms,
  image_build_ms, encode_write_ms
  owned by DeliveryResult

frame summary:
  work_sum_ms, observed_frame_ms, critical_path_ms, instant_fps
  owned by FrameResult
```

Summary timing definitions:

```text
work_sum_ms:
  serial sum of the work attributed to the frame
  e.g. prepare + render + pack + readback + write where applicable

observed_frame_ms:
  wall-clock/completion interval observed by the benchmark or consumer
  maps to the existing frame_total_ms CSV column for compatibility

critical_path_ms:
  estimated overlapped throughput limiter
  for current async RGB/RGB8 delivery:
    max(render_execute_ms + pack_rgb8_ms, readback_host_ms)
```

Keep `frame_timing.csv` flat as an analysis/export format only. Internally,
future cleanup should move toward typed timing blocks with clear ownership, then
flatten those blocks into the existing CSV schema. This is especially important
for async delivery: a simple sum of render + delivery work is not always the
observed frame time when readback overlaps the next render. Preserve
`frame_total_ms` as the CSV export name; use `observed_frame_ms` as the internal
FrameResult summary field if/when typed frame summaries are introduced.

The production/public user API remains consumer-first and should map onto these
internal fields later. Do not force path tracing, video export, realtime preview,
and ordered sensors to share a premature public enum before the internal
RenderSession and delivery contracts stabilize.

### 14.6 Active Lab Render Foundation Plan

This is the current plan guiding the next implementation slices.

The old framing treated the first concrete implementation as a Go2 render
pipeline because the work grew out of Menagerie Go2 video benchmarks. That is no
longer the right architecture. The durable boundary is:

```text
Scene/source adapter:
  converts an external world to OpticalLabRenderSource

Render foundation:
  creates workspace/session/cache/snapshot/BVH/executor from source/options
  exposes begin_frame(...).render(...)

Video/delivery/reporting:
  consumes RenderResult and frame metadata
  handles readback/write/CSV/matrix policy
```

Current implementation status:

```text
I1 complete:
  Go2RenderSession / Go2RenderFrameContext / Go2RenderPipeline were extracted
  from go2_backend.py into a separate lab module.

I2 complete:
  Go2RenderWorkspace(device, stream) exists, and session.device/session.stream
  are compatibility properties over session.workspace.

C1 complete:
  the generic lab render foundation now lives in render_session.py under
  OpticalLabRender* names, with transitional Go2Render* compatibility aliases.

C2 complete:
  OpticalLabRenderSource and OpticalLabRenderOptions exist, and
  OpticalLabRenderPipeline.create_from_source(...) builds a session from the
  source/options boundary. The existing callback-based create(...) path remains
  in place for Go2 until C3.

C3 complete:
  Go2/Menagerie and synthetic lab scenes are built through
  build_go2_render_source(...). The Go2 backend now enters the generic render
  foundation through the source/options factory path, while keeping Go2 CLI,
  preset, camera, video, and reporting vocabulary in go2_backend.py. The old
  callback-based create(...) entrypoint has been removed.

C4 complete:
  OpticalLabRenderWorkspace owns dynamic frame preparation execution:
  snapshot, refit/rebuild, synchronization, and prepare timing. The pipeline
  still owns static/dynamic frame selection and FrameContext construction.
```

The remaining `Go2Render*` names are compatibility aliases only. New generic
render foundation work should use `OpticalLabRender*`.

Recommended next slices:

```text
C5 video loop split:
  later move generic video render/export helpers out of go2_backend.py
  do not mix this with source/session/workspace foundation changes
```

Names that should remain Go2-specific:

```text
go2_backend.py
go2_menagerie_static
go2_video_ordered_static
go2_video_ordered_baseline
go2_video_delivery_smoke
go2_video_ordered_legacy_960
model_dir/model_xml defaults
Menagerie smoke output paths
```

Names that should stop being Go2-specific:

```text
generic tests named test_go2_pipeline_*
```

After C1, `Go2Render*` and `go2_session.py` exist only as transitional aliases.

The key rule:

```text
External systems provide a render source, not a backend adapter.
```

### 14.7 Path Tracing Is Future Work

Direct-light output channels should not be treated as path-tracing contracts.

Future path tracing may need:

```text
radiance
sample_count
accumulation buffers
random seed state
temporal accumulation
denoising payload
different delivery cadence
```

Delivery primitives may be reusable, but output profiles will change.

Path tracing may require a many-to-one mapping from render calls to output

E0/E1 decision:

```text
path tracing is a later Q54-PT branch
not part of the current direct-light/render-session optimization loop
```

Current work stays focused on the direct-light + hard-shadow pipeline:

```text
RenderSession/workspace boundary
render profile and timing semantics
buffer allocation/reuse
shade and first-hit optimization
ordered RGB8 delivery
```

Path tracing should start only after the RenderSession/workspace boundary is
stable enough to carry multiple render backends. Its expected staging is:

```text
PT0 CPU reference path tracer for semantics and tests
PT1 GPU stochastic single-bounce / AO-style preview
PT2 progressive accumulation buffers and reset semantics
PT3 small multi-bounce diffuse path tracing
PT4 materials, denoise, and temporal/progressive delivery
```

Path tracing should appear as a render backend/output-profile family, not as a
change to the delivery API:

```text
render_backend = direct_light | path_tracing
compute payload = linear rgb / radiance / accumulation
delivery payload = rgb8 / rgb / future hdr/encoded video
```
frames:

```text
multiple sample/accumulation renders
  -> one consumer-visible output frame
```

The delivery layer must not permanently assume one render call equals one
consumer frame.

## 15. Staged Roadmap

### Stage A: Architecture Documentation

Deliver:

```text
total pipeline diagram
per-scenario diagrams
CPU/GPU responsibility split
GPU layer interface boundaries
scenario-to-policy matrix
benchmark metric contract
```

This document is the first pass.

### Stage B: Optical Pipeline Lab Foundation

Goal:

```text
turn the current example-based benchmark into reusable tuning infrastructure
```

Tasks:

```text
create tools/optical_pipeline_lab/ or equivalent
define scenario config
define frame timing schema
define phase timing recorder
define summary/report writer
define preset registry
support device selection, warmup, frame count, progress, CSV output
reserve unimplemented config values with loud NotImplementedError
```

Do not:

```text
implement async scheduler yet
freeze final public consumer API
move every example feature in one pass
```

MVP completion standard:

```text
one lab command reproduces the Appendix A.3 baseline rows
unimplemented fields are present but fail loudly
CSV missing values use NaN, not zero
```

Async D2H spike may begin in parallel during Stage B/C as an isolated go/no-go
experiment. It should not block the lab foundation.

### Stage C: Migrate Existing VIDEO_ORDERED_EXPORT Work Into The Lab

Goal:

```text
preserve and formalize the scenario we have already optimized most
```

Tasks:

```text
Go2 static scene preset
camera orbit/fixed camera modes
GPU raygen path
CUDA LBVH build path
Warp direct-light render path
readback_payload none/rgb/full
write_policy none/png_sequence
render profiling support
memory sampling support
shadow/no-shadow switch
GPU1 baseline reproduction
```

Baseline matrix:

```text
160x120 smoke
960x640 shadow readback=none
960x640 no-shadow readback=none
960x640 shadow readback=rgb
960x640 shadow readback=full if needed for debug
optional 1920x1080 scaling
```

### Stage D: First Functional Cleanup And Warp-Based Optimization Pass

Goal:

```text
make implemented paths coherent and measurable before deeper backend work
```

Tasks:

```text
stabilize output_profile/readback_payload mapping
stabilize diagnostics policy for implemented sync/device-only paths
keep Warp render path as correctness/performance baseline
avoid CUDA rewrites unless they are already isolated
confirm render allocation fast path remains correct
add regression checks around lab presets
```

Completion criteria:

```text
same lab command can reproduce current Go2 numbers
CSV schema is stable enough for subsequent optimization comparison
unimplemented paths fail loudly
example is thinner or clearly marked as transitional
```

### Stage E: Async D2H Copy Spike

Goal:

```text
prove or disprove real D2H overlap
```

Tasks:

```text
try Warp-native async copy if available
try Torch pinned host tensor + non_blocking copy
measure render/readback overlap in the lab
report overlap_ratio
```

Minimum evidence before full scheduler work:

```text
standalone double-buffer spike
separate render and copy streams or equivalent
CUDA/event timing that proves real overlap
overlap_ratio > 0.2 to justify scheduler work
```

First D1 spike result:

```text
1080p shadow RGB, warmup_renders=5, torch_async readback:
  all10 frame_mean       ~= 13.75 ms
  all10 render_mean      ~= 6.23 ms
  all10 readback_copy    ~= 9.26 ms
  all10 overlap_ratio    ~= 0.11

  steady frames 3-8:
    frame_mean           ~= 9.20 ms
    render_mean          ~= 6.31 ms
    readback_copy        ~= 9.19 ms
    readback_wait        ~= 2.18 ms
    overlap_ratio        ~= 0.41
```

This clears the overlap go/no-go threshold. The first two frames still include
pinned allocation/submit overhead, so D1 reports should distinguish all-frame
and steady-tail metrics.

D1 ring-slot follow-up:

```text
1080p shadow RGB, warmup_renders=5, torch_async readback, ring_depth=2:
  all10 frame_mean       ~= 10.78 ms
  all10 render_mean      ~= 6.33 ms
  all10 readback_copy    ~= 10.21 ms
  all10 readback_submit  ~= 0.11 ms
  all10 overlap_ratio    ~= 0.35

  steady frames 2-8:
    frame_mean           ~= 10.20 ms
    render_mean          ~= 6.40 ms
    readback_copy        ~= 10.20 ms
    readback_wait        ~= 3.42 ms
    overlap_ratio        ~= 0.39
```

The ring version moves pinned host allocation out of the timed frame loop and
keeps submit overhead around 0.1 ms from the first measured frame.

The ring machinery now lives in `tools/optical_pipeline_lab/async_readback.py`
as a reusable lab helper. Go2 remains responsible for render/camera metadata
and ordered row emission, while the helper owns pinned host slots, copy stream,
submit timing, copy event timing, and host channel access.

Go/no-go:

```text
If no true async D2H backend is available without large custom code,
defer async scheduler and prioritize RGB8 pack or a CUDA copy extension.
```

### Stage F: Minimal RenderSession / Workspace Skeleton

Goal:

```text
avoid putting stream/ring/resource ownership into examples
```

Tasks:

```text
minimal RenderSession or PipelineController
stream ownership:
  render stream
  copy stream
ring ownership:
  ReadbackSlot ring
workspace allocation interface
delivery ring ownership hooks
no public consumer API freeze
```

This is a small skeleton, not the full runtime refactor. It exists so async
ordered delivery does not become example-local infrastructure.

Explicit non-scope:

```text
full buffer reuse
full accel state lifecycle
allocator policy
public consumer API
dynamic geometry scheduling
```

### Stage G: Async Ordered Delivery For Video Export

Goal:

```text
VIDEO_ORDERED_EXPORT reaches max(render, readback) steady-state behavior
```

Tasks:

```text
OpticalDeliveryPolicy
ReadbackPayload
ReadbackSlot ring
ReadbackJob
CompletedFrame borrow_host
async ordered scheduler
lab delivery_policy=async_ordered
CSV timing additions:
  readback_submit_ms
  readback_wait_ms
  readback_lag_frames
  readback_ring_block_count
  overlap_ratio
```

### Stage H: GPU RGB8 Pack

Goal:

```text
reduce readback payload
remove CPU gamma/uint8 materialization from hot path
```

Tasks:

```text
GPU kernel for clip/gamma/uint8 pack
ReadbackPayload.RGB8
lab rgb8 mode
compare sync and async ordered
```

Initial lab implementation:

```text
tools/optical_pipeline_lab/rgb_pack.py
  Warp kernel for linear RGB float32 -> preview RGB8
  clip + NaN sanitize + gamma 1/2.2 + uint8 round
  returns a new OpticalComputeResult with rgb8 channel and ready_event

go2_backend.py
  --video-readback=rgb8
  sync readback supports rgb8 host staging
  torch_async readback supports rgb8 ring delivery
```

This is the first D2 delivery-pack implementation. It is Warp-based rather than
a dedicated CUDA extension, which keeps it close to the existing optical
executor and lets the lab quantify the payload reduction first. A CUDA pack
extension remains a candidate only if the measured pack cost becomes visible
after the async/readback path is stable.

### Stage I: Render Session / Workspace Refactor

Goal:

```text
make the Optical Pipeline Lab render foundation source-driven
stop growing generic infrastructure under Go2 names
centralize render runtime resources behind lab-local source/options/session APIs
```

Current status:

```text
I1 complete:
  render session classes extracted from go2_backend.py

I2 complete:
  minimal workspace owns device/render stream
  session.device/session.stream are compatibility properties

I3/C1 complete:
  tools/optical_pipeline_lab/render_session.py owns OpticalLabRender* classes
  tools/optical_pipeline_lab/go2_session.py is a transitional alias shim

I4/C2 complete:
  introduce OpticalLabRenderSource
  introduce OpticalLabRenderOptions
  add a source/options construction path
  keep callback-based create(...) until the Go2 backend is migrated in C3
```

Active Stage I plan:

```text
I5/C3 complete:
  build Menagerie Go2 and synthetic smoke scenes as OpticalLabRenderSource
  call the generic OpticalLabRenderPipeline entrypoint
  leave Go2 CLI/preset/matrix vocabulary where it describes real Go2 cases

I6/C4 complete:
  move dynamic snapshot/refit/rebuild execution into workspace
  keep FrameContext construction and static/dynamic decision in pipeline

I7/C5 video-loop split:
  split generic video render/export helpers out of go2_backend.py later
```

Stage I is not a public API promotion. `OpticalLabRender*` remains lab-local
until the source/options/session boundary survives Go2, synthetic dynamic, and
physics-simulation callers.

### Stage J: Dynamic Geometry Pipeline

Goal:

```text
support moving rigid-body scenes through the source-driven render foundation
with build/refit/render/delivery overlap
```

Tasks:

```text
physics simulation publishes GpuPublishedFrame
OpticalLabRenderSource supplies registry/base frame and scene hints
begin_frame(frame_inputs=...) borrows the current frame
refit/rebuild policy
accel double/triple buffering
build/render event dependencies
benchmark dynamic robot scene
dynamic preset for refit vs rebuild comparison
```

Stage J should not invent a separate scene adapter pattern. Physics, Go2, and
synthetic dynamic scenes should all enter through the same source vocabulary.

### Stage K: Sensor Runtime Integration

Goal:

```text
formal frame-aligned optical sensor observations
```

Tasks:

```text
sensor ordered/lossless delivery
ack/backpressure semantics
host result handoff
Q52 integration
diagnostic/error policy
```

### Stage L: Realtime Preview Latest Mode

Goal:

```text
low-latency viewer path
```

Tasks:

```text
latest/mailbox delivery
drop counters
newest completed frame polling
viewer integration
```

### Stage M: CUDA/OptiX Backend Evolution

Goal:

```text
align with high-performance rendering/ray tracing projects
```

Tasks:

```text
CUDA fused kernels
CUDA traversal alternative
OptiX adapter spike
BLAS/TLAS layout
path tracing planning
```

## 16. Near-Term Decision Points

1. Is this stage skeleton correct for all current scenarios?

```text
snapshot -> accel -> render -> device result -> delivery -> consumer
```

2. Is VIDEO_ORDERED_EXPORT correctly identified as the currently optimized
scenario?

3. Should next implementation be Optical Pipeline Lab foundation before full
async D2H scheduler work?

Recommended answer:

```text
yes
```

Async D2H spike may run in parallel as an isolated experiment, but it should not
replace the lab foundation as the main implementation slice.

4. Should new performance-critical GPU code prefer CUDA?

Recommended answer:

```text
yes, while preserving Warp reference paths
```

5. Should consumer modes remain vocabulary until RenderSession/runtime exists?

Recommended answer:

```text
yes
```

6. Should examples keep low-level benchmark flags rather than public consumer
presets?

Recommended answer:

```text
yes for now
```

7. Should CUDA-first migration wait until the lab and first functional cleanup
are stable?

Recommended answer:

```text
yes, except for isolated work that already has a clear CUDA boundary
```

8. Should async ordered delivery wait for a minimal RenderSession/workspace
skeleton?

Recommended answer:

```text
yes
```

## 17. Draft Consumer API

This section is a draft, not a frozen public API. It exists so implementation
work can aim toward a coherent user-facing surface while the next concrete
tasks still focus on the Optical Pipeline Lab and internal delivery mechanics.

E0/E1 update:

```text
the next concrete API artifact is an internal RenderSession/Delivery design note
not a frozen public simulator camera API
```

The internal API should be shaped by the objects that have now appeared in the
lab:

```python
session = OpticalRenderSession.create(scene_config, device="cuda:0")
rendered = session.render(request)
delivered = session.deliver(rendered, delivery_request)
```

The important separation is:

```text
render precision/output profile != delivery payload precision
```

For example, direct-light and future path tracing can both compute linear or
HDR-ish device data while delivering `rgb8` for preview/video. This keeps
delivery choices like RGB8, async ordered rings, and future encoders orthogonal
to render backend choices like direct light vs path tracing.

The public API should be consumer-first:

```text
users choose what they are trying to consume
the runtime chooses a recommended delivery policy
advanced users may override lower-level policy explicitly
```

The implementation remains delivery-first:

```text
delivery policy
readback payload
output profile
buffer lifetime
events
backpressure
```

The consumer API should not expose CUDA stream/event details by default.

### 17.1 Draft Modes

```python
class OpticalConsumerMode(Enum):
    PARITY_DEBUG = "parity_debug"
    RENDER_BENCH = "render_bench"
    VIDEO_ORDERED = "video_ordered"
    REALTIME_PREVIEW = "realtime_preview"
    SENSOR_ORDERED = "sensor_ordered"
```

Mode meanings:

```text
PARITY_DEBUG:
  complete host results
  sync/blocking behavior
  strongest diagnostics
  used for CPU/GPU parity and regression tests

RENDER_BENCH:
  device-only timing
  minimal output
  no host readback unless explicitly requested
  used to measure render kernels without delivery noise

VIDEO_ORDERED:
  every frame is preserved
  frames are emitted in order
  consumer may block if delivery/readback/export falls behind
  used for ordered video export and other ordered visual consumers

REALTIME_PREVIEW:
  low latency
  old frames may be dropped
  newest completed frame wins
  reserved until a viewer exists

SENSOR_ORDERED:
  frame-aligned observation stream
  no silent frame loss
  explicit ack/backpressure semantics
  reserved for simulator sensor integration
```

Naming note:

```text
SENSOR_ORDERED is preferred over SENSOR_LOSSLESS for now.
```

"Lossless" can be confused with compression quality. The real requirement is
ordered, frame-complete delivery with explicit backpressure or ack semantics.

Naming note:

```text
VIDEO_ORDERED is the public consumer-mode name.
video_ordered_export is a lab scenario or preset name.
```

### 17.2 Draft Config Shape

```python
@dataclass(frozen=True)
class OpticalConsumerConfig:
    mode: OpticalConsumerMode
    output_profile: OpticalOutputProfile | None = None
    readback_payload: ReadbackPayload | None = None
    delivery_policy: OpticalDeliveryPolicy | None = None
    diagnostics: bool = True
    fail_on_overflow: bool = True
    allow_frame_drop: bool | None = None
    ring_depth: int | None = None
    write_policy: OpticalWritePolicy | None = None
```

Defaults are mode-specific recommendations, not hard constraints:

```text
PARITY_DEBUG:
  output_profile=direct_light_full or geometry_full
  readback_payload=full
  delivery_policy=sync_blocking

RENDER_BENCH:
  output_profile=render_only
  readback_payload=none
  delivery_policy=device_only

VIDEO_ORDERED:
  output_profile=rgb_preview
  readback_payload=rgb or rgb8
  delivery_policy=async_ordered once available
  delivery_policy=sync_blocking until async D2H is implemented
  allow_frame_drop=False

REALTIME_PREVIEW:
  output_profile=rgb_preview
  readback_payload=rgb8 or rgb
  delivery_policy=async_latest
  allow_frame_drop=True

SENSOR_ORDERED:
  output_profile=geometry_full or direct_light_full
  readback_payload=full or sensor-specific payload
  delivery_policy=async_ordered
  allow_frame_drop=False
```

The config should validate impossible combinations early:

```text
readback_payload must be satisfiable from output_profile
RENDER_BENCH should default to no host payload
VIDEO_ORDERED must not silently drop frames
REALTIME_PREVIEW must expose drop counters
SENSOR_ORDERED must define ack semantics before integration
allow_frame_drop=False cannot be combined with async_latest
RENDER_BENCH with readback_payload != none should warn or fail
VIDEO_ORDERED with sync_blocking should warn that render/readback overlap is disabled
```

For async modes, `fail_on_overflow=True` becomes a delayed check: overflow is
reported when the completed frame or diagnostics are drained, not at submit
time.

### 17.3 Draft Runtime Entry Points

The eventual public surface may look like:

```python
session = OpticalRenderSession(scene, device="cuda:0")

with session.consumer(OpticalConsumerConfig(
    mode=OpticalConsumerMode.VIDEO_ORDERED,
    write_policy=OpticalWritePolicy.PNG_SEQUENCE,
)) as consumer:
    for frame in frames:
        consumer.submit(frame, camera)
    consumer.drain()
```

For blocking debug:

```python
result = session.render_once(
    frame,
    camera,
    OpticalConsumerConfig(mode=OpticalConsumerMode.PARITY_DEBUG),
)
```

For render-only timing:

```python
timing = session.profile_render(
    frame,
    camera,
    OpticalConsumerConfig(mode=OpticalConsumerMode.RENDER_BENCH),
)
```

These names are placeholders. The important contract is:

```text
consumer API selects scenario semantics
runtime maps semantics to output/readback/delivery defaults
lower-level lab and developer tooling can still set policies directly
```

### 17.4 Interaction With The Optical Pipeline Lab

The lab should not wait for this public API.

The lab should implement lower-level scenario configs first:

```text
scenario_family
output_profile
readback_payload
delivery_policy
write_policy
diagnostics_policy
```

Later, the public API can map consumer modes onto those same fields.

This prevents premature API freeze while still ensuring that benchmark,
delivery, and runtime work converge on one vocabulary.

## 18. Explicit Non-Goals For The Next Implementation Step

Do not implement all scenarios at once.

Do not freeze final public consumer API yet.

Do not build realtime preview latest mode before a viewer exists.

Do not rewrite all Warp kernels into CUDA in one pass.

Do not treat path tracing as part of the direct-light delivery refactor.

Do not claim async delivery works until real overlap is measured.

Do not move benchmark-only CLI flags into stable public API prematurely.

Do not keep expanding the Menagerie example as the long-term tuning platform.

Do not treat the Optical Pipeline Lab as production runtime.

## 19. Summary

The architecture should be guided by scenarios, not by isolated optimizations.

The shared pipeline is:

```text
frame -> snapshot -> acceleration -> render -> device result -> delivery -> consumer
```

The shared implementation substrate should support overlap, events, queues, and
resource lifetime.

The overlap strategy is scenario-specific:

```text
debug:
  sync

render benchmark:
  device-only

video export:
  async ordered, no-drop

preview:
  latest/drop

sensor:
  ordered/lossless, backpressure
```

The next concrete implementation should not start with a full scheduler or a
CUDA rewrite. It should start with the Optical Pipeline Lab foundation, because
all later tuning needs stable scenario configs, timing schemas, presets, and
reports. After that, VIDEO_ORDERED_EXPORT should be migrated into the lab. The
async D2H spike may run in parallel as an isolated go/no-go test. Before full
async ordered delivery, add a minimal RenderSession/workspace skeleton so stream,
workspace, acceleration, and ring ownership do not stay trapped in examples.
RGB8 and deeper CUDA/OptiX work should follow trustworthy lab data.

## Appendix A. Current VIDEO_ORDERED_EXPORT Tuning Record

This appendix records what has already been done for the current video export
path, what was measured, and what should be tested again after the pipeline is
migrated into the Optical Pipeline Lab.

### A.1 Scenario Under Test

Current optimized scenario:

```text
VIDEO_ORDERED_EXPORT
static Go2 Menagerie scene
960x640 camera
CUDA LBVH acceleration build
GPU camera ray generation
Warp first-hit traversal
Warp direct-light shading
optional inline shadow any-hit
ordered RGB host readback
PNG/video export path
no dropped frames
```

This is an ordered export scenario, not a realtime preview scenario.

Primary constraints:

```text
preserve every frame
preserve frame order
allow blocking/backpressure
measure render, readback, image build, and encode separately
```

### A.2 Work Already Done

Acceleration:

```text
CPU Python BVH builder identified as a major cold/warm bottleneck
CUDA LBVH builder added as the fast build path
Warp traversal retained as correctness/performance reference
BVH output remains compatible with DeviceOpticalBvh-style SoA traversal
```

Render path:

```text
GPU camera raygen added
render_only / rgb_preview / direct_light_full output profiles introduced
unneeded host-materialized buffers avoided on rgb_preview/render_only paths
internal first-hit buffers switched toward wp.empty fast allocation where safe
hit_mask initialization isolated for partial-output paths
render subphase profiling added for raygen, first-hit, and shade
```

Readback and export path:

```text
readback_payload none/rgb/full added for video experiments
selected readback shown to reduce host transfer/materialization work
full readback recognized as debug/parity-oriented, not the hot video path
RGB image build and PNG encoding measured as separate costs
```

Documentation and planning:

```text
output_profile separated from readback_payload
delivery policy separated from consumer-facing scenario vocabulary
Optical Pipeline Lab proposed as the durable tuning platform
```

### A.3 Measured Baselines

Representative measurements from GPU1/Hopper-class hardware with the Go2
Menagerie static scene at 960x640:

```text
shadow, readback=none:
  render_mean ~= 2.58 ms
  frame_p50   ~= 2.75 ms
  frame_p90   ~= 2.84 ms
  fps         ~= 372

no-shadow, readback=none:
  render_mean ~= 1.23 ms
  frame_p50   ~= 1.31 ms
  frame_p90   ~= 1.45 ms
  fps         ~= 750

shadow, readback=rgb:
  render_mean   ~= 2.61 ms
  readback_mean ~= 4.01 ms
  frame_p50     ~= 6.64 ms
  frame_p90     ~= 7.04 ms
  fps           ~= 148
```

These are Optical Pipeline Lab C2 measurements from
`go2_video_ordered_legacy_960` on `cuda:1`.

Diagnostic render profiling with additional synchronization showed roughly:

```text
raygen kernel       ~= 0.10 ms
first-hit kernel    ~= 1.00 ms
shade kernel        ~= 1.54 ms
allocation/init     ~= 0.2 ms scale
profiled render     ~= 2.95 ms
```

Because profiling adds synchronization, these subphase numbers should be used
for attribution, not as the final throughput baseline.

Earlier host/export observations showed PNG writing can dominate the wall time,
but these are not yet formal Stage C lab baselines:

```text
RGB image build ~= 19 ms scale
PNG encode/write ~= 38 ms scale
```

These numbers should be remeasured in the Optical Pipeline Lab because image
build and encoding depend on resolution, filesystem, and whether RGB8 packing
has already occurred. Treat them as prior observations until Stage C records
resolution, CPU, encoder settings, filesystem, and compression behavior.

Memory footprint for the same class of run:

```text
after init              ~= 529 MiB
after device scene      ~= 561 MiB
after snapshot + AABB   ~= 595 MiB
after CUDA LBVH         ~= 747 MiB
render result alive     ~= 813 MiB
after result release    ~= 747 MiB
```

Conclusion:

```text
memory is not the current limiting factor for this scene
render is already low-single-digit milliseconds
blocking RGB readback is now larger than render
PNG/write path can dominate export wall time if enabled
```

### A.4 Current Readback Limitation

Async D2H has not been implemented yet.

The current video path is still effectively:

```text
render frame N
wait for render N
blocking RGB readback N
host image build / optional encode N
render frame N+1
```

This means steady-state frame time is closer to:

```text
render + readback + host work
```

not:

```text
max(render, readback, host work)
```

The measured `readback=rgb` case is therefore not yet testing true delivery
overlap. After the pipeline is migrated into the Optical Pipeline Lab, the first
delivery optimization to test should be real async D2H:

```text
render frame N+1 while copying RGB for frame N
copy into pinned host buffers or another proven async backend
measure readback_submit_ms
measure readback_wait_ms
measure overlap_ratio
measure ring block count
```

This is the key next experiment before deciding whether async ordered delivery
is worth formalizing for VIDEO_ORDERED_EXPORT.

### A.5 Expected Async D2H Hypothesis

For the measured shadow RGB case:

```text
render_mean   ~= 2.61 ms
readback_mean ~= 4.01 ms
frame_p50     ~= 6.64 ms
```

If real overlap works and encode is disabled, the theoretical steady-state
target becomes closer to:

```text
max(2.61 ms, 4.01 ms) + scheduling overhead
```

rather than:

```text
2.61 ms + 4.01 ms
```

This would move the RGB path from roughly 148 FPS toward a higher ceiling before
any RGB8 packing or CUDA render rewrite.

For the new default 1080p baseline, an early C2 matrix run with only two warmup
renders exposed large initial readback spikes. The measured stable tail after
those spikes was approximately:

```text
render_mean   ~= 6.08 ms
readback_mean ~= 44.53 ms
frame_mean    ~= 50.91 ms
```

If D1 achieves true ordered overlap for this case, the useful target is close
to:

```text
max(6.08 ms, 44.53 ms) + scheduling overhead
```

That implies the first async D2H spike should be judged against a ceiling near
44-50 ms steady-state frame time before RGB8 packing, not against the 960x640
readback target.

A follow-up single-case run with `warmup_renders=5` removed the hundreds of
milliseconds readback spikes:

```text
1080p shadow RGB, warmup_renders=5:
  render_mean        ~= 5.56 ms
  readback_mean      ~= 18.74 ms
  frame_mean         ~= 24.42 ms
  last3 readback     ~= 12.26 ms
  last3 frame_mean   ~= 17.65 ms
```

The lab default warmup is therefore five render passes for subsequent C2/D1
baselines. D1 should still record both all-frame and tail metrics because
readback allocation/context effects can otherwise pollute overlap conclusions.

The first D1 `torch_async` readback spike used the same 1080p shadow RGB setup:

```text
all10:
  frame_mean         ~= 13.75 ms
  render_mean        ~= 6.23 ms
  readback_copy      ~= 9.26 ms
  readback_wait      ~= 2.43 ms
  overlap_ratio      ~= 0.11

steady frames 3-8:
  frame_mean         ~= 9.20 ms
  render_mean        ~= 6.31 ms
  readback_copy      ~= 9.19 ms
  readback_wait      ~= 2.18 ms
  overlap_ratio      ~= 0.41
```

The steady result beats the D1 go/no-go threshold (`overlap_ratio > 0.2`) and
shows that a pinned Torch D2H path can overlap with the next render on this
hardware. The remaining work is to turn the spike into a real ordered delivery
primitive with reusable slots, explicit ownership, and cleaner first-frame
allocation behavior.

The immediate ring-slot follow-up preallocated two pinned host slots before the
timed frame loop:

```text
torch_async ring_depth=2:
  all10:
    frame_mean       ~= 10.78 ms
    render_mean      ~= 6.33 ms
    readback_copy    ~= 10.21 ms
    readback_submit  ~= 0.11 ms
    overlap_ratio    ~= 0.35

  steady frames 2-8:
    frame_mean       ~= 10.20 ms
    render_mean      ~= 6.40 ms
    readback_copy    ~= 10.20 ms
    readback_wait    ~= 3.42 ms
    overlap_ratio    ~= 0.39
```

This removes the first-frame pinned allocation penalty from timed rows and is a
better baseline for the next delivery primitive extraction.

The first D2 RGB8 pack run used the same 1080p shadow setup, still on GPU1, with
`torch_async` ring_depth=2:

```text
torch_async rgb8 ring_depth=2:
  all10:
    frame_mean       ~= 6.71 ms
    render+pack_mean ~= 6.04 ms
    readback_copy    ~= 2.33 ms
    readback_submit  ~= 0.11 ms
    overlap_ratio    ~= 0.20

  steady frames 1-8:
    frame_mean       ~= 6.44 ms
    render+pack_mean ~= 6.13 ms
    readback_copy    ~= 2.33 ms
    readback_wait    ~= 0.08 ms
    overlap_ratio    ~= 0.24
```

Note: these D2 rows used the old `render_execute_ms` convention where RGB8 pack
was folded into render timing. E1 splits that into `render_execute_ms` plus
`pack_rgb8_ms`.

The first E1 timing split smoke used the same 1080p shadow RGB8 async ring2
case on an idle GPU3 because GPU1 was occupied by another workload:

```text
torch_async rgb8 ring_depth=2, E1 split timing:
  all10:
    frame_mean       ~= 6.42 ms
    render_mean      ~= 5.66 ms
    pack_rgb8_mean   ~= 0.08 ms
    readback_copy    ~= 2.32 ms
    overlap_ratio    ~= 0.20

  steady frames 1-8:
    frame_mean       ~= 6.14 ms
    render_mean      ~= 5.75 ms
    pack_rgb8_mean   ~= 0.07 ms
    readback_copy    ~= 2.33 ms
    overlap_ratio    ~= 0.25
```

This confirms RGB8 pack is not a visible bottleneck on this setup; the remaining
render-side budget is dominated by executor render, not delivery packing.

The first E1 `--render-profile` diagnostic split for the same case showed:

```text
torch_async rgb8 ring_depth=2, E1 render profile:
  all10:
    frame_mean          ~= 6.22 ms
    render_mean         ~= 5.70 ms
    render_overhead     ~= -0.15 ms
    pack_rgb8_mean      ~= 0.12 ms
    raygen_kernel       ~= 0.18 ms
    first_hit_kernel    ~= 1.97 ms
    shade_kernel        ~= 3.16 ms
    readback_copy       ~= 0.15 ms

  steady frames 1-8:
    frame_mean          ~= 6.23 ms
    render_mean         ~= 5.73 ms
    render_overhead     ~= -0.15 ms
    pack_rgb8_mean      ~= 0.11 ms
    raygen_kernel       ~= 0.17 ms
    first_hit_kernel    ~= 1.96 ms
    shade_kernel        ~= 3.20 ms
    readback_copy       ~= 0.15 ms
```

Because render profiling inserts extra synchronization, these profile-on rows
are diagnostic proportions rather than the primary throughput baseline. They
indicate the next render-side work should start with shade cost and first-hit
traversal before worrying about RGB8 packing.

E2 shadow optimization starts from same-device E1 profile comparisons, not from
a kernel rewrite. The first comparison set should be:

```text
1080p shadow    vs no-shadow, readback=rgb8, torch_async ring_depth=2
1080p shadow    vs no-shadow, readback=none
render-profile on for both pairs
```

The immediate question is whether the shadow delta lives almost entirely inside
`shade_kernel_ms`, or whether first-hit/profile/allocation effects are also
moving. Candidate follow-ups are deliberately ordered from low-risk to larger
kernel changes: reuse point-light distance in the shade kernel, skip unneeded
`intensity` allocation/write for RGB preview, make shadow diagnostics optional
for non-debug output profiles if atomics show up, then benchmark a stack-only
shadow any-hit traversal before considering a separate shadow-ray queue or
specialized shadow shade kernel.

The E2.0 same-device run on GPU1 answered the first question:

```text
1920x1080, frames=20, warmup_renders=5, steady frames 1-18

rgb8 async preview:
  shadow:
    frame_total_ms      ~= 7.69
    render_execute_ms   ~= 7.13
    first_hit_kernel_ms ~= 2.09
    shade_kernel_ms     ~= 3.38
    readback_host_ms    ~= 2.31

  no-shadow:
    frame_total_ms      ~= 4.53
    render_execute_ms   ~= 3.95
    first_hit_kernel_ms ~= 2.09
    shade_kernel_ms     ~= 0.26
    readback_host_ms    ~= 2.35

  shadow delta:
    frame_total_ms      ~= +3.17
    render_execute_ms   ~= +3.18
    shade_kernel_ms     ~= +3.12
```

The repeated `readback=none` pair gave the same conclusion: shadow adds about
3.1 ms, almost entirely inside `shade_kernel_ms`; first-hit and delivery do not
move materially. The first `shadow_none` run had allocation/raygen spikes and is
treated as an outlier; `shadow_none_repeat` is the stable row.

The first E2.1 stack-only shadow any-hit experiment removed the per-shadow-ray
`stack_t` local float array while keeping the integer node stack. Compared with
the E2.0 `shadow_rgb8_async` baseline:

```text
stack-only shadow any-hit, steady frames 1-18:
  frame_total_ms      ~= 7.03  (baseline 7.69)
  render_execute_ms   ~= 6.46  (baseline 7.13)
  shade_kernel_ms     ~= 2.78  (baseline 3.38)
  first_hit_kernel_ms ~= 2.09  (baseline 2.09)
  readback_host_ms    ~= 2.31  (baseline 2.31)
```

So `stack_t` is not the whole shadow bottleneck, but it is a real part of it:
about 0.6 ms recovered from `shade_kernel_ms` on this scene. The remaining
shadow cost is still the any-hit traversal itself.

The E2.2 traversal-counter run then split that remaining cost:

```text
shadow traversal counters, steady frames 1-18:
  shadow rays        ~= 3.82M/frame
  directional rays   ~= 1.92M/frame
  point rays         ~= 1.91M/frame
  occluded rays      ~= 0.18M/frame
  unoccluded rays    ~= 3.65M/frame
  node visits        ~= 12.32M/frame
  leaf visits        ~= 0.95M/frame
  triangle tests     ~= 0.95M/frame
  plane tests        ~= 3.65M/frame

per shadow ray:
  node visits        ~= 3.22
  triangle tests     ~= 0.25
  plane tests        ~= 0.95
  occluded ratio     ~= 4.6%
```

This suggests the remaining cost is not deep BVH traversal. It is mostly shadow
ray volume: the Go2 setup launches almost two shadow rays per hit pixel because
both the directional key light and point fill light cast shadows, and most rays
are unoccluded. The single floor plane fallback is also tested for almost every
unoccluded ray and did not appear to occlude in this run. The next likely levers
are per-light shadow-casting policy and plane-shadow policy before deeper BVH
traversal micro-optimization.

Against the float32 RGB async ring baseline, RGB8 cuts measured copy time from
about 10.21 ms to 2.33 ms (roughly 4.4x lower) and reduces all-frame ordered
frame time from about 10.78 ms to 6.71 ms (roughly 1.6x faster). Against the
sync RGB warmup=5 baseline, it reduces all-frame frame time from about 24.42 ms
to 6.71 ms (roughly 3.6x faster).

The lab should validate this with measured overlap, not assume it.

### A.6 Next Tuning Questions For This Scenario

After migration into the Optical Pipeline Lab:

```text
1. Can async D2H reduce ordered RGB export frame time?
2. Does pinned host memory or Torch/Warp async copy provide real overlap?
3. How much of host time is RGB materialization vs PNG encoding?
4. Does GPU RGB8 packing beat float32 RGB readback plus CPU conversion?
5. At 1920x1080, does readback or render dominate?
6. Does shadow cost remain acceptable at larger resolution and camera paths?
7. Does CUDA LBVH build/refit remain small enough for video without overlap?
8. Which measurements change when geometry becomes dynamic?
```

These questions should be answered with lab scenarios, not ad hoc example
flags.
