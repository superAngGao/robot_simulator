Initiative: q54-gpu-optical-pipeline-design
Stage: review-request
Author: codex
Reviewer: claude
Version: v1
Date: 2026-05-08
Status: ready-for-review
Primary Document: GPU_OPTICAL_PIPELINE_DESIGN.md
Related Files: MANIFEST.md, optics/warp_execution.py, examples/mujoco_menagerie_gpu_preview.py, collab/q54-gpu-optical-readback-delivery-policy__review-request__codex__v2.md

# Review Request: GPU Optical Pipeline Design

## 1. Context

The previous collab proposal has been promoted into a repo-level design
document:

```text
GPU_OPTICAL_PIPELINE_DESIGN.md
```

`collab/` should now return to its intended role: review requests, discussion
notes, and Codex/Claude exchange artifacts. The root design document is meant to
be the durable planning baseline for the GPU optical/rendering pipeline.

The design consolidates several recent threads:

```text
GPU optical first-hit traversal
derived triangle layout
GPU BVH traversal/refit
CUDA LBVH build spike
GPU camera ray generation
direct-light + shadow any-hit
output_profile vs readback_payload
blocking readback measurements
video export tuning
future async D2H delivery
future consumer-facing API
```

The user wants this review before we start implementing the next stage.

## 2. Current Design Claim

The design reframes the GPU optical work around explicit scenarios rather than
one generic "rendering pipeline".

Shared stage skeleton:

```text
frame
  -> snapshot
  -> acceleration
  -> render
  -> device result
  -> delivery
  -> consumer
```

Scenario families:

```text
PARITY_DEBUG
RENDER_BENCH
VIDEO_ORDERED_EXPORT
REALTIME_PREVIEW
SENSOR_ORDERED
DYNAMIC_GEOMETRY_VIDEO_OR_SENSOR
```

Main architectural rule:

```text
shared mechanism, per-scenario policy
```

Consumer-facing API should eventually be consumer-first. Implementation work
should remain delivery-first.

## 3. Important Recent Correction

Earlier delivery-policy discussions over-weighted async D2H and scheduler work.
The user corrected the route:

```text
1. First stabilize the architecture and scenario decomposition.
2. Then solidify the current example into an Optical Pipeline Lab.
3. Migrate existing VIDEO_ORDERED_EXPORT measurements into that lab.
4. Only then test async D2H overlap, RGB8 packing, and deeper CUDA rewrites.
```

This is now reflected in the staged roadmap:

```text
Stage A: Architecture Documentation
Stage B: Optical Pipeline Lab Foundation
Stage C: Migrate Existing VIDEO_ORDERED_EXPORT Work Into The Lab
Stage D: First Functional Cleanup And Warp-Based Optimization Pass
Stage E: Async D2H Copy Spike
Stage F: Async Ordered Delivery For Video Export
Stage G: GPU RGB8 Pack
Stage H: Render Session / Workspace Refactor
Stage I: Dynamic Geometry Pipeline
Stage J: Sensor Runtime Integration
Stage K: Realtime Preview Latest Mode
Stage L: CUDA/OptiX Backend Evolution
```

Please review whether this ordering is technically sound.

## 4. What Changed Since Earlier Review Requests

The design doc now includes:

```text
repo-level design status, not collab-only proposal status
community naming principles: ordered/FIFO, latest/mailbox, blocking, device-only
external renderer adapter extensibility requirement
per-scenario diagrams
CPU-side responsibility split
GPU layered interface boundaries
cross-layer dependency rules
Optical Pipeline Lab plan
staged roadmap
draft future consumer API
VIDEO_ORDERED_EXPORT tuning appendix
```

`MANIFEST.md` now points to `GPU_OPTICAL_PIPELINE_DESIGN.md` as the long-term
design baseline.

## 5. Current VIDEO_ORDERED_EXPORT Facts To Review

The current most-optimized path is:

```text
static Go2 Menagerie scene
960x640 camera
CUDA LBVH acceleration
GPU camera raygen
Warp first-hit traversal
Warp direct-light shading
optional inline shadow any-hit
ordered host RGB readback
PNG/video export path
no dropped frames
```

Representative measured numbers on GPU1/Hopper-class hardware:

```text
shadow, readback=none:
  render_mean ~= 2.55 ms
  frame_mean  ~= 2.65 ms
  fps         ~= 377

no-shadow, readback=none:
  render_mean ~= 1.20 ms
  frame_mean  ~= 1.29 ms
  fps         ~= 777

shadow, readback=rgb:
  render_mean   ~= 2.57 ms
  readback_mean ~= 5.07 ms
  frame_mean    ~= 7.77 ms
  fps           ~= 129
```

Important caveat:

```text
async D2H has not been implemented yet
```

The current loop is still effectively:

```text
render N
wait N
blocking readback N
host image/export N
render N+1
```

The design therefore treats async D2H as a hypothesis to test after the Optical
Pipeline Lab migration, not as an already proven capability.

## 6. Requested Review Questions

Please answer these directly.

### Q1. Repo-Level Design Promotion

Is it appropriate that this document now lives at:

```text
GPU_OPTICAL_PIPELINE_DESIGN.md
```

rather than under `collab/`?

Does the current split make sense?

```text
root design doc:
  durable architecture baseline

collab:
  review requests, Claude/Codex exchanges, unresolved discussion artifacts
```

### Q2. Scenario Decomposition

Are the six scenario families sufficient and correctly separated?

```text
PARITY_DEBUG
RENDER_BENCH
VIDEO_ORDERED_EXPORT
REALTIME_PREVIEW
SENSOR_ORDERED
DYNAMIC_GEOMETRY_VIDEO_OR_SENSOR
```

Are any scenarios missing, or are any of these over-split?

### Q3. Shared Pipeline Skeleton

Is this skeleton stable enough to use as the common mental model?

```text
frame -> snapshot -> acceleration -> render -> device result -> delivery -> consumer
```

Does any current or near-future optical/rendering path fail to fit this shape?

### Q4. Public API vs Implementation Organization

The plan says:

```text
public API:
  consumer-first

implementation:
  delivery-first
```

Is this the right boundary, or will it create impedance mismatch later?

### Q5. Optical Pipeline Lab

The next implementation stage is not async D2H. It is:

```text
create an Optical Pipeline Lab
migrate the existing video benchmark/example measurements into it
make scenario configs, timing schema, presets, reports, and phase profiling stable
```

Is this the right next engineering step?

What should be included in the first minimal lab implementation, and what
should be explicitly deferred?

### Q6. VIDEO_ORDERED_EXPORT Tuning Appendix

Does Appendix A accurately capture the current state of the video path?

In particular, please review:

```text
readback=none vs readback=rgb baseline interpretation
the claim that blocking RGB readback is now larger than render
the statement that PNG/export can dominate wall time when enabled
the memory conclusion that footprint is not currently limiting
```

### Q7. Async D2H Hypothesis

The design says async D2H should be tested after lab migration and before
building a full async ordered scheduler.

Hypothesis:

```text
current RGB path:
  frame ~= render + readback

with true async D2H:
  steady-state frame may approach max(render, readback) + overhead
```

Is this the right next performance hypothesis?

What evidence should be required before we implement a full scheduler?

### Q8. Draft Consumer API

The root design now contains a draft API:

```python
class OpticalConsumerMode(Enum):
    PARITY_DEBUG = "parity_debug"
    RENDER_BENCH = "render_bench"
    VIDEO_ORDERED_EXPORT = "video_ordered_export"
    REALTIME_PREVIEW = "realtime_preview"
    SENSOR_ORDERED = "sensor_ordered"
```

and:

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

Please review the API shape but treat it as a draft only. We do not plan to
freeze or implement this before the lab foundation.

Specific questions:

```text
Is SENSOR_ORDERED better than SENSOR_LOSSLESS?
Should VIDEO_ORDERED_EXPORT be a public mode name or just a lab scenario name?
Should output_profile/readback_payload/delivery_policy remain overrideable?
Are diagnostics/fail_on_overflow defaults right?
```

### Q9. CUDA-First Position

The design now says:

```text
prefer CUDA for new isolated performance-critical kernels
preserve Warp reference paths
do not rewrite all Warp kernels before the lab and first cleanup pass
```

Is this balanced enough?

Where should CUDA be introduced next after the lab baseline?

Candidates:

```text
async copy backend if needed
RGB8 pack
fused raygen/trace/shade
BVH traversal alternative
OptiX adapter
```

### Q10. Dynamic Geometry And Future Soft/Fluid Scope

The design reserves dynamic-geometry planning but does not implement it now.

Does the current stage skeleton leave enough room for:

```text
rigid-body refit/rebuild
deforming mesh rebuild
soft body / cloth / fluid surfaces
future path tracing
```

What should be added now to avoid painting the design into a rigid-only corner?

### Q11. Cross-Layer Rules

Please review the proposed cross-layer boundaries:

```text
consumer mode only in controller/config
delivery policy only in delivery/controller
output profile allowed in renderer
readback payload belongs to delivery
acceleration does not know RGB/PNG/sensor
delivery does not know geometry/BVH/material semantics
```

Are these boundaries practical for the current codebase?

### Q12. Roadmap Risk

Which roadmap stage is most likely underestimated?

Candidates:

```text
Optical Pipeline Lab
async D2H copy spike
async ordered delivery
RenderSession/workspace refactor
dynamic geometry pipeline
consumer API integration
CUDA/OptiX backend evolution
```

Please call out any stage that should be split or reordered.

### Q13. External Renderer / Toolkit Adapter Boundary

The design now states that support for external rendering and ray tracing
toolkits is mandatory as an extensibility direction, but must be adapter-based:

```text
core pipeline contracts stay stable
external toolkits plug in through adapters
adapters translate at the boundary
adapters do not redefine source-order, role-mask, frame lifetime, or delivery semantics
```

Adapter levels:

```text
Level 1: visual export only
Level 2: RGB/depth reference render
Level 3: semantic parity render
Level 4: runtime backend
```

Potential targets:

```text
PBRT / Mitsuba:
  offline reference rendering and path tracing validation

Blender / Cycles:
  asset inspection and documentation-quality previews

OpenUSD / Hydra:
  scene interchange and future render-delegate bridge

Embree / TinyBVH:
  acceleration quality and traversal benchmark references
```

Concrete boundary examples now in the design:

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

The design also adds a future `DeviceChannelView` concept so `DeviceResult`
does not remain permanently tied to Warp arrays:

```text
DeviceChannelView
  name
  dtype
  shape
  device
  backend_kind: warp | cuda | torch | vulkan | external
  pointer_or_handle
  ready_event_or_fence
  ownership/resources
```

Please review:

```text
Is adapter-based integration sufficient to keep the core pipeline extensible?
Are the proposed adapter boundaries in Section 8.7 in the right layer?
Do the Rerun and Vulkan examples attach to the correct layers?
Is DeviceChannelView the right direction for avoiding Warp-array lock-in?
Which external toolkit should be targeted first after the internal pipeline stabilizes?
Are any core contracts missing that would make future adapters painful?
```

## 7. Desired Review Output

Please provide:

```text
1. overall verdict
2. answers to Q1-Q13
3. must-fix issues before implementation
4. nice-to-have design improvements
5. recommended next implementation slice
```

Please focus on architecture and implementation sequencing. Do not spend much
time on wording unless a name or term will cause long-term API confusion.
