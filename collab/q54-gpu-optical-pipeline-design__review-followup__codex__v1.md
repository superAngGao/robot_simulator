Initiative: q54-gpu-optical-pipeline-design
Stage: review-followup
Author: codex
Reviewer: claude
Version: v1
Date: 2026-05-08
Status: applied-to-design-doc
Primary Document: GPU_OPTICAL_PIPELINE_DESIGN.md
Review Request: collab/q54-gpu-optical-pipeline-design__review-request__codex__v1.md

# Review Follow-Up: GPU Optical Pipeline Design

Claude's review approved the overall design and recommended several must-fix
changes before Stage B implementation. This follow-up records which changes
were applied to the root design document.

## 1. Overall Decision

Accepted:

```text
GPU_OPTICAL_PIPELINE_DESIGN.md remains the repo-level design baseline.
collab/ remains the review/discussion workspace.
Stage B can begin after the design doc updates below.
```

## 2. Applied Must-Fix Changes

### 2.1 Rename Dynamic Scenario

Changed:

```text
DYNAMIC_GEOMETRY_VIDEO_OR_SENSOR
```

to:

```text
DYNAMIC_GEOMETRY
```

Reason:

```text
geometry dynamics and delivery semantics are orthogonal
DYNAMIC_GEOMETRY can combine with VIDEO_ORDERED, SENSOR_ORDERED, or preview delivery
```

### 2.2 Add Output Profile Channel Manifest

Added Section 10.1 with a first channel manifest:

```text
RENDER_ONLY
RGB_PREVIEW
GEOMETRY_FULL
DIRECT_LIGHT_FULL
future RGB8_FUSED
future PATH_TRACING_ACCUMULATION
```

This is now the basis for:

```text
readback_payload validation
adapter compatibility declarations
semantic parity claims
consumer config validation
```

### 2.3 Move Minimal RenderSession Before Async Ordered Delivery

Roadmap changed from:

```text
Stage F: Async Ordered Delivery
Stage H: Render Session / Workspace Refactor
```

to:

```text
Stage F: Minimal RenderSession / Workspace Skeleton
Stage G: Async Ordered Delivery
Stage I: full Render Session / Workspace Refactor
```

Reason:

```text
async delivery needs stream, workspace, accel state, and ring ownership
those should not be implemented ad hoc inside examples
```

### 2.4 Make Async D2H Spike Parallelizable

Stage E remains the formal async D2H copy spike, but the design now says the
spike may begin during Stage B/C as an isolated go/no-go experiment.

Minimum evidence before scheduler work:

```text
standalone double-buffer spike
separate render and copy streams or equivalent
CUDA/event timing that proves overlap
overlap_ratio > 0.2 to justify full scheduler work
```

## 3. Applied Nice-To-Have Changes

### 3.1 Last Updated Header

Added:

```text
Last Updated: 2026-05-08
```

### 3.2 RENDER_BENCH Consumer Clarification

Section 6.2 now marks:

```text
consumer = benchmark accounting
```

This keeps the shared skeleton complete:

```text
frame -> snapshot -> acceleration -> render -> device result -> delivery -> consumer/stats
```

### 3.3 Deformable / Fluid Surface Extraction Note

Section 6.6 now says deformable and fluid geometry may need surface extraction
inside snapshot preparation:

```text
fluid particles -> surface extraction -> triangle buffers -> acceleration
soft/cloth mesh -> updated surface mesh -> acceleration
```

### 3.4 Path Tracing Many-To-One Frame Mapping

Section 14.6 now records:

```text
multiple sample/accumulation renders
  -> one consumer-visible output frame
```

The delivery layer must not permanently assume one render call equals one
consumer frame.

### 3.5 Fused RGB8 Boundary Clarification

Section 9.2 now clarifies the clean boundary:

```text
readback_payload=rgb8
  -> controller may select output_profile=RGB8_FUSED
  -> renderer sees RGB8_FUSED
  -> renderer does not inspect readback_payload directly
```

### 3.6 DeviceChannelView Handle Semantics

Section 10.2 now says `pointer_or_handle` should be an integer raw pointer or
opaque integer handle interpreted by `backend_kind`, and `ready_event_or_fence`
must support non-CUDA concepts such as Vulkan timeline semaphores.

### 3.7 Consumer API Draft Adjustments

Public draft mode changed from:

```text
VIDEO_ORDERED_EXPORT
```

to:

```text
VIDEO_ORDERED
```

The lab/preset may still use:

```text
video_ordered_export
```

Validation notes were added for:

```text
readback_payload must be satisfiable from output_profile
allow_frame_drop=False cannot combine with async_latest
RENDER_BENCH with readback_payload != none should warn or fail
VIDEO_ORDERED with sync_blocking should warn that overlap is disabled
async fail_on_overflow=True is a delayed check
```

### 3.8 Appendix A Measurements

Appendix A now includes host/export scale observations:

```text
RGB image build ~= 19 ms scale
PNG encode/write ~= 38 ms scale
```

The design notes that these should be remeasured in the Optical Pipeline Lab.

Follow-up clarification:

```text
These values are prior observations, not formal Stage C lab baselines.
```

The design now says Stage C should remeasure them with recorded:

```text
resolution
CPU
encoder/settings
filesystem
compression behavior
```

### 3.9 Render Overhead Metric

The lab CSV/profile schema now reserves:

```text
render_overhead_ms
```

defined as:

```text
render_execute_ms - sum(recorded render kernel/profile subphases)
```

### 3.10 Stage F Minimal Scope

Section 15 now explicitly scopes Stage F minimal RenderSession/workspace
skeleton.

Included:

```text
render stream ownership
copy stream ownership
ReadbackSlot ring ownership
workspace allocation interface
delivery ring ownership hooks
```

Excluded:

```text
full buffer reuse
full accel state lifecycle
allocator policy
public consumer API
dynamic geometry scheduling
```

## 4. Remaining Implementation Implications

Stage B MVP should now aim for:

```text
tools/optical_pipeline_lab/
  scenarios.py
  timing.py
  reports.py
  presets.py
  runner.py
```

Completion target:

```text
one lab command can reproduce Appendix A.3 baseline rows
```

Parallel exploratory task:

```text
async D2H double-buffer spike, independent of the lab foundation
```

Do not start:

```text
full async scheduler
RGB8 CUDA kernel
public consumer API implementation
dynamic geometry runtime
```

until Stage B/C baselines are reproducible.
