# Q54 Render Video Current Design And R3 Plan

Date: 2026-05-13
Author: Codex
Status: review-request

## Context

R1 and R2 are now reviewed and committed:

```text
6763987 Align optical delivery runtime contract
be9d6ba Bridge lab delivery frames to runtime result
```

R1 added the CPU-safe delivery runtime protocol and clarified
`DeliveryResult`. R2 added a bridge from lab `DeliveredVideoFrame` to runtime
`DeliveryResult` without making the lab facade implement the runtime protocol.

This note reviews the current render-video design before deciding whether R3
should implement `Go2RenderPipeline.create_delivery_runtime(...)`.

## Current Render Video Shape

The current video path is split into four layers.

### 1. Long-Lived Resource Owner

`Go2RenderSession` owns the concrete Go2 lab resources:

```text
scene
device / stream
base GpuPublishedFrame
DeviceOpticalSceneCache
static snapshot
BVH
direct-light executor
RGB8 pack helper
```

This is still lab-local, but its ownership shape matches the intended
RenderSession direction.

### 2. Frame Context

`Go2RenderPipeline.begin_frame(...)` returns `Go2RenderFrameContext`.

For static video frames:

```text
begin_frame(frame_inputs=None)
  -> uses session snapshot/BVH
  -> prepare timing is NaN
```

For dynamic lab frames:

```text
begin_frame(frame_inputs=GpuPublishedFrame)
  -> snapshots the frame
  -> refits or rebuilds acceleration state
  -> records snapshot/refit/rebuild timing
```

`Go2RenderFrameContext.render(RenderRequest)` returns runtime `RenderResult`.
It owns the compute result and render timing. It does not own RGB8 pack,
readback, writer timing, frame CSV rows, or progress output.

### 3. Video Render Envelope

`_render_video_frame(...)` builds the camera/rays, calls the frame context, and
wraps the runtime render output in lab-local `RenderedVideoFrame`.

`RenderedVideoFrame` currently carries:

```text
frame_index
camera
result
camera_rays_ms
render_execute_ms
render_profile_row
include_shadow_traversal_stats
geometry_mode
prepare_timing
```

This is the important extra layer. It is not just `RenderResult`; it is
`RenderResult` plus video-loop metadata needed by delivery, CSV, overflow
diagnostics, progress lines, and dynamic-frame reporting.

### 4. Delivery Facade And CSV Builder

`VideoDeliveryFacade` owns sync and torch async delivery:

```text
submit(RenderedVideoFrame, frame_start=...)
complete_available(latest_rendered_frame_index=...)
flush()
```

It owns:

```text
RGB8 pack
host readback
async readback ring
image build/write
delivery timing
completion identity
lag/ring/block metadata
```

`VideoFrameTimingRowBuilder` owns the lab CSV/progress flattening. It consumes
`DeliveredVideoFrame`, not runtime `DeliveryResult`, because it still needs
lab-only fields:

```text
observed_frame_ms
frame_path
overlap_ratio
camera/video metadata
render_profile_row
prepare_timing
```

## What Is Good About The Current Design

The hot path now has explicit ownership:

```text
begin_frame:
  snapshot/refit/rebuild timing

render:
  RenderRequest -> RenderResult
  render_execute/profile timing

delivery:
  pack/readback/write timing
  completed_frame_index
  ring/lag/block metadata

row builder:
  lab CSV/progress/export formatting
```

The sync and async video loop share one body:

```python
for frame_index in frames:
    rendered = _render_video_frame(...)
    for completed in delivery.complete_available(...):
        record(completed)
    completed = delivery.submit(rendered, frame_start=...)
    if completed is not None:
        record(completed)
    for completed in delivery.complete_available(...):
        record(completed)

for completed in delivery.flush():
    record(completed)
```

This is the right control-flow shape for async ordered delivery because the
blocking/completion points are explicit.

## Current Tension

`OpticalDeliveryRuntime.submit(...)` is intentionally generic:

```python
submit(rendered: RenderResult, *, frame_start: float | None = None)
```

But `VideoDeliveryFacade.submit(...)` needs lab video metadata:

```python
submit(rendered: RenderedVideoFrame, *, frame_start: float)
```

Those are not structurally compatible.

The missing design decision is where this metadata should live if delivery
runtime becomes generic:

```text
camera
camera_rays_ms
geometry_mode
prepare_timing
include_shadow_traversal_stats
render_profile_row / render timing flattening
observed frame start
```

Forcing `VideoDeliveryFacade` to implement `OpticalDeliveryRuntime` now would
either:

- hide side-channel metadata behind mutable maps;
- bloat `RenderResult` with lab CSV fields;
- or make the generic protocol less generic.

All three are worse than the current explicit lab bridge.

## R3 Options

### Option A — Do Not Implement The Pipeline Hook Yet

Keep:

```python
Go2RenderPipeline.create_delivery_runtime(...)
  -> NotImplementedError
```

Use R1/R2 as the stable contract and bridge layer. Run R4 GPU smoke next.

Pros:

- no semantic churn;
- current video loop stays clear;
- avoids a fake protocol conformance.

Cons:

- `create_delivery_runtime(...)` remains a stub;
- generic runtime path is still aspirational.

### Option B — Add A Lab Video Render Envelope Type

Make the current envelope explicit:

```python
@dataclass
class VideoRenderedFrame:
    render: RenderResult
    frame_index: int
    camera: OpticalPinholeCameraSpec
    camera_rays_ms: float
    geometry_mode: str
    prepare_timing: Mapping[str, float]
    include_shadow_traversal_stats: bool
```

Then make `RenderedVideoFrame` either an alias or a transitional name.

Pros:

- documents the missing layer;
- keeps `RenderResult` generic;
- sets up a future adapter from `RenderResult + metadata` to video delivery.

Cons:

- mostly naming/shape cleanup;
- does not by itself make `VideoDeliveryFacade` implement
  `OpticalDeliveryRuntime`.

### Option C — Add A Lab-Specific Delivery Runtime Adapter

Create a wrapper that implements `OpticalDeliveryRuntime`, but internally
requires a side table from completed `RenderResult` to video metadata.

Pros:

- `create_delivery_runtime(...)` can return something.

Cons:

- introduces hidden mutable state;
- makes completion identity and metadata association easier to get wrong;
- obscures the simple current loop.

I do not recommend this option.

## Recommendation

Do not implement R3 as a code change yet.

Recommended next step:

```text
R3a:
  review and accept the current-design note

R4:
  run go2_video_delivery_smoke on GPU when available

R3b:
  if more cleanup is desired, introduce an explicit lab video render envelope
  before attempting create_delivery_runtime(...)
```

In other words:

```text
Current bridge is good:
  DeliveredVideoFrame -> DeliveryResult

Missing bridge is not delivery:
  RenderResult + video metadata -> lab delivery input
```

That missing bridge should be solved explicitly, not smuggled through
`create_delivery_runtime(...)`.

## Review Questions

1. Do you agree that `VideoDeliveryFacade` should not implement
   `OpticalDeliveryRuntime` until the video render envelope is explicit?

2. Should we rename or reshape `RenderedVideoFrame` into a clearer
   `VideoRenderedFrame` / `VideoRenderEnvelope`, or keep the current name?

3. Should `RenderResult` ever grow `camera` or `frame_index`, or should those
   stay in frame/video envelopes?

4. Is R4 GPU smoke the better next action before more R3 code?

5. If we do introduce a video envelope, should it store the full `RenderResult`
   plus lab metadata, or keep the current flattened fields for CSV stability?

## Codex Recommendation

```text
Q1: yes, do not force protocol conformance yet
Q2: keep current name for now; document it as a lab video envelope
Q3: keep camera/frame_index outside RenderResult
Q4: yes, run R4 GPU smoke next if GPU is available
Q5: eventually store RenderResult plus metadata, but not as an urgent change
```
