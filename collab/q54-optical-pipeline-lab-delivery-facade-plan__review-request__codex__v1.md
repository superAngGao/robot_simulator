# Q54 Optical Pipeline Lab Delivery Facade Plan Review Request

Date: 2026-05-12
Author: Codex
Status: review-request

## Context

A8.4 is merged at:

```text
4c1b399 Add dynamic optical smoke preset
```

The lab now has:

- static Go2 video ordered export;
- synthetic dynamic rigid video smoke;
- shared `Go2RenderPipeline.begin_frame(frame_inputs=...)`;
- sync and `torch_async` video delivery paths;
- RGB/RGB8/full/none readback vocabulary;
- stable frame timing CSV columns for render, delivery, and dynamic prepare
  timing.

The next cleanup should make delivery ownership clearer before more consumer
presets or production-facing camera APIs are added.

## Design Source

This plan follows `GPU_OPTICAL_PIPELINE_DESIGN.md`, especially the E0/E1
internal API guidance:

```text
snapshot -> accel -> render -> device result -> delivery -> consumer
```

and the timing ownership split:

```text
RenderFrameContext:
  owns prepare timing: snapshot_ms, accel_refit_ms, accel_rebuild_ms

RenderResult:
  owns render timing: render_execute_ms, render profile phases
  does not own RGB8 pack, host readback, writer, or frame-level timing

DeliveryResult:
  owns delivery timing: pack_rgb8_ms, readback_submit_ms,
  readback_wait_ms, readback_host_ms, image_build_ms, encode_write_ms
  may retain device resources until async copy completion

FrameResult:
  lightweight frame-bound summary
  constructed when delivery completes for async delivery
```

This is an internal lab step, not a frozen simulator API.

## Problem

`tools/optical_pipeline_lab/go2_backend.py` still mixes delivery concerns into
the video loops:

1. Sync and `torch_async` paths each build frame timing rows.
2. Overflow diagnostics, image/write timing, rolling FPS, and CSV defaults are
   spread across both paths.
3. RGB8 packing is triggered from `_render_video_frame(...)`, even though the
   design says RGB8 is a delivery payload/packing choice.
4. `DeliveryRequest` exists, but the loops still interpret delivery policy and
   payload in several places.
5. Async ring ownership lives in a helper, but ordered delivery semantics are
   still mostly example-loop control flow.

This makes the next features risky:

- consumer presets that select delivery behavior;
- dynamic scenes combined with async delivery;
- future `FrameResult` summaries;
- eventual production `session.deliver(rendered, request)` API.

## Proposed Slice

Add a lab-local delivery facade while preserving behavior and CSV schema.

Suggested new module:

```text
tools/optical_pipeline_lab/delivery.py
```

Primary goal:

```text
go2_backend render loop:
  build/render frame
  submit/complete delivery
  write completed timing row
```

The delivery facade should be shaped so it can later migrate toward:

```python
rendered = frame_context.render(render_request)
delivered = session.deliver(rendered, delivery_request)
```

but this patch should not introduce a production `OpticalRenderSession`.

## Non-Goals

Do not do these in this cleanup:

- do not change Q52 `GpuPublishedFrame` borrow/complete lifecycle;
- do not change snapshot, BVH, executor, or dynamic frame ownership;
- do not introduce public `OpticalCameraStream` or simulator-facing camera API;
- do not add `async_latest`, frame dropping, streaming preview, or sensor publish;
- do not change frame timing CSV column names or existing lab CLI flags;
- do not claim final async ordered runtime semantics beyond the current
  `torch_async` ordered lab path.

## Proposed API Shape

### Delivery Request Adapter

Keep `_video_delivery_request(...)` or move it into `delivery.py`:

```python
def video_delivery_request(
    *,
    readback_mode: str,
    delivery_mode: str,
    ring_depth: int,
    write_frames: bool,
) -> DeliveryRequest:
    ...
```

Mapping remains:

```text
readback=none
  -> payload=NONE, policy=DEVICE_ONLY

readback=rgb/rgb8/full + delivery=sync
  -> payload=<mode>, policy=SYNC_HOST

readback=rgb/rgb8 + delivery=torch_async
  -> payload=<mode>, policy=TORCH_ASYNC_ORDERED
```

The lab CSV may keep `delivery_policy=sync|torch_async` for continuity, while
the internal enum remains `SYNC_HOST|DEVICE_ONLY|TORCH_ASYNC_ORDERED`.

### Rendered Video Frame

Keep the existing `_RenderedVideoFrame` concept, but remove RGB8 packing from
render:

```python
@dataclass
class RenderedVideoFrame:
    frame_index: int
    camera: OpticalPinholeCameraSpec
    result: OpticalComputeResult
    camera_rays_ms: float
    render_execute_ms: float
    render_profile_row: dict[str, float]
    include_shadow_traversal_stats: bool
    geometry_mode: str
    prepare_timing: Mapping[str, float]
```

`pack_rgb8_ms` should move to delivery timing.

### Delivered Video Frame

Add a lab-local completed delivery object:

```python
@dataclass
class DeliveredVideoFrame:
    frame_index: int
    camera: OpticalPinholeCameraSpec
    geometry_mode: str
    host_channels: Mapping[str, object]
    delivery_timing: DeliveryTimingSummary
    readback_lag_frames: int = 0
    readback_ring_depth: int = 0
    readback_ring_block_count: int = 0
    completed_frame_index: int | None = None
    frame_path: str = ""
```

The exact type can stay in `tools/` for now. The important part is that delivery
returns a completed object rather than forcing the video loop to know how each
backend stages, waits, writes, and times.

### Delivery Facade

Candidate object:

```python
class VideoDeliveryFacade:
    @classmethod
    def create(
        cls,
        *,
        pipeline: Go2RenderPipeline,
        args: argparse.Namespace,
        out_dir: Path,
        request: DeliveryRequest,
    ) -> "VideoDeliveryFacade":
        ...

    def submit(
        self,
        rendered: RenderedVideoFrame,
        *,
        frame_start: float,
    ) -> Iterable[DeliveredVideoFrame]:
        ...

    def drain(self) -> Iterable[DeliveredVideoFrame]:
        ...
```

Sync implementation:

```text
submit(rendered)
  -> optional RGB8 pack
  -> blocking host readback or no-op for device_only
  -> optional PNG write
  -> return one DeliveredVideoFrame immediately
drain()
  -> no-op
```

Torch async ordered implementation:

```text
create()
  -> warmup representative result
  -> allocate TorchAsyncReadbackRing

submit(rendered)
  -> optional RGB8 pack
  -> submit non-blocking D2H copy
  -> complete previous pending job in order when available
  -> return zero or one completed DeliveredVideoFrame

drain()
  -> complete final pending job
```

This preserves the existing ordered, no-drop behavior. It does not implement
latest/drop delivery.

### Row Builder

Move row construction into one shared helper:

```python
def build_video_frame_timing_row(
    *,
    rendered: RenderedVideoFrame,
    delivered: DeliveredVideoFrame,
    delivery_request: DeliveryRequest,
    args: argparse.Namespace,
    frame_total_ms: float,
    instant_fps: float,
    rolling_fps: float,
) -> dict[str, object]:
    ...
```

The helper owns:

- `readback_mode` / `write_mode` CSV values;
- prepare timing flattening;
- render timing flattening;
- delivery timing flattening;
- overflow scalar extraction;
- shadow traversal fields;
- async lag/ring/completed-frame fields.

`FrameTimingRecorder` remains unchanged.

## Implementation Steps

### F0 — Move Delivery Data Shapes

Create `tools/optical_pipeline_lab/delivery.py` with:

- `RenderedVideoFrame` if moving out of `go2_backend.py` is not too noisy;
- `DeliveredVideoFrame`;
- `video_delivery_request(...)`;
- shared delivery timing helpers.

Keep the first patch mechanical and testable.

### F1 — Sync Delivery Facade

Extract the sync body from `_run_video_benchmark(...)` into the facade:

- host staging;
- RGB8 pack if requested;
- image build/write;
- `readback=none` device-only behavior;
- overflow channel availability.

The sync video loop should still complete one row per rendered frame.

### F2 — Torch Async Ordered Facade

Move async pending-job mechanics behind the same facade:

- ring warmup;
- submit;
- ordered completion of previous pending job;
- final drain;
- ring depth/block count/lag metadata.

Existing `tools/optical_pipeline_lab/async_readback.py` should remain the
low-level pinned-host ring implementation. The new facade owns lab video
semantics, not raw copy mechanics.

### F3 — RGB8 Pack Ownership Cleanup

Move RGB8 pack from `_render_video_frame(...)` to delivery.

Expected result:

```text
render_execute_ms excludes RGB8 pack
pack_rgb8_ms is populated by DeliveryResult/DeliveredVideoFrame
```

This matches `GPU_OPTICAL_PIPELINE_DESIGN.md`. If this changes benchmark
numbers, call it out explicitly in the implementation note. The CSV column is
already present, so the schema does not change.

### F4 — Thin Video Loops

After F1/F2/F3:

```text
_run_video_benchmark:
  create delivery facade
  for frame:
    rendered = _render_video_frame(...)
    for delivered in facade.submit(...):
      rows.add(build_row(...))
  for delivered in facade.drain():
    rows.add(build_row(...))
```

If useful, keep separate sync/async loop functions initially, but both should
call the same delivery facade and row builder.

## Validation Plan

CPU/unit tests:

- `DeliveryRequest` mapping remains unchanged.
- sync `readback=none` still does not stage host channels.
- sync RGB/RGB8/full row fields preserve existing values.
- torch async rejects non-RGB payloads.
- torch async ring depth validation remains unchanged.
- async row construction preserves:
  - `readback_submit_ms`;
  - `readback_wait_ms`;
  - `readback_host_ms`;
  - `readback_lag_frames`;
  - `readback_ring_depth`;
  - `readback_ring_block_count`;
  - `completed_frame_index`.
- RGB8 pack timing is delivery-owned.
- dynamic video frame inputs still propagate `geometry_mode=dynamic_rigid` and
  prepare timing fields.

GPU smoke:

```text
tests/gpu/test_optical_gpu_runtime.py::
  test_optical_lab_dynamic_video_loop_writes_prepare_timing_csv
  test_optical_lab_dynamic_smoke_preset_writes_prepare_timing_csv
```

Add or retain an RGB8 async smoke if the environment supports it:

```text
go2_video_ordered_static, readback=rgb8, delivery=torch_async, ring_depth=2
```

Manual lab run after implementation, on a GPU host:

```bash
python -m tools.optical_pipeline_lab run \
  --preset go2_video_ordered_static \
  --width 160 --height 120 \
  --frames 3 \
  --readback rgb8 \
  --video-readback-delivery torch_async \
  --video-readback-ring-depth 2 \
  --progress-every 0
```

## Compatibility Rules

Preserve these unless a review explicitly approves a change:

- no frame timing CSV column rename;
- no CLI flag rename;
- `readback=none` remains timing-only/device-only and cannot write frames;
- `torch_async` remains ordered/no-drop;
- `full + torch_async` remains rejected;
- dynamic frame inputs remain owned by the video/render path, not delivery;
- `.codex` stays untouched.

## Risks

### Risk: Hidden Benchmark Shift From RGB8 Ownership

Moving RGB8 pack from render to delivery should not change total frame work, but
it will make the ownership clearer. Existing tests should assert that
`render_execute_ms` no longer includes pack time only if current behavior is
updated intentionally.

### Risk: Async Completion Row Identity

Async rows must represent the completed frame, not the latest rendered frame.
This is easy to regress if row construction becomes too generic. Keep
`completed_frame_index` explicit in `DeliveredVideoFrame`.

### Risk: Facade Becomes Production API Too Early

Keep it in `tools/optical_pipeline_lab/`. Do not move it into `optics/` until
the RenderSession/workspace shape has another review.

## Review Questions

1. Is `tools/optical_pipeline_lab/delivery.py` the right temporary home, or
   should the facade stay inside `go2_backend.py` until after one extraction?
2. Should RGB8 pack move to delivery in the same patch as the facade, or should
   it be a follow-up patch after sync/async row building is unified?
3. Is `VideoDeliveryFacade.submit(...)->Iterable[DeliveredVideoFrame]` a good
   enough shape for sync and ordered async, or would separate `submit` and
   `complete_available` calls make async ownership clearer?
4. Should the lab CSV continue to write `delivery_policy=torch_async`, or should
   it switch to the internal enum value `torch_async_ordered`?
5. Are there any delivery semantics in `GPU_OPTICAL_PIPELINE_DESIGN.md` that
   should block this lab-local cleanup before implementation?
