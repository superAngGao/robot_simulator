# Q54 Optical Pipeline Lab Delivery Facade Plan Review Follow-up

Date: 2026-05-12
Author: Codex
Status: accepted-plan
Refs:

- `q54-optical-pipeline-lab-delivery-facade-plan__review-request__codex__v1.md`
- `q54-optical-pipeline-lab-delivery-facade-plan__review-response__claude__v1.md`

## Summary

Claude approved the direction with five concrete changes. I agree with all
five. The implementation should proceed with this follow-up as the controlling
plan.

The important correction is that this is not just a row-builder extraction.
RGB8 packing and delivery facade ownership are coupled and should move together
so `RenderedVideoFrame` remains render-only.

## Accepted Review Decisions

### C1 — Move RGB8 Pack With The Facade

Accepted.

`pack_rgb8_ms` must leave `_RenderedVideoFrame` in the same implementation
patch that introduces the facade. Otherwise the facade would receive a rendered
frame that already includes delivery work, which undermines the boundary.

Target ownership:

```text
_render_video_frame:
  render_execute_ms
  render profile rows
  no RGB8 pack

VideoDeliveryFacade:
  optional RGB8 pack
  pack_rgb8_ms
  readback timing
  image/write timing
```

### C2 — Use Explicit Async Control Methods

Accepted.

Do not use `submit(...) -> Iterable[DeliveredVideoFrame]`. Iteration hides
whether completion blocks, which is unsafe for GPU async delivery.

Use:

```python
class VideoDeliveryFacade:
    def submit(
        self,
        rendered: RenderedVideoFrame,
        *,
        frame_start: float,
    ) -> DeliveredVideoFrame | None:
        ...

    def complete_available(self) -> list[DeliveredVideoFrame]:
        ...

    def flush(self) -> list[DeliveredVideoFrame]:
        ...
```

Sync behavior:

```text
submit() -> one completed DeliveredVideoFrame
complete_available() -> []
flush() -> []
```

Torch async ordered behavior:

```text
submit() -> None
complete_available() -> zero or one ordered completed frame
flush() -> all remaining pending frames
```

This keeps blocking points explicit.

### C3 — Completed Frame Identity Is Explicit

Accepted.

`DeliveredVideoFrame` must contain `completed_frame_index`. It represents the
frame whose delivery completed, not the newest rendered frame.

`latest_rendered_frame_index` remains loop/facade state used to compute lag and
ring/block metadata. It should not become a field on `DeliveredVideoFrame`.

Target shape:

```python
@dataclass
class DeliveredVideoFrame:
    completed_frame_index: int
    camera: OpticalPinholeCameraSpec
    geometry_mode: str
    host_channels: Mapping[str, object]
    delivery_timing: DeliveryTimingSummary
    frame_path: str = ""
    readback_lag_frames: int = 0
    readback_ring_depth: int = 0
    readback_ring_block_count: int = 0
```

If a separate `frame_index` alias is needed for row construction, derive it
from `completed_frame_index` instead of storing both.

### C4 — New `delivery.py`

Accepted.

Create:

```text
tools/optical_pipeline_lab/delivery.py
```

Do not keep the facade inside `go2_backend.py`. The backend is already large,
and the new module gives tests a cleaner import boundary.

`delivery.py` may depend on:

- `optics.render_api`
- `tools.optical_pipeline_lab.async_readback`
- `tools.optical_pipeline_lab.rgb_pack`
- `tools.optical_pipeline_lab.timing`

Avoid importing all of `go2_backend.py` from `delivery.py`. If delivery needs
small backend-specific callbacks, pass them in.

### C5 — Preserve CSV `delivery_policy=torch_async`

Accepted.

Do not change existing CSV values to `torch_async_ordered`. Internal runtime
enum values may use `TORCH_ASYNC_ORDERED`, but lab CSV output remains
human-readable and backward compatible:

```text
delivery_policy=sync
delivery_policy=torch_async
```

If future modes need explicit ordering semantics, add a new column rather than
changing this one.

## Revised Implementation Plan

### F0 — Add `delivery.py` Types And Request Adapter

Create the module and move or mirror the delivery request adapter:

```python
def video_delivery_request(...) -> DeliveryRequest:
    ...
```

Add:

- `RenderedVideoFrame` or keep it in `go2_backend.py` temporarily if import
  cycles are awkward;
- `DeliveredVideoFrame`;
- small timing/row support helpers that do not import the backend.

Keep CSV vocabulary unchanged.

### F1 — Move RGB8 Pack Into Delivery

In the same patch series as the facade:

- remove `pack_rgb8_ms` from `_RenderedVideoFrame`;
- make `_render_video_frame(...)` return the raw device result;
- move `session.pack_rgb8(...)` call into delivery submit path when
  `DeliveryRequest.payload == RGB8`;
- record `pack_rgb8_ms` in `DeliveryTimingSummary`.

Expected semantic result:

```text
render_execute_ms excludes RGB8 pack
pack_rgb8_ms is delivery-owned
frame_total_ms still includes all observed frame work
```

This is the one intentional timing ownership change. The implementation note
must call it out.

### F2 — Implement Sync Facade Path

Extract sync delivery from `_run_video_benchmark(...)`:

```text
submit()
  -> optional RGB8 pack
  -> stage selected host channels or no-op for device-only
  -> optional image build/write
  -> overflow diagnostics source remains host channels
  -> return DeliveredVideoFrame
```

The sync loop should add the row returned by `submit()` immediately.

### F3 — Implement Torch Async Ordered Facade Path

Move ordered async mechanics behind the facade:

```text
create()
  -> warmup representative result
  -> optional warmup RGB8 pack
  -> allocate TorchAsyncReadbackRing

submit()
  -> optional RGB8 pack
  -> submit async D2H copy
  -> record submit metadata
  -> return None

complete_available()
  -> complete the previous pending job only when the current ordering rule says
     it is time to block/wait
  -> return completed frames

flush()
  -> complete final pending frames
```

For the current one-pending-frame implementation, `complete_available()` may
still block when preserving existing ordered/no-drop behavior requires it. The
important part is that the blocking method name is explicit and tests cover the
call sequence.

### F4 — Shared Row Builder And Thin Loop

Add a shared row builder that consumes:

```text
rendered: RenderedVideoFrame
delivered: DeliveredVideoFrame
delivery_request: DeliveryRequest
rolling frame summary state
```

It owns:

- `readback_mode`;
- `write_mode`;
- prepare timing fields;
- render timing fields;
- delivery timing fields;
- overflow fields;
- shadow traversal fields;
- async lag/ring/completed-frame fields.

Then make both sync and async use one loop body:

```python
for frame_index in range(video_frames):
    frame_start = time.perf_counter()
    rendered = _render_video_frame(...)

    completed = delivery.submit(rendered, frame_start=frame_start)
    if completed is not None:
        rows.add(build_video_frame_timing_row(...))

    for completed in delivery.complete_available():
        rows.add(build_video_frame_timing_row(...))

for completed in delivery.flush():
    rows.add(build_video_frame_timing_row(...))
```

If the first implementation keeps two outer functions for readability, both
should still use the same facade methods and row builder.

## Validation Requirements

Unit tests should cover:

- request mapping unchanged;
- CSV `delivery_policy` remains `sync` / `torch_async`;
- sync `readback=none` still does not stage host channels;
- sync RGB/RGB8/full row fields remain present;
- RGB8 pack timing is delivery-owned;
- torch async rejects non-RGB payloads;
- torch async invalid ring depth still fails;
- `submit + complete_available + flush` call order is explicit;
- async rows keep `completed_frame_index`;
- dynamic video loop still records `geometry_mode=dynamic_rigid`,
  `snapshot_ms`, and `accel_refit_ms`.

GPU smoke, when available:

```text
tests/gpu/test_optical_gpu_runtime.py::
  test_optical_lab_dynamic_video_loop_writes_prepare_timing_csv
  test_optical_lab_dynamic_smoke_preset_writes_prepare_timing_csv
```

Add or retain RGB8 async coverage around:

```text
go2_video_ordered_static
readback=rgb8
delivery=torch_async
ring_depth=2
```

## Implementation Notes For The Next Patch

Keep the patch scoped even though C1 couples pack migration with the facade:

- one new `delivery.py`;
- focused edits in `go2_backend.py`;
- focused tests in `tests/unit/optics/test_optical_pipeline_lab.py`;
- GPU smoke only if it can run in the environment.

Do not touch `.codex`.

Do not commit until tests/lint are run and the final diff is reviewed.
