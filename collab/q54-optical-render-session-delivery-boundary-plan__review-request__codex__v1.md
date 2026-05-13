# Q54 Optical RenderSession Delivery Boundary Plan

Date: 2026-05-12
Author: Codex
Status: review-request

## Context

The lab-local delivery cleanup is now implemented and reviewed:

```text
a069ec6 Extract optical lab delivery facade
f72cbf1 Tighten optical lab delivery facade coverage
7c8583f Add optical lab delivery smoke matrix
7dfffae Cover delivery smoke matrix CLI dispatch
```

That work moved RGB8 pack into delivery ownership, replaced duplicated sync and
torch async loop code with `VideoDeliveryFacade`, and added a small delivery
smoke matrix that covers sync readback and torch async RGB8 ring delivery.

This plan is the next design step. It should not move code into production yet.
It records what the lab facade taught us about the formal RenderSession /
delivery boundary.

## Design References

`GPU_OPTICAL_PIPELINE_DESIGN.md` already says the next API work should
formalize an internal render/delivery boundary before a public camera API:

```text
RenderSession:
  owns device, streams, workspace, scene cache, snapshot, BVH, executor

RenderResult:
  owns render-side output and render-side timing summary

DeliveryResult:
  owns delivery-side output and delivery-side timing summary
```

It also says async `FrameResult` should be constructed when delivery completes,
not when render is submitted.

The A5 pipeline-entrypoint plan intentionally left `DeliveryResult` unused:

```text
Promote DeliveryResult only when factoring duplicated row/completion code
removes real complexity.
```

The lab facade now satisfies that condition: the duplicated sync/async row and
completion logic has been factored, and the explicit completion API has proven
useful.

## Current Lab Shape

`tools/optical_pipeline_lab/delivery.py` now has the shape we should learn from:

```python
class VideoDeliveryFacade:
    def submit(
        self,
        rendered: RenderedVideoFrame,
        *,
        frame_start: float,
    ) -> DeliveredVideoFrame | None: ...

    def complete_available(
        self,
        *,
        latest_rendered_frame_index: int | None = None,
    ) -> list[DeliveredVideoFrame]: ...

    def flush(self) -> list[DeliveredVideoFrame]: ...
```

The important semantic decisions:

- `submit()` may complete sync delivery immediately, but async submit returns
  `None`.
- `complete_available()` is the explicit place where async completion may
  become visible or block because a ring is full.
- `flush()` owns end-of-loop drain.
- `DeliveredVideoFrame.completed_frame_index` is explicit.
- `DeliveryTimingSummary` owns `pack_rgb8_ms`, readback timing, image build,
  and encode/write timing.
- CSV labels remain stable even when internal enum names are more specific.

## Problem

The current internal protocol in `optics.render_api.OpticalRenderPipeline`
still has:

```python
def deliver(
    self,
    rendered: RenderResult,
    request: DeliveryRequest | None = None,
) -> DeliveryResult: ...
```

This is fine for blocking debug delivery, but it is the wrong primary shape for
ordered async delivery because it hides where the caller may block. The lab
review already caught the same problem when `submit() -> Iterable[...]` was
considered and rejected.

If we promote `deliver(rendered, request) -> DeliveryResult` directly, we risk
reintroducing the exact ambiguity the facade removed.

## Proposed Boundary

Keep `RenderSession` / `OpticalRenderPipeline` as the long-lived resource owner,
but make delivery execution a runtime/controller object with explicit
submission and drain methods.

Candidate protocol vocabulary:

```python
@runtime_checkable
class OpticalDeliveryRuntime(Protocol):
    @property
    def request(self) -> DeliveryRequest: ...

    def submit(
        self,
        rendered: RenderResult,
        *,
        frame_start: float | None = None,
    ) -> DeliveryResult | None: ...

    def complete_available(
        self,
        *,
        latest_rendered_frame_index: int | None = None,
    ) -> Sequence[DeliveryResult]: ...

    def flush(self) -> Sequence[DeliveryResult]: ...
```

Candidate pipeline/session hook:

```python
@runtime_checkable
class OpticalRenderPipeline(Protocol):
    def begin_frame(...) -> RenderFrameContext: ...

    def create_delivery_runtime(
        self,
        request: DeliveryRequest,
    ) -> OpticalDeliveryRuntime: ...
```

Do not remove `deliver(...)` immediately if it is useful as a blocking helper,
but treat it as sugar for sync/single-frame delivery rather than the async
ordered execution contract.

## Result Type Direction

`optics.render_api` already has CPU-safe vocabulary:

- `RenderResult`
- `DeliveryRequest`
- `DeliveryResult`
- `DeliveryTimingSummary`
- `FrameResult`

The next cleanup should align these with the lab facade without importing lab
code.

Recommended direction:

```python
@dataclass(frozen=True)
class DeliveryResult:
    completed_frame_index: int
    host_channels: Mapping[str, object] = field(default_factory=dict)
    device_result: OpticalComputeResult | None = None
    delivery: DeliveryTimingSummary = field(default_factory=DeliveryTimingSummary)
    lag_frames: int = 0
    ring_depth: int = 0
    ring_block_count: int = 0
    dropped: bool = False
    backpressure_count: int = 0
```

Open compatibility option:

```text
Keep frame_index as a deprecated alias or property for one transition slice.
```

Reason: `frame_index` alone is ambiguous in async paths. The lab facade and
Claude review both converged on the completed-frame identity being explicit.

## Implementation Plan

### R0 — Review This Plan

No code changes. Confirm the target boundary and naming before touching
`optics.render_api`.

### R1 — CPU-Safe Protocol And Type Alignment

In `optics/render_api.py` only:

- add `OpticalDeliveryRuntime` protocol;
- add `OpticalRenderPipeline.create_delivery_runtime(...)` or equivalent;
- adjust `DeliveryResult` toward explicit completed-frame identity and typed
  `DeliveryTimingSummary`;
- keep imports CPU-safe and dependency-light;
- do not export these names publicly from `optics.__init__`.

Tests:

- import-safety tests in `tests/unit/optics/test_render_api.py`;
- validation that async-capable request combinations still fail early when
  invalid.

### R2 — Lab Adapter Produces Runtime Vocabulary

Inside `tools/optical_pipeline_lab`:

- keep `VideoDeliveryFacade` as the lab implementation;
- add conversion or direct construction of `optics.render_api.DeliveryResult`;
- keep `DeliveredVideoFrame` if the row builder still needs lab-specific
  fields such as `observed_frame_ms`, `frame_path`, or `overlap_ratio`;
- explicitly drop lab-only fields from runtime `DeliveryResult`:
  `observed_frame_ms` remains frame-summary/CSV-row data, `frame_path` remains
  consumer-adapter/writer data, and `overlap_ratio` remains lab analysis data;
- ensure CSV output is unchanged.

This slice should be mostly mechanical and testable through existing unit tests.

### R3 — Optional Pipeline Hook

Teach the Go2 lab pipeline/session to create the delivery runtime:

```python
delivery = pipeline.create_delivery_runtime(delivery_request)
```

or, if resource ownership is still awkward:

```python
delivery = VideoDeliveryFacade.create(...)
```

with a TODO noting which resources must move before the generic hook is real.

The goal is to clarify ownership, not to force a broad migration in one patch.

### R4 — GPU Smoke And Matrix Verification

Run the small delivery smoke matrix once on GPU when available:

```text
python -m tools.optical_pipeline_lab matrix \
  --suite go2_video_delivery_smoke \
  --out out/optical_pipeline_lab/go2_video_delivery_smoke_gpu1 \
  --device cuda:1 \
  --frames 5 \
  --warmup-renders 5 \
  --progress-every 1
```

Expected compatibility:

- `frame_timing.csv` schema unchanged;
- `delivery_policy` remains `sync` or `torch_async`;
- async RGB8 case reports ring columns;
- matrix summary includes delivery policy and ring depth.

## Non-Goals

- No public `OpticalCameraStream` API.
- No realtime preview latest/mailbox mode.
- No sensor ack/backpressure policy.
- No path tracing execution changes.
- No CSV column rename.
- No change to `GpuPublishedFrame` borrow/complete lifecycle.
- No move of Pillow image writing or lab CSV row formatting into `optics/`.
- No assumption that lab `VideoDeliveryFacade` is the final production class
  name or module location.

## Risks

### Risk: Over-Promoting Lab Code

The facade is useful because it is concrete, but it also knows about lab CSV,
PNG frames, progress lines, and smoke timing. Only the protocol shape and result
vocabulary should move upward first.

### Risk: Ambiguous Frame Identity

Async submit and completion can refer to different frames. `DeliveryResult`
should name the completed identity explicitly, and `FrameResult` should be
constructed only when delivery completes.

### Risk: Hidden Blocking

Any API that returns an iterable, or a single `DeliveryResult` from an async
submit, can hide blocking. The explicit methods are the guardrail:

```text
submit
complete_available
flush
```

### Risk: Timing Ownership Drift

RGB8 pack must stay delivery-owned. Render timing must not absorb pack/readback
work again.

## Acceptance Criteria

For the first implementation slice after review:

- `optics.render_api` remains importable without Warp, Torch, PIL, or lab code.
- Existing tests for render API and optical pipeline lab pass.
- Existing CSV schema and `delivery_policy` values remain unchanged.
- Existing delivery facade tests still cover sync, async ring depth 1, async
  ring depth 2, and async flush.
- `go2_video_delivery_smoke` remains available from the CLI.
- No `.codex` or unrelated workspace files are touched.

## Review Questions

1. Should we add an `OpticalDeliveryRuntime` protocol now, or keep the explicit
   `submit/complete_available/flush` shape documented until a second production
   consumer exists?

2. Should `OpticalRenderPipeline.deliver(...)` be deprecated as a protocol
   method in favor of `create_delivery_runtime(...)`, or should it remain as
   sync-only convenience sugar?

3. Should `DeliveryResult` replace `frame_index` with
   `completed_frame_index`, or keep both for one transition slice?

4. Should `DeliveryResult` use a typed `delivery: DeliveryTimingSummary` field
   now, while preserving the old `timing: Mapping[str, float]` only as a
   transition helper?

5. Should image/write timing remain delivery-owned even though the writer itself
   is currently lab/consumer-adapter code?

6. Is `OpticalDeliveryRuntime` the right name, or should we use
   `DeliveryController`, `DeliveryQueue`, or `DeliveryScheduler`?

## Codex Recommendation

Proceed conservatively:

```text
R1 first:
  add protocol shape and clarify DeliveryResult identity/timing

R2 second:
  make the lab facade emit/bridge to the runtime vocabulary

Defer:
  moving implementation out of tools/optical_pipeline_lab
  public camera stream API
  latest/mailbox delivery
```

My preferred answers:

```text
Q1: add the protocol now, implementation-free
Q2: keep deliver(...) only as sync convenience or remove from the protocol
Q3: use completed_frame_index explicitly; keep frame_index only as transition
Q4: use typed DeliveryTimingSummary
Q5: yes, timing is delivery-owned even if writer code remains an adapter
Q6: OpticalDeliveryRuntime is acceptable; DeliveryController is also reasonable
```

## Claude Review Result

Claude review: PASS / green light to implement R1 first.

Accepted follow-ups:

```text
1. Add OpticalDeliveryRuntime now as an implementation-free protocol.
2. Keep OpticalRenderPipeline.deliver(...) but document it as sync-only
   convenience, not the async ordered contract.
3. Use DeliveryResult.completed_frame_index, with frame_index retained only as
   a transition alias for one slice.
4. Replace DeliveryResult.timing with typed DeliveryTimingSummary.
5. Keep image/write timing delivery-owned.
6. Use OpticalDeliveryRuntime as the protocol name.
7. Include latest_rendered_frame_index on complete_available(...), because
   ring_depth=1 implementations need it.
8. In R2, explicitly drop DeliveredVideoFrame lab-only fields from runtime
   DeliveryResult: observed_frame_ms, frame_path, and overlap_ratio.
```
