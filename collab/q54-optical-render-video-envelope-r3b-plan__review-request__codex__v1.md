# Q54 Render Video Envelope R3b Plan

Date: 2026-05-13
Author: Codex
Status: review-request

## Context

The delivery boundary sequence is now clean through R4:

```text
R1: protocol/type alignment
R2: DeliveredVideoFrame -> DeliveryResult bridge
R3a: current render-video design note
R4: go2_video_delivery_smoke GPU validation
```

Claude accepted the R3a current-design note and confirmed:

- do not force `VideoDeliveryFacade` to implement `OpticalDeliveryRuntime` yet;
- keep `RenderResult` free of camera and loop identity;
- keep `RenderedVideoFrame` as the current name for now;
- run GPU smoke before more R3 code;
- eventually move toward `RenderResult + metadata`, but not urgently.

R4 then passed on GPU.

This plan scopes the next optional cleanup: making the lab video render
envelope explicit without changing CSV output or delivery behavior.

## Current Shape

`RenderedVideoFrame` currently stores flattened fields:

```python
frame_index: int
camera: object
result: object
camera_rays_ms: float
render_execute_ms: float
render_profile_row: dict[str, float]
include_shadow_traversal_stats: bool
geometry_mode: str = "static"
prepare_timing: Mapping[str, float] = field(default_factory=dict)
```

Most of these fields come from runtime objects:

```text
result:
  RenderResult.compute

render_execute_ms:
  RenderResult.render_timing.execute_ms
  or RenderResult.timing["render_execute_ms"]

render_profile_row:
  flattened RenderResult.timing phases

prepare_timing:
  Go2RenderFrameContext.prepare_timing
```

But several are video-loop metadata, not render output:

```text
frame_index
camera
camera_rays_ms
geometry_mode
include_shadow_traversal_stats
```

This mixed shape is acceptable for the lab, but it is why
`VideoDeliveryFacade.submit(RenderedVideoFrame)` cannot honestly satisfy
`OpticalDeliveryRuntime.submit(RenderResult)`.

## Goal

Make the envelope ownership obvious while keeping the hot path behavior and CSV
schema unchanged.

The target direction is:

```python
@dataclass
class RenderedVideoFrame:
    render: RenderResult
    frame_index: int
    camera: OpticalPinholeCameraSpec
    camera_rays_ms: float
    geometry_mode: str
    include_shadow_traversal_stats: bool
    prepare_timing: Mapping[str, float]
```

Then provide compatibility properties for existing row/delivery code:

```python
result -> render.compute
render_execute_ms -> render.render_timing.execute_ms or render.timing key
render_profile_row -> existing flattened render profile row
```

This makes the lab envelope concept explicit without requiring a broad rewrite.

## Non-Goals

- Do not rename `RenderedVideoFrame` in this slice.
- Do not make `VideoDeliveryFacade` implement `OpticalDeliveryRuntime`.
- Do not change `VideoDeliveryFacade.submit(...)`.
- Do not move CSV row building out of the lab.
- Do not change frame timing CSV columns or values.
- Do not put camera, frame index, or loop metadata onto `RenderResult`.
- Do not remove `Go2RenderPipeline.create_delivery_runtime(...)` stub yet.

## Proposed Steps

### R3b.0 — Documentation Boundary

Add a docstring to `RenderedVideoFrame` explaining that it is a lab video render
envelope, not a generic `RenderResult`.

This can land immediately because it has no behavior impact.

### R3b.1 — Add Runtime RenderResult Field

Add a `render: RenderResult | None = None` field or replace `result` directly
with `render: RenderResult`.

Conservative option:

```python
render: RenderResult | None = None
result: object | None = None
```

Strict option:

```python
render: RenderResult
```

The strict option is cleaner, but the conservative option may reduce test churn
if many unit tests instantiate `RenderedVideoFrame` with bare fake results.

### R3b.2 — Add Compatibility Accessors

Expose current fields as properties if needed:

```python
@property
def result(self):
    return self.render.compute

@property
def render_execute_ms(self):
    ...
```

Keep `VideoFrameTimingRowBuilder` output unchanged.

### R3b.3 — Update `_render_video_frame(...)`

Construct `RenderedVideoFrame(render=render_result, ...)` after
`frame_context.render(render_request)`.

The only intended semantic change is internal ownership clarity. Timings and
CSV values should remain identical within normal noise.

### R3b.4 — Tests

Add focused tests:

- `RenderedVideoFrame.result` returns `RenderResult.compute`;
- render timing compatibility still yields the same row values;
- `VideoDeliveryFacade` still accepts `RenderedVideoFrame`;
- existing `R1/R2/R4` contract tests still pass.

## Open Questions

### OQ1 — Strict Or Conservative Field Migration?

Should R3b switch directly to:

```python
render: RenderResult
```

or temporarily support both `render` and `result`?

Codex recommendation: use the strict shape if unit-test churn is small. If it
touches many fake objects, use the conservative shape for one slice.

### OQ2 — Render Profile Row Storage

Should `render_profile_row` remain stored as a flattened dict, or become a
property derived from `RenderResult.timing`?

Codex recommendation: derive it eventually, but keep stored flattened data in
the first R3b patch if that keeps CSV stability obvious.

### OQ3 — Overlap Ratio Regression

R4 intentionally did not treat `overlap_ratio` as a pass/fail metric. The
160x120 / 5-frame smoke is too short and too small for that value to be stable.

If overlap ratio becomes a regression metric later, add a separate longer-run
matrix case with more warmup and more frames. Do not overload
`go2_video_delivery_smoke`, which is a contract smoke.

## Acceptance Criteria

- `RenderedVideoFrame` explicitly documents or stores the runtime
  `RenderResult` relationship.
- Existing `VideoDeliveryFacade` behavior is unchanged.
- Existing frame timing CSV output is unchanged.
- Existing sync and torch async unit tests pass.
- `go2_video_delivery_smoke` remains available and unchanged.
- No public API is introduced.

## Review Questions

1. Should R3b use the strict `render: RenderResult` shape immediately, or a
   conservative one-slice compatibility shape?

2. Should `render_profile_row` be derived from `RenderResult.timing` in R3b, or
   left as a stored flattened dict until a later CSV cleanup?

3. Is it acceptable for `RenderedVideoFrame.result` to become a compatibility
   property that returns `render.compute`?

4. Should overlap-ratio validation stay out of the smoke matrix unless we add a
   dedicated longer-run case?

## Codex Recommendation

```text
Q1: prefer strict render: RenderResult if test churn is small
Q2: keep render_profile_row stored for now
Q3: yes, property is acceptable during transition
Q4: yes, keep overlap-ratio validation separate from smoke
```

## Claude Review Result

Claude review: PASS / proceed with the conservative R3b.1 path.

Accepted follow-ups:

```text
1. Keep the current RenderedVideoFrame name.
2. Add render: RenderResult | None = None at the end of the dataclass.
3. Keep result as the existing required field for now so unit tests with
   result=object() do not churn.
4. Have the production _render_video_frame(...) path fill render=render_result.
5. Keep render_profile_row stored; do not derive it in this slice.
6. Do not turn result into a compatibility property until a later slice.
7. When render_execute_ms becomes a property later, use explicit fallback:
   render.render_timing.execute_ms, then render.timing["render_execute_ms"],
   then the stored fallback field. Do not rely on truthiness because NaN and
   None have different meanings.
8. Keep overlap-ratio validation out of the smoke matrix unless a separate
   longer-run case is added.
```
