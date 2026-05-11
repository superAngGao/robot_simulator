# Q54 Optical Render API A5 Pipeline Entrypoint Plan

Date: 2026-05-11
Author: Codex
Status: review-request

## Context

A1/A2 introduced CPU-safe runtime request/delivery contracts and taught the Go2
lab session to execute a `RenderRequest`.

A3 added a minimal `DeliveryRequest` adapter in the Go2 lab backend while
preserving benchmark CSV behavior.

A4 routed traversal diagnostic intent through `RenderRequest.diagnostics`
instead of reading lab CLI flags directly in readback channel selection.

The next design question is the internal entrypoint:

```text
How should a caller invoke the render pipeline without calling low-level
executor/snapshot/BVH functions directly?
```

This plan is about the next internal slice only. It is not the public simulator
API and does not add path tracing yet.

## Goal

Create a small internal pipeline facade that makes the lifecycle explicit:

```text
long-lived pipeline/session
  -> begin frame
  -> render camera/rays
  -> deliver result
```

The goal is to give direct-light, RGB8 async delivery, diagnostics, and future
path tracing a shared call shape without freezing a public API.

## Non-Goals

- Do not re-export runtime API names from `optics.__init__`.
- Do not move Go2-specific scene import into `optics/`.
- Do not merge sync and async benchmark loops yet.
- Do not implement path tracing accumulation buffers.
- Do not introduce public names like `OpticalCameraStream` /
  `OpticalSensorStream` yet.
- Do not turn Go2 shadow observations into a generic `ShadowPolicy` yet.

## Proposed A5 Slice

### A5.1: Define Internal Protocol Shape

Add dependency-light protocol-ish contracts near the runtime vocabulary.

Preferred location:

```text
optics/render_api.py
```

Reason:

- the shape is part of the internal runtime boundary;
- it can remain CPU-safe if it is only dataclasses/protocols;
- callers can import explicitly from `optics.render_api`;
- no Warp/Torch/Go2 lab dependency leaks into `optics`.

Recommended names:

```text
OpticalRenderPipeline
RenderFrameContext
```

Use `RenderFrameContext`, not `OpticalRenderFrame`, because this object is a
lifecycle/context handle rather than a rendered image/result.

Possible minimal shape:

```python
from typing import Protocol, runtime_checkable


@runtime_checkable
class RenderFrameContext(Protocol):
    @property
    def frame_id(self) -> int: ...

    @property
    def sim_time(self) -> float: ...

    @property
    def env_idx(self) -> int: ...

    def render(self, request: RenderRequest) -> RenderResult: ...


@runtime_checkable
class OpticalRenderPipeline(Protocol):
    def begin_frame(
        self,
        frame_inputs: object | None = None,
        *,
        env_idx: int = 0,
    ) -> RenderFrameContext: ...

    def deliver(
        self,
        rendered: RenderResult,
        request: DeliveryRequest | None = None,
    ) -> DeliveryResult: ...
```

Important: these should be structural contracts only. They should not require a
generic implementation in `optics/` yet.

`RenderFrameContext` does not need `__enter__` / `__exit__` in A5. A later path
tracing implementation may naturally extend this into:

```python
with pipeline.begin_frame(frame_inputs, env_idx=0) as frame:
    rendered = frame.render(request)
```

That is a useful accumulation-buffer lifecycle hook, but not needed for the
current direct-light lab path.

### A5.2: Add Go2 Adapter Classes In Lab Backend

Add Go2-specific implementations in:

```text
tools/optical_pipeline_lab/go2_backend.py
```

Suggested names:

```text
Go2RenderPipeline
Go2RenderFrameContext
```

The first implementation can be a thin wrapper around existing
`Go2RenderSession`.

Sketch:

```python
@dataclass
class Go2RenderPipeline:
    session: Go2RenderSession
    default_delivery: DeliveryRequest | None = None

    @classmethod
    def create(cls, args, timings):
        return cls(session=Go2RenderSession.create(args, timings))

    def begin_frame(self, frame_inputs: GpuPublishedFrame | None = None, *, env_idx=0):
        return Go2RenderFrameContext(self.session, env_idx=env_idx)

    def deliver(self, rendered, request=None):
        ...
```

Frame context:

```python
@dataclass
class Go2RenderFrameContext:
    session: Go2RenderSession
    env_idx: int = 0

    @property
    def frame_id(self): return self.session.scene.frame.frame_id
    @property
    def sim_time(self): return self.session.scene.frame.sim_time

    def render(self, request: RenderRequest) -> RenderResult:
        ...
```

For A5, `render(...)` may still call `Go2RenderSession.execute_request(...)`
internally. The important boundary change is caller shape, not a new executor.

### A5.3: Keep Existing Video Benchmark Loops Intact

Do not immediately rewrite `_run_video_benchmark` and
`_run_video_benchmark_torch_async` around the new facade.

Instead, use the facade only in one narrow place first:

```text
_render_video_frame(...)
```

Suggested minimal change:

1. `render_many_views(...)` creates a pipeline instead of a bare session.
2. Setup/refit benchmark and warmup may still access `pipeline.session` in A5.
3. `_render_video_frame(...)` obtains a `RenderFrameContext` and calls
   `frame.render(render_request)`.
4. Existing timing fields remain exactly as-is.

This keeps A5 focused and avoids perturbing async delivery timing semantics.

### A5.4: RenderResult Wrapping

Current `Go2RenderSession.execute_request(...)` returns an
`OpticalComputeResult`, not a `RenderResult`.

A5 should introduce a thin wrapping point:

```python
RenderResult(
    compute=compute_result,
    timing={},
    diagnostics={},
)
```

The lab may keep using `rendered.compute` internally or unwrap immediately.

Do not force all timing into `RenderResult.timing` in A5. Existing CSV timing is
already validated and should remain in the lab recorder until a later timing
cleanup.

### A5.5: DeliveryResult Is Optional For This Slice

A5 should not force the full sync/async delivery loops to return
`DeliveryResult`.

Reason:

- sync and async frame timing semantics differ;
- async completion happens one frame later;
- A3 just stabilized CSV compatibility;
- `DeliveryResult` is useful vocabulary but not yet the best execution object.

Use `DeliveryRequest` at the delivery boundary as A3 already does. Promote
`DeliveryResult` only when factoring duplicated row/completion code removes real
complexity.

## Proposed Call Shape After A5

Internal explicit form:

```python
pipeline = Go2RenderPipeline.create(args, timings)
frame = pipeline.begin_frame(env_idx=0)

rendered = frame.render(
    RenderRequest(
        frame_id=camera.frame_id,
        sim_time=camera.sim_time,
        env_idx=camera.env_idx,
        camera=camera,
        backend=RenderBackend.DIRECT_LIGHT,
        output_profile=OpticalOutputProfile.RGB_PREVIEW,
        diagnostics=RenderDiagnosticsRequest(
            profile_timing=True,
            traversal_counters=True,
        ),
    )
)

compute = rendered.compute
```

Future generic form:

```python
pipeline = OpticalRenderPipeline.create(...)
frame = pipeline.begin_frame(frame_inputs, env_idx=0)
rendered = frame.render(request)
delivered = pipeline.deliver(rendered, delivery_request)
```

Future public API remains deferred:

```python
camera_stream = optical.create_camera_stream(...)
reading = camera_stream.render(published_frame, camera)
```

## Path Tracing Implications

A5 should preserve the path tracing slot without implementing it.

Why this shape helps:

- `OpticalRenderPipeline` can own long-lived accumulation state later;
- `RenderFrameContext` can detect scene/frame changes and reset accumulation;
- `RenderRequest.accumulation_id` already gives the future backend a stable
  stream key;
- delivery remains reusable for path-traced preview RGB8 or full debug output.

Do not add `PathTracingSettings` in A5 unless implementation starts.

Open design note: `OpticalRenderPipeline.deliver(rendered, request)` currently
reads like single-frame delivery. Path tracing accumulation is multi-frame by
nature. The likely answer is that accumulation remains pipeline/session-owned,
while `deliver(...)` only transports the current preview or final accumulated
view. A5 should record this as a future design question, not solve it.

## Risks And Guardrails

### Risk: Generic Name Too Early

`OpticalRenderPipeline` sounds public. Guardrail:

- keep it in `optics.render_api` as an explicit internal import;
- do not export from `optics.__init__`;
- document it as an internal protocol shape.

### Risk: Go2 Lab Adapter Becomes The Generic API

Guardrail:

- name concrete classes `Go2RenderPipeline` / `Go2RenderFrameContext`;
- keep Go2 scene import and Menagerie defaults in `tools/`;
- keep generic protocols implementation-free.

### Risk: Timing Semantics Shift

Guardrail:

- do not move timing fields into `RenderResult.timing` yet;
- do not merge sync/async loops in A5;
- require the same CPU tests plus a short RGB8 async GPU smoke.

### Risk: DeliveryResult Premature Refactor

Guardrail:

- A5 may keep delivery result construction out of the hot loop;
- only introduce `DeliveryResult` execution if it clearly simplifies code.

## Implementation Steps

Recommended sequence:

1. Add internal protocol/shape names in `optics/render_api.py`.
2. Add `Go2RenderPipeline` and `Go2RenderFrameContext` wrappers in
   `go2_backend.py`.
3. Make `Go2RenderFrameContext.render(...)` return `RenderResult`.
4. Update `_render_video_frame(...)` to call through the frame context.
5. Keep existing readback/delivery loops and CSV rows unchanged.
6. Add CPU tests:
   - protocol names are import-safe;
   - Go2 frame context wraps compute result in `RenderResult`;
   - unsupported backend still fails loudly through existing logic.
7. Run:
   - ruff check/format;
   - `pytest tests/unit/optics/test_render_api.py tests/unit/optics/test_optical_pipeline_lab.py -q`;
   - short GPU smoke with `readback=rgb8`, `torch_async`, `--render-profile`.

## Acceptance Criteria

- `OpticalRenderPipeline` and `RenderFrameContext` are importable from
  `optics.render_api` but not re-exported from `optics.__init__`.
- Go2 video rendering can flow through a frame context render call.
- `RenderResult.compute` contains the existing `OpticalComputeResult`.
- CSV schema and values remain unchanged.
- `render_execute_ms`, `pack_rgb8_ms`, traversal counters, async ring depth,
  and overlap columns remain populated as before.
- Existing CPU tests pass.
- A short GPU smoke passes.

## Review Questions

1. Should `OpticalRenderPipeline` / `RenderFrameContext` be protocols in
   `optics.render_api.py`, or should A5 only document the shape and implement
   Go2 concrete classes first?

2. Do you agree with `RenderFrameContext` over `FrameHandle` for the internal
   name?

3. Should `Go2RenderFrameContext.render(...)` return `RenderResult` now, even
   if timing remains in the lab recorder, or should it return the raw
   `OpticalComputeResult` until a fuller timing refactor?

4. Is it acceptable for A5 to let setup/warmup paths keep accessing
   `pipeline.session` directly while only `_render_video_frame(...)` uses the
   new frame context?

5. Should `DeliveryResult` stay unused in A5, or should sync delivery create it
   immediately as a low-risk first use?

6. Do you agree that `OpticalCameraStream` / public mode APIs should remain
   deferred until after this internal facade survives the Go2 lab workloads?

7. Any concerns that this plan makes path tracing harder later?

## Recommended Default Answers

Codex recommendation:

```text
Q1: add protocols in optics.render_api.py, implementation-free and explicit import only
Q2: use RenderFrameContext
Q3: return RenderResult now, but keep timing in lab CSV recorder
Q4: yes, direct pipeline.session access is acceptable transitional A5 scope
Q5: keep DeliveryResult unused until delivery loop factoring is justified
Q6: defer OpticalCameraStream/public modes
Q7: no; pipeline/frame/request split preserves path tracing accumulation slot
```

## Claude Review Result

Claude review: PASS / green light to implement.

Accepted follow-ups:

```text
1. Use typing.Protocol and preferably @runtime_checkable for the internal
   pipeline/frame shapes.
2. Keep RenderFrameContext as the name.
3. Record context-manager extension as a future path-tracing accumulation
   lifecycle hook.
4. Return RenderResult from Go2RenderFrameContext.render(...), with timing left
   empty in A5.
5. Keep direct pipeline.session access for setup/warmup as a documented
   transitional state.
6. Keep DeliveryResult unused in A5.
7. Defer OpticalCameraStream / public mode API.
8. Record the multi-frame path tracing accumulation vs single-frame deliver
   semantic question.
9. Use frame_inputs: object | None = None in the protocol; Go2 may type this as
   GpuPublishedFrame | None.
```
