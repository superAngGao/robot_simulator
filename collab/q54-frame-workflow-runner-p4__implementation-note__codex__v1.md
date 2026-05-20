# Q54 Frame Workflow Runner P4 Implementation Note

Date: 2026-05-20
Author: Codex
Status: implementation-note

## Scope

Added the first narrow lab-internal frame workflow runner. This is P4 of the
physics/video boundary plan.

The new runner coordinates:

```text
frame provider lifecycle
→ video-focused frame consumer
→ delivery complete/submit/complete
→ typed frame result
```

It does not enable `FrameSourceKind.PHYSICS_RUNTIME` in `runner.py`, does not
change CLI behavior, and deliberately does not introduce the long-term
`SimulationFrameRuntime` name.

## Changes

`tools/optical_pipeline_lab/frame_runtime.py`

- Added `FrameWorkflowResult`.
  - Typed fields:
    - `frame_index`
    - `video: RenderedVideoFrame | None`
    - `delivered_video: tuple[DeliveredVideoFrame, ...]`
  - Disabled/missing video uses `None`, not absent dictionary keys.
- Added `FrameWorkflowRunner`.
  - Acquires a frame context through `frame_provider.begin_frame(...)`.
  - Runs one video-focused frame consumer inside provider lifetime.
  - Releases provider before delivery submit/complete work.
  - Submits delivery through `VideoDeliveryFacade`.
  - Records delivered video frames through an optional callback.
  - Supports `provider_kwargs` for future physics `published_frame` forwarding.
  - Provides `flush()` for pending delivery products without acquiring a new
    provider frame.

`tests/unit/optics/test_optical_pipeline_lab.py`

- Added a test proving provider exit happens before delivery completion/submit.
- Added a test proving disabled video keeps stable typed result shape:
  `video=None`, `delivered_video=()`.
- Added a test proving provider exit still runs when the video consumer raises.
- Added a test proving `flush()` records pending delivery without acquiring a
  provider frame.

`GPU_OPTICAL_PIPELINE_DESIGN.md`

- Marked P4 complete in the active render foundation plan.

`MANIFEST.md`

- Added `frame_runtime.py` and updated Q54 sensing/optics test count to 252.

## Key Boundary

`FrameWorkflowRunner` is deliberately smaller than the future
`SimulationFrameRuntime`.

It currently knows only one typed product:

```python
video: RenderedVideoFrame | None
```

This avoids the premature `Mapping[str, object]` product bag that Claude flagged
during architecture review, while keeping a stable shape for disabled video
frames.

## Validation

Focused tests:

```text
conda run -n env_tilelang_20260119 python -m pytest \
  tests/unit/optics/test_optical_pipeline_lab.py -q \
  -k "frame_workflow_runner"

4 passed, 98 deselected
```

Static checks:

```text
conda run -n env_tilelang_20260119 python -m ruff check \
  tools/optical_pipeline_lab/frame_runtime.py \
  tests/unit/optics/test_optical_pipeline_lab.py

All checks passed

conda run -n env_tilelang_20260119 python -m py_compile \
  tools/optical_pipeline_lab/frame_runtime.py \
  tests/unit/optics/test_optical_pipeline_lab.py

clean
```

## 关键思考

1. Why not wire `run_video_benchmark_with_frame_contexts(...)` through the new
   runner immediately?

   P4 is meant to establish the workflow contract without destabilizing the
   already-tested P3 benchmark loop. The two shapes are intentionally compatible:
   P3 can migrate to `FrameWorkflowRunner` after review if doing so removes
   duplication without changing behavior.

2. Why keep `FrameWorkflowRunner` video-focused?

   The current implemented product is video delivery. RL observations and sensor
   products are future consumers with different tensor/device ownership. A typed
   video field is safer than a loose product dictionary while those shapes are
   still unknown.

3. Why allow `provider_kwargs`?

   Physics providers need to accept frame-source details such as
   `published_frame` without teaching the workflow runner about physics engine
   types. This preserves the dependency direction: runner coordinates phases,
   providers own source-specific acquisition.

## Residual Risks

- P3 still uses its own private delivery loop helper. This is acceptable for the
  P4 slice; a later cleanup may migrate it to `FrameWorkflowRunner`.
- `FrameWorkflowRunner` intentionally matches P3 delivery ordering:
  `complete_available` before submit, submit, then `complete_available` again.
  This drains already-ready ordered async frames before queuing new delivery and
  then records any frame completed by the current submit.
- `FrameWorkflowRunner` is CPU-unit covered only. GPU coverage should land with
  the P5 physics video smoke path.
- The runner is lab-internal and should not be exported as public simulator API
  yet.
