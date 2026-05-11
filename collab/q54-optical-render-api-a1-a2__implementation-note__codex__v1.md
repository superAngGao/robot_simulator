# Q54 Optical Render API A1/A2 Implementation Note

Date: 2026-05-10
Author: Codex
Status: reviewed-implementation-note

## Scope

Implemented the first internal API slice from the architecture plan:

```text
A1: internal render/delivery dataclasses in optics/render_api.py
A2: thin Go2RenderSession adapter that accepts RenderRequest
```

No public simulator-facing API was added, and the new names are not re-exported
from `optics/__init__.py`.

## A1: `optics/render_api.py`

Added CPU-safe runtime vocabulary:

```text
RenderBackend
ReadbackPayload
DeliveryPolicy
WritePolicy
RenderDiagnosticsRequest
RenderRequest
RenderResult
DeliveryRequest
DeliveryResult
```

Important behavior:

- `RenderRequest` validates exactly one of `camera` or `rays`;
- `RenderRequest` normalizes `backend` and `output_profile`;
- rays-only requests require `use_gpu_raygen=False`;
- `accumulation_id: int | None = None` is present as a reserved path-tracing
  field and has no direct-light behavior;
- `DeliveryRequest` validates ring depth and payload/policy compatibility;
- `RGB8` is modeled as delivery payload, not render backend.

## A2: Go2 Adapter

`Go2RenderSession` now has:

```python
execute_request(request: RenderRequest, *, render_profile: list[tuple[str, float]] | None)
```

The existing lab CLI path now builds a `RenderRequest` through
`_video_render_request(...)`, then dispatches through `execute_request`.

CLI behavior, output files, frame CSV schema, and timing semantics remain
unchanged.

## Validation

CPU:

```text
ruff check: passed
ruff format --check: passed
py_compile: passed
pytest tests/unit/optics/test_render_api.py tests/unit/optics/test_optical_pipeline_lab.py -q
  39 passed
```

GPU smoke:

```text
out/optical_pipeline_lab/a1_a2_render_api_smoke_gpu1/frame_timing.csv

device: cuda:1 / NVIDIA H200
resolution: 1920x1080
frames: 5
readback: rgb8
delivery: torch_async ring_depth=2
render_profile: on

mean:
  render_execute_ms          ~= 6.75
  pack_rgb8_ms               ~= 0.15
  render_shade_kernel_ms     ~= 2.91
  shadow_traversal_ray_count ~= 3.83M
  readback_host_ms           ~= 2.38
```

This confirms the A2 adapter still exercises RGB8 async delivery, render
profile timing, and E2 traversal counters.

## Follow-Up

Claude review: PASS / green light to commit.

Non-blocking follow-ups recorded from review:

```text
1. Keep optics.execution.normalize_output_profile CPU-safe.
   render_api.py depends on it, so execution.py should not gain Warp/Torch
   import-time requirements that would break render_api.py's CPU-safe contract.

2. Revisit RenderRequest sim_time equality before public API exposure.
   A1/A2 construct request/source timestamps from the same object, so exact
   equality is fine here. External callers may need a tolerance or a single
   source-of-truth model.

3. Internalize render_profile allocation in a later session/delivery cleanup.
   A2 still creates render_profile outside execute_request and passes it as a
   separate argument. That is intentionally transitional.

4. Make traversal_counters mapping explicit later.
   The lab currently sets traversal_counters=profile_timing and routes counters
   through render-profile-only diagnostics. A future generic session should make
   that mapping explicit.
```

Next logical slice:

```text
A3 delivery boundary adapter
```

Path tracing and public `OpticalCameraStream` style API remain deferred.
