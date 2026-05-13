# Q54 RenderSession / Workspace Stage I1 Implementation Note

Date: 2026-05-13
Author: Codex
Status: implementation-note

## Summary

Implemented the first Stage I slice from the reviewed plan:

```text
Extract Go2RenderSession / Go2RenderFrameContext / Go2RenderPipeline
from tools/optical_pipeline_lab/go2_backend.py
to tools/optical_pipeline_lab/go2_session.py
```

This is a module boundary cleanup only. It does not introduce
`Go2RenderWorkspace`, does not change delivery ownership, and does not fill
`create_delivery_runtime(...)`.

## Changed Files

```text
tools/optical_pipeline_lab/go2_session.py
tools/optical_pipeline_lab/go2_backend.py
tests/unit/optics/test_optical_pipeline_lab.py
```

## Design Details

### New Module

`go2_session.py` now owns the three tightly coupled runtime classes:

```text
Go2RenderSession
Go2RenderFrameContext
Go2RenderPipeline
```

The classes remain lab-local under `tools/optical_pipeline_lab`. This is not a
promotion to `optics/`.

### Backend Re-Exports

`go2_backend.py` imports the classes from `go2_session.py`, so existing callers
that use:

```python
go2_backend.Go2RenderSession
go2_backend.Go2RenderPipeline
```

continue to work.

`Go2RenderFrameContext` is also intentionally re-exported from the backend
module, even though `go2_backend.py` does not use it directly.

### Helper Ownership

The review guidance was to avoid moving video-loop and CSV helpers.

To keep that boundary, `Go2RenderPipeline.create(...)` now accepts callbacks for
the lab-owned helpers:

```text
scene_factory
base_gpu_frame_factory
pack_rgb8
render_profile_buffer_for_request
render_profile_row
```

Production construction in `go2_backend.render_many_views(...)` wires these to
the existing helpers:

```text
_build_scene_for_preset
_base_gpu_frame_for_scene
_pack_video_rgb8
_render_profile_buffer_for_request
_render_profile_row
```

This keeps scene preset selection, synthetic frame defaults, RGB8 pack policy,
and CSV/profile row formatting in the backend module.

`go2_session.py` contains minimal default render-profile and RGB8-pack helpers
so unit tests, fake sessions, and manually constructed sessions can still
exercise the class methods without building the full backend construction path.

### Delivery Runtime

The transitional stub remains unchanged:

```python
def create_delivery_runtime(self, request: DeliveryRequest):
    raise NotImplementedError(
        "Go2RenderPipeline delivery remains owned by the lab benchmark loops"
    )
```

This preserves the reviewed I1 boundary. Ring/stream ownership is still deferred
to the workspace discussion.

### Tests

Pipeline internals now live in `go2_session.py`, so tests that monkeypatch Warp
or dynamic BVH builders now patch that module directly:

```python
import tools.optical_pipeline_lab.go2_session as go2_session
```

No production behavior assertions were relaxed.

Added a direct `Go2RenderPipeline.create(...)` callback-boundary test with fake
Warp/cache/BVH/executor objects. It confirms scene construction, base GPU frame
construction, RGB8 pack, and render-profile helpers are wired through callbacks
without requiring CUDA.

## Non-Goals

- No `Go2RenderWorkspace` yet.
- No delivery facade promotion.
- No `create_delivery_runtime(...)` implementation.
- No CSV schema changes.
- No `RenderedVideoFrame.result` property migration.
- No `render_profile_row` derivation cleanup.
- No dynamic Go2 visual importer work.

## Validation

```text
python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py \
  tests/unit/optics/test_render_api.py

72 passed in 1.70s
```

```text
ruff check \
  tools/optical_pipeline_lab/go2_backend.py \
  tools/optical_pipeline_lab/go2_session.py \
  tests/unit/optics/test_optical_pipeline_lab.py \
  tests/unit/optics/test_render_api.py

All checks passed.
```

## Next Slice

Recommended next review target:

```text
I2: introduce a lab-local Go2RenderWorkspace after this module boundary is
stable.
```

The first workspace version should stay small and focus on resource ownership,
not delivery scheduling.
