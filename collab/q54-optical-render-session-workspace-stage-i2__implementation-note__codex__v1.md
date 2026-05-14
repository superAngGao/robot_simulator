# Q54 RenderSession / Workspace Stage I2 Implementation Note

Date: 2026-05-14
Author: Codex
Status: implementation-note

## Summary

Implemented the first minimal workspace slice after I1:

```text
Go2RenderWorkspace(device, stream)
```

This is intentionally small. It gives render/copy/scratch ownership a stable
home without changing delivery scheduling, CSV output, video loop behavior, or
public runtime APIs.

## Changed Files

```text
tools/optical_pipeline_lab/go2_session.py
tools/optical_pipeline_lab/go2_backend.py
tests/unit/optics/test_optical_pipeline_lab.py
```

## Design Details

### Workspace Shape

`go2_session.py` now defines:

```python
@dataclass
class Go2RenderWorkspace:
    device: object
    stream: object
```

No copy stream, ring slots, scratch buffers, allocator policy, or delivery
objects were added in this slice.

### Session Ownership

`Go2RenderSession` now stores:

```text
workspace: Go2RenderWorkspace
```

and exposes compatibility properties:

```python
@property
def device(self):
    return self.workspace.device

@property
def stream(self):
    return self.workspace.stream
```

This keeps existing lab code and GPU tests that read `session.device` or
`session.stream` working, while making `workspace` the real owner of those
resources.

### Constructor Compatibility

The session constructor accepts either:

```text
workspace=Go2RenderWorkspace(...)
```

or the previous manual-test style:

```text
device=...
stream=...
```

If `workspace` is omitted, the constructor creates a workspace from
`device/stream`. Passing both forms is rejected to avoid ambiguous ownership.

### Production Construction

`Go2RenderSession.create(...)` creates the workspace immediately after Warp
device/stream creation and then passes `workspace.device` / `workspace.stream`
to cache, snapshot, BVH, and executor construction.

### Backend Re-Export

`go2_backend.py` re-exports:

```text
Go2RenderWorkspace
```

alongside the existing session/pipeline/frame-context names.

## Tests

Added/updated tests to cover:

```text
Go2RenderPipeline.create(...) stores a Go2RenderWorkspace
session.device/session.stream point at workspace.device/workspace.stream
manual Go2RenderSession(workspace=...) construction preserves compatibility
```

Existing manual construction tests that pass `device=` and `stream=` continue to
exercise the compatibility path.

## Non-Goals

- No copy stream.
- No readback ring ownership move.
- No `create_delivery_runtime(...)` implementation.
- No delivery facade promotion.
- No CSV/schema changes.
- No dynamic Go2 visual importer work.

## Validation

```text
python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py \
  tests/unit/optics/test_render_api.py

73 passed in 2.36s
```

```text
ruff check \
  tools/optical_pipeline_lab/go2_backend.py \
  tools/optical_pipeline_lab/go2_session.py \
  tests/unit/optics/test_optical_pipeline_lab.py \
  tests/unit/optics/test_render_api.py

All checks passed.
```

`git diff --check` passed.

## Next Slice

Recommended next review target:

```text
I2.1: decide whether the workspace should own a separate copy_stream placeholder
or whether frame preparation should move onto workspace first.
```

Codex recommendation: move frame preparation helpers toward workspace before
adding copy-stream behavior, unless review wants a placeholder field first.
