# Q54 Optical Lab Render Foundation C1 Rename Implementation Note

Date: 2026-05-14
Author: Codex
Status: implementation-note

## Summary

Implemented the C1 rename-only slice from the refreshed root plan:

```text
tools/optical_pipeline_lab/go2_session.py
  -> tools/optical_pipeline_lab/render_session.py

Go2RenderWorkspace
  -> OpticalLabRenderWorkspace
Go2RenderSession
  -> OpticalLabRenderSession
Go2RenderFrameContext
  -> OpticalLabRenderFrameContext
Go2RenderPipeline
  -> OpticalLabRenderPipeline
```

No source/options types were introduced. No delivery, video loop, workspace frame
prep, or runtime behavior was changed.

## Changed Files

```text
GPU_OPTICAL_PIPELINE_DESIGN.md
MANIFEST.md
tools/optical_pipeline_lab/render_session.py
tools/optical_pipeline_lab/go2_session.py
tools/optical_pipeline_lab/go2_backend.py
tests/unit/optics/test_optical_pipeline_lab.py
```

## Design Details

### New Canonical Module

`render_session.py` now owns the generic lab render foundation classes:

```text
OpticalLabRenderWorkspace
OpticalLabRenderSession
OpticalLabRenderFrameContext
OpticalLabRenderPipeline
```

This module does not import `go2_backend.py`.

### Transitional Alias Shim

`go2_session.py` remains as a compatibility shim:

```python
# transitional: remove after C3
Go2RenderWorkspace = OpticalLabRenderWorkspace
Go2RenderSession = OpticalLabRenderSession
Go2RenderFrameContext = OpticalLabRenderFrameContext
Go2RenderPipeline = OpticalLabRenderPipeline
```

This keeps old imports working while new generic work uses `render_session.py`.

### Backend Re-Exports

`go2_backend.py` imports the canonical classes from `render_session.py` and
re-exports both:

```text
OpticalLabRender*
Go2Render* compatibility aliases
```

The Go2 backend still uses the generic `OpticalLabRenderPipeline` internally.
Actual Go2 names remain for Go2 scene presets, matrix suites, and Menagerie CLI
compatibility.

### Tests

Generic render-session tests were renamed from `test_go2_pipeline_*` /
`test_go2_render_session_*` to `test_lab_render_pipeline_*` /
`test_lab_render_session_*`.

Tests now monkeypatch `tools.optical_pipeline_lab.render_session`, not the
compatibility shim.

## Non-Goals

- No `OpticalLabRenderSource`.
- No `OpticalLabRenderOptions`.
- No `create_from_source(...)`.
- No Go2 source-builder migration.
- No workspace dynamic frame preparation move.
- No video loop split.
- No delivery facade changes.

## Validation

```text
python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py \
  tests/unit/optics/test_render_api.py

74 passed in 1.77s
```

```text
ruff check \
  tools/optical_pipeline_lab/go2_backend.py \
  tools/optical_pipeline_lab/render_session.py \
  tools/optical_pipeline_lab/go2_session.py \
  tests/unit/optics/test_optical_pipeline_lab.py \
  tests/unit/optics/test_render_api.py

All checks passed.
```

`git diff --check` passed.

## Next Slice

Proceed to C2:

```text
introduce OpticalLabRenderSource
introduce OpticalLabRenderOptions
add source/options construction path alongside callback-based create(...)
```

The callback-based `create(...)` should remain until C3 migrates the Go2 backend
to a source builder.
