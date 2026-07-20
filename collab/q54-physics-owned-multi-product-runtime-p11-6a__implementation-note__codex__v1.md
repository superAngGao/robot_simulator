# Q54 P11.6a Implementation Note: Menagerie Static Runner Rename

Owner: Codex
Date: 2026-07-20

## Summary

P11.6a starts retiring `tools/optical_pipeline_lab/go2_backend.py` by moving its
actual implementation to:

```text
tools/optical_pipeline_lab/menagerie_static_runner.py
```

`go2_backend.py` now remains only as a deprecated compatibility shim for legacy
imports. New runner/example code no longer imports it.

## Why This Slice

The P11 design says Go2/Menagerie should be treated as a concrete static scene
or example, not as the Optical Pipeline Lab backend. Direct deletion is not a
good first step because older unit/GPU tests and the legacy static preview CLI
still reference private helper names through `go2_backend.py`.

This slice removes the misleading ownership while keeping compatibility:

- the implementation is no longer in a file named `go2_backend.py`;
- `runner.py` delegates legacy static runs to `menagerie_static_runner`;
- `examples/mujoco_menagerie_gpu_preview.py` imports `menagerie_static_runner`;
- `go2_backend.py` forwards legacy names, including private helpers, so old
  tests and benchmark call sites do not break during the migration.

## Changed Files

- `tools/optical_pipeline_lab/menagerie_static_runner.py`
  - renamed from `go2_backend.py`;
  - module docstring now describes a Menagerie static runner, not a backend.
- `tools/optical_pipeline_lab/go2_backend.py`
  - new deprecated shim that forwards to `menagerie_static_runner`;
  - preserves legacy monkeypatch behavior for `_build_video_camera` through
    wrappers around `_render_video_frame`, `_run_video_benchmark`, and
    `_build_torch_async_warmup_result`.
- `tools/optical_pipeline_lab/runner.py`
  - static runner-clocked scenarios now import `render_many_views` from
    `menagerie_static_runner`.
- `examples/mujoco_menagerie_gpu_preview.py`
  - imports `main` from `menagerie_static_runner`;
  - docstring describes the CLI as legacy static preview.
- `tests/unit/optics/test_optical_pipeline_lab.py`
  - the lab-runner delegation test now patches `menagerie_static_runner`.
- `MANIFEST.md`
  - records the new module and the deprecated shim.

## Boundaries

This does not implement a Go2 P11 preset workflow yet. The Menagerie static
runner still owns the legacy static asset CLI path. A later P11/P12 slice can
move static asset runs behind the same public preset workflow surface once that
workflow can own static scenes as products.

This also does not remove old tests that intentionally exercise compatibility
aliases through `go2_backend.py`. Those tests should migrate after the static
workflow has a non-legacy public entry point.

## Review Questions

1. Is `menagerie_static_runner.py` the right target name, or should this be
   `menagerie_preview_runner.py` to emphasize CLI/example status?
2. Is the dynamic compatibility shim acceptable for this migration step, given
   that existing tests still touch private helper names?
3. Should the next slice migrate unit tests from `go2_backend` to
   `menagerie_static_runner`, or wait until static workflows are routed through
   P11/P12?
