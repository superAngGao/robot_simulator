# Q54 Optical Lab Render Foundation C5 Video Loop Split Implementation Note

Author: codex
Date: 2026-05-14
Status: implemented

## Scope

C5 moved generic video render-loop helpers out of `go2_backend.py` and into
`tools/optical_pipeline_lab/video_loop.py`.

Implemented:

- `video_loop.render_video_frame(...)`
- `video_loop.run_video_benchmark(...)`
- video request/profile/readback helper functions in `video_loop.py`
- Go2 adapter wrappers that inject `_build_video_camera(...)`
- compatibility aliases in `go2_backend.py` for existing lab-private helper
  call sites and tests

Intentionally not changed:

- Go2 source builder, presets, CLI, camera construction, and reporting stay in
  `go2_backend.py`
- video delivery primitives remain in `delivery.py`
- `Go2Render*` aliases remain for a later alias-deletion cleanup
- no public API promotion was included

## Design Notes

`video_loop.py` owns the generic video loop mechanics:

- per-frame render request construction
- frame input timestamp adaptation
- render/profile timing extraction
- delivery request construction
- sync and ordered async delivery loop
- torch async warmup rendering
- optional host ray precompute

The Go2 backend still owns camera semantics. It passes `_build_video_camera(...)`
into the generic helpers, so `video_loop.py` does not import or depend on
Menagerie/Go2 code.

## Verification

Focused unit coverage continues to exercise the Go2 adapter path and now checks
that Go2's lab-private helper aliases delegate to `video_loop.py`.
