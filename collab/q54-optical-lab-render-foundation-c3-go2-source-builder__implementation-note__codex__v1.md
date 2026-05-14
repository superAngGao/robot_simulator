# Q54 Optical Lab Render Foundation C3 Go2 Source Builder Implementation Note

Author: codex
Date: 2026-05-14
Status: implemented

## Scope

C3 migrated the Go2 backend onto the source/options render foundation without
splitting video helpers or moving workspace frame preparation.

Implemented:

- `OpticalLabRenderSession.source`
- `OpticalLabRenderSession.create_from_source_factory(...)`
- `OpticalLabRenderPipeline.create_from_source_factory(...)`
- `go2_backend.build_go2_render_source(...)`
- Go2 `render_many_views(...)` now calls the generic source/options factory path
- the old callback-based `create(...)` entrypoint was removed after Go2 moved
  to source/options

Intentionally not changed:

- video render/export helpers remain in `go2_backend.py`
- dynamic snapshot/refit/rebuild remains in `OpticalLabRenderPipeline`
- `Go2Render*` aliases remain transitional compatibility names

## Design Notes

The Go2 source builder needs the resolved Warp device to create the base
`GpuPublishedFrame`. For that reason C3 added a generic source factory entrypoint
that initializes Warp/workspace first, then calls the source builder, then
continues through the same source/options session construction path.

`session.source` now records the canonical `OpticalLabRenderSource`. `session.scene`
remains the scene view consumed by existing Go2 camera/video/reporting code. For
Go2 this scene view is the original imported/synthetic scene stored in source
metadata; for direct `create_from_source(...)` calls it remains the source itself.

## Verification

Focused unit coverage was added for:

- source factory construction with a separate scene view
- Go2 source builder wrapping scene, base frame, bounds, and metadata
- Go2 args mapping to `OpticalLabRenderOptions`
