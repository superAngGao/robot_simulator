# Q54 Optical Lab Render Foundation C2 Source/Options Implementation Note

Author: codex
Date: 2026-05-14
Status: implemented

## Scope

C2 added the canonical source/options construction path for the Optical Pipeline
Lab render foundation without migrating the Go2 backend yet.

Implemented:

- `OpticalLabRenderSource`
- `OpticalLabRenderOptions`
- `OpticalLabRenderSession.create_from_source(...)`
- `OpticalLabRenderPipeline.create_from_source(...)`
- shared session construction for callback-based `create(...)` and the new
  source/options entrypoint

Intentionally not changed:

- Go2 backend still calls the callback-based `OpticalLabRenderPipeline.create(...)`
- no Go2 source builder yet
- no workspace frame-prep extraction
- no video-loop helper extraction

## Boundary Notes

`OpticalLabRenderSource.bounds_min` and `bounds_max` are documented in code as
scene world-space AABB hints for setup, not camera/frustum hints. If later frame
prep work shows these need to be camera- or frame-derived, they should move to a
frame options/query object instead of remaining on the source.

The legacy callback path now wraps its scene/frame pair in an internal
`OpticalLabRenderSource` for shared construction, but it preserves
`session.scene is scene` for current Go2 camera/video code.

`OpticalLabRenderOptions.verbose_warp` preserves the old callback path's default
quiet Warp initialization behavior for the new source/options entrypoint.

## Verification

Focused unit coverage was added for:

- source identity and frame access
- `create_from_source(...)` building workspace/cache/snapshot/BVH/executor
- default quiet Warp initialization through `OpticalLabRenderOptions`
- old callback-based construction still using the backend callback boundary
