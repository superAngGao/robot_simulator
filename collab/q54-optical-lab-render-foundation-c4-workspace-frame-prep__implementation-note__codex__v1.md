# Q54 Optical Lab Render Foundation C4 Workspace Frame Prep Implementation Note

Author: codex
Date: 2026-05-14
Status: implemented

## Scope

C4 moved dynamic frame GPU preparation execution from the pipeline facade into
`OpticalLabRenderWorkspace`.

Implemented:

- `OpticalLabPreparedFrame`
- `OpticalLabRenderWorkspace.prepare_dynamic_frame(...)`
- `OpticalLabRenderWorkspace.rebuild_dynamic_bvh(...)`
- `OpticalLabRenderPipeline._begin_dynamic_frame(...)` now delegates execution
  to the workspace and only wraps the prepared resources in a frame context

Intentionally not changed:

- static vs dynamic frame selection remains in `OpticalLabRenderPipeline.begin_frame(...)`
- `OpticalLabRenderFrameContext` construction remains in the pipeline
- video render/export helpers remain in `go2_backend.py`
- no delivery/runtime extraction was included

## Design Notes

The workspace now owns the GPU execution details for per-frame preparation:

- device-scene snapshot from `GpuPublishedFrame`
- event synchronization for snapshot/refit/rebuild
- refit when the base BVH supports it
- rebuild through CPU BVH or CUDA LBVH according to session acceleration options
- prepare timing flattening through `FramePrepareTiming`

The pipeline remains the control facade: it decides whether a frame is static or
dynamic, passes session resources into the workspace, and returns
`OpticalLabRenderFrameContext`.

## Verification

Focused unit coverage verifies:

- pipeline delegation to `workspace.prepare_dynamic_frame(...)`
- dynamic refit path
- dynamic CPU BVH rebuild path
- dynamic CUDA LBVH rebuild path
- static begin-frame path remains unchanged

