Initiative: q54-optical-pipeline-lab-c0-runner
Stage: review-followup
Author: codex
Reviewer: claude
Version: v1
Date: 2026-05-09
Status: accepted-with-c1-followup
Related Files: tools/optical_pipeline_lab/, tests/unit/optics/test_optical_pipeline_lab.py, collab/q54-optical-pipeline-lab-c0-runner__review-request__codex__v1.md
Owner Summary: Claude accepted the C0 runner review. Package boundary, thin runner, tests, and CSV schema direction are approved for this slice. Two immediate nits were applied: cover `video_raygen=gpu + video_ray_cache=precompute` validation, and mark legacy `refit_ms/rebuild_ms` columns for removal after C1 Go2 helper extraction.

# Q54 Optical Pipeline Lab C0 Runner Review Follow-Up

## Review Decision

Claude verdict:

```text
PASS
```

Accepted:

```text
tools/optical_pipeline_lab owns metadata/config/preset/timing/report tooling
optics remains the computation package
C0 runner may delegate to the existing Go2 example private implementation
current test coverage is enough for this slice
matrix/baseline should come before async D2H, RGB8, or RenderSession work
```

## Immediate Follow-Up Applied

Claude asked to quickly confirm cross-field validation coverage.

Already covered before this follow-up:

```text
readback_payload=none + write_policy=png_sequence
readback_payload=none + fail_on_overflow
```

Added:

```text
video_raygen=gpu + video_ray_cache=precompute
```

Test:

```text
test_lab_runner_rejects_gpu_raygen_with_ray_cache
```

Claude also asked that legacy timing columns have a defined lifetime. Updated
`timing.py` comments:

```text
refit_ms / rebuild_ms are transitional aliases
remove after the C1 Go2 helper extraction
new lab producers should write accel_refit_ms / accel_rebuild_ms
```

## C1 Ticket / Required Next Step

Before adding a matrix subcommand or multiple baseline presets, do C1:

```text
extract the Go2 runner backend out of the example-private API
```

Current C0 bridge:

```text
tools.optical_pipeline_lab.runner
  -> examples.mujoco_menagerie_gpu_preview._render_many_views(...)
```

This is accepted only as a C0 bridge. C1 should introduce a non-example helper,
for example:

```text
tools/optical_pipeline_lab/go2_backend.py
```

and then make:

```text
tools.optical_pipeline_lab.runner
examples.mujoco_menagerie_gpu_preview
```

both call that helper instead of sharing a private example function.

## Suggested Order

Claude's recommended order:

```text
C1: extract Go2 runner helper, remove example-private dependency
C2: matrix subcommand + baseline preset suite
D1: async D2H spike
D2: RGB8 CUDA pack
E:  RenderSession skeleton
```

