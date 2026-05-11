# Q54 A8.1 Synthetic Body-Bound Smoke Implementation Note

Date: 2026-05-11
Author: Codex
Status: implemented

## Scope

A8.1 validates the physics-published-frame shape with a tiny synthetic
body-bound optical scene.

This is still not the full dynamic video benchmark loop. The goal is narrower:

```text
body-bound optical registry
  + pose-only GpuPublishedFrame
  + perturbed clone
  -> DeviceOpticalSceneCache.snapshot_from_gpu_frame(...)
  -> changed world geometry / AABB
```

## Implementation

Extended `tools.optical_pipeline_lab.dynamic_frames` with:

```text
make_body_bound_triangle_registry(...)
make_gpu_pose_frame(...)
```

`make_body_bound_triangle_registry(...)` creates one body-bound triangle with a
small `X_body_geometry` z offset. `make_gpu_pose_frame(...)` creates a pose-only
`GpuPublishedFrame` from host translation/rotation arrays using an injected
Warp-like module.

Added GPU smoke:

```text
tests/gpu/test_optical_gpu_runtime.py::test_optical_lab_synthetic_body_bound_frame_moves_snapshot_geometry
```

The smoke:

1. creates a base pose frame at translation `[0, 0, 0]`;
2. clones and perturbs body 0 by `[0, 0, 0.4]`;
3. builds device snapshots for both frames;
4. verifies the moved triangle is translated by `[0, 0, 0.4]`;
5. verifies the moved AABB is updated;
6. verifies the source frame translation remains unchanged.

## Why Synthetic First

The current Go2 Menagerie import path is world-baked: visual instances do not
carry `body_index`, and the static Go2 frame has zero pose bodies. A perturbed
Go2 `GpuPublishedFrame` would therefore not move visual geometry.

The synthetic smoke proves the optical side of the contract independently:

```text
begin_frame(frame_inputs) can eventually consume a pose-bearing frame
DeviceOpticalSceneCache can update body-bound geometry from that frame
snapshot/AABB data changes when the frame pose changes
```

The later Go2 dynamic path still requires either:

- an importer change that preserves body-bound visual instances; or
- a separate synthetic dynamic backend/preset for lab benchmarking.

## Validation

```text
conda run -n env_tilelang_20260119 python -m pytest \
  tests/unit/optics/test_optical_pipeline_lab.py -q

43 passed

conda run -n env_tilelang_20260119 python -m pytest \
  tests/gpu/test_optical_gpu_runtime.py::test_optical_lab_synthetic_body_bound_frame_moves_snapshot_geometry -q

1 passed

conda run -n env_tilelang_20260119 python -m ruff check \
  tools/optical_pipeline_lab/dynamic_frames.py \
  tests/unit/optics/test_optical_pipeline_lab.py \
  tests/gpu/test_optical_gpu_runtime.py

All checks passed.
```

## Next Step

Wire dynamic `begin_frame(frame_inputs=other_frame)` for a synthetic body-bound
pipeline/backend and populate `FramePrepareTiming` with non-NaN `snapshot_ms`
and acceleration timing.
