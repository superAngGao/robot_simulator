# Q54 A8.4 Dynamic Smoke Preset Implementation Note

Date: 2026-05-12
Author: Codex
Status: implemented

## Scope

A8.4 promotes the A8.3 internal video-frame hook into a public lab preset:

```text
synthetic_body_triangle_dynamic_smoke
  -> scene_preset=synthetic_body_triangle
  -> geometry_mode=dynamic_rigid
  -> accel_backend=cpu_bvh
  -> accel_policy=refit_each_frame
  -> per-frame GpuPublishedFrame inputs
```

This intentionally remains a synthetic body-bound scene. The Go2 Menagerie
visual importer still produces world-static geometry, so this preset does not
claim Go2 body-bound visual motion.

## Implementation

`OpticalLabScenarioConfig.validate_implemented()` now admits one dynamic
configuration: `synthetic_body_triangle` with dynamic rigid geometry and CPU BVH
refit. Other dynamic/deformable/fluid modes still fail loudly.

`tools/optical_pipeline_lab/presets.py` adds:

```text
synthetic_body_triangle_dynamic_smoke
```

The runner passes `scene_preset` and `video_geometry_mode` into the backend.
For the synthetic preset, the backend:

- builds a tiny body-bound triangle registry with a directional light;
- creates a one-body pose-only `GpuPublishedFrame`;
- generates a deterministic per-frame `video_frame_inputs` sequence;
- reuses the existing video loop, `begin_frame(frame_inputs=...)`, and timing
  CSV schema.

The preset defaults to RGB readback so the CLI default `fail_on_overflow=True`
is satisfiable. Faster timing-only runs can still override to `--readback none`
and `--no-fail-on-overflow`.

## Validation

CPU tests cover preset validation, runner argument translation, and deterministic
dynamic frame input generation:

```text
python -m pytest -q tests/unit/optics/test_optical_pipeline_lab.py
50 passed
```

The GPU smoke now includes a public-preset path:

```text
tests/gpu/test_optical_gpu_runtime.py::
  test_optical_lab_dynamic_smoke_preset_writes_prepare_timing_csv
```

It runs `run_scenario(get_preset("synthetic_body_triangle_dynamic_smoke"), ...)`
with a small frame count and checks that `frame_timing.csv` records:

- `scenario_name=synthetic_body_triangle_dynamic_smoke`;
- `scene_preset=synthetic_body_triangle`;
- `geometry_mode=dynamic_rigid`;
- non-NaN `snapshot_ms`;
- non-NaN `accel_refit_ms`;
- NaN `accel_rebuild_ms`.

On this CPU-only session the GPU-selected smoke was skipped by the existing
`Warp or CUDA not available` pytest guard.

Lint:

```text
ruff check tools/optical_pipeline_lab \
  tests/unit/optics/test_optical_pipeline_lab.py \
  tests/gpu/test_optical_gpu_runtime.py
All checks passed!
```
