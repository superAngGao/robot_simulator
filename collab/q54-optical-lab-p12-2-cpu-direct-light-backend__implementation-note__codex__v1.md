# Q54 P12.2 Implementation Note: CPU Direct-Light Backend

## Related Files

- `tools/optical_pipeline_lab/scenarios.py`
- `tools/optical_pipeline_lab/render_session.py`
- `tools/optical_pipeline_lab/static_asset_source.py`
- `tools/optical_pipeline_lab/runner.py`
- `tools/optical_pipeline_lab/delivery.py`
- `tools/optical_pipeline_lab/video_products.py`
- `tests/unit/optics/test_optical_pipeline_lab.py`
- `MANIFEST.md`

## Owner Summary

Implemented the P12.2 CPU direct-light Optical Lab backend path as the
correctness/reference member of the direct-light backend family. The new path
is intentionally narrow: static lab frames, CPU BVH acceleration, host ray
generation, sync delivery, and non-RGB8 readback.

This does not implement `cuda_direct_light`; P12.3 remains the hand-written CUDA
kernel path.

## What Changed

### Scenario and validation

- Added `RenderBackend.CPU_DIRECT_LIGHT = "cpu_direct_light"`.
- `OpticalLabScenarioConfig.validate_implemented()` now accepts static
  `cpu_direct_light` configs when `accel_backend == cpu_bvh`.
- Invalid CPU backend combinations fail fast:
  - `cpu_direct_light` requires `cpu_bvh`;
  - `cpu_direct_light` requires `video_raygen="host"`;
  - `cpu_direct_light` requires sync delivery;
  - `cpu_direct_light` does not support `readback_payload="rgb8"` yet.

### Render session dispatch

- Added `OpticalLabRenderOptions.render_backend`.
- `OpticalLabRenderSession.create_from_source(...)` and
  `create_from_source_factory(...)` now dispatch:
  - `warp_bvh_direct_light` -> existing Warp/device snapshot + device BVH path;
  - `cpu_direct_light` -> `OpticalSceneCache(...).snapshot_from_published_frame(...,
    acceleration="cpu_bvh")` + `CpuDirectLightOpticalExecutor`.
- CPU direct-light renders only host-ray requests. GPU camera raygen remains
  part of the Warp path.
- CPU direct-light dynamic frame contexts fail fast for now. P12.2 covers
  static reference rendering only.

### Static asset source

- Static asset render sources now carry CPU base frames as metadata.
- When the render workspace device is `"cpu"`, the static source uses a
  `CpuPublishedFrame` as its base frame. Existing GPU/Warp paths still use
  `GpuPublishedFrame`.

### Delivery

- Sync video readback now accepts host `OpticalComputeResult` values by reading
  requested channels directly.
- `synchronize_ready_event(None)` is now a no-op, matching host result
  semantics.

## Tests Added

- CPU direct-light config validation and compatibility fail-fast tests.
- CPU direct-light render-session smoke over the tiny body triangle scene.
- CPU direct-light P10/P11 product workflow smoke:
  - explicit static runtime;
  - debug + video products;
  - host RGB delivery;
  - `scenario_config.json` records `render_backend=cpu_direct_light`;
  - `frame_timing.csv` records `render_backend=cpu_direct_light` and
    `accel_backend=cpu_bvh`.

## Verification

```bash
ruff check tools/optical_pipeline_lab/render_session.py \
  tools/optical_pipeline_lab/static_asset_source.py \
  tools/optical_pipeline_lab/scenarios.py \
  tools/optical_pipeline_lab/runner.py \
  tools/optical_pipeline_lab/delivery.py \
  tools/optical_pipeline_lab/video_products.py \
  tests/unit/optics/test_optical_pipeline_lab.py
```

Result: all checks passed.

```bash
PYTHONPATH=. pytest -q tests/unit/optics/test_optical_pipeline_lab.py
```

Result: `191 passed`.

## Non-Goals

- No CUDA direct-light render kernels.
- No public `run_optical_lab_preset(..., render_backend=...)` override.
- No backend registry extraction.
- No path tracing, accumulation, OptiX, denoising, or multisample semantics.
- No CPU dynamic/physics published-frame rendering.

## Review Questions

1. Is the CPU path narrow enough for P12.2, especially the static-only and
   host-raygen-only limits?
2. Should `readback_payload="rgb8"` stay fail-fast for CPU until a host rgb8
   pack path is explicitly designed?
3. Is it acceptable that `OpticalLabRenderSession.gpu_frame` continues to hold
   the base frame for legacy compatibility, even when the CPU backend stores a
   `CpuPublishedFrame` there?
4. Should P12.3 reuse this `render_backend` dispatch point for
   `cuda_direct_light`, or should CUDA first land behind a separate experimental
   session constructor before joining this path?
