# Q54 P12.3a Implementation Note: CUDA Direct-Light Skeleton

Author: Codex
Status: ready for review
Related design:
`collab/q54-optical-lab-p12-3-cuda-direct-light-backend__implementation-plan__codex__v1.md`

## Owner Summary

This implements the P12.3a skeleton and validation slice for
`cuda_direct_light`.

The commit makes `cuda_direct_light` a known, constrained Optical Lab backend
configuration, wires an import-safe CUDA executor skeleton into the render
session dispatch point, and locks the initial compatibility rules in tests.
`run_scenario(...)` still rejects the backend until the P12.3 CUDA first-hit
kernel lands.

This does not implement CUDA first-hit traversal, direct-light shading, shadow
any-hit, camera raygen, or a Lab GPU smoke. Those remain P12.3b through P12.3f.

## What Changed

### `optics/cuda_direct_light.py`

Added `CudaDeviceBvhDirectLightOpticalExecutor` as an import-safe skeleton.

Properties:

- supports the same direct-light output profiles intended for the final CUDA
  backend:
  - `DIRECT_LIGHT_FULL`
  - `RGB_PREVIEW`
  - `RENDER_ONLY`
- uses optional imports for Torch, Warp, and `torch.utils.cpp_extension`;
- exposes `cuda_direct_light_available()` as a lightweight dependency probe;
- validates constructor parameters such as RGB tuple length and `shadow_bias`;
- raises explicit `NotImplementedError` from `execute(...)` and
  `execute_camera(...)` until P12.3b/P12.3e land.

### `optics/__init__.py`

Exports:

- `CudaDeviceBvhDirectLightOpticalExecutor`
- `cuda_direct_light_available`

The module remains CPU-import safe because `optics/cuda_direct_light.py` guards
all optional CUDA imports.

### `tools/optical_pipeline_lab/scenarios.py`

`OpticalLabScenarioConfig.validate_implemented()` now recognizes:

```text
render_backend='cuda_direct_light'
accel_backend='cuda_lbvh'
```

Invalid CUDA combinations fail fast:

```text
render_backend='cuda_direct_light' requires accel_backend='cuda_lbvh'
```

Other reserved render backends remain rejected.

### `tools/optical_pipeline_lab/runner.py`

Run-option validation now constrains the initial P12.3a CUDA surface:

```text
cuda_direct_light requires video_raygen='host' until P12.3e
cuda_direct_light requires video_readback_delivery='sync'
cuda_direct_light does not support readback_payload='rgb8' yet
```

This intentionally accepts only the P12.3a/P12.3b host-ray, sync-readback path.

`validate_run_scenario_supported(...)` still returns unsupported for
`cuda_direct_light`, because P12.3a does not yet include the first-hit kernel
required for an executable Lab scenario.

### `tools/optical_pipeline_lab/render_session.py`

The existing P12.2 render-session dispatch point now selects:

```text
cpu_direct_light       -> CpuDirectLightOpticalExecutor
cuda_direct_light      -> CudaDeviceBvhDirectLightOpticalExecutor
warp_bvh_direct_light  -> GpuDeviceBvhDirectLightOpticalExecutor
```

CUDA still reuses the existing device-scene and CUDA LBVH setup path. The
executor itself is the only new render-session resource.

### `MANIFEST.md`

Registers `optics/cuda_direct_light.py`.

## Tests Added

`tests/unit/optics/test_device_optical.py`

- import-safe CUDA direct-light skeleton probe;
- dependency probe behavior matches `cuda_direct_light_available()`.

`tests/unit/optics/test_optical_pipeline_lab.py`

- `cuda_direct_light + cuda_lbvh` validates for host raygen + sync;
- render options map to `render_backend='cuda_direct_light'` and
  `bvh_backend='cuda_lbvh'`;
- `cuda_direct_light` rejects default GPU raygen until P12.3e;
- `cuda_direct_light` rejects torch async readback in this initial slice;
- `cuda_direct_light` rejects `readback_payload='rgb8'` until RGB8 packing is
  verified;
- `cuda_direct_light` rejects non-CUDA-LBVH acceleration.
- `run_scenario(...)` support remains false until the CUDA first-hit kernel
  lands.

## Validation

Commands run:

```bash
PYTHONPATH=. ruff check \
  optics/__init__.py optics/cuda_direct_light.py \
  tools/optical_pipeline_lab/render_session.py \
  tools/optical_pipeline_lab/scenarios.py \
  tools/optical_pipeline_lab/runner.py \
  tests/unit/optics/test_device_optical.py \
  tests/unit/optics/test_optical_pipeline_lab.py

PYTHONPATH=. pytest -q \
  tests/unit/optics/test_device_optical.py \
  tests/unit/optics/test_optical_pipeline_lab.py
```

Result:

- ruff: clean
- pytest: 207 passed

## Review Questions

1. Is it acceptable for `validate_implemented()` to recognize
   `cuda_direct_light + cuda_lbvh` while the executor still raises
   `NotImplementedError` until P12.3b?
2. Are the initial run-option restrictions right for P12.3a:
   host raygen, sync readback, and no RGB8?
3. Should `cuda_direct_light_available()` stay in `optics/cuda_direct_light.py`,
   or should it move to a future P12.5 backend compatibility helper once CPU,
   Warp, and CUDA direct-light behavior all exists?
