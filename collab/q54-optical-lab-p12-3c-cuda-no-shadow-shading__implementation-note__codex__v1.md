# Q54 P12.3c Implementation Note: CUDA No-Shadow Direct-Light Shading

Author: Codex
Status: ready for review
Related design:
`collab/q54-optical-lab-p12-3-cuda-direct-light-backend__implementation-plan__codex__v1.md`

## Owner Summary

This implements P12.3c: CUDA direct-light shading for the no-shadow path.

`CudaDeviceBvhDirectLightOpticalExecutor.execute(...)` now composes:

```text
CudaDeviceBvhOpticalExecutor first-hit
  -> shade_direct_light_no_shadow CUDA kernel
  -> OpticalComputeResult(location="device")
```

The backend now produces `rgb` and `intensity` as native Torch CUDA tensor
channels for host-ray direct-light rendering when `shadows=False`.

`shadows=True` still raises `NotImplementedError` and remains P12.3d.
`execute_camera(...)` still raises `NotImplementedError` and remains P12.3e.
`run_scenario(...)` still rejects `cuda_direct_light` until the Lab smoke is
accepted.

## What Changed

### `optics/cuda_direct_light.py`

Added a second CUDA extension function:

```text
shade_direct_light_no_shadow(...)
```

It consumes the first-hit geometry channels:

- `hit_mask`
- `position_world`
- `normal_world`
- `material_index`

and scene lighting/material buffers:

- `material_albedo_rgb`
- `light_kind`
- `light_position_or_direction_world`
- `light_intensity`
- `light_color_rgb`

It outputs:

- `rgb`
- `intensity`
- `shadow_stack_overflow_count`
- `shadow_max_stack_depth`

The no-shadow shader supports:

- ambient term;
- background color for misses;
- directional lights;
- point lights with inverse-square attenuation;
- multiple lights;
- BT.709 luminance intensity.

The output profile contract is preserved:

- `DIRECT_LIGHT_FULL` keeps all first-hit and shading channels;
- `RGB_PREVIEW` filters to the guaranteed preview channels;
- `RENDER_ONLY` filters to diagnostic counters.

### Stream / Synchronization Decision

The P12.3b review correctly noted that the extension launches on
`at::cuda::getCurrentCUDAStream()`, not on the incoming Warp stream.

P12.3c does not remove the current `torch.cuda.synchronize(device)` behavior.
That means correctness is preserved, but the implementation still performs
device-wide synchronization and can block unrelated streams on the selected
device.

Before P12 removes device-wide sync or enables async delivery for
`cuda_direct_light`, it must bind Torch and Warp stream semantics explicitly.
The preferred direction is a scoped Torch current-stream bridge if the Warp
stream can expose a compatible CUDA stream handle; otherwise keep the sync until
a reviewed bridge exists.

## Tests Added

`tests/gpu/test_optical_gpu_runtime.py`

Added `test_cuda_direct_light_no_shadow_matches_cpu`.

It verifies:

- CUDA no-shadow direct-light matches `CpuDirectLightOpticalExecutor` RGB and
  intensity on a tiny lit triangle scene;
- first-hit channels still match CPU for hit mask and range;
- `rgb` is a native Torch CUDA tensor;
- `DIRECT_LIGHT_FULL` is produced by default;
- `RGB_PREVIEW` returns exactly the guaranteed preview channels and matches CPU
  RGB;
- `shadows=True` fails fast with a P12.3d message;
- shadow diagnostic counters exist and are zero in the no-shadow path.

## Validation

Commands run:

```bash
PYTHONPATH=. ruff check optics/cuda_direct_light.py \
  tests/gpu/test_optical_gpu_runtime.py \
  tests/unit/optics/test_device_optical.py

PYTHONPATH=. pytest -q tests/unit/optics/

conda run -n env_tilelang_20260119 env PYTHONPATH=. pytest -q \
  tests/gpu/test_optical_gpu_runtime.py \
  -k "cuda_direct_light_no_shadow_matches_cpu"

conda run -n env_tilelang_20260119 env PYTHONPATH=. pytest -q \
  tests/gpu/test_optical_gpu_runtime.py \
  -k "cuda_direct_light_first_hit_matches_warp_bvh_executor or cuda_direct_light_no_shadow_matches_cpu"
```

Result:

- ruff: clean
- unit optics: 274 passed
- CUDA no-shadow shading: 1 passed, 38 deselected
- CUDA first-hit + no-shadow shading: 2 passed, 37 deselected

## Known Limits

- No hard-shadow any-hit yet.
- No CUDA camera raygen yet.
- No `rgb8` readback path for CUDA direct-light yet.
- No Lab video smoke yet.
- Device-wide Torch CUDA synchronization remains correctness-first.

## Review Questions

1. Is the composed first-hit -> no-shadow shading shape the right P12.3c
   boundary, instead of fusing traversal and shading immediately?
2. Is it acceptable that P12.3c supports point/multiple lights in the no-shadow
   shader, even though P12.3c only requires at least one directional light?
3. Should P12.3d add hard shadows as a separate kernel over this geometry result,
   or should it replace no-shadow shading with one shade kernel that optionally
   calls any-hit traversal?
