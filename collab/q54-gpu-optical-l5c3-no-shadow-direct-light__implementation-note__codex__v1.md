# Q54 GPU Optical L5C.3 No-Shadow Direct Light Implementation Note

Date: 2026-05-05

## Scope

Added the first GPU direct-light shading path without shadows.

Implemented:

- Device material/light layout:
  - `material_albedo_rgb`
  - enabled lights as `light_kind`, `light_position_or_direction_world`, `light_intensity`, `light_color_rgb`
  - per-triangle and per-plane `material_index`
- BVH first-hit now writes a `material_index` channel for hit rays.
- Added `GpuDeviceBvhDirectLightOpticalExecutor`.
- Added a no-shadow direct-light shade kernel that emits:
  - `rgb`
  - `intensity`

The shading model matches the CPU direct-light executor for `shadows=False`:

- ambient term: `ambient_rgb * albedo`
- directional lights: normalized direction, no attenuation
- point lights: inverse-square attenuation
- miss rays: background RGB and zero intensity

## Not Included Yet

- Shadow rays / any-hit occlusion.
- RGB preview example using the new GPU executor.
- Direct-light support for the non-BVH device-scene executor.

## Verification

Commands run:

```bash
python -m py_compile optics/device_scene.py optics/warp_execution.py optics/__init__.py \
  tests/gpu/test_optical_gpu_runtime.py
conda run -n env_tilelang_20260119 python -m pytest tests/gpu/test_optical_gpu_runtime.py -q
conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics -q
conda run -n env_tilelang_20260119 ruff check optics/device_scene.py optics/warp_execution.py \
  optics/__init__.py tests/gpu/test_optical_gpu_runtime.py
```

Results:

- `tests/gpu/test_optical_gpu_runtime.py`: 18 passed
- `tests/unit/optics`: 57 passed
- `ruff check`: passed
