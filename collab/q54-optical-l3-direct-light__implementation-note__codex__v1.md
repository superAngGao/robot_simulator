Initiative: q54-optical-l3-direct-light
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-05-02
Status: implemented
Related Files: optics/execution.py, optics/registry.py, optics/__init__.py, tests/unit/optics/test_direct_light_executor.py
Owner Summary: Implemented the accepted L3 direct-light/simple RGB executor. The new `CpuDirectLightOpticalExecutor` shades first-hit geometry with deterministic two-sided Lambertian direct lighting, emits unbounded linear `rgb` plus BT.709 `intensity`, and uses module-level BVH/plane any-hit shadow occlusion over snapshot acceleration.

# Q54 L3 Direct-Light Implementation Note

## 1. Implemented Path

Implemented:

```text
CpuDirectLightOpticalExecutor
Lambertian diffuse direct light
point and directional OpticalLightSpec support
optional shadow rays
rgb and intensity result channels
```

The executor is still not a full renderer. It does not implement PBR, indirect
light, tone mapping, exposure, reflection/refraction, textures, rasterization,
or GPU execution.

## 2. Public Behavior

Default:

```text
CpuDirectLightOpticalExecutor()
```

uses:

```text
geometric_executor = CpuBvhOpticalExecutor()
shadows = True
ambient_rgb = (0, 0, 0)
background_rgb = (0, 0, 0)
shadow_bias = 1e-6
```

Missing BVH acceleration raises `MissingAccelerationError`. Tests may inject
`CpuReferenceOpticalExecutor` with `shadows=False`, but the default path uses
BVH.

## 3. Result Channels

The executor preserves first-hit geometry channels:

```text
range_m
hit_mask
position_world
normal_world
material_id
instance_id
numeric_instance_id
```

and adds:

```text
rgb: float64[num_rays, 3]
intensity: float64[num_rays]
```

`rgb` is unbounded linear RGB. Values may exceed `1.0`; clipping, tone mapping,
and display conversion belong to consumers.

`intensity` is BT.709 luminance of `rgb`:

```text
[0.2126, 0.7152, 0.0722]
```

Misses receive `background_rgb` and `intensity=0`.

## 4. Light Semantics

Directional lights:

```text
position_or_direction_world = direction from shaded point toward the light
```

Point lights:

```text
position_or_direction_world = world-space light position
attenuation = 1 / max(distance^2, 1e-12)
```

If a point light is exactly at the hit position, the light direction is
undefined and the L3 executor returns zero contribution for that singular light.

The simple L3 model accepts that point and directional `intensity` have
different practical unit semantics.

## 5. Shadow Occlusion

Shadow rays use the primary `spec.sensor_role`.

Implemented module-level helpers:

```text
_is_occluded(...)
_is_occluded_by_bvh(...)
_is_occluded_by_planes(...)
```

These helpers read `snapshot.acceleration` directly. They do not call private
methods on `CpuBvhOpticalExecutor` and do not allocate full
`OpticalComputeResult` objects for shadow rays.

## 6. Normal Policy

The executor uses `normal_world` from the geometric executor. Because triangle
normals are oriented against the primary ray, the first L3 version is a
two-sided Lambertian model.

## 7. Test Coverage

Added `tests/unit/optics/test_direct_light_executor.py` covering:

```text
directional light front-lit Lambertian rgb and BT.709 intensity
directional light back-facing surface gives zero
point light inverse-square attenuation
ambient-only hit and miss background
disabled light ignored
directional shadow occlusion
point shadow occlusion before light distance
point shadow not blocked behind light
shadow rays use primary sensor role
camera postprocess reshapes rgb/intensity
default executor requires BVH acceleration
explicit reference executor with shadows=False for tests
zero directional vector raises ValueError
point light at hit position returns finite zero contribution
```

## 8. Verification

```text
PYTHONPATH=. pytest tests/unit/optics tests/unit/sensing -q
81 passed

ruff check optics sensing tests/unit/optics tests/unit/sensing
All checks passed

python -m compileall optics sensing tests/unit/optics tests/unit/sensing
passed

git diff --check
passed
```

## 9. Deferred

Still deferred:

```text
PBR / BRDF
specular highlights
reflection/refraction
indirect light
texture sampling
exposure / tone mapping
gamma conversion
raster framebuffer
GPU direct lighting
soft shadows / area lights
scene-scale-aware shadow bias
```
