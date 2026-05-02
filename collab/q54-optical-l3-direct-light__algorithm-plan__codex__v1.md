Initiative: q54-optical-l3-direct-light
Stage: algorithm-plan
Author: codex
Version: v1
Date: 2026-05-01
Status: draft-for-review
Related Files: optics/execution.py, optics/scene.py, optics/registry.py, sensing/optical.py
Owner Summary: This document proposes the first L3 direct-light/simple RGB optical executor. The recommended path is a deterministic host-side Lambertian direct-light executor built on the existing first-hit/BVH contract, with optional shadow rays, no indirect light, no exposure/tone mapping, and no physically based material model.

# Q54 L3 Direct-Light / Simple RGB Algorithm Plan

## 1. Decision

Implement the first L3 RGB capability as:

```text
host-side deterministic direct-light executor
Lambertian diffuse only
uses material.albedo_rgb
uses OpticalLightSpec point/directional lights
requires first-hit geometry result
uses CPU BVH for optional shadow rays
outputs existing geometry channels + rgb + intensity
```

Do not implement a full renderer, path tracer, PBR material model, raster
framebuffer, exposure, tone mapping, reflection, refraction, or indirect light.

## 2. Why This Path

Now that L2 provides CPU BVH acceleration, direct-light RGB can be introduced
without making every shadow ray an `O(num_triangles)` scan.

The goal is not photorealism. The goal is to add a small, deterministic optical
model that exercises:

```text
light registry semantics
material albedo usage
shadow-ray visibility
rgb/intensity result schema
camera postprocessing compatibility
```

This gives useful debug/RL observations while keeping the simulator-owned
optical contract independent from future Embree, OptiX, Mitsuba, or raster
adapters.

## 3. Proposed Public Shape

Add a new executor:

```text
CpuDirectLightOpticalExecutor
```

Public call remains:

```text
execute(snapshot: OpticalSceneSnapshot, spec: OpticalRaySensorSpec)
  -> OpticalComputeResult
```

Recommended constructor options:

```text
CpuDirectLightOpticalExecutor(
    geometric_executor: OpticalExecutor | None = None,
    shadows: bool = True,
    ambient_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
    background_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
    shadow_bias: float = 1e-6,
)
```

Default `geometric_executor` should be `CpuBvhOpticalExecutor`, meaning the
default direct-light path requires `snapshot.acceleration.kind == "cpu_bvh"`.
Tests may inject `CpuReferenceOpticalExecutor` with `shadows=False`, but the
production default should use BVH.

## 4. Inputs

Inputs:

```text
OpticalSceneSnapshot
OpticalRaySensorSpec
snapshot.instances
snapshot.lights
instance.material.albedo_rgb
```

Expected sensor role:

```text
spec.sensor_role == "rgb"
```

The executor should not force `"rgb"` internally. It should honor
`spec.sensor_role` exactly, just like the geometry executors. This lets tests
and future sensors deliberately choose their visibility role.

## 5. Outputs

Return all first-hit geometry channels:

```text
range_m
hit_mask
position_world
normal_world
material_id
instance_id
numeric_instance_id
```

plus:

```text
rgb: float64[num_rays, 3]
intensity: float64[num_rays]
```

Miss semantics:

```text
hit_mask=False
range_m=np.inf
position_world/normal_world=NaN
material_id=None
instance_id=None
numeric_instance_id=0
rgb=background_rgb
intensity=0.0
```

For camera results, `build_pinhole_camera_image_result(...)` can reshape `rgb`
and `intensity` like any other flat channel.

## 6. Light Semantics

Current schema:

```text
OpticalLightSpec
  kind: "point" | "directional"
  position_or_direction_world: object
  intensity: float = 1.0
  color_rgb: tuple[float, float, float] = (1, 1, 1)
  enabled: bool = True
```

Proposed interpretation:

### Directional Light

For `kind == "directional"`:

```text
position_or_direction_world = direction from shaded point toward the light
```

The vector is normalized by the executor.

Contribution:

```text
L = normalize(light.direction_world)
n_dot_l = max(dot(normal_world, L), 0)
radiance_rgb = albedo_rgb * light.color_rgb * light.intensity * n_dot_l
```

No distance attenuation.

### Point Light

For `kind == "point"`:

```text
position_or_direction_world = light position in world frame
```

Contribution:

```text
to_light = light_position - hit_position
distance = norm(to_light)
L = to_light / distance
n_dot_l = max(dot(normal_world, L), 0)
attenuation = 1 / max(distance * distance, attenuation_eps)
radiance_rgb = albedo_rgb * light.color_rgb * light.intensity * attenuation * n_dot_l
```

Recommended:

```text
attenuation_eps = 1e-12
```

This is intentionally simple inverse-square attenuation. It is not a calibrated
photometric model. Directional and point light `intensity` therefore have
different practical unit semantics in L3: directional intensity behaves like a
dimensionless multiplier, while point intensity is distance-attenuated.

## 7. Shading Algorithm

High-level algorithm:

```text
geometry = geometric_executor.execute(snapshot, spec)
rgb = full([num_rays, 3], background_rgb)
intensity = zeros([num_rays])

for ray_index where geometry.hit_mask:
    albedo = material.albedo_rgb
    normal = geometry.normal_world[ray_index]
    position = geometry.position_world[ray_index]
    accumulated = ambient_rgb * albedo

    for light in snapshot.lights:
        if not light.enabled:
            continue
        contribution = evaluate_direct_light(...)
        if shadows and contribution is potentially nonzero:
            if shadow_ray_occluded(...):
                continue
        accumulated += contribution

    rgb[ray_index] = accumulated
    intensity[ray_index] = luminance(accumulated)
```

Recommended luminance:

```text
intensity = dot(rgb, [0.2126, 0.7152, 0.0722])
```

These are BT.709 luminance coefficients. This keeps `intensity` deterministic
and tied to `rgb`.

## 8. Normal Policy

Use `normal_world` returned by the geometric executor.

Current first-hit executors orient triangle normals against the primary ray.
Therefore L3 first version is effectively a two-sided Lambertian model. This is
acceptable for simple sensor/debug output.

Deferred:

```text
geometric normal vs shading normal
one-sided material policy
backface culling
vertex normal interpolation
normal maps
```

## 9. Shadow Rays

If `shadows=True`, cast one shadow ray per visible light contribution.

Shadow origin:

```text
shadow_origin = hit_position + normal_world * shadow_bias
```

Directional light shadow ray:

```text
direction = L
max_distance = np.inf
```

Point light shadow ray:

```text
direction = L
max_distance = distance_to_light - shadow_bias
```

If `max_distance <= 0`, treat as unoccluded.

Occlusion query:

```text
any hit before max_distance -> light is blocked
```

The shadow ray should use the same visibility role as the primary spec:

```text
shadow_spec.sensor_role = spec.sensor_role
```

Rationale: current `roles` are the only visibility model. Objects invisible to
the current sensor role should not cast shadows for that role in L3.

## 10. Shadow-Ray Implementation Option

Do not call full `execute(...)` for every shadow ray in the first implementation.
That would allocate full result channels repeatedly.

Instead, add a module-level helper that reads snapshot acceleration directly:

```text
_is_occluded(snapshot, origin, direction, max_distance, sensor_role) -> bool
```

It can reuse:

```text
BVH traversal
triangle scalar intersection
analytical plane pass
role filtering
```

but it only needs a boolean any-hit answer.
It should not call private methods on `CpuBvhOpticalExecutor`; this avoids
coupling direct-light shading to a specific executor instance.

## 11. Result Schema

Recommended `capabilities` for direct-light executor:

```text
{
  "range_m",
  "hit_mask",
  "position_world",
  "normal_world",
  "material_id",
  "instance_id",
  "numeric_instance_id",
  "rgb",
  "intensity",
}
```

`rgb` dtype:

```text
float64
```

No clipping in the executor. Values may exceed 1.0 if light intensity is high.
Tone mapping and display conversion are consumer/viewer responsibilities.

## 12. Edge Cases

No lights:

```text
rgb = ambient_rgb * albedo for hits
intensity = luminance(rgb)
```

Disabled lights:

```text
ignored
```

Zero-length directional light vector:

```text
raise ValueError or skip diagnostic?
```

Recommendation: raise during executor evaluation, because the current
`OpticalLightSpec` accepts only shape, not nonzero direction.

Point light at hit position:

```text
if distance <= direction_eps:
    contribution = 0
```

The light direction is undefined at the exact hit point. Returning zero is a
deterministic singular-case policy for L3 and avoids very large finite values
from `1 / attenuation_eps`.

Shadow self-intersection:

```text
shadow_bias moves origin along normal_world
```

Known limitation: thin geometry and extremely small scenes may need a
scene-scale-aware bias later.

## 13. Tests

Required tests:

```text
directional light front-lit Lambertian rgb
directional light back-facing surface contributes zero
point light inverse-square attenuation
ambient-only hit
miss gets background rgb and intensity 0
disabled light ignored
shadow ray blocks directional light
shadow ray blocks point light before light distance
shadow ray does not block when occluder is behind light
roles filtering affects both primary and shadow rays
camera postprocess reshapes rgb/intensity
missing BVH acceleration raises MissingAccelerationError for default executor
schema test: channel names, dtype, shapes
```

No image-quality or visual regression tests in the first L3 implementation.

## 14. Explicit Non-Goals

Do not implement:

```text
PBR / BRDF library
specular highlights
reflection/refraction
indirect bounce
global illumination
ambient occlusion
texture sampling
exposure / tone mapping
gamma conversion
color management
raster framebuffer
Mitsuba integration
GPU direct lighting
soft shadows
area lights
participating media
```

## 15. Review Questions

1. Should directional light direction mean "from shaded point toward the light",
   as proposed, or "direction light travels"?
2. Should L3 default to `CpuBvhOpticalExecutor` and require acceleration, or
   allow reference traversal with `shadows=False` as the default?
3. Should `rgb` be unbounded float64 linear RGB, leaving clipping/tone mapping to
   consumers?
4. Should `intensity` be luminance of `rgb`, or a separate scalar return-strength
   concept?
5. Should point lights use inverse-square attenuation in the first version, or
   should point-light attenuation be deferred?
6. Should shadow rays use the same `sensor_role` as the primary ray, or a
   separate visibility role such as `"shadow"`?
7. Should `ambient_rgb` live on the executor constructor, the registry, or wait
   entirely?
8. Is two-sided Lambertian shading using oriented hit normals acceptable for the
   first version?

## 16. Review Outcome

Claude accepted the plan on 2026-05-02 with the following concrete decisions:

```text
directional light direction = from shaded point toward the light
default geometric executor = CpuBvhOpticalExecutor
missing acceleration = MissingAccelerationError
rgb = unbounded float64 linear RGB
intensity = BT.709 luminance of rgb
point light uses inverse-square attenuation
shadow rays use primary spec.sensor_role
ambient_rgb belongs on executor constructor
first version is two-sided Lambertian
shadow occlusion helper is module-level and reads snapshot.acceleration directly
```

Review archive:

```text
collab/q54-optical-l3-direct-light__review__claude__v1.md
```
