Initiative: q54-optical-l3-direct-light
Stage: review
Author: claude
Version: v1
Date: 2026-05-02
Status: accepted-with-required-clarifications
Related Files: collab/q54-optical-l3-direct-light__algorithm-plan__codex__v1.md, optics/execution.py, optics/registry.py
Owner Summary: Claude accepted the L3 direct-light/simple RGB plan. Required clarifications: directional light direction means "from shaded point toward the light"; default direct-light executor should use BVH and raise on missing acceleration; RGB is unbounded float64 linear light; intensity is BT.709 luminance; point/directional light intensity units differ in the simple model; shadow occlusion should be a module-level helper that reads snapshot acceleration directly.

# Q54 L3 Direct-Light Algorithm Plan Review

## Conclusion

The L3 algorithm plan is accepted for implementation.

The architecture boundary is correct:

```text
geometric executor -> first hit
direct-light executor -> shading channels
```

The executor should remain a deterministic simple optical model, not a full
renderer.

## Accepted Decisions

1. Directional light direction convention:

```text
position_or_direction_world = direction from shaded point toward the light
```

2. Default executor:

```text
CpuDirectLightOpticalExecutor defaults to CpuBvhOpticalExecutor
missing acceleration raises MissingAccelerationError
```

3. RGB channel:

```text
float64 linear RGB
unbounded
values may exceed 1.0
no clipping or tone mapping in executor
```

4. Intensity channel:

```text
BT.709 luminance of rgb
[0.2126, 0.7152, 0.0722]
```

5. Point-light attenuation:

```text
1 / max(distance^2, attenuation_eps)
```

The simple L3 model accepts that directional and point `intensity` have
different practical unit semantics.

6. Shadow visibility:

```text
shadow rays use the primary spec.sensor_role
```

7. Ambient:

```text
ambient_rgb belongs on the executor constructor
```

8. Normal policy:

```text
first version is two-sided Lambertian using oriented hit normals
```

## Required Implementation Note

Shadow occlusion should be implemented as a module-level helper:

```text
_is_occluded(snapshot, origin, direction, max_distance, sensor_role) -> bool
```

It should read `snapshot.acceleration` directly and should not call private
methods on `CpuBvhOpticalExecutor`.

## Additional Notes

- Validate zero-length directional light vectors in
  `CpuDirectLightOpticalExecutor._validate(...)`, not in `OpticalLightSpec`.
- If point-light `max_distance = distance_to_light - shadow_bias <= 0`, treat
  the shadow ray as unoccluded and explain this in code comments.
- Add docstrings/comments clarifying:
  - directional light convention;
  - unbounded linear RGB;
  - BT.709 intensity;
  - point vs directional intensity semantics;
  - two-sided Lambertian behavior.
