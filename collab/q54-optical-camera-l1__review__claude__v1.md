Initiative: q54-optical-camera-l1
Stage: review
Author: claude
Version: v1
Date: 2026-04-30
Status: accepted-with-minor-fix
Related Files: sensing/optical.py, tests/unit/sensing/test_optical_camera.py, collab/q54-optical-camera-l1__implementation-note__codex__v1.md
Owner Summary: Claude accepted the L1 camera implementation, including the OpenCV camera convention and `OpticalCameraImageResult` naming. The only suggested fix is to avoid rebuilding camera rays in the postprocessor when the caller already has the ray spec.

# Q54 L1 Optical Camera Review

## Conclusion

The L1 implementation is correct. The camera convention and
`OpticalCameraImageResult` naming are accepted.

One minor implementation issue should be fixed before commit:

```text
build_pinhole_camera_image_result(...)
```

currently rebuilds camera rays even though callers normally already have them.
Add an optional `rays: OpticalRaySensorSpec | None = None` parameter and reuse it
when supplied.

## Camera Frame Convention

The implementation uses the OpenCV convention:

```text
+X right
+Y down
+Z optical axis
```

This is correct for the current row-vector convention:

```text
directions_world = directions_camera @ R.T
```

which is equivalent to `R @ d_camera` with column vectors. The optical axis
calculation:

```text
[0, 0, 1] @ R.T
```

is consistent with the same convention.

The `rot_y(pi / 2)` test correctly maps camera `+Z` to world `+X`.

## Depth Semantics

The projected depth formula is correct:

```text
depth_m = range_m * dot(ray_direction_world, optical_axis_world)
```

The integration test covers the oblique case:

```text
corner range = 2 * sqrt(3)
corner depth = 2
```

## Result Naming

Keep `OpticalCameraImageResult`.

Including `Camera` distinguishes projected camera images from future image-shaped
outputs with different semantics, such as LiDAR range images.

## Required Minor Fix

Change:

```text
build_pinhole_camera_image_result(result, spec)
```

to:

```text
build_pinhole_camera_image_result(result, spec, *, rays=None)
```

If `rays` is supplied, use `rays.directions_world` for projection instead of
calling `build_pinhole_camera_rays(spec)` again.

This avoids unnecessary `H * W` ray reconstruction for high-resolution cameras.
