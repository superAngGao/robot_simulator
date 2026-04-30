Initiative: q54-optical-camera-l1
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-04-30
Status: implemented
Related Files: sensing/optical.py, sensing/__init__.py, tests/unit/sensing/test_optical_camera.py
Owner Summary: Implemented L1 image-shaped pinhole camera semantics in `sensing/` while keeping `CpuReferenceOpticalExecutor` ray-batch based and unchanged. Camera ray generation lowers to `OpticalRaySensorSpec`; postprocessing reshapes executor channels and adds projected `depth_m`.

# Q54 L1 Optical Camera Implementation Note

## 1. Decision Applied

Claude accepted the algorithm-path review and recommended L1:

```text
sensing.OpticalPinholeCameraSpec
camera ray builder
image-shaped schema tests
depth_m postprocessor
```

The implementation follows that boundary:

```text
OpticalPinholeCameraSpec
  -> build_pinhole_camera_rays(...)
  -> OpticalRaySensorSpec
  -> OpticalExecutor.execute(snapshot, ray_spec)
  -> build_pinhole_camera_image_result(..., rays=ray_spec)
  -> OpticalCameraImageResult
```

`CpuReferenceOpticalExecutor` remains camera-agnostic.

## 2. Added API

Added to `sensing.optical`:

```text
OpticalPinholeCameraSpec
OpticalCameraImageResult
build_pinhole_camera_rays(spec)
build_pinhole_camera_image_result(result, spec)
```

`OpticalRaySensorSpec` now accepts an optional `ray_shape` metadata tuple. The
flat ray contract is unchanged; `ray_shape` only records how consumers can
reshape the result.

`build_pinhole_camera_image_result(...)` also accepts an optional `rays`
argument. Supplying it avoids rebuilding high-resolution camera rays during
postprocessing.

## 3. Camera Convention

The pinhole camera uses an OpenCV-style camera frame:

```text
+X right
+Y down
+Z optical axis
```

`X_world_camera` places the camera in world coordinates. Generated rays are
stored as world-frame origins and normalized world-frame directions.

## 4. Depth Semantics

The flat executor still outputs `range_m`: true distance along each normalized
ray.

The camera postprocessor adds `depth_m`:

```text
depth_m = range_m * dot(ray_direction_world, optical_axis_world)
```

This keeps projected camera depth outside executor backends and makes the rule
reusable for CPU reference, Embree, OptiX, and future implementations.

## 5. Test Coverage

Added `tests/unit/sensing/test_optical_camera.py` covering:

- image-shaped ray batch generation;
- `ray_shape == (height, width)`;
- camera world rotation;
- invalid intrinsics rejection;
- full registry/cache/executor/postprocess flow;
- postprocess reuse of caller-provided rays;
- `range_m` vs projected `depth_m` on an oblique pinhole ray;
- image-shaped material / instance / numeric instance channels;
- postprocessor timeline/sensor-id mismatch rejection.
- mismatched postprocessor `rays` rejection.

## 6. Deferred

Still deferred:

- RGB/direct-light;
- camera noise/response;
- semantic-id channel generation;
- Rerun image sink;
- GPU/device image result ownership;
- raster framebuffer bridging.
