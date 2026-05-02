Initiative: q54-optical-camera-reading
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-05-02
Status: implemented
Related Files: sensing/readings.py, sensing/builders.py, sensing/__init__.py, tests/unit/sensing/test_optical_camera.py
Owner Summary: Added the first narrow optical consumer bridge: `OpticalCameraImageResult -> OpticalCameraReading`. The builder requires host image-shaped channels, copies selected channels into a sensor-facing reading, and leaves Rerun/RenderScene integration deferred.

# Q54 Optical Camera Reading Implementation Note

## 1. Why This Step

After L1/L2/L3, the optical pipeline can produce camera-shaped channels such as
`range_m`, `depth_m`, `rgb`, `intensity`, and segmentation ids. The missing
piece was a sensor-facing host reading that downstream consumers can retain
without depending on executor result lifetimes.

This pass adds that bridge without changing execution:

```text
OpticalExecutor.execute(...)
  -> OpticalComputeResult
  -> build_pinhole_camera_image_result(...)
  -> OpticalCameraImageResult
  -> build_optical_camera_reading(...)
  -> OpticalCameraReading
```

## 2. Boundary Decision

`OpticalCameraReading` lives in `sensing.readings`, and the builder lives in
`sensing.builders`.

It does not import or depend on `rendering/`. This keeps the Q53 boundary intact:
`sensing/` owns sensor-facing readings, while `rendering/` and Rerun remain
visualization/export sinks.

## 3. Builder Semantics

`build_optical_camera_reading(result, channels=None)`:

- requires `result.location == "host"`;
- validates `result.image_shape` is positive `(height, width)`;
- validates every selected channel starts with that image shape;
- copies selected channels into owned NumPy arrays;
- preserves channel names from the optical result contract.

The builder intentionally does not:

- generate rays;
- reshape flat executor output;
- compute projected depth;
- apply tone mapping, gamma, camera response, or noise;
- log to Rerun;
- add camera payloads to `RenderScene`.

## 4. Why Keep A Generic Channel Map

The reading keeps a `channels: dict[str, object]` rather than fixed fields. The
optical result contract is still expanding across CPU, GPU, RGB, segmentation,
and future LiDAR/camera variants. A channel map lets consumers opt into the
channels they understand while preserving schema tests around shape, dtype, and
miss values at the executor/postprocessor level.

## 5. Tests

Added coverage for:

- package export of `OpticalCameraReading`;
- owned-copy behavior from `OpticalCameraImageResult`;
- channel selection;
- host-only result requirement;
- rejection of non-image-shaped channels.

Verification:

```text
PYTHONPATH=. pytest tests/unit/optics tests/unit/sensing -q
84 passed

ruff check sensing tests/unit/sensing/test_optical_camera.py tests/unit/sensing/test_readings.py
All checks passed

git diff --check
passed
```

## 6. Deferred

- Rerun image sink;
- RL observation adapter;
- device-result staging into camera readings;
- camera noise/response model;
- `RenderScene` debug overlays for rays/frustums/images.
