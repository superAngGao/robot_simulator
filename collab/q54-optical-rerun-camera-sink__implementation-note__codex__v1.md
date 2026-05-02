Initiative: q54-optical-rerun-camera-sink
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-05-02
Status: implemented
Related Files: rendering/backends/rerun_backend.py, tests/rendering/test_rerun_backend.py
Owner Summary: Added an explicit Rerun sink for `OpticalCameraReading` without routing camera payloads through `RenderScene`. The sink logs metric depth/range, segmentation ids, and debug-preview RGB/intensity images while preserving Q53/Q54 execution boundaries.

# Q54 Optical Rerun Camera Sink Implementation Note

## 1. Why This Step

The optical pipeline now has a sensor-facing host reading:

```text
OpticalComputeResult
  -> OpticalCameraImageResult
  -> OpticalCameraReading
```

The next useful consumer is Rerun/debug visualization. This pass adds that
consumer as an explicit sink method on `RerunBackend`, not as a `RenderScene`
field.

## 2. API

Added:

```text
RerunBackend.log_optical_camera_reading(
    reading,
    *,
    timestamp: float | None = None,
    env_index: int | None = None,
    channels: Iterable[str] | None = None,
    entity_prefix: str | None = None,
) -> None
```

Default entity path:

```text
env_{reading.env_idx}/sensors/optical/{safe_sensor_id}/{channel}
```

`timestamp` defaults to `reading.sim_time`. `channels` defaults to the supported
channels found in the reading; unsupported channels are skipped unless explicitly
requested, in which case the method raises.

## 3. Channel Mapping

| Reading channel | Rerun archetype | Conversion |
| --- | --- | --- |
| `depth_m` | `DepthImage` | float32, meter=1.0 |
| `range_m` | `DepthImage` | float32, meter=1.0 |
| `numeric_instance_id` | `SegmentationImage` | uint32 |
| `numeric_material_id` | `SegmentationImage` | uint32 |
| `semantic_id` | `SegmentationImage` | uint32 |
| `rgb` | `Image` | clipped uint8 debug preview |
| `intensity` | `Image` | normalized uint8 debug preview |

The RGB and intensity conversions are intentionally debug-display conversions.
They do not redefine the canonical optical result contract: `rgb` remains
unbounded linear float data in the reading, and tone mapping remains a consumer
responsibility.

## 4. Boundary

This does not change `RenderBackend.render_frame(...)` and does not add camera
payloads to `RenderScene.sensor_data`.

The boundary stays:

```text
sensing/optics own execution and readings
rendering/Rerun consumes already-computed readings as a sink
```

`RerunBackend` uses duck typing and does not import `sensing`, keeping
`rendering/` from depending directly on sensor reading classes.

## 5. Tests

Added mock-Rerun coverage for:

- logging supported channels;
- safe sensor-id entity naming;
- metric channel dtype and `meter=1.0`;
- segmentation dtype;
- RGB clipping preview behavior;
- explicit prefix/channel selection;
- explicit unsupported channel rejection;
- malformed image-shaped channel rejection.

Verification:

```text
PYTHONPATH=. pytest tests/rendering/test_rerun_backend.py tests/unit/sensing/test_optical_camera.py tests/unit/sensing/test_readings.py -q
40 passed, 1 skipped

ruff check rendering/backends/rerun_backend.py tests/rendering/test_rerun_backend.py sensing tests/unit/sensing/test_optical_camera.py tests/unit/sensing/test_readings.py
All checks passed
```

The skipped test is the existing real `.rrd` save test; this environment does
not have the optional `rerun` package installed.
