Initiative: q50-render-backend
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-04-27
Status: ready-for-review
Related Files: OPEN_QUESTIONS.md#Q50, OPEN_QUESTIONS.md#Q51, rendering/render_scene.py, rendering/scene_builder.py, sensing/readings.py, tests/unit/rendering/test_published_frame_bridge.py
Owner Summary: Q50 Step 4 is implemented as a narrow numeric/state sensor-data bridge on `RenderScene`. It reuses the frozen sensing phase-1 readings, avoids camera/LiDAR payloads, and does not turn `RenderScene` into the canonical sensor execution contract.

---

## 1. Scope

Implemented:

- `RenderSensorData`
- `RenderScene.sensor_data: RenderSensorData | None`
- published-frame bridge population through sensing phase-1 builders
- CPU and GPU unit coverage

Not implemented:

- camera / image / segmentation payloads
- LiDAR / surface-query payloads
- Rerun-specific sensor plotting
- RL observation vector schema

Those remain outside Q50 Step 4.

---

## 2. API Shape

`RenderSensorData` is a debug/export container:

```python
@dataclass
class RenderSensorData:
    frame_id: int
    sim_time: float
    env_idx: int

    imu_readings: list = field(default_factory=list)
    joint_state: object | None = None
    force: object | None = None
    contact: object | None = None
```

The contained objects are produced by `sensing/`:

- `IMUReading`
- `JointStateReading`
- `ForceSensorReading`
- `ContactStateReading`

The type annotations intentionally stay broad in `rendering/` so `RenderScene`
does not re-own or duplicate the sensing schema.

---

## 3. Builder Behavior

`build_render_scene_from_published_frame(...)` now defaults to:

```python
include_sensor_data=True
```

When enabled, it builds:

```text
PublishedFrame -> StateSampleView -> phase-1 readings -> RenderSensorData
```

The visual-only path remains available:

```python
build_render_scene_from_published_frame(..., include_sensor_data=False)
```

Plain `build_render_scene(...)` and `build_render_scene_from_tree(...)` still
default to `sensor_data=None`.

---

## 4. Boundary Notes

This implementation follows the Q53 decision:

- no imaging data in `RenderScene.sensor_data`
- no surface-query result in `RenderScene.sensor_data`
- no `sensing -> rendering` dependency
- `RenderScene` may carry debug/export sensor readings, but is not the
  authoritative sensor execution model

The one intentional dependency is in the bridge direction:

```text
rendering.scene_builder -> sensing builders
```

This is limited to published-frame materialization and only happens when
`include_sensor_data=True`.

---

## 5. Test Coverage

Added / updated tests cover:

- `RenderScene.sensor_data` defaults to `None`
- CPU published frame builds numeric sensor data by default
- CPU published frame supports `include_sensor_data=False`
- GPU published frame maps `force_sensor_wp` to
  `ForceSensorReading.contact_force`
- GPU generalized force fields remain `None` when not published

Focused verification command:

```bash
PYTHONPATH=. pytest \
  tests/unit/rendering/test_render_scene.py \
  tests/unit/rendering/test_published_frame_bridge.py \
  tests/unit/sensing \
  -q
```

Result:

```text
29 passed
```

Extended verification command:

```bash
PYTHONPATH=. pytest \
  tests/unit/rendering \
  tests/integration/test_published_frame_render_backend_integration.py \
  tests/unit/sensing \
  tests/unit/physics/test_telemetry_snapshot.py \
  tests/unit/physics/test_cpu_publish_runtime.py \
  tests/unit/physics/test_publish.py \
  -q
```

Result:

```text
68 passed
```

---

## 6. Remaining Work

Possible follow-ups:

1. Decide whether Rerun should log selected numeric sensor values as scalar
   timelines.
2. Keep RL observation vector schema separate from `RenderScene.sensor_data`.
3. Do not add camera/LiDAR fields here; use Q53 `sensor_rendering/` and
   `SurfaceQuerySpec` directions instead.
