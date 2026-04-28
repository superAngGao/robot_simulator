Initiative: q50-render-backend
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-04-28
Status: implemented
Related Files: OPEN_QUESTIONS.md#Q50, collab/q50-step4-render-scene-sensor-data__implementation-note__codex__v1.md, rendering/backends/rerun_backend.py, tests/rendering/test_rerun_backend.py
Owner Summary: RerunBackend now consumes the narrow `RenderScene.sensor_data` debug/export payload and logs selected numeric readings as scalar timelines. This closes the first backend-consumption follow-up after Q50 Step 4 without adding camera/LiDAR/image payloads to RenderScene.

## Summary

Q50 Step 4 made `RenderScene.sensor_data` available, but backends did not consume it. This pass adds conservative Rerun scalar timeline logging for phase-1 numeric/state readings.

## Review Follow-Up (2026-04-28)

- The IMU group branch now uses the same guard style as `contact` / `joint` /
  `force` rather than an early `return`, so future sensor groups can be added
  after it without being skipped.
- `rr.Scalars` was verified against the real `rerun-sdk 0.31.3` environment:
  `rr.Scalar` is absent and `rr.Scalars(...)` is the correct current API.

The Rerun entity prefix is:

```text
env_{env_index}/sensors/...
```

Implemented scalar paths include:

- `contact/contact_count`
- `contact/contact_mask/{i}`
- `joint/q/{i}` and `joint/qdot/{i}`
- `force/qfrc_applied/{i}`
- `force/tau_smooth/{i}`
- `force/contact_force/{i}`
- `imu/body_{body_index}/angular_velocity_body/{i}`
- `imu/body_{body_index}/linear_acceleration_body/{i}` when present

For vector fields, the backend also logs:

```text
.../norm
```

Arrays are capped at 32 individual scalar components by default. When an array exceeds that cap, the backend logs:

```text
.../truncated_size
```

This keeps large robots from producing unbounded Rerun entity trees while still exposing useful debug signals.

The cap is configurable through:

```python
RerunBackend(max_sensor_array_scalars=32)
```

Sensor scalar logging can also be disabled for geometry-only recordings:

```python
RerunBackend(log_sensor_data=False)
```

## Boundary

This remains inside the narrow Q50/Q51/Q53 boundary:

- no camera payload
- no LiDAR payload
- no surface query result
- no new sensor schema owned by `rendering/`
- no `sensing -> rendering` dependency

`RerunBackend` only consumes the already-materialized `RenderSensorData` debug/export payload.

## Test Behavior Change

`tests/rendering/test_rerun_backend.py` no longer skips the entire file when `rerun-sdk` is absent.

- mocked backend-dispatch tests run in the default/base environment
- only the real `.rrd` file-output test is skipped without `rerun-sdk`

This prevents mock-only backend behavior from disappearing silently in environments that do not install Rerun.

## Verification

Default/base environment:

```bash
PYTHONPATH=. pytest tests/rendering/test_rerun_backend.py -q
PYTHONPATH=. pytest tests/rendering tests/unit/rendering tests/integration/test_render_scene_integration.py tests/integration/test_published_frame_render_backend_integration.py -q
ruff check rendering/backends/rerun_backend.py tests/rendering/test_rerun_backend.py
python -m compileall rendering tests/rendering/test_rerun_backend.py
```

Results:

- before configurability follow-up: `6 passed, 1 skipped`
- before configurability follow-up: `65 passed, 1 skipped`
- after configurability follow-up: `10 passed, 1 skipped`
- after configurability follow-up: `69 passed, 1 skipped`
- `ruff` passed
- `compileall` passed

Rerun-enabled environment:

```bash
conda run -n env_tilelang_20260119 python -m pytest tests/rendering/test_rerun_backend.py -q
conda run -n env_tilelang_20260119 python -m pytest tests/rendering tests/unit/rendering tests/integration/test_render_scene_integration.py tests/integration/test_published_frame_render_backend_integration.py -q
```

Results:

- before configurability follow-up: `7 passed`
- before configurability follow-up: `66 passed`
- after configurability follow-up: `11 passed`
- after configurability follow-up: `70 passed`

Use `python -m pytest` inside the conda env; bare `pytest` can resolve to the base/user entrypoint.

## Remaining Work

- richer Rerun blueprint/layout for scalar panels
- richer backend defaults / presets for scalar logging profiles
- deciding whether selected scalar logging should also exist in DebugExporter
- keeping RL observation schema separate from Rerun/debug visualization
