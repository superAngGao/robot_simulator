Initiative: sensing-phase1
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-04-26
Status: review-request
Related Files: collab/sensing-phase1__discussion-summary__codex__v1.md, OPEN_QUESTIONS.md#Q51, OPEN_QUESTIONS.md#Q53
Owner Summary: Phase-1 `sensing/` has been implemented as a conservative host-side sensor-facing layer over `PublishedFrame` and `TelemetrySnapshot`. This note summarizes what changed and where Claude should challenge the design.

---

## 1. Scope

This implementation only covers numeric / state-like sensing:

- `StateSampleView`
- `IMUReading`
- `JointStateReading`
- `ForceSensorReading`
- `ContactStateReading`
- builders that derive those readings from `StateSampleView`

It intentionally does not implement:

- `SurfaceQueryView`
- raycast / lidar / depth query runtime
- `ImagingView`
- camera / render-backed sensing

Those remain under `OPEN_QUESTIONS.md#Q53`.

---

## 2. Files Added

- `sensing/__init__.py`
- `sensing/state_sample.py`
- `sensing/readings.py`
- `sensing/builders.py`
- `tests/unit/sensing/test_state_sample.py`
- `tests/unit/sensing/test_readings.py`

---

## 3. Public API

Package exports:

```python
from sensing import (
    StateSampleView,
    build_state_sample_view,
    IMUReading,
    JointStateReading,
    ForceSensorReading,
    ContactStateReading,
    build_imu_reading,
    build_joint_state_reading,
    build_force_sensor_reading,
    build_contact_state_reading,
)
```

---

## 4. `StateSampleView`

`StateSampleView` is a host-side view derived from one published frame and one env:

```python
@dataclass
class StateSampleView:
    frame_id: int
    step_index: int
    sim_time: float
    env_idx: int

    q: np.ndarray | None
    qdot: np.ndarray | None

    X_world: object | None
    v_bodies: np.ndarray | None
    contact_count: int | None

    telemetry: TelemetrySnapshot | None
```

Implementation rule:

- consume only `CpuPublishedFrame` / `GpuPublishedFrame`
- use `build_telemetry_snapshot_from_published_frame(...)` for telemetry
- do not read engine-private scratch buffers
- copy numeric arrays crossing into the view
- use `None` when a field is not published

CPU path:

- copies `q`, `qdot`, `v_bodies`
- keeps `X_world` from the CPU published frame
- copies / normalizes `contact_count`
- composes `TelemetrySnapshot`

GPU path:

- reads published Warp buffers via `.numpy()`
- reconstructs `SpatialTransform` objects from `x_world_R_wp` and `x_world_r_wp`
- slices by `env_idx`
- composes `TelemetrySnapshot`

---

## 5. Reading Builders

### `build_joint_state_reading(...)`

- returns copies of `q` and `qdot`
- supports `joint_indices`
- missing state fields remain `None`

### `build_contact_state_reading(...)`

- returns `contact_count`

Rationale: per-body active contact mask is not directly available from the current published contract. Deriving it would require interpreting contact pair cache semantics, so phase-1 does not expose that field yet.

### `build_imu_reading(...)`

- takes `body_index`
- `orientation_world_R = X_world[body_index].R`
- `angular_velocity_body = v_bodies[body_index][3:6]`
- `linear_acceleration_body = None`

Important convention:

- this repo uses spatial velocity layout `[linear; angular]`
- therefore angular velocity is `v[3:6]`, not `v[:3]`

Rationale for `linear_acceleration_body=None`:

- no per-body acceleration source is currently published into `StateSampleView`
- using qacc directly would not be equivalent to body-frame IMU acceleration
- phase-1 preserves the "do not guess un-published values" rule

### `build_force_sensor_reading(...)`

- `qfrc_applied = telemetry.qfrc_applied`
- `tau_smooth = telemetry.tau_smooth`
- `contact_force = telemetry.force_sensor`
- `body_force = None`
- supports `sensor_indices` for `contact_force`

Current asymmetry:

- CPU `TelemetrySnapshot.force_sensor` is currently `None`
- GPU telemetry can provide `force_sensor`
- CPU generalized forces are available through `ForceState`
- GPU generalized force fields are not yet published in the same way

Claude review follow-up:

- `generalized_force` was removed because it ambiguously hid the fact that `qfrc_applied` excludes actuator torque.
- `qfrc_applied` and `tau_smooth` are now separate fields.

---

## 6. Test Coverage

Added unit coverage for:

- CPU `build_state_sample_view(...)`
- GPU-style `GpuPublishedFrame` view construction using array wrappers
- joint reading full and subset extraction
- contact reading exposing `contact_count`
- IMU orientation and angular velocity extraction
- IMU missing fields returning `None`
- force reading from telemetry
- force sensor subset extraction
- force reading missing telemetry returning `None`
- real CPU engine force reading with `contact_force is None`
- package-level exports

Verification command:

```bash
PYTHONPATH=. pytest \
  tests/unit/sensing \
  tests/unit/physics/test_telemetry_snapshot.py \
  tests/unit/physics/test_cpu_publish_runtime.py \
  tests/unit/physics/test_publish.py \
  -q
```

Result:

```text
35 passed
```

---

## 7. Review Targets For Claude

Please challenge these points first:

1. Is `StateSampleView` correctly scoped as a host-side view over published frames, or is it already too close to a sensor-specific abstraction?

2. Is composing `TelemetrySnapshot` inside `StateSampleView` better than flattening telemetry fields into the view?

3. Should `build_imu_reading(...)` expose `orientation_world_R` as a rotation matrix (`X_world.R`) for phase-1, or should the reading schema prefer quaternion from the start?

4. Is `angular_velocity_body = v_bodies[body_index][3:6]` correct given the repo-wide `[linear; angular]` spatial convention and the GPU kernel conventions?

5. Is `linear_acceleration_body=None` the right conservative choice, or should phase-1 add an explicit published body-acceleration field before calling the IMU builder "real"?

6. Are `ForceSensorReading.qfrc_applied` and `ForceSensorReading.tau_smooth` the right phase-1 generalized-force fields, or should one of them be left in `TelemetrySnapshot` only?

7. Should `ContactStateReading` stay minimal with only `contact_count` until the published contract can support per-body contact masks directly?

8. Are the CPU/GPU asymmetries documented clearly enough for downstream RL observation code?

---

## 8. Known Non-Goals / Deferred Work

- no noise model
- no sensor frame transform
- no named joint/body/sensor selection layer
- no schema registry
- no async sensor queue
- no ray queries
- no camera/imaging path
- no migration of `physics/telemetry.py`

These are intentionally left out to keep this pass focused on published-frame consumption.
