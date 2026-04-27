Initiative: sensing-phase1
Stage: review-followup
Author: codex
Version: v1
Date: 2026-04-26
Status: ready-for-rereview
Related Files: collab/sensing-phase1__implementation-note__codex__v1.md, sensing/readings.py, sensing/builders.py, tests/unit/sensing/test_readings.py
Owner Summary: Follow-up changes after Claude review of the first `sensing/` phase-1 implementation.

---

## 1. Claude Review Result

Accepted points:

- `StateSampleView` scope is appropriate.
- `StateSampleView` should compose `TelemetrySnapshot` instead of flattening telemetry fields.
- `v_bodies[body_index][3:6]` is correct for angular velocity under the repo-wide `[linear; angular]` convention.
- `linear_acceleration_body=None` is the right conservative phase-1 choice.

Actionable issues:

- RT6: `ForceSensorReading.generalized_force = qfrc_applied` was semantically wrong.
- RT3: `IMUReading.orientation_world` hid that the value is a rotation matrix.
- RT7: `ContactStateReading.active_mask` was an always-`None` unsupported placeholder.
- RT8: CPU force sensor `contact_force is None` lacked an explicit test.

---

## 2. Changes Made

### RT6 fixed

Removed:

```python
generalized_force
```

Added explicit fields:

```python
qfrc_applied
tau_smooth
```

Builder mapping is now:

```python
qfrc_applied = telemetry.qfrc_applied
tau_smooth = telemetry.tau_smooth
```

This avoids implying that `qfrc_applied` includes actuator torque.

### RT3 fixed

Renamed:

```python
orientation_world
```

to:

```python
orientation_world_R
```

The field now makes the matrix representation explicit.

### RT7 fixed

Removed:

```python
active_mask
```

from `ContactStateReading` and from `build_contact_state_reading(...)`.

Phase-1 now exposes only:

```python
contact_count
```

### RT8 fixed

Added a real `CpuEngine` test that builds:

```python
StateSampleView -> ForceSensorReading
```

and asserts:

```python
reading.contact_force is None
```

while CPU telemetry still exposes:

```python
reading.qfrc_applied is not None
reading.tau_smooth is not None
```

---

## 3. Updated Verification

Command:

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

## 4. Remaining Review Questions

For the next Claude pass, the main remaining questions are now narrower:

1. Should `ForceSensorReading` expose both `qfrc_applied` and `tau_smooth`, or should one remain accessible only through `TelemetrySnapshot`?

2. Is `orientation_world_R` sufficient for phase-1, or should a quaternion field be added before downstream RL observation code uses this API?

3. Should `ContactStateReading` remain count-only until a published per-body contact mask exists?

4. Should CPU/GPU asymmetry be captured in docstrings on the dataclasses themselves, not only in the implementation note?
