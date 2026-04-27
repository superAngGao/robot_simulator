Initiative: sensing-phase1
Stage: review-followup
Author: codex
Version: v1
Date: 2026-04-26
Status: ready-for-rereview-decision-recorded
Related Files: collab/sensing-phase1__implementation-note__codex__v1.md, collab/sensing-phase1__decision__codex__v1.md, sensing/readings.py, sensing/builders.py, tests/unit/sensing/test_readings.py
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
- Remaining question: CPU/GPU asymmetry should be visible on public dataclasses,
  not only in this implementation note.

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

### Public dataclass docs clarified

Added public docstrings to:

- `StateSampleView`
- `IMUReading`
- `JointStateReading`
- `ForceSensorReading`
- `ContactStateReading`

These now document:

- `StateSampleView` is a conservative host-side view over published physics data.
- missing phase-1 fields remain `None` until published by the shared contract.
- `IMUReading.orientation_world_R` is a rotation matrix.
- `IMUReading.linear_acceleration_body` is intentionally absent in phase-1.
- CPU/GPU force telemetry is asymmetric and remains explicit through optional
  fields.
- `ContactStateReading` is count-only until a contact-mask contract exists.

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

For the next Claude pass, the main remaining questions have phase-1 decisions
recorded in:

- `collab/sensing-phase1__decision__codex__v1.md`

Current phase-1 stance:

1. Keep both `ForceSensorReading.qfrc_applied` and
   `ForceSensorReading.tau_smooth`; do not collapse them into a generic force
   field.

2. Keep `IMUReading.orientation_world_R` as the only phase-1 orientation field;
   defer quaternion output until a named downstream consumer requires it.

3. Keep `ContactStateReading` count-only until a backend-neutral published
   contact-mask or contact-pair block exists.

4. Resolved in this follow-up: CPU/GPU asymmetry and phase-1 omissions are now
   captured in public dataclass docstrings, not only in the implementation note.
