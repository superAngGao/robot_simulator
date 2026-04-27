Initiative: sensing-phase1
Stage: decision
Author: codex
Version: v1
Date: 2026-04-27
Status: accepted-for-phase1
Related Files: collab/sensing-phase1__implementation-note__codex__v1.md, collab/sensing-phase1__review-followup__codex__v1.md, sensing/readings.py, sensing/builders.py
Owner Summary: Phase-1 sensing API decisions after the first Claude review follow-up. These decisions freeze the current numeric/state reading shape until a later published-contract or RL-observation pass creates enough pressure to revise it.

---

## 1. Decision Summary

For phase-1:

1. Keep both `ForceSensorReading.qfrc_applied` and
   `ForceSensorReading.tau_smooth`.
2. Keep `IMUReading.orientation_world_R` as a rotation matrix and do not add a
   quaternion field yet.
3. Keep `ContactStateReading` count-only.

These are API-freeze decisions for the current phase, not claims that the
schemas are final sensor models.

---

## 2. Force Reading Fields

Decision:

- keep `qfrc_applied`
- keep `tau_smooth`
- keep both optional
- do not collapse either into a generic `generalized_force`

Rationale:

- `qfrc_applied` and `tau_smooth` mean different things in the existing force
  pipeline.
- hiding either behind `generalized_force` loses important provenance.
- CPU currently publishes generalized force terms through `TelemetrySnapshot`.
- GPU currently has a different telemetry surface and can expose force-sensor
  values without the same generalized force terms.
- optional fields make the backend asymmetry visible instead of pretending that
  both paths publish the same force schema.

Deferred:

- A later named force-sensor API may split generalized-force telemetry from
  physical contact-force sensors.
- A later GPU telemetry expansion may add `qfrc_*` / `tau_smooth` parity.

---

## 3. IMU Orientation Representation

Decision:

- keep `orientation_world_R`
- represent it as a rotation matrix copied from `X_world[body_index].R`
- do not add `orientation_world_quat` in phase-1

Rationale:

- `StateSampleView` already carries `SpatialTransform` data, whose orientation
  is matrix-shaped.
- adding a quaternion now duplicates orientation state without adding a new
  published source.
- quaternion convention would need to be documented carefully across consumers
  because this repo uses scalar-first `[w, x, y, z]`, while some downstream
  interfaces prefer `[x, y, z, w]`.
- downstream RL observations can derive quaternions later with
  `physics.spatial.rot_to_quat` if the observation schema wants them.

Deferred:

- Add a quaternion field only when a named observation/sensor consumer requires
  it, and document its ordering at that point.

---

## 4. Contact Reading Scope

Decision:

- keep `ContactStateReading.contact_count`
- do not expose `active_mask` or per-body contact flags in phase-1

Rationale:

- `contact_count` is directly part of the current published contract.
- a per-body active mask would need interpretation of contact pairs / slots and
  backend-specific cache details.
- deriving that mask in `sensing/` today would violate the rule that phase-1
  views should not infer unpublished values from private internals.

Deferred:

- Add per-body or per-shape contact masks only after the published contract
  exposes a backend-neutral contact-mask or contact-pair block.

---

## 5. Phase-1 Boundary

These decisions keep phase-1 intentionally conservative:

- no inferred sensor values
- no duplicate orientation representation
- no synthetic contact masks
- backend asymmetry remains visible through `None`

The next implementation pressure should come from either:

- an RL observation schema that needs a particular representation, or
- a published-contract expansion that makes a missing field first-class.
