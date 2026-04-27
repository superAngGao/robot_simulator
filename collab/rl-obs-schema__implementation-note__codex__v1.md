Initiative: rl-obs-schema
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-04-27
Status: ready-for-review
Related Files: OPEN_QUESTIONS.md#Q31, OPEN_QUESTIONS.md#Q51, rl_env/obs.py, rl_env/obs_terms.py, tests/unit/rl_env/test_obs_schema.py
Owner Summary: Adds a small executable observation-schema contract for the next RL phase. It fixes the quaternion convention at the schema boundary, records explicit normalization semantics, and names the sensing/publish pressure around contact masks without implementing the full GPU RLEnv yet.

---

## 1. Scope

Implemented:

- `ObsFieldSpec`
- `ObsSchema`
- `locomotion_obs_schema(...)`
- `obs_cfg_from_schema(...)`
- unit coverage for field order, slices, quaternion convention, optional contact
  mask, and scale validation

Not implemented:

- manager-based GPU `RLEnv`
- force/contact-force observation fields
- robot-specific normalization ranges
- contact-mask inference from private contact scratch

Those remain out of this schema pass.

---

## 2. Default Field Order

The default locomotion schema is:

```text
base_lin_vel_body
base_ang_vel_body
base_orientation_quat_wxyz
joint_pos
joint_vel
optional contact_mask
```

This order is now queryable through `ObsSchema.slices`, which gives downstream
training code stable names instead of hard-coded offsets.

---

## 3. Quaternion Convention

Decision:

- RL schema uses scalar-first quaternion `[w, x, y, z]`
- the field name is explicit: `base_orientation_quat_wxyz`
- this matches `FreeJoint` and `physics.spatial`

Follow-up pressure on sensing phase-2:

- if observations are built from sensing readings instead of direct `q`, derive
  `base_orientation_quat_wxyz` via `physics.spatial.rot_to_quat` from
  `IMUReading.orientation_world_R`.

This keeps phase-1 `IMUReading.orientation_world_R` unchanged while making the
consumer requirement explicit.

---

## 4. Normalization Convention

Schema normalization is explicit and mechanical:

```text
obs[field] = raw_term(field) * field.scale
```

Default scales are `1.0`. The common RL ranges are robot/task dependent, so this
pass avoids baking global velocity or joint-position scales into the library.
Task configs can override field scales later without changing the canonical
field order.

---

## 5. Contact Mask Boundary

`contact_mask` is optional and has binary semantics:

```text
0.0 / 1.0 in contact_body_names order
```

The current CPU debug `Env` can produce this from its legacy
`active_contacts`. As of the contact-mask published-contract follow-up,
CPU/GPU published frames also expose a backend-neutral `contact_mask`:

- `StateSampleView.contact_mask`
- `ContactStateReading.contact_mask`

GPU/RLEnv should consume that published field instead of reading private
contact scratch.

---

## 6. Verification

Command:

```bash
PYTHONPATH=. pytest tests/unit/rl_env -q
```

Result:

```text
31 passed
```

Local note: the active Python environment was missing the declared `.[rl]`
dependency `gymnasium`; installing `gymnasium>=0.29` allowed the existing RL
tests and the new schema tests to run.
