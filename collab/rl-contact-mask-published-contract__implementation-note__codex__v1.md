Initiative: rl-contact-mask
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-04-27
Status: ready-for-review
Related Files: OPEN_QUESTIONS.md#Q31, OPEN_QUESTIONS.md#Q51, OPEN_QUESTIONS.md#Q52, physics/publish.py, physics/cpu_engine.py, physics/gpu_engine.py, sensing/state_sample.py, sensing/readings.py, rl_env/obs.py
Owner Summary: Adds the first narrow sensing phase-2 published contract needed by RL obs: a per-contact-body binary `contact_mask`. The mask is published by CPU/GPU frames, carried through `StateSampleView`, exposed on `ContactStateReading`, and aligned with `rl_env/obs.py` contact-mask schema.

---

## 1. Scope

Implemented:

- `CpuPublishedFrame.contact_mask`
- `GpuPublishedFrame.contact_mask_wp`
- GPU published slot buffer for `contact_mask`
- CPU host snapshot field `"contact_mask"`
- GPU host snapshot field `"contact_mask"`
- `StateSampleView.contact_mask`
- `ContactStateReading.contact_mask`
- RL schema wording updated to consume published `contact_mask`

Not implemented:

- compact contact-pair block
- contact-force mask or force magnitude observations
- body-pair matrix
- async host staging

---

## 2. Mask Semantics

The mask is binary and ordered by the published contact-body list:

```text
0.0 / 1.0 in contact_body_names order
```

CPU derives it from published `ContactInfo` entries.

GPU derives it from active published contact slots:

```text
contact_active + contact_bi/contact_bj + contact_body_idx -> contact_mask
```

Both `body_i` and `body_j` are checked so body-body contacts can activate named
contact bodies on either side.

---

## 3. Boundary

This does not reopen the dense `RigidBlock` question. `contact_mask` is a
lightweight summary block that can be published every frame without requiring
the full dense contact cache.

This also does not make RL obs read GPU private scratch. The path is:

```text
PublishedFrame -> StateSampleView -> ContactStateReading -> RL obs schema
```

---

## 4. Verification

Command:

```bash
PYTHONPATH=. pytest \
  tests/unit/sensing \
  tests/unit/rendering/test_published_frame_bridge.py \
  tests/unit/physics/test_cpu_publish_runtime.py \
  tests/unit/physics/test_telemetry_snapshot.py \
  tests/unit/physics/test_publish.py \
  -q
```

Result:

```text
46 passed
```

Extended command:

```bash
PYTHONPATH=. pytest \
  tests/unit/sensing \
  tests/unit/rendering/test_published_frame_bridge.py \
  tests/unit/physics/test_cpu_publish_runtime.py \
  tests/unit/physics/test_telemetry_snapshot.py \
  tests/unit/physics/test_publish.py \
  tests/unit/rl_env \
  -q
```

Result:

```text
77 passed
```

GPU contact-mask accessor check was also collected with the existing Warp/CUDA
skip guard:

```bash
PYTHONPATH=. pytest tests/gpu/test_gpu_engine_api.py::TestStateAccessors::test_contact_mask_shape -q
```

Result in this environment:

```text
1 skipped
```
