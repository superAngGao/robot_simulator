Initiative: q54-multiphysics-optical-scene
Stage: review-followup
Author: codex
Version: v1
Date: 2026-04-30
Status: required-edit-applied
Related Files: collab/q54-multiphysics-optical-scene__review-request__codex__v1.md, collab/q54-multiphysics-optical-scene__review__claude__v1.md, optics/scene.py, optics/__init__.py, tests/unit/optics/test_optics_phase_a.py
Owner Summary: Claude's blocking multi-physics scene edit has been applied. `OpticalSceneCache` now has a `snapshot_from_frame_inputs(...)` entrypoint that consumes an `OpticalFrameInputs` aggregate. The existing `snapshot_from_published_frame(...)` API remains as a Phase-A rigid-body convenience wrapper.

# Q54 Multi-Physics Optical Scene Review Follow-Up

## 1. Applied Blocking Edit

Claude requested an `OpticalFrameInputs` aggregate before registry-builder work.
Applied.

New Phase-A shape:

```python
@dataclass(frozen=True)
class OpticalFrameInputs:
    frame_id: int
    sim_time: float
    env_idx: int
    rigid: CpuPublishedFrame | None = None
```

Current behavior:

- `OpticalFrameInputs.from_published_frame(frame, env_idx=0)` builds the rigid
  Phase-A aggregate;
- `OpticalSceneCache.snapshot_from_frame_inputs(inputs)` is now the main scene
  construction entrypoint;
- `OpticalSceneCache.snapshot_from_published_frame(frame, env_idx=0)` is
  retained as a convenience wrapper;
- `OpticalFrameInputs` validates rigid `frame_id` and `sim_time` alignment;
- Phase A still rejects `env_idx != 0`.

## 2. Boundary Preserved

The change does not alter executor/result behavior. Scene/cache still prepares
frame-aligned executable scene data; executor still performs optical
computation.

The `OpticalFrameInputs` docstring records Claude's sensor-independent geometry
realization rule:

```text
particle/level-set -> surface mesh conversion can be scene preparation;
ray-direction-dependent volume integration remains executor work.
```

## 3. Deferred Items

Deferred per Claude review:

- `binding_kind` / `dynamic_source_id`, until the first non-rigid producer;
- medium registry, while `OpticalMaterialSpec.extension` is enough;
- fluid representation-kind dirty rule, until fluid producer implementation.

## 4. Verification

Commands:

```text
PYTHONPATH=. pytest tests/unit/optics -q
ruff check optics tests/unit/optics
python -m compileall optics tests/unit/optics
```

Results:

```text
14 passed
All checks passed
compileall passed
```
