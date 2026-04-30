Initiative: q54-optical-executor-phase-b
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-04-30
Status: implemented
Related Files: optics/execution.py, tests/unit/optics/test_executor_schema.py
Owner Summary: Executor Phase B is implemented. `CpuReferenceOpticalExecutor.execute(...)` now keeps the same public API while internally splitting validation, workload preparation, intersection, channel resolution, and result construction. New schema tests pin channel names, dtype/shape contracts, and miss values for future backend replacement.

# Q54 Optical Executor Phase B Implementation Note

## 1. What Changed

`CpuReferenceOpticalExecutor.execute(snapshot, spec)` now delegates to explicit
internal steps:

```text
_validate(snapshot, spec)
_prepare_workload(spec)
_intersect(snapshot, workload)
_resolve_channels(hits)
_build_result(snapshot, spec, channels)
```

The public contract remains unchanged:

```python
OpticalExecutor.execute(snapshot, spec) -> OpticalComputeResult
```

## 2. Why

This prepares the executor boundary for future backends:

- CPU BVH / Embree can replace the `_intersect(...)` step;
- direct-light/RGB can add channel resolution after first hit;
- GPU backends can keep the same `OpticalComputeResult` schema while changing
  buffer ownership/readiness.

## 3. Capability Declaration

`CpuReferenceOpticalExecutor.capabilities` now declares the channels it returns:

```text
range_m
hit_mask
position_world
normal_world
material_id
instance_id
numeric_instance_id
```

Tests verify returned channels match capabilities.

## 4. Schema Tests

Added `tests/unit/optics/test_executor_schema.py`.

It verifies:

- returned channel names;
- channel shapes;
- dtypes;
- miss semantics:
  - `hit_mask=False`;
  - `range_m=np.inf`;
  - `position_world` / `normal_world` are NaN;
  - human-readable ids are `None`;
  - `numeric_instance_id=0`.

These tests are intended to protect the contract when adding Embree, Warp/CUDA,
OptiX, or direct-light/RGB executors.

## 5. Verification

```text
PYTHONPATH=. pytest tests/unit/optics tests/unit/sensing -q
48 passed

ruff check optics sensing tests/unit/optics tests/unit/sensing
All checks passed

git diff --check
passed

python -m compileall optics sensing tests/unit/optics tests/unit/sensing
passed
```
