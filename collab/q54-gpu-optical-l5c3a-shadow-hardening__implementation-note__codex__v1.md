# Q54 GPU Optical L5C.3a Shadow Hardening Implementation Note

Date: 2026-05-05

## Scope

Implemented the first L5C.3a hardening pass after Claude's review.

Added:

- aggregate shadow traversal diagnostics on `GpuDeviceBvhDirectLightOpticalExecutor`:
  - `shadow_stack_overflow_count`
  - `shadow_max_stack_depth`
- explicit docs that:
  - `GpuDeviceSceneOpticalExecutor` is first-hit only;
  - GPU shaded RGB should use `GpuDeviceBvhDirectLightOpticalExecutor`;
  - `rgb` / `intensity` are deterministic direct-light channels, not future
    path-tracing radiance/sample-accumulation contracts.
- GPU direct-light parity coverage for:
  - zero-light scenes with ambient plus background miss color;
  - multi-light additive direct lighting;
  - point-light shadow occlusion by a triangle blocker;
  - analytical plane shadow occlusion separate from triangle BVH shadow
    occlusion;
  - shadow diagnostic channels for no-shadow and shadow-enabled paths.

## Design Notes

The shadow diagnostics follow the existing primary BVH traversal model: they are
returned as device result channels instead of forcing the executor to
synchronize and raise. This keeps the Q52 async/lifetime shape intact. Preview,
benchmark, and staged tests can inspect the aggregate counter and fail loudly
when it is non-zero.

`shadow_bias` was already an explicit constructor parameter:

```python
GpuDeviceBvhDirectLightOpticalExecutor(shadow_bias=1.0e-6)
```

This pass kept that API and added documentation/tests around the behavior rather
than changing the signature.

## Verification

Commands run:

```bash
python -m py_compile optics/execution.py optics/warp_execution.py \
  tests/gpu/test_optical_gpu_runtime.py

conda run -n env_tilelang_20260119 python -m pytest \
  tests/gpu/test_optical_gpu_runtime.py -q

conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics -q

conda run -n env_tilelang_20260119 ruff check \
  optics/execution.py optics/warp_execution.py tests/gpu/test_optical_gpu_runtime.py
```

Results:

- GPU optical runtime tests: 24 passed.
- Unit optics tests: 57 passed.
- Ruff: passed.

## Remaining L5C.3a Items

- L5C.3a parity/hardening items from the review are complete.
- Future optional cases: role-filtered shadow rays with a visible primary
  surface and invisible occluder, and equal-distance shadow blockers if shadow
  source-order semantics ever become observable.
