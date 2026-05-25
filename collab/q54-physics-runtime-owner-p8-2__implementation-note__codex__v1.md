# Q54 Physics Runtime Owner P8.2 Implementation Note

Author: Codex
Date: 2026-05-25
Status: implemented locally, not pushed

## Summary

Implemented P8.2: a focused GPU smoke that wires the explicit
`PhysicsLabScenarioRuntime` owner into the existing stepped video runner.

P8.1 introduced the lifecycle owner. P8.2 proves it can drive the real
render/video/delivery path:

```text
create_physics_body_triangle_lab_runtime(...)
  -> runtime.step_frame(i)
  -> run_physics_stepped_video_scenario(...)
  -> delivered full-readback frame
```

## Code Changes

- Added `test_optical_lab_physics_runtime_owner_drives_stepped_video_runner`.
- Imported `tools.optical_pipeline_lab.physics_runtime` into the GPU runtime
  test module.
- Updated `GPU_OPTICAL_PIPELINE_DESIGN.md` and `MANIFEST.md`.

## Test Coverage

The GPU smoke:

- creates `PhysicsLabScenarioRuntime` with scripted body heights;
- uses it as a context manager;
- passes `runtime.engine`, `runtime.registry`, `runtime.base_frame`,
  `runtime.step_frame`, `runtime.bounds_min`, `runtime.bounds_max`, and
  `runtime.metadata` into `run_physics_stepped_video_scenario(...)`;
- captures delivered full-readback frames by wrapping
  `runner.record_delivered_video_frame(...)`;
- asserts delivered `range_m` follows the scripted runtime body heights;
- asserts dynamic prepare timings are present;
- asserts `frame_timing.csv` keeps `frame_source == "physics_runtime"`;
- asserts the runtime is open inside the context and closed after exit.

## Verification

Focused GPU smoke:

```bash
conda run -n env_tilelang_20260119 \
  python -m pytest tests/gpu/test_optical_gpu_runtime.py -q \
  -k "runtime_owner_drives"
```

Result:

```text
1 passed, 36 deselected
```

Broader verification:

```bash
conda run -n env_tilelang_20260119 python -m pytest tests/gpu/test_optical_gpu_runtime.py -q
conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics -q
ruff check tools/optical_pipeline_lab/physics_runtime.py tests/gpu/test_optical_gpu_runtime.py
ruff format --check tools/optical_pipeline_lab/physics_runtime.py tests/gpu/test_optical_gpu_runtime.py
```

Result:

```text
37 passed
185 passed
ruff check clean
ruff format --check clean
```

## Boundaries

- No `run_scenario(...)` physics execution was enabled.
- No CLI physics engine construction was added.
- No runner API change was needed.
- No `torch_async` physics delivery was enabled.
- No static asset or Go2 path changed.

## 关键思考

1. Why make this a separate GPU smoke from P7.2?

   P7.2 proved the stepped runner can consume frames from a raw stepper
   callback. P8.2 proves the new lifecycle owner can supply the same contract
   without leaking engine construction into runner/video/render code.

2. Why capture delivered frames the same way as P7.2?

   The runner returns timing rows by design. Full-readback channels are only
   needed for the smoke assertion, so the test wraps
   `record_delivered_video_frame(...)` rather than adding production API just
   for test observation.

3. Why keep `run_scenario(...)` guarded?

   The runtime owner is now real, but the user-facing ownership question remains
   open: CLI, sensor-loop, and future RL runtime have different action/reset
   semantics. P8.2 proves the owner can drive explicit lab callers without
   making that decision prematurely.
