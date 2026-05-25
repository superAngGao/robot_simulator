# Q54 Physics-Owned Workflow P7.2 Implementation Note

Author: Codex
Date: 2026-05-25
Status: implemented locally, not pushed

## Summary

Added the P7.2 GPU smoke for the physics-owned stepped video runner.

P7.1 introduced the helper:

```text
step_physics_frame(i)
  -> published frame
  -> run_physics_video_scenario(...) bridge
```

P7.2 proves the important GPU behavior: the stepper can perform real physics
stepping before render, and the rendered output comes from the stepped frame
rather than the stale base frame used to initialize the render runtime.

## Code Changes

- Added
  `test_optical_lab_physics_stepped_video_runner_advances_before_render`.
- Imported `run_physics_stepped_video_scenario(...)` into the GPU runtime test.
- Temporarily wrapped `runner.record_delivered_video_frame(...)` inside the
  test to capture delivered full-readback frames without changing production
  APIs.
- Added a clarifying comment near `PhysicsPublishedFrameStepper`:
  the current call shape is `step_physics_frame(frame_index)`, while the wide
  callable type leaves room for future action/control inputs.
- Updated `GPU_OPTICAL_PIPELINE_DESIGN.md` and `MANIFEST.md`.

## Test Coverage

The GPU smoke:

- initializes a base physics frame at one body height;
- runs `run_physics_stepped_video_scenario(...)`;
- uses `step_physics_frame(frame_index)` to call real `GpuEngine.step(...)` at
  two different body heights;
- captures delivered full-readback frames;
- asserts:
  - the delivered `range_m` matches the current stepped body height plus the
    body-bound triangle offset;
  - the two frames have different body heights and different ranges;
  - delivered camera `frame_id` / `sim_time` come from the stepped frame;
  - dynamic frame-preparation timing is present;
  - `frame_timing.csv` keeps `frame_source == "physics_runtime"`.

## Verification

Focused GPU smoke:

```bash
conda run -n env_tilelang_20260119 \
  python -m pytest tests/gpu/test_optical_gpu_runtime.py -q \
  -k "physics_stepped_video_runner"
```

Result:

```text
1 passed, 35 deselected
```

Broader verification:

```bash
conda run -n env_tilelang_20260119 python -m pytest tests/gpu/test_optical_gpu_runtime.py -q
conda run -n env_tilelang_20260119 python -m pytest tests/gpu/test_optical_warp_executor.py -q
conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics -q
conda run -n env_tilelang_20260119 python -m pytest tests/unit/sensing -q
ruff check tools/optical_pipeline_lab/runner.py tests/gpu/test_optical_gpu_runtime.py
ruff format --check tools/optical_pipeline_lab/runner.py tests/gpu/test_optical_gpu_runtime.py
```

Result:

```text
36 passed
5 passed
183 passed
40 passed
ruff check clean
ruff format --check clean
```

## Boundaries

- No `run_scenario(...)` physics execution was enabled.
- No CLI physics engine construction was added.
- No production API was added just to expose delivered frames.
- No `torch_async` physics delivery was enabled.
- No static asset or Go2 path was changed.

## 关键思考

1. Why capture delivered frames by wrapping `record_delivered_video_frame(...)`?

   `run_physics_stepped_video_scenario(...)` correctly returns timing rows, not
   render products. The test needs full-readback channels only to prove P7.2's
   stale-frame regression condition. Wrapping the existing recorder keeps the
   production helper narrow and avoids adding a test-only callback to the public
   runner signature.

2. Why assert `range_m` rather than only timing CSV metadata?

   P6 already proved `frame_source == "physics_runtime"` timing metadata. P7.2
   needs to prove data ownership: render must consume the frame returned by the
   stepper. The body-bound triangle range is a direct observable that changes
   with the current physics body height.

3. Why keep this as one GPU test?

   The CPU/unit tests already cover ordering and exception edges. This slice
   needs only one GPU smoke to connect real `GpuEngine.step(...)` to the stepped
   runner entry and verify render output changes with physics state.
