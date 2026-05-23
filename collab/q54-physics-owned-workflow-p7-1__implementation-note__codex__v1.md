# Q54 Physics-Owned Workflow P7.1 Implementation Note

Author: Codex
Date: 2026-05-23
Status: implemented locally, not pushed

## Summary

Implemented the first P7 slice: a physics-owned stepped video entry above the
existing P6 published-frame bridge.

P6 remains the lower-level reproducible entry:

```text
published_frame_for_index(i)
  -> run_physics_video_scenario(...)
```

P7.1 adds the stronger stepper vocabulary:

```text
step_physics_frame(i)
  -> published frame
  -> existing P6 provider/render/delivery bridge
```

The new helper does not construct a physics engine, does not enable the plain
CLI `run_scenario(...)` path, and does not change render/session ownership.

## Code Changes

- Added `PhysicsPublishedFrameStepper = Callable[..., object]` in
  `tools/optical_pipeline_lab/runner.py`.
- Added `run_physics_stepped_video_scenario(...)`.
  - It calls `step_physics_frame(frame_index)`.
  - It forwards the returned frame through `run_physics_video_scenario(...)`.
  - It reuses the existing P6 runtime/provider/video/delivery/timing path.
- Updated the guarded `run_scenario(...)` error message to point callers at both
  explicit physics entries:
  - `run_physics_video_scenario(...)`
  - `run_physics_stepped_video_scenario(...)`
- Updated `GPU_OPTICAL_PIPELINE_DESIGN.md` with P7.1 completion status.
- Updated `MANIFEST.md` with the new helper and test counts.

## Tests

Added three unit tests:

- `test_physics_stepped_video_runner_steps_before_provider_borrow`
  - verifies the stepper runs before provider `begin_frame(...)`;
  - verifies the returned published frame is passed into the existing provider
    bridge;
  - verifies delivery flush runs after all per-frame provider contexts exit.
- `test_physics_stepped_video_runner_stepper_exception_stops_before_provider_borrow`
  - verifies a stepper exception propagates;
  - verifies provider borrow is never attempted after the stepper fails.
- `test_physics_stepped_video_runner_render_exception_completes_provider_borrow`
  - verifies render failure inside the provider context exits the provider;
  - verifies the exception is re-raised rather than suppressed.

## Verification

Focused unit tests:

```bash
conda run -n env_tilelang_20260119 \
  python -m pytest tests/unit/optics/test_optical_pipeline_lab.py -q \
  -k "physics_stepped_video_runner or physics_video_runner_rejects or run_scenario_physics"
```

Result:

```text
5 passed, 105 deselected
```

Broader verification:

```bash
conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics -q
conda run -n env_tilelang_20260119 python -m pytest tests/gpu/test_optical_gpu_runtime.py -q
ruff check tools/optical_pipeline_lab/runner.py tests/unit/optics/test_optical_pipeline_lab.py
ruff format --check tools/optical_pipeline_lab/runner.py tests/unit/optics/test_optical_pipeline_lab.py
```

Result:

```text
183 passed
35 passed
ruff check clean
ruff format --check clean
```

## Boundaries

- No `run_scenario(...)` physics execution was enabled.
- No CLI physics engine construction was added.
- No `torch_async` physics delivery was enabled.
- No static asset or Go2 code was touched.
- No generic `SimulationFrameRuntime` was exported.

## 关键思考

1. Why make the new helper a thin wrapper over P6?

   The important new boundary is the stepper semantics, not a second copy of the
   provider/render/delivery loop. Reusing `run_physics_video_scenario(...)`
   keeps P6 as the single assembly bridge and preserves its debugging value: a
   failing stepped run can still be reduced to a known
   `published_frame_for_index(...)` replay.

2. Why keep `PhysicsPublishedFrameStepper` duck-typed?

   `runner.py` is lab tooling and already treats physics runtime objects as
   explicit caller-owned inputs. Keeping the alias as `Callable[..., object]`
   avoids tightening the dependency on a concrete physics type before the future
   RL/action interface is designed.

3. Why not expose action/control yet?

   The stepped helper's contract is only "advance or select physics time and
   return a published frame". Action selection can live inside the callback for
   now. Exposing an `action_for_index(...)` hook before observation products
   exist would make the workflow API look more mature than it is.

4. Debugging note: exception ordering needed two separate tests.

   Stepper failure happens before provider borrow, while render failure happens
   inside provider ownership. The tests intentionally cover both sides so future
   refactors do not accidentally acquire a physics borrow before stepping
   succeeds or suppress render exceptions while completing the provider.
