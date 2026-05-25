# Q54 Physics Runtime Owner P8.1 Implementation Note

Author: Codex
Date: 2026-05-25
Status: implemented locally, not pushed

## Summary

Implemented P8.1: a narrow lab-internal physics runtime owner.

P7 proved that physics-owned frames can drive render/video/delivery. P8.1 adds
the missing lifecycle owner for explicit lab callers:

```text
PhysicsLabScenarioRuntime
  owns engine/registry/base_frame/bounds/metadata
  exposes step_frame(frame_index)
  exposes idempotent close/context-manager cleanup
```

This does not enable `run_scenario(...)`, does not add CLI engine construction,
and does not introduce a generic `SimulationFrameRuntime`.

## Code Changes

- Added `tools/optical_pipeline_lab/physics_runtime.py`.
- Added `PhysicsLabScenarioRuntime`.
  - Holds engine, registry, base frame, bounds, metadata.
  - Calls a supplied `step_frame_fn(frame_index)`.
  - Rejects stepping after close.
  - Provides idempotent `close()`.
  - Calls an explicit `close_fn` if supplied, otherwise calls engine
    `close()`/`destroy()` when available.
- Review follow-up:
  - `_closed` uses `field(init=False)` so it is not part of the public
    dataclass constructor;
  - scripted stepping documents that it teleports `q[6]`;
  - `qdot` now follows the dtype of the default-state `q`.
- Added `create_physics_body_triangle_lab_runtime(...)`.
  - Lazily constructs the one-body synthetic physics scene.
  - Creates a `GpuEngine`.
  - Publishes a base frame at `initial_height`.
  - Exposes scripted `height_for_frame(frame_index)` stepping.
  - Carries the existing body-bound triangle registry/bounds/metadata.
- Updated `GPU_OPTICAL_PIPELINE_DESIGN.md` and `MANIFEST.md`.

## Tests

Added two unit tests:

- `test_physics_lab_scenario_runtime_steps_and_closes_once`
  - verifies `step_frame(...)`;
  - verifies context-manager cleanup on exception;
  - verifies `close()` is idempotent;
  - verifies stepping after close is rejected.
- `test_create_physics_body_triangle_lab_runtime_owns_reset_and_step`
  - monkeypatches the lazy physics construction helpers;
  - verifies factory-time base frame publication at `initial_height`;
  - verifies `height_for_frame(...)` drives later `step_frame(...)`;
  - verifies optional frame synchronization and engine cleanup.

## Verification

Focused unit tests:

```bash
conda run -n env_tilelang_20260119 \
  python -m pytest tests/unit/optics/test_optical_pipeline_lab.py -q \
  -k "physics_lab_scenario_runtime or physics_body_triangle_lab_runtime"
```

Result:

```text
2 passed, 110 deselected
```

Focused lint/format:

```bash
ruff check tools/optical_pipeline_lab/physics_runtime.py tests/unit/optics/test_optical_pipeline_lab.py
ruff format --check tools/optical_pipeline_lab/physics_runtime.py tests/unit/optics/test_optical_pipeline_lab.py
```

Result:

```text
ruff check clean
ruff format --check clean
```

Broader unit coverage:

```bash
conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics -q
```

Result:

```text
185 passed
```

## Boundaries

- No `run_scenario(...)` physics execution was enabled.
- No CLI physics engine construction was added.
- No RL/action observation API was introduced.
- No static asset or Go2 path changed.
- No `torch_async` physics delivery was enabled.

## 关键思考

1. Why add `physics_runtime.py` instead of putting this in `runner.py`?

   `runner.py` coordinates explicit lab runs, but engine construction and
   cleanup are lifecycle ownership. Keeping the owner in a separate narrow
   module prevents `runner.py` from becoming the place where physics scenes,
   action policy, render products, and CLI concerns all accumulate.

2. Why make cleanup idempotent and mostly placeholder-like?

   `GpuEngine` does not currently expose a public close/destroy contract. The
   runtime owner therefore calls `close()` or `destroy()` only if the engine has
   one. This records the lifecycle boundary now without inventing a fake GPU
   resource release API.

3. Why keep `height_for_frame(...)` as the first scripted control hook?

   The body-triangle smoke already uses scripted heights to prove stale-frame
   behavior. A height callback is narrow enough for P8.1, while still proving
   that the runtime owner, not render/video, owns time advancement.

4. Debugging note: factory tests should not require Warp.

   The real factory lazily imports physics/GPU dependencies, but the unit test
   monkeypatches `_build_ball_model`, `_merge_single_ball_model`, and
   `_create_gpu_engine`. This keeps lifecycle semantics unit-testable without a
   GPU.
