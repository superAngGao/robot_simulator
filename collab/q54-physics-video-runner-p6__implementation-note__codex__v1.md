# Q54 Physics Video Runner P6 Implementation Note

Author: Codex
Date: 2026-05-21
Status: implemented locally, not pushed

## Summary

Added the P6 explicit physics video runner path.

The new path is intentionally not a generic CLI physics loop. It requires the
caller to provide the physics-owned objects and callback:

```text
engine
registry
base_frame
published_frame_for_index(frame_index)
build_video_camera(scene, args, frame_index)
pack_rgb8(result)
synchronize_event(event)
```

This preserves the design boundary: physics owns time and dynamic published
frames; render/video/delivery consume already selected frames.

## Code Changes

- Added `physics_body_triangle_video_smoke` preset.
- Relaxed `OpticalLabScenarioConfig.validate_implemented()` for only the tiny
  physics runtime smoke shape.
- Added `runner.run_physics_video_scenario(...)`.
  - Validates the explicit physics path.
  - Writes `scenario_config.json`.
  - Creates a physics render runtime through
    `create_physics_render_runtime_for_config(...)`.
  - Uses `PhysicsFrameContextProvider`.
  - Uses `FrameWorkflowRunner` for provider lifecycle, video rendering, and
    delivery.
  - Writes `frame_timing.csv`.
- Added `runner.build_physics_video_args(...)`.
- Kept plain `run_scenario(...)` guarded for physics runtime configs; it now
  points callers to `run_physics_video_scenario(...)`.
- Guarded `build_menagerie_example_args(...)` against physics runtime configs;
  it remains only for static/synthetic transitional paths.
- Moved the P6 `frame_source` metadata tracking item from `OPEN_QUESTIONS.md`
  into `REFLECTIONS.md` after closing it.

## Tests

- Unit coverage:
  - physics smoke preset is currently implemented;
  - physics runtime configs outside the explicit smoke remain reserved;
  - physics runner args use `frame_defaults_for_config(...)`;
  - Menagerie arg builder rejects physics runtime configs;
  - physics runner rejects `torch_async` until a physics warmup frame source
    exists;
  - plain `run_scenario(...)` requires explicit physics runtime inputs.
- GPU coverage:
  - `test_optical_lab_physics_video_runner_writes_frame_source_csv`
    drives real `GpuEngine.step(...)` frames through
    `run_physics_video_scenario(...)`;
  - asserts `frame_timing.csv` includes
    `frame_source == "physics_runtime"`;
  - asserts width/height/scene preset/geometry metadata and dynamic
    snapshot/refit timings.

## Verification

Focused unit tests:

```bash
conda run -n env_tilelang_20260119 \
  python -m pytest tests/unit/optics/test_optical_pipeline_lab.py -q \
  -k "physics_body_triangle or physics_runtime or physics_smoke or run_scenario_physics or translates_physics"
```

Result:

```text
6 passed, 101 deselected
```

Focused GPU runner smoke:

```bash
conda run -n env_tilelang_20260119 \
  python -m pytest tests/gpu/test_optical_gpu_runtime.py -q \
  -k "physics_video_runner_writes_frame_source_csv"
```

Result:

```text
1 passed, 34 deselected
```

Full Q54 sensing/optics set:

```bash
conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics -q
conda run -n env_tilelang_20260119 python -m pytest tests/unit/sensing -q
conda run -n env_tilelang_20260119 python -m pytest tests/gpu/test_optical_warp_executor.py -q
conda run -n env_tilelang_20260119 python -m pytest tests/gpu/test_optical_gpu_runtime.py -q
```

Result:

```text
180 passed
40 passed
5 passed
35 passed
```

## Boundaries

- No CLI physics command was added.
- Plain `run_scenario(...)` does not construct a physics engine.
- `torch_async` remains rejected for physics runtime video until
  provider-backed warmup for physics has a real warmup frame source.
- The future user-facing runtime still needs a decision about who owns engine
  construction: CLI, sensor-loop module, or RL/runtime wrapper.

## 关键思考

### 非显而易见的技术决策

1. Why not let `run_scenario(...)` construct a physics engine?

   That would make the lab runner implicitly own physics runtime lifecycle:
   engine construction, publish policy, base-frame selection, and per-frame time
   advancement. That ownership belongs to physics/simulation for now. The P6
   helper therefore requires `engine`, `registry`, `base_frame`, and
   `published_frame_for_index(...)` as explicit inputs. The runner assembles
   render/video/delivery around a selected frame; it does not decide how physics
   time advances.

   Alternative considered: route `FrameSourceKind.PHYSICS_RUNTIME` directly
   through `run_scenario(...)`. Rejected for this slice because the existing CLI
   scenario vocabulary has no way to describe or construct a physics engine.

2. Why reject `torch_async` on the physics runtime video path?

   The async delivery ring needs a warmup render result. For physics-backed
   frames, warmup must also go through a valid provider lifecycle and a valid
   published frame. P3 added provider-backed warmup mechanics, but the physics
   runtime path still needs a real warmup frame source policy. Allowing
   `torch_async` now would risk bypassing the physics borrow/complete lifecycle
   during warmup.

   Alternative considered: use frame index `0` as warmup implicitly. Rejected
   here because the caller owns physics time and may not want frame `0` consumed
   before the main loop.

3. Why keep the result video-focused rather than promote to
   `SimulationFrameRuntime`?

   P6 still only proves the video product path. RL observation products,
   multi-consumer dispatch, and engine reset/action semantics are not defined
   yet. Keeping the helper explicit and narrow avoids locking in a too-large
   runtime interface before those contracts exist.

### 调试困难与诊断

1. `validate_implemented()` guard ordering was easy to get wrong.

   `_is_implemented_physics_runtime_smoke()` must run before the generic
   `frame_source is PHYSICS_RUNTIME` reserved-mode guard. If the generic guard
   runs first, the new preset fails before it can be recognized as the one
   allowed physics smoke shape.

2. The P5 metadata gap shaped the P6 test.

   P5 proved provider/runtime/delivery ordering but constructed timing defaults
   directly in the test. P6 therefore needed a runner-level smoke that writes
   `frame_timing.csv` through `frame_defaults_for_config(...)` and asserts
   `frame_source == "physics_runtime"` in the actual CSV output.
