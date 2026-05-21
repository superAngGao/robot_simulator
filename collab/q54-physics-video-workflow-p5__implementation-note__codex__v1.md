# Q54 Physics Video Workflow P5 Implementation Note

Author: Codex
Date: 2026-05-21
Status: implemented locally, not pushed

## Summary

Added the P5 test-only physics video workflow smoke.

This slice proves the architecture path before enabling any CLI or
`run_scenario(...)` physics runtime entry:

```text
GpuEngine.step()
-> create_physics_render_runtime_for_config(...)
-> PhysicsFrameContextProvider
-> FrameWorkflowRunner
-> build_video_render_plan(...)
-> render_video_frame_from_context(...)
-> VideoDeliveryFacade
-> FrameTimingRecorder / frame_timing.csv
```

`FrameSourceKind.PHYSICS_RUNTIME` remains guarded for the main runner.

## Code Changes

- Added `test_optical_lab_physics_video_workflow_uses_provider_runtime_delivery`
  in `tests/gpu/test_optical_gpu_runtime.py`.
- The test drives two real physics heights through `GpuEngine.step(...)`.
- Each published GPU frame is passed through
  `PhysicsFrameContextProvider.begin_frame(..., published_frame=...)`.
- The video consumer builds `FrameIdentity` from the provider-owned
  `OpticalLabRenderFrameContext`, not from the base scene.
- The frame is rendered through `render_video_frame_from_context(...)`.
- `FrameWorkflowRunner` exits the provider lifecycle before submitting delivery.
- Sync full readback verifies the body-bound triangle range for both frames.
- `FrameTimingRecorder` verifies dynamic `snapshot_ms` and `accel_refit_ms`
  appear in `frame_timing.csv`.
- Added `frame_source` to the stable timing CSV schema. The full GPU runtime
  test file exposed that scenario defaults already write this field, but the
  fieldnames table had not caught up.

## Boundaries Preserved

- No CLI path was enabled.
- No `run_scenario(...)` physics runtime route was added.
- No relaxation of `FrameSourceKind.PHYSICS_RUNTIME` guards.
- No new public `SimulationFrameRuntime` API.
- Static asset builder and Go2/Menagerie static video paths remain unchanged.

## Verification

Focused P5 smoke:

```bash
conda run -n env_tilelang_20260119 \
  python -m pytest tests/gpu/test_optical_gpu_runtime.py -q \
  -k "physics_video_workflow"
```

Result:

```text
1 passed, 33 deselected
```

Adjacent physics/video GPU smokes:

```bash
conda run -n env_tilelang_20260119 \
  python -m pytest tests/gpu/test_optical_gpu_runtime.py -q \
  -k "physics_published_frame or physics_video_workflow or dynamic_video_loop_writes_prepare_timing_csv"
```

Result:

```text
4 passed, 30 deselected
```

Full GPU runtime smoke file after the timing schema fix:

```bash
conda run -n env_tilelang_20260119 \
  python -m pytest tests/gpu/test_optical_gpu_runtime.py -q
```

Result:

```text
34 passed
```

Focused unit coverage around provider/runtime/frame-source/timing:

```bash
conda run -n env_tilelang_20260119 \
  python -m pytest tests/unit/optics/test_optical_pipeline_lab.py -q \
  -k "frame_source or frame_timing or provider_backed or frame_workflow_runner"
```

Result:

```text
13 passed, 89 deselected
```

## Review Notes

- The test uses `video_readback="full"` intentionally so the delivered frame
  exercises real sync host delivery and the staged result still exposes
  `hit_mask`, `range_m`, and `numeric_instance_id`.
- `frame_source` is now a first-class `FrameTimingRecorder` field because
  scenario defaults already populate it.
- P6 can now focus on the smallest guarded runner integration for
  `FrameSourceKind.PHYSICS_RUNTIME`.
