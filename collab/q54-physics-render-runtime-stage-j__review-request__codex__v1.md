# Q54 Physics Render Runtime Stage J — Review Request

Author: Codex
Date: 2026-05-18
Status: local commits only, not pushed

## Context

This is the next Stage J slice after the Optical Pipeline Lab source/session
cleanup and the first physics-published GPU frame smoke tests.

The design intent is still:

- Physics/simulator systems provide `GpuPublishedFrame`.
- Static asset builders only build non-simulated optical registries/assets.
- External renderer/backend integration is the place for "adapter" terminology.
- The lab render pipeline stays generic and consumes a source/options/session
  boundary.

This series does not introduce a production runner loop yet. It adds a small
physics-backed runtime boundary that can be reused by future runner/sensor work.

## Commits To Review

```text
11925d0 Add physics render frame lease helper
dac4017 Add physics render pipeline factory
1bbaefa Add physics render consumer registration helper
530ee23 Add physics render runtime bundle
6ee6f4b Add lab frame source scenario vocabulary
```

Files touched:

```text
tools/optical_pipeline_lab/physics_source.py
tools/optical_pipeline_lab/scenarios.py
tools/optical_pipeline_lab/presets.py
tools/optical_pipeline_lab/runner.py
tools/optical_pipeline_lab/__init__.py
tests/unit/optics/test_optical_pipeline_lab.py
tests/gpu/test_optical_gpu_runtime.py
MANIFEST.md
```

## What Changed

### 1. Physics frame lease helper

`PhysicsLabFrameLease` and `begin_physics_render_frame(...)` now wrap:

```text
engine.borrow_device_frame(...)
  -> pipeline.begin_frame(frame_inputs=borrowed_frame)
  -> engine.complete_device_consumer(...)
```

Important lifecycle behavior:

- `complete()` is idempotent.
- `__exit__` calls `complete()`.
- `done_event` is populated only after `complete()` runs.
- If `pipeline.begin_frame(...)` raises, including `BaseException`, the helper
  completes the borrowed device consumer before re-raising.

The two GPU physics smoke tests now use this lease path.

### 2. Physics render pipeline factory

`create_physics_render_pipeline(...)` wraps the fixed physics source construction:

```text
build_physics_render_source(...)
scene_from_physics_render_source(...)
OpticalLabRenderPipeline.create_from_source_factory(...)
```

`scene=None` remains safe: `build_physics_render_source(...)` creates a
`PhysicsLabRenderScene` and stores it in source metadata.

### 3. Physics render consumer registration helper

`physics_render_consumer(...)` builds the default device-borrow consumer:

```text
consumer_kind="render_backed_sensing"
qos_mode="lossless"
access_mode="borrow"
consumer_location="device"
```

`register_physics_render_consumer(...)` registers either a provided
`ConsumerState` or a default one.

### 4. Physics render runtime bundle

`PhysicsLabRenderRuntime` holds:

```text
engine
pipeline
consumer
```

and exposes:

```text
runtime.begin_frame(published_frame=..., env_idx=...)
```

`create_physics_render_runtime(...)` composes pipeline creation and consumer
registration. It does not own the engine, mutate publish policy, or run a step
loop.

The GPU physics smoke tests now use:

```text
runtime = create_physics_render_runtime(...)
with runtime.begin_frame(published_frame=latest_frame) as lease:
    ...
```

### 5. Scenario vocabulary: `frame_source`

`FrameSourceKind` was added to separate "where frames come from" from
`scene_preset`:

```text
static_asset_builder
synthetic_frame_sequence
physics_runtime
```

Current presets:

- Go2 Menagerie static -> `static_asset_builder`
- synthetic dynamic smoke -> `synthetic_frame_sequence`

`physics_runtime` is now vocabulary only. Validation fails loudly until the lab
runner owns a real physics engine loop.

There is also a guard so `synthetic_frame_sequence` is implemented only by the
existing `synthetic_body_triangle_dynamic_smoke` preset.

## Tests / Verification

Focused and collection checks run locally:

```text
conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics/test_optical_pipeline_lab.py -q
# 83 passed

CUDA_VISIBLE_DEVICES=0 conda run -n env_tilelang_20260119 python -m pytest \
  tests/gpu/test_optical_gpu_runtime.py -q -k "optical_lab_physics_published_frame"
# 2 passed, 31 deselected

conda run -n env_tilelang_20260119 python -m pytest --collect-only -q \
  tests/unit/optics tests/unit/sensing \
  tests/gpu/test_optical_warp_executor.py tests/gpu/test_optical_gpu_runtime.py
# 234 tests collected

conda run -n env_tilelang_20260119 python -m ruff check ...
# clean

python -m py_compile ...
# clean

git diff --check
# clean
```

Pre-commit hooks on each local commit:

```text
ruff passed
ruff format passed
```

## Review Questions

1. Is `PhysicsLabFrameLease` the right ownership boundary for
   borrow/begin/complete, especially the `BaseException` cleanup path?

2. Is `PhysicsLabRenderRuntime` appropriately small, or is this premature
   wrapping before runner integration?

3. Are `physics_render_consumer(...)` and
   `register_physics_render_consumer(...)` correctly located in
   `physics_source.py`, or should consumer registration live in a future runner
   module?

4. Does `FrameSourceKind` correctly separate frame source from `scene_preset`,
   and are the names right?

5. Any concern that `frame_source=physics_runtime` is in the vocabulary before
   the runner can execute it, even though validation reserves it?

6. Any missing unit/GPU coverage before we push this local series?

