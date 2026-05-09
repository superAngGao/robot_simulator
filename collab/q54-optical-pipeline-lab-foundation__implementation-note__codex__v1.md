Initiative: q54-optical-pipeline-lab-foundation
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-05-09
Status: implemented-b0-b1
Related Files: tools/optical_pipeline_lab/, examples/mujoco_menagerie_gpu_preview.py, tests/unit/optics/test_optical_pipeline_lab.py
Owner Summary: Stage B0/B1 foundation is in place. Added the first `tools.optical_pipeline_lab` package with scenario vocabulary, Go2 static video preset metadata, stable frame timing schema, timing/report helpers, and a small `describe` CLI. The Menagerie GPU example now reuses the lab timing recorder instead of carrying local timing classes.

# Q54 Optical Pipeline Lab Foundation Implementation Note

## What Landed

Added:

```text
tools/
tools/optical_pipeline_lab/
  __init__.py
  __main__.py
  scenarios.py
  presets.py
  timing.py
  reports.py
  runner.py
```

The package is intentionally a lab/tooling layer, not the production optical
runtime.

## Scenario Vocabulary

`scenarios.py` defines the first structured lab config:

```text
OpticalLabScenarioConfig
OpticalLabScenarioFamily
GeometryMode
AccelBackend
AccelPolicy
RenderBackend
ReadbackPayload
DeliveryPolicy
WritePolicy
```

Reserved future modes fail loudly through:

```python
config.validate_implemented()
```

Current implemented lab modes are deliberately narrow:

```text
geometry_mode=static
accel_policy=build_once
render_backend=warp_bvh_direct_light
delivery_policy=sync or device_only
readback_payload=none/rgb/full
write_policy=none/png_sequence
```

## Preset

Added first preset:

```text
go2_video_ordered_static
```

It describes the current Stage C target:

```text
VIDEO_ORDERED_EXPORT
go2_menagerie_static
static geometry
cuda_lbvh build_once
warp_bvh_direct_light
rgb_preview
readback_payload=rgb
delivery_policy=sync
```

The new CLI can describe it:

```bash
python -m tools.optical_pipeline_lab describe --preset go2_video_ordered_static
```

## Timing Extraction

Moved the reusable timing pieces out of:

```text
examples/mujoco_menagerie_gpu_preview.py
```

into:

```text
tools/optical_pipeline_lab/timing.py
```

The example now imports:

```text
TimingRecorder
FrameTimingRecorder
RENDER_PROFILE_PHASES
NAN
```

No render kernels, BVH logic, readback logic, output profiles, or example CLI
behavior were intentionally changed.

## Frame CSV Direction

`FrameTimingRecorder` now owns the lab-level frame schema. It keeps the current
example fields and reserves the Stage B/C fields from the design doc:

```text
scenario_name
device
width / height
scene_preset
accel_backend / accel_policy
render_backend
output_profile
readback_payload
delivery_policy
write_policy
render_overhead_ms
readback_submit_ms
readback_wait_ms
future async overlap/ring fields
```

Missing values are normalized to `NaN`.

## Tests

Added:

```text
tests/unit/optics/test_optical_pipeline_lab.py
```

Coverage:

```text
percentile interpolation
TimingRecorder summary CSV
FrameTimingRecorder schema normalization and video summary
Go2 static preset validation
reserved future modes fail loudly
```

## Validation

Ran:

```bash
ruff check tools/optical_pipeline_lab examples/mujoco_menagerie_gpu_preview.py tests/unit/optics/test_optical_pipeline_lab.py
python -m py_compile examples/mujoco_menagerie_gpu_preview.py tools/optical_pipeline_lab/*.py tests/unit/optics/test_optical_pipeline_lab.py
python -m pytest tests/unit/optics/test_optical_pipeline_lab.py -q
python -m tools.optical_pipeline_lab describe --preset go2_video_ordered_static
```

Results:

```text
ruff: passed
py_compile: passed
pytest: 5 passed
describe CLI: printed the Go2 static video preset
```

## Next Natural Step

Stage C0:

```text
add `python -m tools.optical_pipeline_lab run --preset go2_video_ordered_static`
```

The first runner can delegate to the existing mature Menagerie GPU example path
or share a thin helper, but should keep the lab responsible for:

```text
scenario config
output directory layout
timing CSV/report paths
baseline matrix orchestration
reserved-mode validation
```

Do not start async ordered delivery, RGB8 pack, public consumer API, dynamic
geometry scheduling, or CUDA/OptiX backend work until the lab can reproduce the
current Go2 static baselines.

