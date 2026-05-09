Initiative: q54-optical-pipeline-lab-c0-runner
Stage: review-request
Author: codex
Reviewer: claude
Version: v1
Date: 2026-05-09
Status: ready-for-review
Related Files: tools/optical_pipeline_lab/, examples/mujoco_menagerie_gpu_preview.py, tests/unit/optics/test_optical_pipeline_lab.py, MANIFEST.md, collab/q54-optical-pipeline-lab-foundation__implementation-note__codex__v1.md, collab/q54-optical-pipeline-lab-runner__implementation-note__codex__v1.md, collab/q54-optical-pipeline-lab-review-followup__codex__v1.md
Owner Summary: Requesting review for the Stage B/C0 Optical Pipeline Lab work. The lab now has scenario vocabulary, preset metadata, stable timing CSV helpers, a thin `run` CLI for the Go2 static video scenario, mock-covered runner/CLI/report paths, MANIFEST updates, and a successful 160x120 GPU smoke through the new lab command.

# Review Request: Q54 Optical Pipeline Lab C0 Runner

## 1. Context

The durable architecture baseline is:

```text
GPU_OPTICAL_PIPELINE_DESIGN.md
```

It says the next implementation slice should be:

```text
Stage B: Optical Pipeline Lab foundation
Stage C: migrate current VIDEO_ORDERED_EXPORT work into the lab
```

and should not start with:

```text
async ordered scheduler
RGB8 CUDA pack
public OpticalConsumerMode API
dynamic geometry scheduling
OptiX/CUDA fused renderer
```

This review request covers the first implemented slice of that direction:

```text
B0/B1 foundation
C0 thin Go2 runner
review feedback follow-up
```

## 2. What Landed

### 2.1 New Lab Package

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

The package is developer tooling for scenario config, timing schema, presets,
reports, and benchmark/lab execution. It is not the production optical runtime.

### 2.2 Scenario Vocabulary

`scenarios.py` defines:

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

Reserved modes fail loudly through:

```python
config.validate_implemented()
```

Currently allowed:

```text
geometry_mode=static
accel_policy=build_once
render_backend=warp_bvh_direct_light
delivery_policy=sync or device_only
readback_payload=none/rgb/full
write_policy=none/png_sequence
```

### 2.3 First Preset

Added:

```text
go2_video_ordered_static
```

It describes:

```text
VIDEO_ORDERED_EXPORT
Go2 Menagerie static scene
cuda_lbvh build_once
warp_bvh_direct_light
rgb_preview default
sync delivery for now
```

### 2.4 Timing Helpers

Moved reusable timing pieces out of:

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

No render kernels, BVH logic, readback helper behavior, or output profile logic
was intentionally changed.

### 2.5 C0 Thin Runner

Added:

```bash
python -m tools.optical_pipeline_lab run --preset go2_video_ordered_static
```

Default output:

```text
out/optical_pipeline_lab/go2_video_ordered_static/
  scenario_config.json
  timing.csv
  frame_timing.csv
```

For now, the runner is intentionally transitional:

```text
lab config / validation / output metadata
  -> existing examples.mujoco_menagerie_gpu_preview._render_many_views(...)
```

This keeps the mature Go2 GPU path in one place while letting the lab own
scenario metadata and CSV/report layout.

The runner validates combinations before GPU work starts, including:

```text
readback_payload=none + write_policy=png_sequence
readback_payload=none + fail_on_overflow
video_raygen=gpu + video_ray_cache!=off
negative frame/progress counts
non-positive fps
```

## 3. Claude Review Follow-Up Already Applied

Claude's previous B0/B1 review marked this as light NEEDS WORK and asked for:

```text
runner.py / reports.py / __main__.py smoke tests
FRAME_TIMING_FIELDNAMES clarification for refit_ms vs accel_refit_ms
percentile doc/contract cleanup
MANIFEST.md update
```

Applied:

```text
run_scenario smoke test with fake Menagerie module
reports format_summary_rows test
__main__.py describe/run dispatch tests
percentile(samples, q) now sorts internally and accepts unsorted input
timing.py comments mark refit_ms/rebuild_ms as legacy transitional columns
MANIFEST.md includes tools/optical_pipeline_lab/ and current Q54 test count
```

The lab test file now has:

```text
14 tests
```

## 4. Validation Performed

### 4.1 Static / Unit Validation

Ran:

```bash
ruff check tools/optical_pipeline_lab examples/mujoco_menagerie_gpu_preview.py tests/unit/optics/test_optical_pipeline_lab.py
python -m py_compile tools/optical_pipeline_lab/*.py tests/unit/optics/test_optical_pipeline_lab.py examples/mujoco_menagerie_gpu_preview.py
python -m pytest tests/unit/optics/test_optical_pipeline_lab.py -q
python -m pytest --collect-only -q tests/unit/optics tests/unit/sensing tests/gpu/test_optical_warp_executor.py tests/gpu/test_optical_gpu_runtime.py
```

Results:

```text
ruff: passed
py_compile: passed
lab tests: 14 passed
Q54 collect-only: 148 tests collected
```

Current split:

```text
116 CPU optics/sensing/lab
32 GPU optical
148 total
```

### 4.2 Real GPU Smoke

Default Python environment lacks Warp/CUDA. The direct runner path now reports
the same clean requirement message as the original example instead of an
AttributeError.

In the GPU environment, this passed:

```bash
conda run -n env_tilelang_20260119 python -m tools.optical_pipeline_lab run \
  --preset go2_video_ordered_static \
  --width 160 --height 120 --frames 1 \
  --readback none --no-fail-on-overflow \
  --out out/optical_pipeline_lab/smoke_160_readback_none \
  --progress-every 1
```

Observed:

```text
render_execute_ms ~= 1.30 ms
frame_total_ms    ~= 1.42 ms
```

Artifacts were produced:

```text
scenario_config.json
timing.csv
frame_timing.csv
```

`frame_timing.csv` now includes lab scenario metadata columns populated from
the preset:

```text
scenario_name
device
width / height
scene_preset
camera_mode
accel_backend / accel_policy
render_backend
output_profile
readback_payload
delivery_policy
write_policy
```

## 5. Intentional Non-Goals

This implementation deliberately does not:

```text
create RenderSession / PipelineController
implement async ordered readback
implement RGB8 pack
introduce public consumer-mode API
change GPU kernels
change CUDA LBVH builder
change output profile semantics
change readback helper semantics
implement dynamic geometry presets
```

## 6. Known Transitional Choices

### 6.1 Runner Delegates To Example

The lab runner currently calls:

```python
examples.mujoco_menagerie_gpu_preview._render_many_views(...)
```

This is intentional for C0. It avoids copying the mature Go2 setup/render path
while moving scenario metadata and output layout into the lab.

Question for review:

```text
Is this acceptable as a C0 bridge, or should the render path be extracted into
a non-example helper before adding matrix runs?
```

### 6.2 Legacy Timing Columns

The frame CSV currently contains both:

```text
accel_refit_ms / accel_rebuild_ms
refit_ms / rebuild_ms
```

The code now comments that:

```text
refit_ms / rebuild_ms are legacy transitional columns from the Menagerie example
new lab producers should write accel_refit_ms / accel_rebuild_ms
```

Question for review:

```text
Is this compatibility period acceptable, or should we immediately remove the
legacy columns and update the example row names now?
```

### 6.3 CLI Error Boundary

The CLI catches:

```text
ValueError
NotImplementedError
```

and turns them into argparse errors. Runtime failures from the delegated GPU
example still propagate normally.

Question for review:

```text
Is this the right CLI boundary for lab validation vs execution failures?
```

## 7. Requested Review Questions

1. Is the package boundary right?

```text
tools/optical_pipeline_lab owns scenario/timing/preset/report/runner tooling
optics remains the computation/runtime package
```

2. Is the C0 thin runner acceptable, or should we extract a shared Go2 runner
helper before adding matrix/baseline commands?

3. Are the current tests enough for this slice?

```text
timing/scenarios/presets
runner translation and fake-delegation smoke
reports formatting
CLI describe/run dispatch
bad config validation
```

4. Is the frame CSV schema direction acceptable, including reserved async
columns and temporary legacy refit/rebuild aliases?

5. Should the next implementation be:

```text
matrix subcommand / baseline suite runner
```

before:

```text
async D2H spike
RGB8 pack
RenderSession skeleton
```

Recommended answer from Codex:

```text
yes
```

