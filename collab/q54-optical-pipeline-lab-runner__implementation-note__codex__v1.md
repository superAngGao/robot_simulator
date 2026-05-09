Initiative: q54-optical-pipeline-lab-runner
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-05-09
Status: implemented-c0-thin-runner
Related Files: tools/optical_pipeline_lab/__main__.py, tools/optical_pipeline_lab/runner.py, tests/unit/optics/test_optical_pipeline_lab.py
Owner Summary: Added the first Stage C0 `run` entry point for the Optical Pipeline Lab. The runner validates a preset, writes `scenario_config.json`, maps lab scenario semantics to the existing Menagerie GPU example arguments, and delegates to the mature static Go2 video path without introducing a RenderSession or changing render kernels.

# Q54 Optical Pipeline Lab Runner Implementation Note

## What Landed

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
  frames/ only when --write-frames is requested
```

The runner is intentionally transitional:

```text
lab config / output metadata / validation
  -> existing examples.mujoco_menagerie_gpu_preview._render_many_views(...)
```

No render kernel, BVH builder, readback helper, output profile, or async
delivery logic was changed.

## New Runner Pieces

`runner.py` now provides:

```text
LabRunOptions
apply_run_overrides(...)
run_scenario(...)
build_menagerie_example_args(...)
write_scenario_config(...)
validate_run(...)
```

`apply_run_overrides(...)` lets CLI flags override only the intended run fields:

```text
device
width / height
readback payload
shadows
write frames
```

`build_menagerie_example_args(...)` maps:

```text
AccelBackend.CUDA_LBVH -> --bvh-backend cuda_lbvh
ReadbackPayload.RGB   -> --video-readback rgb
WritePolicy.NONE      -> --no-write-frames behavior
```

and reserves:

```text
timing.csv
frame_timing.csv
```

under the lab output directory.

## Validation

The lab now rejects unsatisfiable combinations before importing/running the GPU
example:

```text
readback_payload=none + write_policy=png_sequence
readback_payload=none + fail_on_overflow
video_raygen=gpu + video_ray_cache!=off
negative frame/progress counts
non-positive fps
```

CLI errors are reported through argparse instead of Python tracebacks.

Example:

```bash
python -m tools.optical_pipeline_lab run \
  --preset go2_video_ordered_static \
  --readback none
```

prints a clean error unless paired with:

```text
--no-fail-on-overflow
```

## Tests

Extended:

```text
tests/unit/optics/test_optical_pipeline_lab.py
```

New coverage:

```text
run overrides do not mutate preset source
lab config maps to Menagerie example args
lab frame defaults populate scenario metadata columns
scenario_config.json serializes enum/path values
bad readback/write/fail combinations are rejected
```

## Validation Commands

Ran:

```bash
ruff check tools/optical_pipeline_lab examples/mujoco_menagerie_gpu_preview.py tests/unit/optics/test_optical_pipeline_lab.py
python -m py_compile tools/optical_pipeline_lab/*.py tests/unit/optics/test_optical_pipeline_lab.py examples/mujoco_menagerie_gpu_preview.py
python -m pytest tests/unit/optics/test_optical_pipeline_lab.py -q
python -m tools.optical_pipeline_lab describe --preset go2_video_ordered_static
python -m tools.optical_pipeline_lab run --preset go2_video_ordered_static --readback none --out /tmp/optical_lab_should_not_run
conda run -n env_tilelang_20260119 python -m tools.optical_pipeline_lab run \
  --preset go2_video_ordered_static \
  --width 160 --height 120 --frames 1 \
  --readback none --no-fail-on-overflow \
  --out out/optical_pipeline_lab/smoke_160_readback_none \
  --progress-every 1
```

Results:

```text
ruff: passed
py_compile: passed
pytest: 10 passed
describe CLI: printed preset
invalid run CLI: clean argparse error
160x120 GPU smoke: passed in env_tilelang_20260119
  render_execute_ms ~= 1.30 ms
  frame_total_ms ~= 1.42 ms
```

The default Python environment does not have Warp/CUDA available. The direct
runner path now reports the same clean "requires warp with CUDA support" message
as the original example entry point instead of an AttributeError.

## Next Natural Step

Run the actual GPU baseline matrix through the new lab command:

```text
160x120 smoke
960x640 shadow readback=none with --no-fail-on-overflow
960x640 no-shadow readback=none with --no-fail-on-overflow
960x640 shadow readback=rgb
optional shadow readback=full debug row
```

After those artifacts exist, the next implementation slice should add a matrix
subcommand and/or a small baseline report writer, still before async D2H or
RGB8 work.
