Initiative: q54-optical-pipeline-lab-c2-review-followup
Stage: review-followup
Author: codex
Version: v1
Date: 2026-05-09
Status: implemented
Related Files: tools/optical_pipeline_lab/matrix.py, tools/optical_pipeline_lab/runner.py, tools/optical_pipeline_lab/__main__.py, GPU_OPTICAL_PIPELINE_DESIGN.md, tests/unit/optics/test_optical_pipeline_lab.py
Owner Summary: Applied Claude C2 review follow-ups: documented the encode/write schema fallback, confirmed fail_on_overflow inference coverage, updated the durable design doc with new GPU1 960x640 measurements, and changed lab default warmup to five render passes after confirming it removes the large 1080p RGB readback spikes.

# Q54 Optical Pipeline Lab C2 Review Follow-Up

## Code Follow-Ups

Added a comment in `matrix.py` explaining why matrix summary falls back from:

```text
encode_write_ms
```

to:

```text
encode_or_write_ms
```

The latter is retained for older frame CSV compatibility.

Confirmed `test_matrix_suite_runs_cases_and_writes_summary` already verifies:

```text
readback_payload=none -> fail_on_overflow=False
readback_payload=rgb  -> fail_on_overflow=True
```

## Design Doc Updates

Updated `GPU_OPTICAL_PIPELINE_DESIGN.md` with the C2 GPU1 960x640 suite:

```text
legacy_960x640_shadow_readback_none:
  frame_p50 ~= 2.75 ms
  render_mean ~= 2.58 ms

legacy_960x640_no_shadow_readback_none:
  frame_p50 ~= 1.31 ms
  render_mean ~= 1.23 ms

legacy_960x640_shadow_readback_rgb:
  frame_p50 ~= 6.64 ms
  readback_mean ~= 4.01 ms
  render_mean ~= 2.61 ms
```

The old `7.77 ms / 5.07 ms` RGB readback record is no longer the canonical
baseline.

## Warmup 5 Check

Ran:

```bash
conda run -n env_tilelang_20260119 python -m tools.optical_pipeline_lab run \
  --preset go2_video_ordered_static \
  --out out/optical_pipeline_lab/go2_1080p_shadow_rgb_warmup5_gpu1 \
  --device cuda:1 \
  --width 1920 \
  --height 1080 \
  --frames 10 \
  --warmup-renders 5 \
  --progress-every 1 \
  --readback rgb
```

Result:

```text
all10 frame_total_mean_ms:    24.421
all10 readback_host_mean_ms:  18.735
all10 render_execute_mean_ms: 5.558

last6 frame_total_mean_ms:    24.813
last6 readback_host_mean_ms:  19.104
last6 render_execute_mean_ms: 5.586

last3 frame_total_mean_ms:    17.646
last3 readback_host_mean_ms:  12.257
last3 render_execute_mean_ms: 5.267
```

The previous 1080p matrix run with two warmup renders had first-frame readback
spikes of 156/198/315/180 ms. With five warmup renders, those spikes disappear.

## Default Change

Changed the Optical Pipeline Lab default warmup to:

```text
DEFAULT_LAB_WARMUP_RENDERS = 5
```

This applies to the lab `run` and `matrix` commands. The lower-level Go2 backend
CLI keeps its own explicit default, so example behavior remains opt-in.

## D1 Implication

D1 async D2H should use `warmup_renders >= 5` for baseline and spike runs, and
should report tail metrics as well as all-frame metrics. Otherwise allocation
or context warmup can masquerade as delivery latency.

## Validation

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
lab tests: 22 passed
Q54 collect-only: 156 tests collected
```
