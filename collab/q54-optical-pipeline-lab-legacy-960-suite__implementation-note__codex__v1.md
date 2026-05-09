Initiative: q54-optical-pipeline-lab-legacy-960-suite
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-05-09
Status: implemented
Related Files: tools/optical_pipeline_lab/matrix.py, tools/optical_pipeline_lab/__main__.py, tests/unit/optics/test_optical_pipeline_lab.py, MANIFEST.md, collab/q54-optical-pipeline-lab-c2-matrix__implementation-note__codex__v1.md
Owner Summary: Added a named 960x640 matrix suite for direct GPU1 comparison against the older VIDEO_ORDERED_EXPORT measurements in GPU_OPTICAL_PIPELINE_DESIGN.md, while preserving 1080p as the default render rule.

# Q54 Optical Pipeline Lab Legacy 960 Suite

## Why

The new lab default resolution is 1080p:

```text
1920x1080
```

The older plan measurements were recorded at:

```text
GPU1 / Go2 static / 960x640
```

Those two runs are useful together, but only the 960x640 suite is directly
comparable to the historical latency table.

## What Landed

New suite:

```bash
python -m tools.optical_pipeline_lab matrix --suite go2_video_ordered_legacy_960
```

Default cases:

```text
legacy_960x640_shadow_readback_none
legacy_960x640_no_shadow_readback_none
legacy_960x640_shadow_readback_rgb
```

Optional debug row:

```bash
python -m tools.optical_pipeline_lab matrix \
  --suite go2_video_ordered_legacy_960 \
  --include-full-debug
```

adds:

```text
legacy_960x640_shadow_readback_full
```

## Expected GPU1 Commands

Historical comparison:

```bash
conda run -n env_tilelang_20260119 python -m tools.optical_pipeline_lab matrix \
  --suite go2_video_ordered_legacy_960 \
  --out out/optical_pipeline_lab/go2_video_ordered_legacy_960_gpu1 \
  --device cuda:1 \
  --frames 10 \
  --progress-every 1
```

Current 1080p baseline:

```bash
conda run -n env_tilelang_20260119 python -m tools.optical_pipeline_lab matrix \
  --suite go2_video_ordered_baseline \
  --out out/optical_pipeline_lab/go2_video_ordered_baseline_1080p_gpu1 \
  --device cuda:1 \
  --frames 10 \
  --progress-every 1
```

`--include-full-debug` is intentionally omitted from the default commands
because full readback is a heavier diagnostic path.

## Validation

Ran:

```bash
ruff check tools/optical_pipeline_lab examples/mujoco_menagerie_gpu_preview.py tests/unit/optics/test_optical_pipeline_lab.py
python -m py_compile tools/optical_pipeline_lab/*.py tests/unit/optics/test_optical_pipeline_lab.py examples/mujoco_menagerie_gpu_preview.py
python -m pytest tests/unit/optics/test_optical_pipeline_lab.py -q
python -m pytest --collect-only -q tests/unit/optics tests/unit/sensing tests/gpu/test_optical_warp_executor.py tests/gpu/test_optical_gpu_runtime.py
python -m tools.optical_pipeline_lab matrix --suite go2_video_ordered_legacy_960 --out /tmp/should_not_run_legacy_matrix --frames -1
```

Results:

```text
ruff: passed
py_compile: passed
lab tests: 22 passed
Q54 collect-only: 156 tests collected
invalid legacy matrix CLI: clean argparse error
```
