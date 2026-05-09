Initiative: q54-optical-pipeline-lab-c2-matrix
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-05-09
Status: implemented-c2-matrix
Related Files: tools/optical_pipeline_lab/matrix.py, tools/optical_pipeline_lab/__main__.py, tools/optical_pipeline_lab/timing.py, tools/optical_pipeline_lab/go2_backend.py, tests/unit/optics/test_optical_pipeline_lab.py, MANIFEST.md
Owner Summary: Implemented the C2 matrix baseline runner. Added a serial `matrix` CLI, a `go2_video_ordered_baseline` suite, a `go2_video_ordered_legacy_960` historical-comparison suite, per-case output directories, `suite_config.json`, `matrix_summary.csv`, fake-runner tests for success/failure aggregation, and removed legacy `refit_ms/rebuild_ms` frame columns now that the C1 Go2 backend extraction is complete.

# Q54 Optical Pipeline Lab C2 Matrix Implementation Note

## What Landed

Added:

```text
tools/optical_pipeline_lab/matrix.py
```

New CLI:

```bash
python -m tools.optical_pipeline_lab matrix --suite go2_video_ordered_baseline
```

Default output:

```text
out/optical_pipeline_lab/go2_video_ordered_baseline/
  suite_config.json
  matrix_summary.csv
  smoke_160x120_shadow_readback_none/
  1080p_shadow_readback_none/
  1080p_no_shadow_readback_none/
  1080p_shadow_readback_rgb/
```

Optional debug full-readback row:

```bash
python -m tools.optical_pipeline_lab matrix \
  --suite go2_video_ordered_baseline \
  --include-full-debug
```

Historical 960x640 comparison suite:

```bash
python -m tools.optical_pipeline_lab matrix --suite go2_video_ordered_legacy_960
```

## Baseline Suite

Default suite cases:

```text
smoke_160x120_shadow_readback_none
1080p_shadow_readback_none
1080p_no_shadow_readback_none
1080p_shadow_readback_rgb
```

`--include-full-debug` adds:

```text
1080p_shadow_readback_full
```

The 1080p cases use:

```text
DEFAULT_RENDER_WIDTH=1920
DEFAULT_RENDER_HEIGHT=1080
```

## Legacy 960 Suite

This suite exists for direct comparison against the older
VIDEO_ORDERED_EXPORT measurements recorded in `GPU_OPTICAL_PIPELINE_DESIGN.md`.
Those records used GPU1, Go2 static, and 960x640 resolution, so this suite keeps
the same resolution instead of inheriting the new 1080p default.

Default suite cases:

```text
legacy_960x640_shadow_readback_none
legacy_960x640_no_shadow_readback_none
legacy_960x640_shadow_readback_rgb
```

`--include-full-debug` adds:

```text
legacy_960x640_shadow_readback_full
```

## Matrix Summary

`matrix_summary.csv` records one row per case:

```text
case_name
status
error
output_dir
width / height
shadows
readback_payload
write_policy
frames
fps_mean
frame_p50_ms
frame_p90_ms
render_execute_mean_ms
readback_host_mean_ms
image_build_mean_ms
encode_write_mean_ms
```

Failed cases are retained with:

```text
status=failed
error=<exception text>
```

and the matrix continues by default. `--stop-on-failure` is available for
fail-fast runs.

## Validation Rules

Matrix cases reuse existing `run_scenario(...)` validation.

Readback-none cases automatically set:

```text
fail_on_overflow=False
```

because no diagnostics are staged.

## Legacy Column Cleanup

Now that C1 extracted the Go2 backend, the legacy frame columns were removed:

```text
refit_ms
rebuild_ms
```

The current schema keeps:

```text
accel_refit_ms
accel_rebuild_ms
```

and `go2_backend.py` writes those names.

## Tests

Extended:

```text
tests/unit/optics/test_optical_pipeline_lab.py
```

New coverage:

```text
baseline suite case order and 1080p defaults
legacy 960 suite case order and historical comparison dimensions
optional full-debug case
matrix success aggregation from fake frame_timing.csv
readback=none disables fail_on_overflow
failed case is recorded and matrix continues
CLI matrix dispatch
```

No GPU is required for these tests.

## Validation

Ran:

```bash
ruff check tools/optical_pipeline_lab examples/mujoco_menagerie_gpu_preview.py tests/unit/optics/test_optical_pipeline_lab.py
python -m py_compile tools/optical_pipeline_lab/*.py tests/unit/optics/test_optical_pipeline_lab.py examples/mujoco_menagerie_gpu_preview.py
python -m pytest tests/unit/optics/test_optical_pipeline_lab.py -q
python -m pytest --collect-only -q tests/unit/optics tests/unit/sensing tests/gpu/test_optical_warp_executor.py tests/gpu/test_optical_gpu_runtime.py
python -m tools.optical_pipeline_lab matrix --suite go2_video_ordered_baseline --out /tmp/should_not_run_matrix --frames -1
```

Results:

```text
ruff: passed
py_compile: passed
lab tests: 22 passed
Q54 collect-only: 156 tests collected
invalid matrix CLI: clean argparse error
```

## Next

C2 gives D1/D2 a stable comparison surface.

Recommended next steps:

```text
run the real legacy 960 matrix on cuda:1 in env_tilelang_20260119
run the real 1080p baseline matrix on cuda:1 in env_tilelang_20260119
then begin D1 async D2H spike with matrix_summary.csv as the comparison target
```
