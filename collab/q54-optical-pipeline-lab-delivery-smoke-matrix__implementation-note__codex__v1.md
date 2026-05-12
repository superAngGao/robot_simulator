# Q54 Optical Pipeline Lab Delivery Smoke Matrix Implementation Note

Date: 2026-05-12
Author: Codex
Status: implemented

## Scope

Added a small reusable matrix suite for exercising the lab video delivery
facade across the main CSV/readback policy combinations:

```text
go2_video_delivery_smoke
```

This is a lab regression entry point. It does not change the production optics
API or introduce a new render session abstraction.

## Implementation

`tools/optical_pipeline_lab/matrix.py` now supports per-case delivery settings:

- `video_readback_delivery`
- `video_readback_ring_depth`

Those values flow from `MatrixCase` into `LabRunOptions` and are written to
`matrix_summary.csv` for analysis.

The new suite uses the static Go2 ordered video preset and includes:

- `readback=none`, sync delivery
- `readback=rgb`, sync delivery
- `readback=rgb8`, torch async delivery with `ring_depth=2`

`--include-full-debug` appends the existing heavy `readback=full` sync case.

## Validation

CPU/unit:

```text
python -m pytest -q tests/unit/optics/test_optical_pipeline_lab.py
57 passed
```

Lint:

```text
ruff check tools/optical_pipeline_lab \
  tests/unit/optics/test_optical_pipeline_lab.py \
  tests/gpu/test_optical_gpu_runtime.py
All checks passed!
```

CLI discovery:

```text
python -m tools.optical_pipeline_lab matrix --help
```

The help output lists `go2_video_delivery_smoke` in the accepted matrix suites.

## Follow-up

The unit CLI dispatch test now routes `--suite go2_video_delivery_smoke
--include-full-debug` through `tools.optical_pipeline_lab.__main__` and asserts
that the suite passed to `run_matrix_suite(...)` includes both the torch async
RGB8 case and the full debug case.
