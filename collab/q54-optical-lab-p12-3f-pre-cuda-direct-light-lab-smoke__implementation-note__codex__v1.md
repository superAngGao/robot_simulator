# Q54 P12.3f-pre Implementation Note: CUDA Direct-Light Lab Smoke

Author: Codex
Status: ready for review
Related design:
`collab/q54-optical-lab-p12-3f-pre-cuda-direct-light-lab-smoke__design-request__codex__v2.md`

## Owner Summary

This implements the narrow P12.3f-pre smoke slice. `run_scenario(...)` can now
execute an explicit no-shadow CUDA direct-light static/synthetic Lab config:

```text
scene_preset="synthetic_body_triangle"
render_backend="cuda_direct_light"
accel_backend="cuda_lbvh"
shadows=False
video_raygen="host"
video_readback_delivery="sync"
readback_payload="rgb"
```

The slice intentionally does not expose a public preset/backend override and
does not enable Go2, hard shadows, GPU camera raygen, rgb8, async delivery, or
matrix expansion.

## What Changed

### `tools/optical_pipeline_lab/runner.py`

- `validate_run_scenario_supported(...)` now allows
  `render_backend='cuda_direct_light'` only for
  `scene_preset='synthetic_body_triangle'` and `shadows=False`.
- Go2 CUDA direct-light remains rejected until the full P12.3f visual check.
- `shadows=True` is rejected at validation time with a P12.3d message.
- `build_menagerie_example_args(...)` now forwards `render_backend` through the
  existing static runner args object.

### `tools/optical_pipeline_lab/menagerie_static_runner.py`

- `_render_options_from_args(...)` now reads `args.render_backend`, defaulting
  to `warp_bvh_direct_light` for legacy CLI/default behavior.
- Warmup/render benchmark result synchronization is nil-safe for CUDA
  direct-light results that are already synchronized and carry
  `ready_event=None`.

### Tests

Unit tests cover:

- accepted synthetic CUDA no-shadow host-ray config;
- existing Warp and CPU direct-light `run_scenario(...)` support still passing;
- Go2 CUDA direct-light rejected for this pre-slice;
- `shadows=True` rejected until P12.3d;
- static runner args include `render_backend`.

GPU smoke covers:

- `run_scenario(...)` end-to-end with
  `cuda_direct_light + cuda_lbvh + synthetic_body_triangle + shadows=False`;
- one warmup render;
- RGB sync readback;
- PNG write through normal delivery path;
- `scenario_config.json` metadata;
- `frame_timing.csv` metadata;
- frame output path via timing metadata, without assuming the writer filename.

## Validation

Commands run:

```bash
PYTHONPATH=. ruff check tools/optical_pipeline_lab/runner.py \
  tools/optical_pipeline_lab/menagerie_static_runner.py \
  tests/unit/optics/test_optical_pipeline_lab.py \
  tests/gpu/test_optical_gpu_runtime.py

PYTHONPATH=. pytest -q tests/unit/optics/test_optical_pipeline_lab.py

conda run -n env_tilelang_20260119 env PYTHONPATH=. pytest -q \
  tests/gpu/test_optical_gpu_runtime.py \
  -k "cuda_direct_light_first_hit_matches_warp_bvh_executor or cuda_direct_light_no_shadow_matches_cpu or cuda_direct_light_lab_smoke"
```

Result:

- ruff: clean
- optical lab unit: 197 passed
- CUDA direct-light targeted GPU: 3 passed, 37 deselected

The GPU smoke printed a pytest `tmp_path` artifact root for manual inspection
and did not create repo-local `out/` or `runs/` artifacts.

## Known Limits

- CUDA hard shadows remain P12.3d.
- CUDA camera raygen remains P12.3e.
- Go2 CUDA direct-light remains deferred to full P12.3f after this tiny smoke.
- Public backend override API remains deferred to P12.6.
- The transitional static runner still owns this `run_scenario(...)` path; the
  P11 product workflow remains the preferred public workflow layer.
