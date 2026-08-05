# Q54 P12.3d Implementation Note: CUDA Shadow Any-Hit

Author: Codex
Status: ready for review
Related design:
`collab/q54-optical-lab-p12-3d-cuda-shadow-any-hit__design-request__codex__v2.md`

## Owner Summary

This implements P12.3d functional CUDA hard shadows for the host-ray
`cuda_direct_light` executor.

`CudaDeviceBvhDirectLightOpticalExecutor.execute(...)` now supports:

```text
shadows=False:
  CUDA LBVH first-hit -> CUDA no-shadow direct-light shade

shadows=True:
  CUDA LBVH first-hit -> CUDA direct-light shade with triangle any-hit shadows
```

The implementation remains inside the production executor boundary. No
benchmark-only kernel caller, temporary script, or mixed execution path was
added.

## What Changed

### `optics/cuda_direct_light.py`

Added CUDA extension entry point:

```text
shade_direct_light_with_shadows(...)
```

It consumes the same first-hit geometry result and the same scene/BVH buffer
layout already used by the CUDA first-hit path:

- first-hit channels: `hit_mask`, `position_world`, `normal_world`,
  `material_index`;
- material/light buffers;
- triangle world buffers and triangle role masks;
- CUDA LBVH node/primitive buffers.

The shadow-capable shade kernel performs per-light any-hit traversal and
returns:

- `rgb`;
- `intensity`;
- `shadow_stack_overflow_count`;
- `shadow_max_stack_depth`.

P12.3d intentionally treats triangle meshes as CUDA shadow occluders. Planes can
still be primary hits, but are non-occluders in this CUDA shadow slice; the code
comment lives next to the shadow traversal call.

### Validator

`run_scenario(...)` now allows synthetic CUDA direct-light with
`shadows=True` after parity passed:

```text
scene_preset="synthetic_body_triangle"
render_backend="cuda_direct_light"
accel_backend="cuda_lbvh"
video_raygen="host"
video_readback_delivery="sync"
readback_payload="rgb"
```

Go2 CUDA direct-light remains rejected until the full P12.3f visual check.

### Tests

Added CUDA parity coverage:

- `test_cuda_direct_light_shadowed_direct_light_matches_cpu_and_warp`
  compares CPU, Warp, and CUDA shadowed direct-light output on a tiny scene with
  a triangle occluder.
- The test uses non-zero `shadow_bias=1e-6`, matching the CPU/Warp default.
- It asserts the no-shadow CUDA path is brighter than the shadowed path for the
  occluded ray.

Added Lab workflow coverage:

- `test_cuda_direct_light_shadowed_lab_smoke` runs `run_scenario(...)` through
  the normal Lab static/synthetic path with `shadows=True`.
- Artifact checks use scenario/timing metadata and pytest `tmp_path`; no
  generated artifacts are committed.

Unit coverage now confirms:

- synthetic `cuda_direct_light + shadows=True` can run;
- Go2 CUDA direct-light remains rejected;
- existing CPU/Warp run_scenario paths remain accepted.

## Validation

Commands run:

```bash
PYTHONPATH=. ruff check optics/cuda_direct_light.py \
  tools/optical_pipeline_lab/runner.py \
  tests/unit/optics/test_optical_pipeline_lab.py \
  tests/gpu/test_optical_gpu_runtime.py

PYTHONPATH=. pytest -q tests/unit/optics/test_optical_pipeline_lab.py \
  -k "cuda_direct_light"

PYTHONPATH=. pytest -q tests/unit/optics/test_device_optical.py \
  -k "cuda_direct_light"

PYTHONPATH=. pytest -q tests/unit/optics/

conda run -n env_tilelang_20260119 env PYTHONPATH=. pytest -q \
  tests/gpu/test_optical_gpu_runtime.py \
  -k "cuda_direct_light_first_hit_matches_warp_bvh_executor or cuda_direct_light_no_shadow_matches_cpu or cuda_direct_light_shadowed_direct_light_matches_cpu_and_warp or cuda_direct_light_lab_smoke or cuda_direct_light_shadowed_lab_smoke"
```

Result:

- ruff: clean
- optical lab CUDA unit slice: 5 passed, 192 deselected
- device optical CUDA unit slice: 3 passed, 11 deselected
- full optics unit: 276 passed
- CUDA direct-light targeted GPU: 5 passed, 37 deselected

## Deferred

- No benchmark suite yet. P12.3d-benchmark should collect performance only
  through production executor/workflow boundaries.
- No optimization work yet.
- No CUDA camera raygen.
- No Go2 CUDA shadow visual.
- No `rgb8`, async delivery, public preset override, or matrix expansion.
