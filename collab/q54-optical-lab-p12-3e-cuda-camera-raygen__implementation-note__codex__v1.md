Initiative: q54-optical-lab-p12-3e-cuda-camera-raygen
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-08-07
Status: implemented
Related Files: collab/q54-optical-lab-p12-3e-cuda-camera-raygen__proposal__codex__v2.md; collab/q54-optical-lab-p12-3e-cuda-camera-raygen__proposal__codex__v3.md; optics/cuda_direct_light.py; tools/optical_pipeline_lab/render_session.py; tools/optical_pipeline_lab/runner.py; tests/gpu/test_optical_gpu_runtime.py; deploy/robot_sim_cuda128/README.md
Owner Summary: P12.3e is implemented in commit 39a8a13. CUDA camera rays now feed the shared CUDA LBVH first-hit and direct-light paths on an explicit target device without accepting a Warp stream. Raygen, first-hit, and shade extension entry points use device guards and the target device's current Torch stream. The implementation preserves the reviewed two-stage blocking behavior, adds direct ray and multi-GPU restoration tests, covers all supported profiles, enables the narrow Lab GPU-raygen configuration, and records the relayed environment manifests. CUDA is not yet the default and the Warp optical path remains available pending P12.4.

## Open Questions Addressed

None in `OPEN_QUESTIONS.md`. P12.3e was an already-defined Q54 sub-stage.

## REFLECTIONS.md / PROGRESS.md Impact

Do not update `REFLECTIONS.md` or `PROGRESS.md` before implementation review.
After approval, record the CUDA-first/no-bridging optical direction and mark
P12.3e complete. P12.4 remains responsible for the performance/default-backend
decision and any Warp optical retirement milestone.

## What Changed

### CUDA camera ray generation

- Added a flat row-major pinhole CUDA kernel with one thread per pixel.
- Added an output-buffer binding anchored by explicit `origins` and `directions`
  Torch tensors on the requested CUDA device.
- Added a private Python helper that allocates contiguous float32 `[N, 3]`
  outputs on an explicit `torch.device` and launches on that device's current
  Torch stream.
- Added CUDA-event raygen timing. The event is resolved after the existing
  post-first-hit geometry synchronization as logical phase `raygen_kernel`,
  which the Lab writes as `render_raygen_kernel_ms`.

### Device and stream contract

- Removed `stream` from both CUDA optical executor constructors.
- Kept Warp executor stream arguments unchanged.
- Branched Lab executor construction so `workspace.stream` is passed only to
  the Warp executor.
- Added CUDA device guards and indexed current-stream lookup to raygen,
  first-hit, no-shadow shade, and shadow shade extension entry points.
- Added same-device validation across every tensor passed to each entry point.
- Preserved and tested restoration of the caller's ambient current device.
- Waited for the Warp-owned snapshot and BVH ready events before starting CUDA
  camera raygen. This remains a blocking ownership boundary, not stream
  interoperation.

### Shared camera render path

- Factored CUDA first-hit execution over already-device-resident ray tensors.
- Host-ray execution now uploads to the explicit target and reuses that helper.
- Geometry `execute_camera(...)` generates CUDA rays and reuses the same
  first-hit implementation.
- Direct-light `execute_camera(...)` reuses the geometry camera path and the
  existing shadow/no-shadow shade helpers.
- Generated ray tensors stay in `OpticalComputeResult.resources` through
  shading and delivery.

### Synchronization behavior

The implementation matches proposal v3:

```text
wait Warp inputs
-> raygen -> first-hit -> geometry torch.cuda.synchronize(target)
-> resolve raygen event
-> shade -> shade torch.cuda.synchronize(target)
-> blocking result with ready_event=None
```

No internal non-blocking geometry API, Warp/Torch stream bridge, or additional
raygen-only synchronization was added.

### Lab and diagnostics

- Removed only the temporary CUDA `video_raygen='host'` restriction.
- Retained CUDA LBVH, sync delivery, non-RGB8, ray-cache, and supported-scenario
  restrictions.
- Added the nine shadow traversal diagnostic channels required by profiled Lab
  readback. No-shadow values are zero; the shadow kernel records ray kind,
  occlusion, node/leaf visits, and triangle tests. Plane shadow tests remain zero
  because P12.3d intentionally supports triangle shadow occluders only.

### Reproducible environment evidence

Added `deploy/robot_sim_cuda128/` containing:

- exact Conda explicit package/build inventory;
- exact non-editable Python package inventory;
- SHA-256 manifest for all 62 relayed wheels;
- offline relay, verification, extension-cache, and test commands.

The wheel binaries are not committed. Server verification against the tracked
manifest completed successfully for all 62 files.

## Files Touched

- `optics/cuda_direct_light.py`
- `tools/optical_pipeline_lab/render_session.py`
- `tools/optical_pipeline_lab/runner.py`
- `tests/gpu/test_optical_gpu_runtime.py`
- `tests/unit/optics/test_device_optical.py`
- `tests/unit/optics/test_optical_pipeline_lab.py`
- `deploy/robot_sim_cuda128/README.md`
- `deploy/robot_sim_cuda128/conda-explicit.txt`
- `deploy/robot_sim_cuda128/python-packages.txt`
- `deploy/robot_sim_cuda128/wheelhouse.sha256`

## Tests Added / Updated

### Direct ray contract

The low-level test uses a 7x3 camera with `fx != fy`, off-center principal
point, non-zero translation, and compound non-identity rotation. It directly
compares every CUDA origin and direction with
`build_pinhole_camera_rays(...)`, checks unit lengths and explicit
`py * width + px` row boundaries, and verifies shape/dtype/contiguity/device.

### Geometry, profiles, and shadows

- Compared CUDA-camera first-hit with materialized host-ray CUDA first-hit on an
  asymmetric triangle grid whose hit pattern distinguishes neighboring pixels.
- Exercised `direct_light_full`, `rgb_preview`, and `render_only` with both
  shadow modes.
- Compared every returned camera channel with the matching materialized-ray
  execution.
- Verified raygen event timing is finite.
- Verified shadow traversal counters are internally consistent and non-trivial
  in the shadow fixture.

### Multi-GPU device guard

On two visible A100 devices, the test constructs the scene/BVH and executor for
logical `cuda:1`, then explicitly resets and asserts ambient `cuda:0`
immediately before and after each of:

1. low-level raygen;
2. geometry camera execution;
3. direct-light camera execution.

It verifies outputs/resources remain on `cuda:1`, numerical parity holds, and
the ambient current device is restored to `cuda:0`.

### Lab workflow

The existing CUDA Lab smoke now runs with GPU raygen and profiling enabled. It
verifies `raygen_mode=gpu` and finite `render_raygen_kernel_ms` through the
normal runner, delivery, PNG, and CSV path.

## Verification Evidence

The non-GPU commit gate and ruff command were rerun after commit `39a8a13` was
created. The final CUDA commands ran on the identical final worktree immediately
before that commit. Earlier candidate-iteration evidence is labeled separately.

### Exact-commit gates

```text
conda run -n robot_sim env PYTHONPATH=. \
  python -m pytest tests/ -m 'not (slow or gpu)' -q

1158 passed, 39 skipped, 466 deselected, 1 pre-existing pytest deprecation warning
```

```text
conda run -n robot_sim python -m ruff check \
  optics/cuda_direct_light.py \
  tools/optical_pipeline_lab/render_session.py \
  tools/optical_pipeline_lab/runner.py \
  tests/unit/optics/test_device_optical.py \
  tests/unit/optics/test_optical_pipeline_lab.py \
  tests/gpu/test_optical_gpu_runtime.py

All checks passed
```

### Final-worktree CUDA camera/profile/Lab/shadow selection

```text
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/gpu/test_optical_gpu_runtime.py \
  -k 'cuda_camera_geometry_and_direct_light_profiles or \
      cuda_direct_light_shadowed_direct_light_matches_cpu_and_warp or \
      (cuda_direct_light_lab_smoke and not shadowed)' -q

3 passed, 42 deselected
```

### Final-worktree two-device restoration test

```text
CUDA_VISIBLE_DEVICES=0,1 python -m pytest tests/gpu/test_optical_gpu_runtime.py \
  -k 'guard_target_device_and_restore_ambient_device' -q

1 passed, 44 deselected
```

### Earlier candidate-iteration regression evidence

```text
python -m pytest tests/unit/optics -q
277 passed

CUDA_VISIBLE_DEVICES=0 python -m pytest tests/gpu/test_optical_gpu_runtime.py \
  -k 'cuda_direct_light or cuda_pinhole_raygen or cuda_camera_' -q
7 passed, 1 skipped, 37 deselected
```

The skipped case in the one-device command is the intentional two-device test;
it passed separately with two devices visible.

### Wheelhouse integrity

```text
sha256sum --check --quiet \
  /data3/ang.gao/robot_simulator/deploy/robot_sim_cuda128/wheelhouse.sha256

wheelhouse_sha256_verified=62
```

## Known Limitations

- The direct-light camera path retains two global target-device
  synchronizations: one after first-hit and one after shade.
- Scene snapshot and LBVH ownership still cross the temporary Warp-backed
  blocking boundary.
- The CUDA result remains blocking and reports `ready_event=None`.
- CUDA GPU raygen is not enabled for async delivery, RGB8, or Go2 acceptance.
- Only the existing pinhole camera model is implemented.
- Warp optical rendering remains available as an oracle and current fallback.
- CUDA is not made the default and Warp optical code is not removed in P12.3e.
- P12.4 must measure representative end-to-end workloads against an
  owner-defined budget before default selection or retirement work.

## Commit

Implementation commit:

```text
39a8a13 feat: add CUDA camera ray generation
```

The branch is not pushed. At implementation-note creation time, local `main`
is four commits ahead of `origin/main` and the implementation worktree is clean.
