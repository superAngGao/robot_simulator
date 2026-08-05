# Q54 P12.3f-pre Design Request V2: CUDA Direct-Light Lab Smoke Without Repo Pollution

Author: Codex
Status: ready for review
Related plan:
`collab/q54-optical-lab-p12-3-cuda-direct-light-backend__implementation-plan__codex__v1.md`
Supersedes:
`collab/q54-optical-lab-p12-3f-pre-cuda-direct-light-lab-smoke__design-request__codex__v1.md`

## Owner Summary

P12.3c proved the CUDA direct-light executor can produce no-shadow RGB and
intensity channels from a device scene plus CUDA LBVH:

```text
host rays
  -> CUDA LBVH first-hit
  -> CUDA no-shadow direct-light shading
  -> Torch CUDA tensor result channels
```

The next slice should be a narrow Lab workflow smoke, not a broad backend
expansion. The goal is to let an explicit Lab config render a tiny no-shadow
`cuda_direct_light` run through the existing video/debug product path and write
normal artifacts. This gives us a real end-to-end visual check before adding
hard shadows, GPU camera raygen, rgb8 packing, public preset overrides, or
benchmark matrices.

## Current State

Already implemented:

- `RenderBackend.CUDA_DIRECT_LIGHT`;
- `AccelBackend.CUDA_LBVH` validation for CUDA direct-light configs;
- `DeviceOpticalSceneCache` and `build_cuda_lbvh_from_snapshot(...)`;
- `CudaDeviceBvhOpticalExecutor` first-hit host-ray path;
- `CudaDeviceBvhDirectLightOpticalExecutor(shadows=False)` no-shadow shading;
- device channel staging for Torch CUDA tensors;
- render-session executor dispatch:

```python
executor_cls = (
    CudaDeviceBvhDirectLightOpticalExecutor
    if _is_cuda_direct_light_backend(options)
    else GpuDeviceBvhDirectLightOpticalExecutor
)
```

Current blocker:

```python
validate_run_scenario_supported(...)
```

still rejects `render_backend='cuda_direct_light'` with an old P12.3b-era
message. The rest of the run-option validation already keeps the CUDA path
narrow:

- `video_raygen='host'` only;
- `video_readback_delivery='sync'` only;
- no `readback_payload='rgb8'`;
- `accel_backend='cuda_lbvh'` required.

## Slice Goal

Add the smallest reviewed path that can run:

```text
OpticalLabScenarioConfig(
    scene_preset="synthetic_body_triangle",
    render_backend="cuda_direct_light",
    accel_backend="cuda_lbvh",
    shadows=False,
    readback_payload="rgb",
    ...
)
  -> run_scenario(...)
  -> OpticalLabRenderSession
  -> CudaDeviceBvhDirectLightOpticalExecutor
  -> video/debug artifacts
```

The accepted first scene is `synthetic_body_triangle`, not Go2. It is smaller,
deterministic, and matches the current no-shadow executor capability. Go2 must
remain a follow-up once this tiny workflow path is accepted.

## Non-Goals

This slice must not implement:

- hard shadows (`shadows=True`);
- CUDA camera raygen (`video_raygen='gpu'`);
- `rgb8` packing for CUDA direct-light;
- async readback for CUDA direct-light;
- public `run_optical_lab_preset(..., render_backend=...)` override;
- benchmark matrix expansion;
- Go2 CUDA direct-light video as the first acceptance case;
- ad hoc standalone scripts for manual experiments.

## Proposed Code Changes

### 1. Update `validate_run_scenario_supported(...)`

Allow `RenderBackend.CUDA_DIRECT_LIGHT` in
`validate_run_scenario_supported(...)` only when the config is inside the
supported pre-slice subset:

```text
render_backend == cuda_direct_light
accel_backend == cuda_lbvh
scene_preset == synthetic_body_triangle
shadows == False
```

This scene/backend whitelist belongs in `validate_run_scenario_supported(...)`,
not in `OpticalLabScenarioConfig` schema validation. Schema validation should
continue to describe generally valid configs, while `run_scenario(...)` should
describe what the runner can execute today.

Do not relax Go2, gpu raygen, rgb8, or async delivery in this slice. Reject
`shadows=True` at validation time with a precise "pending P12.3d" style message
rather than relying only on executor-level fail-fast.

### 2. Add a Unit Validation Test

Add a CPU-safe unit test that confirms:

- explicit `cuda_direct_light + cuda_lbvh + synthetic_body_triangle + no-shadow`
  config is accepted by scenario/run validation;
- existing accepted backends still pass through the same validator:
  - `warp_bvh_direct_light`;
  - `cpu_direct_light`;
- incompatible cases still fail:
  - `shadows=True`;
  - `scene_preset="go2_menagerie_static"` for this pre-slice;
  - `video_raygen="gpu"`;
  - `video_readback_delivery="torch_async"`;
  - `readback_payload="rgb8"`.

This protects the narrow CUDA boundary and confirms the shared validator does
not regress existing CPU/Warp paths.

### 3. Add a GPU Lab Smoke Test

Add one GPU test under the existing GPU optical test suite. It should run a very
small explicit config:

```text
frames: 1 or 2
scene_preset: synthetic_body_triangle
render_backend: cuda_direct_light
accel_backend: cuda_lbvh
shadows: false
readback_payload: rgb
video_raygen: host
video_readback_delivery: sync
warmup_renders: 1
```

Assertions:

- `scenario_config.json` exists and records:
  - `render_backend == "cuda_direct_light"`;
  - `accel_backend == "cuda_lbvh"`;
  - `shadows == false`;
- `frame_timing.csv` exists and has at least one data row;
- timing row records `render_backend == "cuda_direct_light"`;
- video/debug artifacts exist according to product result metadata;
- no output is written outside pytest `tmp_path`.

The test should avoid directly asserting writer-internal filenames unless those
paths are already part of the product result contract. Product metadata is the
stable assertion surface.

The smoke should use the installed Warp/CUDA conda environment during manual
validation:

```bash
conda run -n env_tilelang_20260119 env PYTHONPATH=. pytest -q \
  tests/gpu/test_optical_gpu_runtime.py \
  -k "cuda_direct_light_lab_smoke"
```

### 4. Optional Manual Render Command

If review accepts the smoke and we want a human-viewable result, use the same
GPU-test-driven path with pytest output inspection. Do not add a new private
helper or user-facing script for this pre-slice.

```bash
conda run -n env_tilelang_20260119 env PYTHONPATH=. pytest -q -s \
  tests/gpu/test_optical_gpu_runtime.py \
  -k "cuda_direct_light_lab_smoke"
```

The GPU smoke may print the temporary artifact root for manual inspection, but
the test must still write only under pytest `tmp_path`.

## Repo Pollution Guardrails

Allowed in this slice:

- narrowly-scoped changes to validation/dispatch needed for the existing
  `CudaDeviceBvhDirectLightOpticalExecutor` to run through Lab;
- one unit validation test;
- one GPU smoke test;
- one implementation note after code lands;
- MANIFEST update only if file descriptions become stale.

Not allowed:

- committing generated `out/`, `runs/`, `tmp/`, videos, PNGs, or CSV artifacts;
- adding unreviewed examples;
- adding public API flags before P12.6;
- expanding matrix suites;
- broad refactors of delivery, presets, or product workflow;
- compatibility shims whose only purpose is a local experiment;
- changing Go2 defaults.

Before commit:

```bash
git status --short
git diff --check
PYTHONPATH=. ruff check <changed python files>
PYTHONPATH=. pytest -q tests/unit/optics/test_optical_pipeline_lab.py -k "cuda_direct_light"
conda run -n env_tilelang_20260119 env PYTHONPATH=. pytest -q \
  tests/gpu/test_optical_gpu_runtime.py \
  -k "cuda_direct_light_lab_smoke"
```

## Acceptance Criteria

P12.3f-pre is accepted only if:

1. A reviewed explicit config can run `cuda_direct_light` through Lab with
   `shadows=False`, host raygen, sync delivery, RGB readback, and one warmup
   render.
2. The resulting artifacts are produced by the normal product path, not a
   standalone experimental script.
3. The smoke validates scenario/timing metadata.
4. Existing `cpu_direct_light` and `warp_bvh_direct_light` tests remain green.
5. The repo contains no generated render artifacts.

## Follow-Ups After This Slice

After P12.3f-pre:

1. P12.3d: CUDA shadow any-hit.
2. P12.3e: CUDA camera raygen.
3. P12.3f: full Lab workflow smoke, including a reviewed Go2 CUDA direct-light
   visual check if capability matches the scene.
4. P12.6: public backend override API and examples once backend behavior is
   stable.
5. P13: tracing/path-tracing backend work remains deferred.

## Resolved Review Decisions

1. Go2 static is not allowed in this pre-slice. It belongs to the later full
   P12.3f visual check.
2. `shadows=True` should be rejected in validation until P12.3d, with a clear
   message. Executor-level fail-fast can remain as a second line of defense.
3. Manual inspection should be GPU-test-driven. Do not add a private helper or
   example script.
4. Artifact assertions should use product result metadata, not writer-internal
   filename assumptions.
