# Q54 P12 Design Request: Direct-Light Backend Family

Owner: Codex
Date: 2026-07-21
Status: Draft for review
Related Files:
- `tools/optical_pipeline_lab/scenarios.py`
- `tools/optical_pipeline_lab/render_session.py`
- `tools/optical_pipeline_lab/presets.py`
- `tools/optical_pipeline_lab/matrix.py`
- `tools/optical_pipeline_lab/preset_workflows.py`
- `tools/optical_pipeline_lab/product_workflow.py`
- `optics/render_api.py`
- `tests/unit/optics/test_optical_pipeline_lab.py`
- `tests/gpu/test_optical_gpu_runtime.py`
- `examples/optical_lab/go2_video_ordered_static.py`
- `collab/q54-optical-lab-p12__open-questions__codex__v1.md`

## Summary

P12 should not start with a registry-only refactor. With only one implemented
Optical Lab backend (`warp_bvh_direct_light`), a broad backend registry would
risk over-design.

Instead, P12 should first make the direct-light backend family real:

```text
cpu_direct_light        # CPU reference / correctness / CI-friendly path
warp_bvh_direct_light   # current implemented GPU path
cuda_direct_light       # CUDA direct-light performance evolution path
```

The review discussion clarified one important priority: `cuda_direct_light`
should be treated as a real P12 implementation target, not as a reserved alias
or a Warp pipeline variant. The project goal is high-performance rendering, so
P12 should invest in the CUDA direct-light path even if that makes the slice
larger. CPU direct light remains important as a reference/parity backend, but
CUDA direct light is the main performance-oriented backend work.

After CPU and CUDA direct-light paths exist, P12 can extract backend selection
and benchmark contracts from real implementation pressure rather than imagined
future needs.

## Current State

There are three different "backend" vocabularies in the repo:

1. `rendering/backends/RenderBackend`
   - Matplotlib / Rerun visualization backend.
   - Not part of this P12.

2. `tools.optical_pipeline_lab.scenarios.RenderBackend`
   - Lab scenario-level backend vocabulary.
   - Current values:
     - `warp_bvh_direct_light`
     - `cuda_direct_light`
     - `cuda_fused_rgb`
     - `optix_first_hit`
     - `path_tracer`
   - Only `warp_bvh_direct_light` is implemented today.

3. `optics.render_api.RenderBackend`
   - Runtime request-level compute vocabulary.
   - Current values:
     - `direct_light`
     - `path_tracing`
   - `OpticalLabRenderSession.execute_request(...)` supports only
     `direct_light` today.

P12 should connect (2) to real execution paths without confusing it with (1).

## Existing Infrastructure Audit

The current codebase already has useful direct-light building blocks:

```text
CPU:
  CpuReferenceOpticalExecutor
  CpuBvhOpticalExecutor
  CpuDirectLightOpticalExecutor

GPU/Warp:
  GpuDeviceBvhOpticalExecutor
  GpuDeviceBvhDirectLightOpticalExecutor

Acceleration:
  CPU BVH snapshot acceleration
  CUDA LBVH builder

Benchmark:
  tools/optical_pipeline_lab/matrix.py
  matrix_summary.csv
  frame_timing.csv
  scenario_config.json
```

Missing today:

```text
hand-written CUDA direct-light render kernel path
Optical Lab runtime dispatch for cpu_direct_light / cuda_direct_light
backend compatibility validation beyond current reserved fail-fast
reviewed backend matrix rows for CPU/Warp/CUDA direct-light
```

This means P12.2 can build on existing CPU direct-light infrastructure, while
P12.3 is the larger new work: a real CUDA direct-light render path.

## P12 Goals

1. Add a CPU direct-light Optical Lab backend path.
2. Add a CUDA direct-light Optical Lab backend path with hand-written CUDA
   kernels, not a Warp alias.
3. Keep the existing Warp BVH direct-light path stable.
4. Upgrade benchmark/matrix/test support so backend choice is explicit and
   reviewable.
5. Defer path tracing, OptiX, and accumulation semantics to P13.
6. Refactor backend selection only after CPU/Warp/CUDA direct-light behavior is
   concrete.

## Non-Goals

P12 should not:

- implement path tracing;
- implement OptiX;
- implement `cuda_fused_rgb`;
- add path-tracing accumulation buffers;
- redesign P11 product APIs;
- change physics ownership of the tick stream;
- mix Q50 Matplotlib/Rerun visualization backends into Optical Lab backend
  selection;
- expose low-level benchmark flag clutter in P11 examples before backend
  selection stabilizes.

## Backend Roles

### `cpu_direct_light`

Role:

```text
reference / correctness / CPU-only test backend
```

Expected properties:

- CI-friendly on small scenes;
- no Warp/CUDA requirement for tiny parity tests;
- not performance-oriented;
- useful for validating output shape, payload schema, and approximate numeric
  parity.

Design question:

Should this be added as a new `tools.optical_pipeline_lab.scenarios.RenderBackend`
enum value?

Recommended answer:

```text
Yes. CPU direct light should not pretend to be warp_bvh_direct_light with a CPU
accel backend. It is a distinct render execution backend.
```

### `warp_bvh_direct_light`

Role:

```text
current implemented GPU backend / performance baseline / P11 smoke path
```

Expected properties:

- preserve current Go2 static and physics body-triangle P11 workflow behavior;
- continue supporting existing timing CSV and delivery paths;
- remain the default reviewed backend for current presets.

### `cuda_direct_light`

Role:

```text
CUDA direct-light performance evolution path
```

Expected properties:

- real execution path, not just a reserved enum;
- uses hand-written CUDA kernels rather than Warp kernels;
- target same direct-light output semantics as `warp_bvh_direct_light`;
- first implementation can be minimal and correctness-oriented, but it should
  establish the CUDA ownership/resource path;
- does not need to be fused RGB or final-performance in P12.

Technical positioning:

```text
cuda_direct_light = CUDA LBVH + hand-written CUDA first-hit/direct-light kernels
```

It should not mean:

```text
cuda_direct_light = warp_bvh_direct_light alias
cuda_direct_light = Warp kernel with a different pipeline wrapper
cuda_direct_light = cuda_fused_rgb preview-only optimization
```

Design question:

How is `cuda_direct_light` different from `cuda_fused_rgb`?

Recommended answer:

```text
cuda_direct_light preserves the normal direct-light output/profile/readback
contract. cuda_fused_rgb is a future specialized RGB-preview optimization that
may not provide full payloads or diagnostics. P12 implements cuda_direct_light,
not cuda_fused_rgb.
```

Implementation risk:

```text
Accepted. P12 may take longer because CUDA direct light is central to the
project's high-performance rendering goal.
```

## Acceleration Backend Compatibility

P12 should validate render backend and acceleration backend combinations
explicitly.

Initial expected compatibility:

```text
cpu_direct_light:
  accel_backend = cpu_bvh

warp_bvh_direct_light:
  accel_backend = cpu_bvh | cuda_lbvh

cuda_direct_light:
  accel_backend = cuda_lbvh
```

Open question:

Should `warp_bvh_direct_light + cpu_bvh` remain supported as a CPU-built BVH plus
Warp render path, or should it be treated as legacy/dev-only?

Recommended answer:

```text
Keep it supported if it already works. It is useful for isolating BVH build
backend behavior from render backend behavior.
```

## Runtime Backend Mapping

P12 should not force a one-to-one mapping between Lab backend values and
`optics.render_api.RenderBackend`.

Expected mapping for P12:

```text
tools.optical_pipeline_lab.scenarios.RenderBackend.CPU_DIRECT_LIGHT
  -> optics.render_api.RenderBackend.DIRECT_LIGHT

tools.optical_pipeline_lab.scenarios.RenderBackend.WARP_BVH_DIRECT_LIGHT
  -> optics.render_api.RenderBackend.DIRECT_LIGHT

tools.optical_pipeline_lab.scenarios.RenderBackend.CUDA_DIRECT_LIGHT
  -> optics.render_api.RenderBackend.DIRECT_LIGHT
```

The runtime `DIRECT_LIGHT` vocabulary names the optical transport algorithm,
while the lab-level backend names the implementation family.

`PATH_TRACING` remains reserved for P13.

## User-Facing Selection

P12 should keep backend selection centered on:

```python
OpticalLabScenarioConfig.render_backend
```

Rationale:

- backend selection is scenario behavior, not artifact output behavior;
- `scenario_config.json` already serializes `render_backend`;
- `frame_timing.csv` and matrix summaries already record `render_backend`;
- presets already declare backend intent;
- P11 public workflow should not gain broad override knobs before backend
  compatibility is proven.

P12 should not start by adding:

```python
run_optical_lab_preset(..., render_backend="cuda_direct_light")
```

That can be reconsidered after benchmark/matrix behavior is stable.

## Benchmark And Example Contract

P12 should upgrade backend tests and benchmark support together with backend
implementation. Backend support is incomplete unless it is visible in:

- scenario config serialization;
- frame timing CSV;
- matrix summary CSV;
- reviewed example/backend table;
- fail-fast validation for unsupported combinations.

### Reviewed Example Table

Start with a small reviewed table for a stable example/preset. Candidate:

```text
example/preset: go2_video_ordered_static
products:       video, debug
```

Initial rows:

```text
render_backend          accel_backend  expected
----------------------  -------------  ----------------
warp_bvh_direct_light   cuda_lbvh      implemented
cpu_direct_light        cpu_bvh        implemented tiny/reference smoke
cuda_direct_light       cuda_lbvh      implemented minimal smoke
optix_first_hit         optix          reserved fail-fast
path_tracer             optix/cuda     reserved for P13 fail-fast
cuda_fused_rgb          cuda_lbvh      reserved fail-fast
```

The first table does not need to run full Go2 performance for every backend.
CPU and CUDA can use tiny synthetic scenes for CI, while Go2 GPU smoke can remain
an explicit manual/optional verification if runtime cost or dependencies are too
high.

### Matrix Support

P12 should ensure matrix output records:

- `render_backend`;
- `accel_backend`;
- `frame_source`;
- `clock_owner`;
- `readback_payload`;
- `delivery_policy`;
- pass/fail and fail-fast reason.

Open question:

Should matrix CSV add `backend_status` / `backend_reason` columns?

Recommended answer:

```text
Not initially. Keep stable summary fields unless review shows fail-fast reasons
are hard to analyze.
```

## Proposed P12 Slices

### P12.1: Design Review

Produce and review this design:

- direct-light backend family;
- CPU/Warp/CUDA roles;
- benchmark/example table contract;
- P13 deferral for path tracing;
- non-goals and public API boundaries.

### P12.2: CPU Direct-Light Backend

Implement or expose:

- `RenderBackend.CPU_DIRECT_LIGHT`;
- CPU direct-light render session/executor path;
- tiny static scene smoke;
- parity-tolerant output assertions;
- fail-fast for incompatible accel/backend combinations.

P12.2 can proceed before or in parallel with CUDA work. Its role is to provide a
small CPU-only correctness reference, not to become the main P12 performance
deliverable.

### P12.3: CUDA Direct-Light Backend

Implement a real CUDA direct-light backend:

- `RenderBackend.CUDA_DIRECT_LIGHT`;
- CUDA LBVH-backed first-hit traversal;
- CUDA direct-light shading with the same direct-light contract as the Warp path;
- tiny GPU smoke;
- schema/timing parity with `warp_bvh_direct_light`;
- no `cuda_fused_rgb` behavior yet.

Minimum acceptance criteria:

```text
1. The backend does not call Warp render kernels.
2. A tiny static scene renders through the Optical Lab runtime path.
3. Output channels match the expected direct-light schema.
4. Numeric parity with CPU/Warp direct light is within reviewed tolerance.
5. scenario_config.json and frame_timing.csv record render_backend=cuda_direct_light.
```

### P12.4: Benchmark Contract

Upgrade benchmark/matrix support:

- backend rows for CPU/Warp/CUDA direct light;
- reserved backend fail-fast rows;
- stable scenario/timing/summary CSV assertions;
- reviewed example/backend table.

### P12.5: Backend Selection Refactor

After CPU/Warp/CUDA direct-light paths exist:

- extract backend capability/validation helpers if the duplication is real;
- keep helpers narrow and grounded in implemented behavior;
- avoid a broad registry until it removes real complexity.

### P12.6: Optional Public Override

Only after P12.4/P12.5:

```python
run_optical_lab_preset(..., render_backend="cuda_direct_light")
```

This should remain optional. If config/preset-based selection is sufficient,
skip this slice.

### P12.7: Examples Usability Follow-Up

Deferred until backend behavior is stable.

Candidate work:

- artifacts guide;
- backend-specific example flags;
- `--help` coverage;
- recommended conda environment notes;
- small backend table in `examples/optical_lab/README.md`.

## P13: Path Tracing / OptiX / Accumulation

P13 should own path tracing and other non-direct-light rendering work.

P13 candidate scope:

```text
path_tracer backend
optix_first_hit backend
path-tracing accumulation buffers
sample count / seed / accumulation_id semantics
multi-frame accumulation lifecycle
path-tracing output profiles
denoising / tone mapping hooks
```

Reasons to defer to P13:

- path tracing changes render semantics, not just implementation backend;
- accumulation is stateful and session-owned;
- delivery remains similar, but render requests may become multi-sample or
  multi-frame;
- adding it before CPU/CUDA direct-light would conflate backend selection with a
  new light transport algorithm.

P12 should keep:

```text
RenderBackend.PATH_TRACER -> reserved fail-fast
optics.render_api.RenderBackend.PATH_TRACING -> reserved request vocabulary
RenderRequest.accumulation_id -> reserved field
```

## Review Questions

1. Is it correct to make P12 implementation-driven with CPU/Warp/CUDA
   direct-light before extracting a backend registry?
2. Should `cpu_direct_light` be a first-class lab `RenderBackend` enum value?
3. Is the CUDA direct-light scope correctly defined as hand-written CUDA
   first-hit/direct-light kernels rather than a Warp alias or `cuda_fused_rgb`
   preview path?
4. Are the proposed accel/backend compatibility rules correct?
5. Should P12 expose public `run_optical_lab_preset(..., render_backend=...)`,
   or keep selection in config/presets until later?
6. Is P13 the right home for path tracing, OptiX, and accumulation semantics?
