Initiative: q54-optical-lab-p12-4-backend-parity-performance-gate
Stage: proposal
Author: codex
Version: v1
Date: 2026-08-10
Status: in_review
Related Files: collab/q54-optical-lab-p12-direct-light-backend-family__design-request__codex__v1.md; collab/q54-optical-lab-p12-3e-cuda-camera-raygen__decision__owner__v1.md; collab/q54-optical-lab-p12-3e-cuda-camera-raygen__implementation-note__codex__v1.md; tools/optical_pipeline_lab/matrix.py; tools/optical_pipeline_lab/timing.py; tools/optical_pipeline_lab/runner.py; tests/unit/optics/test_optical_pipeline_lab.py; tests/gpu/test_optical_gpu_runtime.py
Owner Summary: P12.4 turns the existing Optical Lab matrix into an evidence gate for the CUDA-first optical decision. It records an owner-defined absolute workload budget before measurements, compares common CPU/Warp/CUDA capabilities and outputs directly, measures warmed end-to-end rows with compilation separated, and emits an explicit default/retirement recommendation. No Warp/Torch bridging, default switch, or Warp deletion is part of the measurement patch.

---

## Purpose

P12.3e proved that CUDA camera rays can drive the CUDA LBVH first-hit and
direct-light kernels correctly on an explicit target device. It did not prove
that CUDA is ready to replace Warp in product workloads.

P12.4 will answer four separate questions:

1. Which direct-light capabilities are actually retained product requirements?
2. Do CPU, Warp, and CUDA agree on those common capabilities?
3. Does warmed CUDA end-to-end execution meet the owner's absolute product
   budget, independent of whether it is faster or slower than Warp?
4. Which remaining ownership/synchronization gaps block a default switch or
   Warp optical retirement?

The result is an evidence-backed decision, not a benchmark leaderboard.

## Current Baseline

The implemented backend family is:

```text
cpu_direct_light
  accel: cpu_bvh
  role: small correctness oracle / CPU CI
  current limits: host raygen, sync delivery, no RGB8

warp_bvh_direct_light
  accel: cpu_bvh or cuda_lbvh where supported
  role: current GPU optical oracle/fallback
  current ownership: Warp device scene, streams, and render kernels

cuda_direct_light
  accel: cuda_lbvh
  role: CUDA-first performance candidate
  current limits: sync delivery, no RGB8, Lab acceptance currently limited to
                  synthetic_body_triangle
  current ownership: Warp-backed input readiness plus native Torch CUDA output
  current blocking path: geometry sync followed by shade sync
```

The existing `matrix.py` is not yet a backend comparison contract:

- a suite selects one preset whose backend is implicit;
- `MatrixCase` has no explicit accel/render backend axis;
- `matrix_summary.csv` omits accel backend, render backend, output profile,
  raygen mode, warmup/sample identity, and kernel-phase percentiles;
- existing baselines used only 10 frames and cannot establish tail stability;
- failed/reserved combinations are tested in places, but not summarized as a
  capability matrix;
- compile/JIT, warmup, and steady-state evidence are not separated strongly
  enough for a retirement decision.

## Owner Gate Must Be Frozen Before Performance Runs

P12.4 must not invent “fast enough” after seeing results. Add a small tracked
decision input, proposed as:

```text
benchmarks/results/q54_p12_4/gate.json
```

Minimum fields:

```json
{
  "target_workload": "owner-selected workload id",
  "retained_capabilities": [
    "static_scene",
    "dynamic_scene_updates",
    "shadows_on_off",
    "render_only",
    "rgb_preview_sync",
    "direct_light_full_sync"
  ],
  "target_resolution": [1920, 1080],
  "max_frame_p90_ms": null,
  "min_sustained_fps": null,
  "max_peak_memory_mb": null,
  "minimum_measured_frames": 120,
  "independent_repeats": 3
}
```

At least one of `max_frame_p90_ms` or `min_sustained_fps` must be non-null
before the performance gate is run. Memory may remain informational only if the
owner explicitly records that choice.

The current 30 FPS Lab default implies a candidate `max_frame_p90_ms` of
33.333 ms, but this proposal does not silently adopt that number. The owner
must accept or replace it.

Relative CUDA-vs-Warp speed remains diagnostic. It is not the pass condition:
if CUDA meets the absolute product budget and retained requirements, a slower
relative result does not automatically block the CUDA-first direction.

## Evidence Model

P12.4 separates capability, correctness, and performance instead of combining
them into one ambiguous status.

### 1. Capability matrix

Generate a small `capability_matrix.csv` from executable config validation plus
targeted runtime probes. Each row contains:

```text
scene/workload
accel_backend
render_backend
output_profile
raygen_mode
shadows
delivery/readback mode
status = supported | unsupported | reserved
reason
```

Unsupported rows are evidence, not skipped success. Reserved backends
(`cuda_fused_rgb`, `optix_first_hit`, and `path_tracer`) must retain precise
fail-fast behavior and are not executed as benchmark rows.

Full Warp optical retirement is blocked by any retained capability marked
unsupported for CUDA. The owner may remove a capability from the retained list,
but that is an explicit product-scope decision, not a benchmark convenience.

### 2. Numerical parity report

Create direct parity fixtures independent of the end-to-end performance scene.
Use asymmetric geometry/material/light layouts and compare backend outputs at
identical camera parameters.

Required comparisons on common supported rows:

- `hit_mask`, numeric instance/material IDs: exact;
- range and positions: reviewed absolute tolerance;
- normals: reviewed vector/cosine tolerance;
- RGB/intensity: reviewed float tolerance;
- required overflow and traversal-counter semantics;
- direct-light full, RGB preview, and render-only channel sets;
- shadows on and off.

CPU may run a smaller image/scene and serves as the independent numerical
oracle. Warp-vs-CUDA at product resolution is supplemental and must not replace
the CPU comparison.

Output: `parity_summary.json` with per-channel maximum/percentile errors,
status, tolerance, and exact workload identity.

### 3. Performance matrix

Use the ordinary Optical Lab runtime/product/delivery path. Do not benchmark an
isolated kernel and call it a frame result.

Required GPU workload families:

1. synthetic asymmetric static scene, for controlled CPU/Warp/CUDA comparison;
2. Go2 static scene, because it is the current representative static asset
   workflow;
3. dynamic rigid scene update, including snapshot plus acceleration refit or
   rebuild, because the future simulator product is not static-only.

If CUDA cannot execute Go2 or dynamic rows, record them as retirement blockers.
P12.4 must not remove a guard merely to produce a green matrix; capability
support requires a feature audit, parity coverage, and runtime smoke.

Within each applicable GPU workload, cover:

- shadows off and on;
- `render_only` / no readback;
- `rgb_preview` / sync RGB readback;
- `direct_light_full` / sync full readback at a practical diagnostic size;
- the owner-selected target resolution plus a small smoke resolution;
- CUDA GPU raygen; retain host-ray rows only as parity/diagnostic controls.

CPU performance rows are limited to the small correctness workload. They are
not required to finish product-resolution GPU throughput cases.

### 4. Environment and run identity

Every tracked measurement set records:

- git commit and worktree-clean flag;
- Conda/Python manifest identity and wheel-manifest SHA-256;
- GPU name, UUID or PCI identity, driver, runtime, Torch, and Warp versions;
- selected target device and relevant `CUDA_VISIBLE_DEVICES` mapping;
- matrix suite version, gate file hash, command line, repeat index, and random
  seed where applicable;
- whether the JIT cache was cold or warm.

Reuse `deploy/robot_sim_cuda128/`. P12.4 does not install or update packages.
If a missing artifact must be relayed, it is copied from the local machine and
verified against a tracked hash before use.

## Timing Semantics

### Separate three timing regimes

1. **Cold setup/compile**: extension compilation and first construction. Report
   separately; never include it in steady-state FPS.
2. **Warmup**: execute at least the gate's configured warmup count. Discard all
   warmup samples from performance statistics.
3. **Measured steady state**: at least the gate's configured frame count and
   repeats, with per-frame rows retained.

The primary throughput pass runs with `render_profile=False` so profiling does
not alter the winner. A second diagnostic pass uses `render_profile=True` for
kernel/event breakdown. The two datasets must not be merged.

### End-to-end ownership

Primary `frame_total_ms` includes the real workload's:

```text
frame/snapshot preparation
-> LBVH build/refit/rebuild
-> raygen
-> first-hit and geometry synchronization
-> shade and shade synchronization
-> requested delivery/readback
-> image construction/write only when the workload explicitly requires it
```

PNG/encoding is not a renderer-throughput gate unless selected in
`gate.json`; if measured, it is a separate product row.

The current CUDA two-barrier path is measured as implemented. No extra
raygen-only synchronization is introduced. The profiled pass reports at least:

- snapshot and accel build/refit/rebuild;
- `render_raygen_kernel_ms`;
- first-hit and shade kernel phases;
- render execute and overhead;
- readback submit/wait/host time;
- frame p50/p90 and a tail metric (p99 or max);
- peak and delta device memory where available.

## Matrix API and Schema Changes

Extend the matrix case or introduce a dedicated backend-gate case with explicit
fields for:

```text
accel_backend
render_backend
output_profile/readback_payload
raygen_mode
workload_id
repeat_index
```

Do not overload the existing Go2 delivery suite with hidden backend mutation.
Existing suite names and outputs remain backward compatible.

Add to the new P12.4 summary schema:

- workload and repeat identity;
- accel/render backend, output profile, raygen, shadow, delivery, and readback;
- warmup and measured sample counts;
- frame/render phase p50/p90/tail values;
- throughput and memory values;
- hashes or references to gate/environment/parity evidence.

Keep `status/error` for actual run success. Capability and budget decisions live
in their own explicit fields/artifacts rather than overloading runtime status.

Stable CSV tests must assert exact field order and round-trip behavior. Existing
historical matrix CSV readers remain supported or receive an explicit schema
version.

## Implementation Slices

### P12.4a — Gate and Matrix Contract

- add the owner gate schema and validation;
- add explicit backend/workload axes without changing existing suites;
- add capability and result artifact schemas;
- add unit tests for stable fields, unsupported/reserved reasons, failure rows,
  and backward compatibility;
- add a dry-run/list mode that does not require CUDA.

Acceptance: the intended rows and pass rules can be inspected before any GPU
measurement, and missing owner budget fails before running expensive cases.

### P12.4b — Capability and Direct Parity

- audit CUDA against the retained feature list;
- add shared synthetic fixtures and direct CPU/Warp/CUDA channel comparisons;
- enable Go2 or dynamic CUDA rows only after their actual features pass the
  audit and targeted smoke;
- retain explicit unsupported rows for real gaps.

Acceptance: every retained capability is either demonstrated or listed as a
retirement blocker; no missing CUDA row is silently skipped.

### P12.4c — Controlled Measurement

- capture cold setup/compile evidence;
- run warmup and unprofiled steady-state repeats;
- run separate profiled diagnostics;
- write normalized summaries plus raw-artifact hashes;
- rerun correctness and overflow gates on the exact measured commit.

Acceptance: results are reproducible, steady-state samples exclude warmup, and
the current synchronization/ownership costs are visible.

### P12.4d — Owner Decision

Produce a decision request with one of these outcomes:

```text
PASS
  CUDA meets the absolute budget and all retained capabilities.
  Recommend making CUDA default for the validated workload and freezing Warp
  optical implementation changes. Deletion still waits for CUDA-native
  ownership migration plus rollback evidence.

PARTIAL
  CUDA meets the target workload budget but has retained capability gaps.
  Recommend CUDA default only for explicitly validated workloads; retain Warp
  for the listed gaps.

FAIL
  CUDA misses correctness, stability, capability, or absolute budget gates.
  Keep Warp as oracle/fallback and propose focused CUDA-native optimization.
```

No default is changed and no Warp optical code is deleted until the owner
accepts this decision request.

## CUDA-Native Follow-Up Rule

If performance fails, use the phase breakdown to choose one bounded follow-up:

- remove or narrow one of the two global synchronizations through an explicit
  internal async contract;
- migrate scene/LBVH readiness and ownership from Warp to CUDA-native types;
- reuse output/ray buffers;
- fuse kernels only when measurements show launch/buffer traffic is material.

Do not build a Warp/Torch stream bridge. Do not pre-select kernel fusion before
the baseline identifies the bottleneck.

## Non-Goals

- making CUDA default in the matrix-contract implementation commit;
- deleting Warp optical code or tests;
- changing Warp physics or other non-optical Warp ownership;
- adding async/RGB8 support merely to make a row pass;
- path tracing, OptiX, denoising, accumulation, or multisampling;
- a broad backend registry or public `render_backend=` workflow override;
- treating PNG encoding as renderer performance by default;
- changing environment packages during the benchmark stage.

## Required Tests

CPU/unit:

- gate schema accepts complete input and rejects an unset performance budget;
- case expansion is deterministic and preserves existing suite behavior;
- summary/capability/parity schemas have exact field order and stable
  serialization;
- reserved/incompatible backends fail with precise reasons;
- failed rows are retained when `keep_going=True`.

GPU:

- direct channel parity across every common supported backend/profile/shadow
  row;
- exact target-device placement and ambient-device restoration remain covered;
- small end-to-end backend matrix smoke writes the declared artifacts;
- overflow counters remain valid;
- unsupported retained rows appear as blockers rather than skips.

Measurement acceptance:

- exact commit, clean worktree, environment identity, gate hash, commands, and
  raw-summary hashes recorded;
- cold compile, warmup, unprofiled, and profiled passes separated;
- at least the owner-selected frame count and repeats completed;
- the final recommendation is mechanically derived from recorded gate values
  and evidence statuses.

## Review Questions for Claude

1. Is it correct to separate capability, numerical parity, and performance into
   distinct artifacts instead of overloading `matrix_summary.csv.status`?
2. Is `gate.json` sufficiently explicit to prevent post-hoc “fast enough”
   criteria, and which memory/tail fields must be mandatory?
3. Should Go2 static and dynamic rigid rows both block full Warp optical
   retirement, or should either be explicitly removed from retained product
   requirements?
4. Are the proposed unprofiled/profiling split and current two-barrier timing
   treatment valid?
5. Does P12.4b keep capability enablement narrow enough, or should any missing
   capability be deferred to a separate implementation proposal?
6. Are the PASS/PARTIAL/FAIL outcomes strict enough to prevent a premature
   default switch or Warp deletion?

## Recommendation

Approve P12.4 as four reviewed slices. Implement P12.4a first, but do not run
the performance gate until the owner has committed the absolute target budget
and retained capability list. This preserves the CUDA-first decision while
making “fast enough” and “safe to retire Warp” explicit, reproducible, and
separate from relative benchmark enthusiasm.
