Initiative: q54-optical-lab-p12-4-backend-parity-performance-gate
Stage: proposal
Author: codex
Version: v2
Date: 2026-08-10
Status: in_review
Related Files: collab/q54-optical-lab-p12-4-backend-parity-performance-gate__proposal__codex__v1.md; collab/q54-optical-lab-p12-4-backend-parity-performance-gate__decision__owner__v1.md; optics/cuda_direct_light.py; optics/cuda_lbvh.py; optics/device_scene.py; optics/warp_execution.py; tools/optical_pipeline_lab/render_session.py; tools/optical_pipeline_lab/runner.py; tools/optical_pipeline_lab/rgb_pack.py; tools/optical_pipeline_lab/async_readback.py; tools/optical_pipeline_lab/matrix.py; tests/gpu/test_optical_gpu_runtime.py
Owner Summary: Supersede proposal v1. P12.4 is now a CUDA optical replacement program, not only a benchmark-contract stage. CUDA must preserve the current retained Warp optical feature set, pass direct correctness/stability gates, meet the absolute product budget, and be no slower than Warp on matched end-to-end workloads. The work proceeds through a same-process baseline, functional gap closure, measurement-driven CUDA-native optimization, and a final replacement decision. Warp/Torch stream bridging remains forbidden.

---

## Revision Decision

This v2 supersedes proposal v1 for review.

V1 was too weak in two ways:

1. it made relative CUDA-vs-Warp speed informational, even though the owner had
   already required CUDA not to regress from Warp;
2. it treated missing CUDA features mostly as matrix blockers instead of
   planning the work required to close them.

V2 adopts a dual performance gate and turns P12.4 into the complete CUDA
optical replacement stage.

## Exit Condition

P12.4 is complete only when all four gates pass:

```text
Gate F: retained functional parity
Gate C: numerical correctness and stability
Gate R: CUDA no slower than Warp on matched workloads
Gate A: CUDA meets the absolute product budget
```

Formally, for every mandatory GPU row:

```text
median_repeat_p90(CUDA frame_total) <= median_repeat_p90(Warp frame_total)
median_repeat_p90(CUDA render_execute) <= median_repeat_p90(Warp render_execute)
```

and the owner-selected absolute frame-time/throughput/memory limits must pass.

If an observed difference is within a predeclared noise band and the ordering
is unresolved, the row is `inconclusive`, not passed. Increase frames and
repeats until the result resolves or keep Warp.

## Current Gap Audit

### Functional gaps

| Area | Warp | CUDA now | Required P12.4 result |
|---|---|---|---|
| Synthetic static Lab | supported | supported | preserve |
| Go2 static Lab | supported | explicitly rejected | add audited CUDA path |
| Dynamic rigid Lab | supported with current path | scenario rejects CUDA | add CUDA-native update path |
| Physics-published frame | supported with current path | scenario rejects CUDA | add CUDA runtime coverage |
| GPU pinhole raygen | supported | supported | keep direct parity |
| Output profiles | all three | all three | preserve channels and optimize work |
| Triangle shadows | supported | supported | preserve parity/counters |
| Plane shadow occlusion | supported | not traversed by CUDA shade | implement and test |
| Directional/point lights | supported | implemented | broaden parity fixtures |
| RGB/full/none sync | supported | supported | matched end-to-end rows |
| RGB8 pack | Warp kernel | runner rejects CUDA | implement Torch/CUDA-native pack |
| Torch async readback | supported | runner rejects CUDA | support native CUDA readiness/copy |
| Device-only readiness | Warp event | blocking `None` | define native Torch CUDA event |
| Multi-GPU restore | covered | covered | retain regression tests |

Reserved diagnostics/custom payloads and future path tracing/OptiX remain out
of scope because Warp does not provide a retained implementation for them.

### Performance and ownership gaps

The current CUDA camera direct-light path is:

```text
host wait on Warp snapshot/BVH
-> allocate ray buffers
-> raygen
-> allocate and initialize full geometry buffers
-> first-hit
-> torch.cuda.synchronize(target)
-> allocate RGB/intensity/diagnostics
-> shade
-> torch.cuda.synchronize(target)
-> blocking result, ready_event=None
```

Known costs or risks:

1. two target-device-wide synchronizations serialize unrelated work;
2. the geometry barrier is unnecessary for first-hit-to-shade ordering because
   both kernels already use the same target Torch current stream;
3. `render_only` and `rgb_preview` still allocate/initialize full geometry and
   intensity intermediates before filtering channels;
4. ray/output buffers are allocated every frame;
5. Warp-owned snapshot/BVH events are host-synchronized before Torch work;
6. `cuda_lbvh.py` creates native Torch tensors, globally synchronizes, wraps
   them into Warp views, and records a Warp event;
7. RGB8 packing returns to a Warp kernel;
8. the CUDA output cannot participate in a native asynchronous readiness and
   readback chain.

Kernel fusion may help, but it is not assumed to be the first answer.

## Historical Warp Performance Anchors

These figures show the scale the CUDA backend must beat; they do not replace a
matched rerun:

| Workload | Warp result |
|---|---:|
| 960x640 shadow render-only | p50 2.752 ms / p90 2.843 ms |
| 960x640 no-shadow render-only | p50 1.306 ms / p90 1.447 ms |
| 1080p shadow render-only | p50 5.507 ms / p90 5.993 ms |
| 1080p no-shadow render-only | p50 2.471 ms / p90 2.764 ms |
| 1080p shadow RGB8 async | roughly 6.1-7.7 ms/frame in later runs |

The candidate 30 FPS absolute budget (`33.333 ms`) is therefore only a coarse
product floor. It cannot authorize a regression from a roughly 6 ms Warp path.

## Workstream A — Matched Comparator and Frozen Gate

### A1. Gate artifact

Track a gate file before optimization results are known:

```text
benchmarks/results/q54_p12_4/gate.json
```

It contains:

- retained capability list, defaulting to all current Warp optical features;
- target workloads/resolutions;
- minimum warmup, frames, and repeats;
- absolute frame p90/FPS and memory budgets;
- relative limit fixed at `1.00`;
- predeclared noise/inconclusive policy;
- exact primary metrics.

Capability removal requires a separate owner decision and cannot be edited into
this gate after results arrive.

### A2. Same-process paired execution

Add a dedicated backend comparison suite instead of mutating the existing Go2
matrix implicitly.

Required controls:

- construct one deterministic workload definition and feed identical cameras,
  lights, materials, roles, and frame transforms to both backends;
- compile both backends before measured warmup;
- alternate order `Warp->CUDA`, then `CUDA->Warp` across paired repeats;
- same physical GPU, environment, commit, and application process;
- record GPU UUID, clocks/utilization where available, and visible-device map;
- no concurrent benchmark rows;
- unprofiled throughput pass and separate profiled diagnostic pass.

Initial measurements cover the currently common synthetic subset. Their role is
to locate gross CUDA costs early, not to decide replacement.

### A3. Statistics

Default minimum:

```text
warmup frames:       10 per backend after compilation
measured frames:    120 per repeat
paired repeats:       3
inconclusive rerun: 600 frames x 5 repeats
```

Record raw per-frame rows plus repeat p50/p90/p99, mean, standard deviation,
memory, overflow, and paired CUDA/Warp ratios.

## Workstream B — Retained Functional Parity

### B1. Scene and lighting parity

- implement plane occlusion in CUDA shadow traversal;
- directly compare triangle and plane shadow cases with CPU and Warp;
- cover directional and point lights, multiple lights, roles, shadow bias, and
  equal-distance/tie semantics;
- retain traversal/overflow counter contracts.

### B2. Go2 static acceptance

- audit the complete Go2 registry/scene feature set before removing the guard;
- run direct CPU/Warp/CUDA channel parity at a diagnostic resolution;
- add the normal Go2 product/runtime/CSV/PNG smoke for CUDA;
- only then allow CUDA Go2 matrix rows.

### B3. CUDA-native RGB8 and async delivery

- add Torch/CUDA RGB-to-preview-uint8 packing for native Torch tensors;
- return a native Torch CUDA readiness event instead of relying on a Warp event;
- let the readback ring consume Torch tensors/events without requiring Warp;
- test ordering, pinned D2H copies, ring backpressure, and caller-device
  restoration;
- preserve RGB8 numeric parity with the Warp packer.

This is a native delivery path, not a stream bridge.

### B4. Dynamic and physics-published frames

- add CUDA-compatible scenario/backend combinations deliberately;
- preserve physics ownership and Q52 consumer lifetime semantics;
- support CUDA LBVH rebuild first; add refit only if the topology/measurement
  justifies it;
- compare snapshot, accel update, output channels, and frame metadata with Warp;
- include a real multi-frame changing-transform test, not only a static frame
  reused under a dynamic label.

## Workstream C — CUDA-Native Performance Closure

Workstream C is measurement-driven, but the first three candidates are already
justified by code inspection.

### C1. Profile-aware outputs

- pass output intent into CUDA first-hit/shade entry points;
- avoid full NaN/inf geometry initialization for `rgb_preview/render_only`;
- do not allocate/write `intensity` when the requested profile does not retain
  it, unless a fused local computation needs no persistent buffer;
- keep only guaranteed diagnostics for lightweight profiles;
- verify that `direct_light_full` remains unchanged.

Measure allocation, initialization, memory, and kernel deltas separately.

### C2. Remove the internal geometry barrier

- launch raygen, first-hit, and shade on one explicit target Torch stream;
- make the geometry helper internally non-blocking;
- remove the post-first-hit global synchronization from the direct-light path;
- record a native Torch CUDA completion event after the final requested work;
- resolve CUDA-event phase timing after completion without inserting a
  raygen-only/global barrier;
- keep a blocking public compatibility wrapper only where a caller explicitly
  requests synchronous delivery.

### C3. CUDA-native scene and LBVH ownership

- stop wrapping CUDA LBVH Torch tensors back into Warp views for the CUDA
  executor;
- define native scene/BVH buffer ownership for CUDA without duplicating public
  optical semantics;
- produce/consume native Torch readiness events;
- remove per-frame host waits on Warp snapshot/BVH readiness from the final CUDA
  path;
- keep a temporary import/parity adapter only for tests while migration lands,
  not as runtime bridging architecture.

### C4. Buffer reuse and allocation stability

- introduce executor/session-owned reusable ray, geometry, RGB, diagnostics,
  and optional RGB8 buffers keyed by shape/profile/device;
- define aliasing/lifetime rules before reuse;
- verify multiple in-flight readback slots do not observe overwritten buffers;
- measure peak memory and allocation time.

### C5. Conditional fusion

Only after C1-C4 measurements:

- fuse raygen+first-hit if ray buffers/launch overhead remain material;
- fuse first-hit+shade for `render_only/rgb_preview` only if it preserves the
  retained result/readiness contract and beats the non-fused CUDA path;
- keep a directly testable geometry path for parity and non-camera rays.

Do not fuse merely because hand-written CUDA makes it possible.

## Workstream D — Final Capability and Performance Gate

Mandatory matched workload families:

1. asymmetric synthetic static correctness scene;
2. Go2 static 960x640 historical anchor;
3. Go2 static 1920x1080 target;
4. dynamic rigid changing-transform sequence;
5. physics-published-frame sequence.

Mandatory product rows where applicable:

- shadows off/on;
- render-only;
- RGB preview sync;
- RGB8 Torch async ring2;
- direct-light full at diagnostic resolution;
- GPU raygen;
- no PNG/encoder in renderer throughput rows, with a separate artifact smoke.

Every retained row must pass functional/parity/overflow checks before its
performance ratio is considered.

### Final decision states

```text
REPLACE
  All F/C/R/A gates pass. Make CUDA the optical default, freeze Warp optical
  feature work, capture a rollback tag/artifact, and schedule deletion once
  CUDA-native ownership no longer depends on Warp runtime types.

CONTINUE CUDA
  Correctness is sound but one or more capability/performance rows fail or are
  inconclusive. Keep Warp as oracle/default and continue the measured CUDA
  workstream. No bridge.

REASSESS
  A fundamental CUDA design or correctness issue invalidates the approach.
  Requires an explicit owner architecture decision; do not silently fall back
  to indefinite dual-backend maintenance.
```

There is no general “partial replacement” exit for P12.4. Experimental
workload-specific CUDA selection may remain available, but Warp optical
retirement and the project default wait for the full gate.

## Implementation Slices and Review Boundaries

```text
P12.4a  comparator + gate schema + current common-subset baseline
P12.4b1 plane-shadow and lighting parity
P12.4b2 Go2 static CUDA acceptance
P12.4b3 native RGB8/readiness/async delivery
P12.4b4 dynamic and physics-published CUDA path
P12.4c1 profile-aware output work
P12.4c2 single-stream non-blocking internal pipeline
P12.4c3 CUDA-native scene/LBVH ownership
P12.4c4 buffer reuse
P12.4c5 conditional fusion only if triggered
P12.4d  full matched gate + owner replacement decision
```

Each implementation slice gets its own implementation note and Claude review.
Do not batch all functional and synchronization changes into one unreviewable
commit.

## Tests and Evidence

Required throughout:

- CPU-only import/validation tests remain green;
- direct CPU/Warp/CUDA parity for every newly enabled feature;
- all three output profile channel sets and work-elision behavior;
- native Torch readiness event ordering and lifetime tests;
- one- and two-device guard/restoration tests;
- RGB8 exact/one-LSB parity and async ring ordering/backpressure;
- dynamic multi-frame transform and acceleration update parity;
- zero overflow for acceptance fixtures;
- exact commit, clean worktree, environment manifest, commands, raw CSV hashes,
  and paired-repeat summaries;
- full non-GPU gate plus targeted GPU gate at every accepted slice.

## Non-Goals

- Warp/Torch stream bridging;
- deleting or changing Warp physics;
- path tracing, OptiX, accumulation, denoising, or multisampling;
- public API expansion unrelated to backend replacement;
- accepting a slower CUDA backend because it meets 30 FPS;
- dropping retained functionality to improve the CUDA benchmark;
- installing new environment packages during measurement.

## Review Questions for Claude

1. Does the F/C/R/A exit condition correctly encode both “no slower than Warp”
   and the absolute product budget?
2. Is the retained functional list complete, especially plane shadows,
   Go2/static, dynamic/physics, RGB8, async readiness, and diagnostics?
3. Is the paired same-process benchmark protocol strict enough to make a 1.00
   non-regression threshold meaningful?
4. Is the B/C ordering correct, or should native scene/LBVH ownership precede
   dynamic capability work?
5. Does C2 define a true Torch-native asynchronous result without reintroducing
   prohibited Warp/Torch stream bridging?
6. Should C1 profile-aware work precede B2 Go2 acceptance so the first 1080p
   CUDA baseline is not dominated by obviously unnecessary outputs?
7. Are any retained Warp optical capabilities missing from the final matrix?
8. Are the slice boundaries small enough for implementation/review while still
   converging on actual Warp retirement?

## Recommendation

Approve the replacement program, then implement P12.4a first. Use its
common-subset data to prioritize C1/C2 while B1/B2 close the first product
capability blockers. Do not run the final replacement gate until all retained
functional rows exist, and do not accept CUDA until it is both sufficiently
fast in absolute terms and no slower than Warp under matched conditions.
