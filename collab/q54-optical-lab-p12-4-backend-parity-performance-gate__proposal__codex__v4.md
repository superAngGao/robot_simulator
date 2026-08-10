Initiative: q54-optical-lab-p12-4-backend-parity-performance-gate
Stage: proposal
Author: codex
Version: v4
Date: 2026-08-10
Status: in_review
Related Files: collab/q54-optical-lab-p12-4-backend-parity-performance-gate__proposal__codex__v3.md; collab/q54-optical-lab-p12-4-backend-parity-performance-gate__decision__owner__v1.md; collab/q54-optical-lab-p12-3e-cuda-camera-raygen__implementation-note__codex__v1.md; PLAN.md; PROGRESS.md; physics/gpu_engine.py; physics/publish.py; optics/render_api.py; optics/device_channel.py; optics/device_scene.py; optics/cuda_direct_light.py; optics/cuda_lbvh.py; optics/warp_execution.py; tools/optical_pipeline_lab/physics_source.py; tools/optical_pipeline_lab/render_session.py; tools/optical_pipeline_lab/delivery.py; tools/optical_pipeline_lab/async_readback.py; tools/optical_pipeline_lab/rgb_pack.py
Owner Summary: Supersede proposal v3. Compare Warp and CUDA optical implementations only from independently prepared, backend-native ready inputs: Warp arrays for Warp and CUDA Torch tensors for CUDA. Keep product integration as a separate shipping gate from `GpuEngine.step(...)` through host RGB8 image readiness. The benchmark core, canonical workloads, schemas, and the selected backend's native plus physics-to-image regressions become durable infrastructure; P12.4-specific paired decision logic and the losing lane are deleted. The current physics path is explicitly Warp kernels running on CUDA, so CUDA optical may use one measured physics-producer-to-optical-consumer zero-copy CUDA interop boundary, but may not reuse Warp optical scene/BVH preparation. Continuation budgets and the independent `cuda_lbvh.py` retention gate are now concrete.

---

## Revision Decision

This v4 supersedes proposal v3 for review. It resolves the routed review and
owner discussion in four ways:

1. corrects the architecture record: `GpuEngine` is the sole GPU physics
   engine, runs Warp kernels on CUDA, and publishes CUDA-resident Warp arrays
   and Warp CUDA events; there is no separate native-CUDA/Torch physics backend;
2. moves the relative optical comparison boundary to independently prepared
   backend-native ready inputs, so CUDA is not measured through Warp optical
   preparation and Warp is not measured through Torch inputs;
3. separates temporary dual-backend selection machinery from a durable
   benchmark core and durable selected-backend performance monitoring;
4. freezes executable continuation limits and a quantitative Warp-selected
   retention gate for `optics/cuda_lbvh.py`.

V4 retains the accepted requirements that CUDA preserve the current product-
relevant Warp optical capabilities, be numerically correct and stable, be no
slower than Warp, and meet the absolute product budget before replacement.

## Verified Current Architecture

The plan must use the repo's actual terminology rather than treating “CUDA” and
“Warp” as mutually exclusive hardware paths.

Verified facts:

- `PLAN.md` and `PROGRESS.md` record that Q31 removed the former TileLang/CUDA/
  NumPy batch physics backends; `GpuEngine` is the sole GPU physics engine;
- `physics/backends/` contains only the Warp GPU implementation;
- `physics/gpu_engine.py` launches Warp kernels and allocates `wp.array` storage
  on its default `device="cuda:0"`;
- the publish ring copies physics state into CUDA-resident Warp arrays and
  records a Warp `Event`, which wraps a native CUDA event;
- Warp 1.12 exposes native `cuda_stream` and `cuda_event` handles, and
  `wp.to_torch()` creates a zero-copy CUDA tensor view while retaining the Warp
  allocation as the owner;
- the current CUDA optical path is not CUDA-native before render execution: it
  first uses Warp `DeviceOpticalSceneCache` scene-update kernels, Warp snapshot
  buffers/events, and often Warp-wrapped BVH buffers, then host-synchronizes the
  Warp readiness event and creates Torch views for CUDA execution.

Therefore P12.4 may retire the Warp optical data plane, but it does not migrate
or delete Warp GPU physics.

## Completion Gates

P12.4 replacement requires all gates:

```text
Gate F: retained optical functional parity
Gate C: numerical correctness, stability, lifetime, and overflow safety
Gate R-native: CUDA no slower than Warp from native-ready input
Gate R-ship: CUDA no slower than Warp on the actual physics-to-image path
Gate A-ship: CUDA meets the absolute product latency/throughput/memory budget
Gate E: selected-backend consolidation and repository cleanup complete
```

Gate R-native answers whether the optical implementation is strong. Gate
R-ship answers whether it is worth shipping in the current product. A CUDA
kernel win cannot waive a slower shipping path, and a shipping-path loss must
not be misdiagnosed as a kernel loss.

For every mandatory native row:

```text
median_repeat_p90(CUDA primary native metric)
    <= median_repeat_p90(Warp primary native metric)
```

For every mandatory image-producing shipping row:

```text
median_repeat_p90(CUDA sim_step_to_host_image)
    <= median_repeat_p90(Warp sim_step_to_host_image)
```

Device-only and full-channel diagnostic product rows use their predeclared
row-specific completion endpoint (`sim_step_to_device_output` or completed full
host payload) rather than pretending every result is an RGB image. The durable
product SLO centers on host RGB8 image readiness; other profiles remain covered
by F/C and native R0-R2.

An ordering unresolved inside the predeclared noise band is inconclusive and
does not pass.

## Benchmark Boundary Contract

Fairness means identical semantics and equivalent readiness, not identical
device types or implementation steps.

### Canonical logical case

Every workload starts from one dependency-light logical definition containing:

- deterministic local geometry and instance order;
- body-transform sequences;
- camera sequence and frame identity;
- lights, materials, roles, shadows, output profile, and diagnostics;
- expected channel set, numeric tolerances, payload bytes, and delivery policy;
- fixed seeds and content hashes.

The canonical case independently prepares two lanes:

```text
CanonicalCase
  -> prepare_warp_case() -> Warp-owned CUDA allocations
  -> prepare_cuda_case() -> Torch-owned CUDA tensors
```

Neither prepared lane is derived from the other lane's device objects.

### Native-ready input

The relative timer may start only after:

- all inputs are resident on the target CUDA device;
- dtype, shape, stride/layout, precision, and semantic values are validated;
- producer work for the replay input is complete;
- extension loading, JIT compilation, warmup, and initial pool allocation are
  complete;
- the lane can execute without `wp.to_torch`, `wp.from_torch`, host upload,
  host materialization, or cross-lane event/stream waits in the timed region.

Dynamic transform sequences are pre-generated from the canonical case and
uploaded into independent native replay buffers before timing. Selecting the
next already-ready native transform frame is outside the timed region.

### Native output completion

The compute endpoint is the requested native output complete on the lane's
native CUDA timeline:

```text
Warp: Warp output arrays + Warp CUDA completion event
CUDA: Torch CUDA tensors + Torch CUDA completion event
```

Output values are synchronized and staged for parity only in an unprofiled
validation pass outside the primary timed samples.

## Native Benchmark Rows

### R0 — Steady renderer

Input:

```text
native world-space scene
+ fixed normalized BVH topology and primitive order
+ native camera request
```

Timed work:

```text
raygen -> first-hit/shadow traversal -> shade -> native output completion
```

All ray, geometry, shade, diagnostics, and requested output buffers are
preallocated. R0 isolates renderer kernels, launch structure, and memory access.

### R1 — Dynamic native optical transaction

Input:

```text
native body transforms
+ native static local optical scene
+ camera/light/profile request
```

Timed work:

```text
world-space scene update
-> native acceleration refit/rebuild
-> render
-> buffer lease/reclaim bookkeeping
-> native output completion
```

Each lane uses the production acceleration strategy proposed for that optical
implementation. R1 is where scene ownership, acceleration quality, allocation,
completion, and buffer lifetime become part of the relative backend result.

### R2 — Native delivery

Input is the same native-ready boundary as R1. Timed work extends through:

```text
render
-> backend-native RGB8 pack
-> ordered pinned-host D2H
-> contiguous host uint8 RGB image ready
-> final drain for asynchronous batches
```

Both lanes use the same payload shape/bytes, ring depth, ordering,
backpressure/drop policy, and host consumer work. PNG/video encoding and disk
write are excluded from the primary metric and covered by an artifact smoke.

### Cold/setup diagnostics

Record separately, never mix into steady-state R0-R2:

- extension load and cold compile;
- static scene upload;
- initial acceleration build;
- initial buffer-arena allocation;
- first kernel launch and warmup.

Cold costs still require a production packaging decision; exclusion from the
steady-state ratio is not permission for an unbounded first-use stall.

## Acceleration Is a Separate Axis

`optics/cuda_lbvh.py` currently uses Torch/CUDA to build a BVH and wraps its
outputs as Warp arrays. Its gain must not be attributed silently to the Warp
renderer.

Required attribution:

```text
Renderer axis:
  warp_bvh_direct_light
  cuda_direct_light

Acceleration axis:
  fixed_bvh_fixture        # R0 only
  production_cpu/warp_bvh
  cuda_lbvh
```

- R0 fixes topology and primitive order to compare traversal/render kernels.
- R1 lets each candidate use its proposed production acceleration path.
- the independent Warp-selected `cuda_lbvh` retention experiment fixes the
  Warp renderer and changes only the acceleration strategy.

Every report records both renderer and acceleration identifiers.

## Physics Integration and Shipping Gate

### Allowed boundary

The no-bridge rule is now precise:

```text
Forbidden:
  Warp optical lane <-> CUDA optical lane adaptation or fallback
  CUDA optical execution through Warp optical scene/BVH preparation

Allowed and measured:
  Warp-on-CUDA physics producer
      -> selected CUDA optical consumer
  one zero-copy, same-device producer/consumer interop boundary
```

The CUDA optical executor itself consumes Torch-native scene/BVH inputs. A
narrow integration adapter above the executor may create zero-copy Torch views
of the CUDA-resident physics transform arrays, enqueue a device-side wait on the
native CUDA publish event, hold the `GpuPublishedFrame` lease, and return a
native CUDA completion for ring reclaim.

Rules:

- the adapter is the only product location allowed to know both Warp physics
  publication objects and Torch CUDA optical inputs;
- no Warp optical scene-update, BVH, packer, renderer, or readiness helper is
  called by the CUDA product lane;
- physics kernels and numerical state remain unchanged;
- the existing publish-ring ownership and lossless consumer semantics remain
  authoritative;
- zero-copy is the planned path; a D2D-copy fallback or host synchronization
  requires a new owner decision and is not silently introduced;
- C2-C4 receive one joint ownership/stream/lifetime review before dynamic CUDA
  product enablement.

### Integration diagnostic I

Measure separately:

```text
published frame/event available
-> native optical frame input bound
-> required stream dependency enqueued
```

Report view setup, event wait/coordination, lease pressure, and any copy bytes.
I is diagnostic and excluded from R-native, but included in R-ship/A-ship.

### Shipping path S

The permanent product metric starts immediately before the real physics step:

```text
GpuEngine.step(...)
-> publish ring
-> optical consumer acquire
-> scene update
-> acceleration update
-> render
-> RGB8 pack
-> ordered D2H
-> contiguous host uint8 RGB image ready
-> consumer completion recorded and leases safely reclaimable
```

Primary endpoints:

```text
sim_step_to_device_image_ms
sim_step_to_host_image_ms       # primary synchronous latency
publish_to_host_image_ms
batch_first_step_to_final_drain_ms  # primary asynchronous throughput
```

Do not sum overlapping phase timings to obtain the asynchronous total.

### Paired shipping execution

Run independent deterministic product sequences rather than timing two optical
consumers on one already-completed publish event:

```text
repeat 1: Warp product run -> reset -> CUDA product run
repeat 2: CUDA product run -> reset -> Warp product run
repeat 3: Warp product run -> reset -> CUDA product run
```

Reset physics state, action sequence, seed, camera sequence, and environment
between lanes. Validate per-frame transform/output hashes or direct numeric
parity. A separate two-consumer test covers lease/reclaim correctness but is not
a primary performance row.

## Durable Benchmark Infrastructure

P12.4 does not build a disposable benchmark, nor does it retain a permanent
dual-backend product system. The durable and temporary surfaces are explicit.

Recommended durable package:

```text
benchmarks/optical/
  cases.py         # canonical workloads and stable hashes
  schema.py        # versioned metrics/artifact schema
  timing.py        # native and product timing helpers
  validation.py    # channel/counter/image parity
  artifacts.py     # manifest, raw samples, summaries, hashes
  runner.py        # single-lane suite execution
  regression.py    # active baseline comparison
  lanes/
    <selected>.py  # thin adapter to selected production backend
```

During P12.4 only, add the second lane and paired coordinator needed for the
decision. The exact split may be smaller, but the dependency rule is mandatory:

```text
benchmarks -> production modules      allowed
production modules -> benchmarks      forbidden
```

The benchmark package must not copy production kernels or become a product
backend registry.

### Durable assets after selection

Keep:

- canonical cases, transform sequences, hashes, and correctness validators;
- native-ready timing boundaries and versioned result schema;
- environment/build manifest and artifact hashing;
- selected backend R0/R1/R2 regression lane;
- selected backend physics-to-image S regression;
- one small GPU smoke and one full scheduled/release benchmark command;
- approved selected-backend baseline and controlled baseline-update process.

### Temporary P12.4 assets

Remove after the decision:

- the losing benchmark lane;
- P12.4-specific paired-order coordinator and decision report code that cannot
  run a useful single-backend regression;
- migration-only backend selectors, fallback, and interop probes;
- any public/shared API introduced only to compare Warp and CUDA.

Preserve compact gate configuration, summaries, environment manifests, hashes,
and external raw-artifact locations. Git history preserves deleted losing-lane
code.

### Retention gate for the benchmark core

The durable benchmark core itself remains only while:

- CPU-only schema/case/validation tests run in normal CI;
- a documented active-backend GPU smoke is regularly executed;
- a full hardware-pinned run is scheduled or required before releases;
- production code never imports it;
- it has no dependency on the losing optical implementation; if CUDA optical
  is selected, Warp may remain only through the independently retained physics
  producer and its one audited handoff boundary;
- it calls production implementations rather than maintaining alternate
  kernels;
- baseline changes require reviewed justification instead of refreshing away a
  regression.

If those uses disappear, the unused benchmark layer must be deleted.

## Permanent Performance Monitoring

The selected backend's consolidation commit creates the initial approved
baseline containing:

- selected optical implementation and acceleration strategy;
- commit, GPU UUID/model, driver, environment and build manifests;
- canonical workload hashes;
- R0/R1/R2 and S p50/p90/p99, throughput, memory, overflow, dropped frames,
  ring pressure, and raw artifact hashes.

Long-term gates:

```text
native regression:
  current selected-backend R0/R1/R2
  <= approved baseline + predeclared hardware noise allowance

product regression:
  current selected-backend S
  <= approved baseline + predeclared hardware noise allowance

absolute product:
  selected-backend S remains inside absolute latency/FPS/memory budgets

correctness:
  no channel/image mismatch, overflow, dropped frame, or lease error
```

PNG/video encoding remains a separate smoke unless encoder performance becomes
an explicit product SLO.

## Retained Capability Gate

CUDA must preserve all current product-relevant Warp optical behavior:

- Go2 static rendering;
- dynamic rigid and real physics-published changing-frame workflows;
- GPU camera ray generation;
- triangle and plane shadows;
- directional/point lights, multiple lights, roles, bias, and tie semantics;
- direct-light full, RGB preview, and render-only profiles;
- none/RGB/full/RGB8 delivery;
- synchronous and ordered Torch async delivery where Warp supports it;
- target-device restoration, readiness, result lifetime, ring backpressure,
  traversal counters, overflow behavior, CSV, and artifact contracts.

Capability removal requires a separate owner product-scope decision and cannot
be used to make CUDA pass.

## CUDA Closure Work

### C0 — Joint ownership design

Before C2-C4 implementation, freeze one design for:

- Torch-native static scene and per-frame transform input ownership;
- CUDA scene/LBVH/result buffer arenas;
- Torch current stream and native completion event;
- physics published-frame lease duration;
- scene-update completion and physics slot reclaim;
- async readback in-flight result leases;
- exception, resize, device-change, and final-drain cleanup.

### C1 — Profile-aware work

- avoid full geometry/intermediate allocation and initialization for lighter
  profiles;
- do not retain/write unrequested intensity or geometry channels;
- preserve direct-light full and required diagnostics.

### C2 — Native asynchronous execution

- launch scene update, raygen, first-hit, shade, and pack on the intended Torch
  CUDA stream;
- remove post-first-hit and post-shade target-wide synchronization from the
  internal path;
- return a Torch CUDA completion event;
- keep blocking behavior only at an explicitly synchronous boundary.

### C3 — CUDA-native scene and acceleration

- build static optical scene as Torch-owned CUDA tensors;
- update world geometry directly from native body transforms;
- make CUDA LBVH output Torch-native for the CUDA lane;
- remove Warp snapshot/BVH waits and Warp wrapping from the CUDA product path;
- support dynamic rebuild first and refit only when measured/implemented.

### C4 — Buffer arenas and leases

- reuse ray, geometry, RGB, diagnostics, RGB8, scene-update, and acceleration
  buffers by shape/profile/device;
- prevent reuse until renderer and any async readback consumer complete;
- cover ring pressure, resize, profile changes, exceptions, and shutdown.

### C5 — Conditional fusion

Fuse only if R0/R1 evidence shows launch or intermediate-buffer cost remains
material after C1-C4. Preserve a directly testable non-camera geometry path and
the required output/lifetime contract.

Dependencies:

- B3 CUDA RGB8/async delivery waits for C0, C2, and C4;
- B4 dynamic/physics product enablement waits for C0, C2, C3, and C4;
- no temporary Warp optical scene/BVH bridge is allowed to unblock B3/B4.

## Workload Matrix

Mandatory native and shipping families:

1. asymmetric synthetic correctness scene;
2. Go2 static 960x640 historical anchor;
3. Go2 static 1920x1080 target;
4. deterministic dynamic rigid transform replay;
5. real `GpuEngine.step(...)` physics-to-image sequence.

Mandatory rows where applicable:

- shadows off/on;
- render-only;
- RGB preview synchronous;
- RGB8 ordered async ring depth 2;
- direct-light full at diagnostic resolution;
- GPU camera raygen;
- device-only native completion;
- host-image readiness;
- separate artifact smoke.

Every performance row requires prior direct correctness, counter, overflow, and
payload checks.

## Comparator and Statistics Controls

Freeze `benchmarks/results/q54_p12_4/gate.json` before inspecting results. It
contains:

- exact case hashes and mandatory row IDs;
- R0/R1/R2/I/S boundary versions;
- primary metrics and absolute budgets;
- warmup, frames, repeats, and alternating order;
- relative limit `1.00`;
- noise band and inconclusive procedure;
- continuation budgets and hard failure behavior;
- GPU/environment requirements and memory limits.

Default measurement minimum:

```text
warmup frames:       10 per lane after compile/setup
measured frames:    120 per repeat
paired repeats:       3
inconclusive rerun: 600 frames x 5 repeats
```

Other controls:

- same clean commit, process, physical GPU, visible-device map, workload, and
  environment for native paired runs;
- independent reset product runs with alternating order for S;
- no concurrent rows;
- record GPU UUID and clock/power state where available;
- run unprofiled primary and profiled diagnostic passes separately;
- run an empty/copy calibration to detect asymmetric runner/timer overhead;
- keep bulky raw samples outside git when appropriate and track hashes plus
  stable artifact locations.

Historical Warp measurements remain scale anchors, not pass thresholds.

## Bounded Continuation Rule

The following defaults are frozen in `gate.json` before the first baseline:

```json
{
  "max_rerun_campaigns_per_row": 2,
  "max_cuda_optimization_slices": 5,
  "max_correctness_repair_slices_per_failure": 1,
  "continuation_deadline": "2026-10-15T23:59:59+08:00",
  "inconclusive_resolution": "warp_selected",
  "default_terminal_branch": "warp_selected"
}
```

Definitions:

- the initial gate run is not a rerun campaign;
- one optimization slice is one predeclared hypothesis, regardless of how many
  commits implement it; splitting commits cannot evade the limit;
- C1-C5 are the maximum five planned optimization hypotheses;
- the first reached limit among reruns, optimization slices, or deadline forces
  a terminal decision.

Hard failure conditions:

- a retained capability requires a prohibited Warp-optical/CUDA-optical bridge;
- physics-to-CUDA handoff cannot preserve device ordering, lease safety, or
  slot reclaim without an unapproved host synchronization/copy path;
- a mandatory workload OOMs inside the frozen memory budget;
- a numeric corruption, wrong-frame result, buffer overwrite, or use-after-
  reclaim remains after one dedicated repair slice;
- a mandatory row remains inconclusive after two full rerun campaigns;
- C1-C5 are exhausted and any F/C/R-native/R-ship/A-ship gate still fails.

On hard failure or budget exhaustion, execute the Warp-selected cleanup branch.
A future CUDA attempt requires a new owner initiative after P12.4 cleanup; it
cannot keep the current dual-backend tree alive by repeatedly extending the
same initiative. Changing the proposed deadline requires an owner decision
before the first measured baseline, not after results are known.

## Independent `cuda_lbvh.py` Retention Gate

If Warp is selected, `optics/cuda_lbvh.py` is evaluated independently with the
renderer fixed to Warp:

```text
baseline:
  Warp renderer + production CPU/Warp BVH build/refit

candidate:
  Warp renderer + CUDA LBVH build/rebuild
```

Use matched static cold/warm, dynamic transform replay, and physics-published S
workloads. Record acceleration update p90, native R1, shipping S, peak memory,
cold compile/load, correctness, topology/primitive coverage, and overflow.

The module remains only if all conditions pass:

1. all correctness, stability, primitive coverage, lifetime, and overflow
   checks pass;
2. at least one mandatory dynamic acceleration-update row improves by at least
   20% (`candidate_ratio <= 0.80`);
3. at least one mandatory dynamic shipping S row improves by at least 10%
   (`candidate_ratio <= 0.90`);
4. no mandatory R1 or S row regresses outside the predeclared noise band;
5. peak memory stays inside the frozen budget;
6. production CUDA source moves out of
   `examples/cuda_lbvh_extension_spike.py` into an owned production module;
7. unneeded Torch-to-Warp wrappers, global synchronization, spike naming, and
   experimental loading surfaces are removed;
8. a named production owner, static/dynamic GPU tests, packaging/build
   manifest, and regular performance coverage exist.

An inconclusive retention row is a failure to retain. If any condition fails,
delete `cuda_lbvh.py` and its CUDA extension surface with the CUDA optical lane.

If CUDA optical is selected, `cuda_lbvh.py` must instead become CUDA-native and
stop returning Warp-wrapped product outputs; its performance is already covered
by R1/R2/S.

## Terminal Repository States

P12.4 completes only after one selected-backend cleanup branch lands.

### CUDA selected

- make CUDA the sole supported production optical implementation;
- remove the Warp optical executor, scene-update/cache implementation where it
  has no non-optical use, Warp optical RGB8/readback wiring, fallback, product
  selectors, exports, examples, and Warp-only optical tests;
- remove Warp conversions from optical channel helpers when no production
  optical path needs them;
- retain Warp GPU physics and its publish ring;
- keep only the narrow measured physics-publication-to-CUDA-optical adapter;
- make scene, acceleration, result, RGB8, and delivery ownership CUDA-native;
- delete the temporary Warp benchmark lane and P12.4-only paired decision code;
- retain the durable CUDA R0/R1/R2 and physics-to-image S regressions.

### Warp selected

- make Warp the sole supported production optical implementation;
- remove CUDA optical raygen/traversal/shade, CUDA-native scene/delivery code,
  extension loading, product guards/selectors, examples, exports, and CUDA-only
  optical tests;
- remove shared API surface introduced only for the failed CUDA experiment;
- delete the temporary CUDA benchmark lane and P12.4-only paired decision code;
- apply the independent `cuda_lbvh.py` gate and retain or delete it exactly as
  specified;
- retain the durable Warp R0/R1/R2 and physics-to-image S regressions.

There is no permanent dual-backend or automatic-fallback terminal state.

## Cleanup Acceptance Checks

- reviewed commit has a clean worktree and `git diff --check` passes;
- full non-GPU and relevant GPU suites pass in the verified environment;
- CPU-only imports remain safe;
- package exports, CLI help, docs, examples, environment manifests, and product
  configuration expose only the selected optical implementation;
- repository search finds no losing executor factory, fallback, selector,
  packer, event adapter, stale marker, extension loader, or test-only product
  branch;
- repository search finds no product import of `benchmarks/optical`;
- every retained backend-specific module has a production or active regression
  caller, owner, tests, and measured reason to exist;
- durable benchmarks run without installing or importing the losing backend;
- tracked evidence is compact and reproducible from its manifest;
- `PROGRESS.md`, `REFLECTIONS.md`, `OPEN_QUESTIONS.md`, and `MANIFEST.md` are
  updated according to their documented milestone rules;
- completed collaboration history may be moved to `collab/archive/`.

## Implementation and Review Slices

```text
P12.4a0 verified architecture inventory + cleanup ledger
P12.4a1 canonical cases + native/product boundary schema + frozen gate
P12.4a2 durable benchmark core + independent native preparation
P12.4a3 Warp/CUDA R0-R2 lanes + acceleration attribution
P12.4a4 comparator validity review + common-subset baseline

P12.4b1 plane-shadow and lighting parity
P12.4b2 Go2 static CUDA acceptance
P12.4c0 joint ownership/stream/lifetime design
P12.4c1 profile-aware work
P12.4c2 native asynchronous execution
P12.4c3 CUDA-native scene/LBVH ownership
P12.4c4 buffer arenas and leases
P12.4b3 CUDA-native RGB8/readiness/async delivery
P12.4b4 dynamic and physics-published CUDA product path
P12.4c5 conditional fusion, only if triggered

P12.4d native R0-R2 + shipping S full F/C/R/A decision gate
P12.4e selected-backend cleanup + durable regression promotion
```

Every slice gets an implementation note and Claude review. P12.4a4 must be
approved before benchmark results drive optimization. C0 must be approved
before C2-C4 implementation. P12.4e is a blocking part of completion.

## Test and Evidence Requirements

- CPU-only tests for canonical cases, schema, artifact parsing, and baseline
  comparison;
- direct verification that independently prepared Warp/Torch inputs match the
  canonical values, shapes, precision, and layout contract;
- guards/tests proving no conversion, upload, hidden synchronization, or
  cross-lane call occurs inside R0-R2 timing boundaries;
- direct CPU/Warp/CUDA output parity before performance acceptance;
- all profiles, shadows, lights, roles, tie behavior, counters, and overflow;
- native completion ordering and caller-device restoration;
- buffer arena reuse, resize, profile/device change, in-flight readback,
  backpressure, exception cleanup, shutdown, and final drain;
- dynamic replay equality plus real physics reset/action/frame parity;
- one- and two-device coverage where hardware is available;
- separate unprofiled primary and profiled diagnostic artifacts;
- exact commit, clean worktree, commands, environment/build manifest, GPU
  identity, case hashes, raw-result hashes, and summary ratios;
- final cleanup search/dependency checks for the selected terminal branch.

## Durable Documentation Policy

This proposal remains in review, so it does not yet rewrite durable project
status documents. After owner acceptance:

- record the native-ready comparison and permanent selected-path monitoring
  decisions in `REFLECTIONS.md`;
- mark P12.4 active with its bounded continuation policy in `PROGRESS.md`;
- add only genuinely long-lived unresolved questions to `OPEN_QUESTIONS.md`;
- update `MANIFEST.md` when the selected backend and permanent performance
  monitoring become a milestone.

## References Consulted

- `PLAN.md` / `PROGRESS.md` Q31 decisions: `GpuEngine` with Warp kernels is the
  sole GPU physics engine; former native CUDA batch physics backends were
  removed to avoid duplicate implementations.
- `physics/gpu_engine.py` and `physics/publish.py`: CUDA-resident Warp arrays,
  device-side stream waits, published-ring leases, completion, and slot reclaim
  are existing production semantics.
- Warp 1.12 installed source: Warp CUDA arrays convert to Torch without a copy,
  and Warp streams/events expose native CUDA handles suitable for a narrowly
  reviewed producer/consumer interop boundary.
- Existing `optics/render_api.py`: semantic requests, output profiles, delivery
  policies, and result schemas can stay CPU-safe and implementation-neutral.
- Repo Newton/MuJoCo Warp notes in `REFERENCES.md`: hot GPU phases should use
  native flat storage and fixed-phase execution rather than per-component
  conversions in the hot loop.

## Review Questions for Claude

1. Does R-native now begin at a genuinely fair boundary: independent Warp
   arrays versus independent Torch CUDA tensors with identical semantics and
   readiness?
2. Do R0/R1/R2 separate renderer, dynamic optical transaction, and native
   delivery without omitting work that one backend must perform?
3. Does the acceleration axis prevent CUDA LBVH gains or costs from being
   misattributed to the Warp renderer?
4. Is the verified Warp-on-CUDA physics description accurate, including the
   fact that P12.4 retires Warp optical rather than Warp physics?
5. Is the single physics-producer-to-CUDA-optical zero-copy interop boundary
   sufficiently narrow, owned, measurable, and compatible with publish-ring
   lease/reclaim semantics?
6. Are S start/end points correct for both synchronous latency and asynchronous
   batch throughput, especially host image readiness and final drain?
7. Does the durable-core/temporary-lane split keep long-term regression value
   without retaining the losing backend or creating a product registry?
8. Are the continuation deadline, two rerun campaigns, five optimization
   slices, hard failures, and Warp default terminal branch executable and hard
   to evade?
9. Is the quantitative `cuda_lbvh.py` retention gate strict and complete enough
   for a Warp-selected outcome?
10. Do either terminal cleanup branch or the permanent selected-backend
    physics-to-image monitor leave an unowned dependency or dead abstraction?

## Recommendation

Approve v4 before implementation. Build the durable benchmark core and native
input preparation first, validate R0-R2 before using results, then close CUDA
capability and ownership gaps within the frozen continuation budget. Decide on
both native optical strength and the real physics-to-image shipping path. P12.4
is complete only after the losing lane and product implementation are removed
and the selected backend's native plus physics-to-image regressions are active.
