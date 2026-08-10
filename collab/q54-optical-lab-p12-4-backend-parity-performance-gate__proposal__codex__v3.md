Initiative: q54-optical-lab-p12-4-backend-parity-performance-gate
Stage: proposal
Author: codex
Version: v3
Date: 2026-08-10
Status: in_review
Related Files: collab/q54-optical-lab-p12-4-backend-parity-performance-gate__proposal__codex__v2.md; collab/q54-optical-lab-p12-4-backend-parity-performance-gate__decision__owner__v1.md; collab/q54-optical-lab-p12-3e-cuda-camera-raygen__implementation-note__codex__v1.md; optics/render_api.py; optics/execution.py; optics/device_channel.py; optics/cuda_direct_light.py; optics/cuda_lbvh.py; optics/device_scene.py; optics/warp_execution.py; tools/optical_pipeline_lab/delivery.py; tools/optical_pipeline_lab/async_readback.py; tools/optical_pipeline_lab/rgb_pack.py; tools/optical_pipeline_lab/render_session.py; tools/optical_pipeline_lab/runner.py; physics/gpu_engine.py
Owner Summary: Supersede proposal v2. Preserve the F/C/R/A replacement rule, but do not judge CUDA through Warp-oriented delivery plumbing. P12.4 first builds a host-neutral comparator with isolated Warp-native and CUDA-native lanes and measures native compute, render transaction, and product delivery separately. P12.4 has only two clean terminal states: one production optical implementation remains, and the losing implementation, fallback branches, adapters, tests, flags, and unowned benchmark scaffolding are removed. An open-ended dual-backend repo is not an accepted completion state.

---

## Revision Decision

This v3 supersedes proposal v2 for review. It keeps the accepted replacement
rule:

```text
Gate F: retained functional parity
Gate C: numerical correctness and stability
Gate R: CUDA no slower than Warp on matched workloads
Gate A: CUDA meets the absolute product budget
```

V3 corrects two architectural risks left in v2:

1. the current Lab delivery path is not a neutral foundation for measuring a
   CUDA renderer because its readiness, RGB8 packing, and some scene/BVH
   ownership were designed around Warp;
2. v2 described CUDA replacement cleanup, but did not require an equally clean
   outcome if CUDA fails and Warp remains.

The comparison must share semantics and workload, not physical buffers,
streams, events, or backend-specific delivery code.

## Problem Statement

The current CUDA product-path timing is a mixture:

```text
CUDA kernels
+ Warp-owned scene/BVH input waits
+ target-wide synchronization barriers
+ per-frame allocation/initialization
+ Warp-oriented RGB8/readiness/delivery assumptions
```

The Warp renderer uses its native path. Comparing those totals alone cannot
answer whether CUDA compute is weak or whether the surrounding data path is
penalizing it. Optimizing against that mixed number can improve an adapter or a
barrier while leaving the renderer bottleneck untouched.

The opposite mistake is also unacceptable: an isolated CUDA kernel may beat a
Warp kernel while the native CUDA render transaction or delivered product path
is slower. P12.4 therefore needs diagnostic layers and a final product layer.

## Required Clean-Repository Invariant

P12.4 implementation is reversible until the final owner decision. Its final
cleanup is not optional.

At P12.4 completion:

- exactly one optical implementation is on the supported production path;
- there is no automatic or silent runtime fallback to the losing
  implementation;
- product code has no cross-implementation buffer, stream, or event bridge;
- no public API exists only to preserve the comparison experiment;
- the losing executor, native pack/readback path, flags, branches, tests, and
  exports are deleted unless another production subsystem has a separately
  documented use for a component;
- every retained backend-specific module has a production caller, an owner,
  tests, and a measured reason to exist;
- benchmark evidence is preserved as compact manifests, summaries, hashes, and
  artifact locations, not as an indefinitely active second runtime;
- CPU optical references may remain for correctness, but they are not a hidden
  product fallback;
- Warp physics is outside this decision and must not be deleted or coupled to
  optical cleanup.

Git history and the recorded benchmark artifacts provide rollback. Dead
runtime code is not retained as a rollback mechanism.

## Architecture Boundary

### Shared semantic control plane

The following concepts are shared because they describe product meaning rather
than a GPU implementation:

- logical scene/snapshot, camera, light, material, role, and output-profile
  inputs;
- `RenderRequest`, `DeliveryRequest`, delivery policy, output channel names,
  timing schema, diagnostics, and artifact schema;
- frame ordering, backpressure policy, ring depth, payload bytes, and consumer
  behavior;
- correctness tolerances and benchmark case definitions.

`optics/render_api.py` remains CPU-safe. Its `RenderBackend` describes rendering
algorithms such as direct light versus future path tracing; it must not be
overloaded with the temporary Warp-versus-CUDA implementation choice.

### Backend-native data plane

Each lane owns its complete device-side chain:

```text
Warp lane
  Warp scene/BVH -> Warp buffers -> Warp stream/events
  -> Warp RGB8/readback preparation -> common host result schema

CUDA lane
  Torch/CUDA scene/BVH -> Torch buffers -> Torch CUDA stream/events
  -> CUDA-native RGB8/readback preparation -> common host result schema
```

Rules:

- no lane calls the other lane's executor, packer, event waiter, or allocator;
- no measured region converts one backend's device storage into the other's
  view;
- no shared physical device buffer is required for fairness;
- backend-native completion types do not need a common physical event class;
  the host comparator normalizes their completed observations;
- shared delivery orchestration may express policy, ordering, and result
  metadata, but buffer acquisition, completion, and copy scheduling stay in
  the native lane;
- a generic runtime backend registry is not introduced merely for P12.4.

The robust published-ring borrow/completion semantics in
`physics/gpu_engine.py` are a design reference for leases and reclaim points.
Its Warp buffers/events are not reused as the optical CUDA implementation.

## Three-Layer Measurement Contract

### L0 — Native compute floor

Purpose: identify kernel and memory-access capability independently of product
delivery.

Boundary:

- use one immutable logical fixture and one normalized primitive/BVH topology;
- upload equivalent native arrays to each lane before timing;
- compile, warm up, and preallocate all buffers before timing;
- exclude scene build, allocation, readback, image conversion, and encoding;
- time ray generation, first-hit, shadow traversal, shade, and total device work
  with each backend's native GPU events;
- run an unprofiled throughput pass separately from the profiled diagnostic
  pass.

L0 is diagnostic. It tells us where a renderer loses, but it does not authorize
replacement by itself.

### L1 — Native render transaction

Purpose: measure the usable executor rather than only kernels.

Boundary:

- include native buffer-arena acquire/release, requested-profile allocation or
  reuse, render enqueue, completion recording, and result/lease lifetime;
- exclude host readback, image conversion, and encoding;
- measure synchronous single-frame latency with one terminal native completion
  wait outside the submitted work;
- also measure multi-frame enqueue/completion throughput without adding a
  per-phase or per-frame global device synchronization;
- use common host monotonic timing around equivalent completion boundaries;
  native event phases remain diagnostic columns.

Gate R applies to the matched L1 primary metrics. If L0 wins and L1 loses, the
problem is ownership, allocation, or synchronization rather than core kernels.

### L2 — Native product path

Purpose: decide which implementation the repo should keep.

Boundary:

- include the applicable snapshot/scene update, acceleration update, render,
  RGB8 pack, readback, ordered delivery, and final drain;
- use each backend's native data plane while enforcing the same payload,
  delivery policy, ring depth, frame sequence, backpressure behavior, sink, and
  consumer work;
- measure synchronous latency separately from asynchronous batch throughput and
  delivered-frame latency;
- keep PNG/encoder cost out of renderer throughput rows and cover it in a
  separate artifact smoke;
- the primary asynchronous total is wall time from first submit through final
  drain, not the sum of overlapping phase durations.

Gate F/C/R/A applies to L2. If L0 and L1 pass but L2 fails, CUDA delivery or
scene ownership must be fixed; the renderer is not blamed and the gate is not
waived.

## Comparator Validity Requirements

The P12.4 comparator is a host-neutral benchmark tool, not a new product
distribution layer.

Recommended isolation under `benchmarks/`:

```text
benchmarks/optical_backend_compare/
  cases.py        # immutable logical workloads and expected semantics
  schema.py       # backend-neutral measurements/results only
  warp_lane.py    # Warp-native setup, execute, completion, delivery
  cuda_lane.py    # CUDA-native setup, execute, completion, delivery
  runner.py       # paired order, warmup, collection, validation
```

The exact file split may be reduced during implementation, but lane isolation
and dependency direction are mandatory.

Required controls:

- same clean commit, process, physical GPU, visible-device map, environment,
  camera/frame sequence, scene values, output profile, shadows, and delivery
  semantics;
- compile both lanes before measured warmup;
- alternate paired order `Warp -> CUDA`, then `CUDA -> Warp`;
- no concurrent benchmark rows;
- predeclare frames, repeats, noise band, rerun rule, primary metrics, memory
  limits, and absolute budgets in `benchmarks/results/q54_p12_4/gate.json`
  before results are inspected;
- record GPU UUID, relevant clock/power state when available, package/build
  manifest, commit, commands, raw-result hashes, and artifact locations;
- run a minimal empty/copy calibration case to detect asymmetric harness,
  completion, or timer overhead;
- assert that diagnostic profiling is disabled in the primary pass and does
  not insert hidden synchronization;
- validate direct output parity before accepting any performance row.

Historical Warp numbers remain scale anchors only. The pass/fail comparison is
always a paired rerun.

## Workstream A — Trustworthy Measurement Spine

### P12.4a0 — Inventory and cleanup ledger

Before code changes, classify every affected symbol/file as:

```text
shared-semantic
warp-native
cuda-native
temporary-comparator
candidate-for-deletion
independent-production-use
```

Track the classification and expected final action in the P12.4 implementation
notes. New cross-lane imports are a blocking review finding.

Special case: `optics/cuda_lbvh.py` is not retained or deleted based on its
name. If Warp is selected, it may remain only if the production Warp optical
path uses it, its Torch-to-Warp ownership is explicitly documented, and it
passes a standalone benefit/maintenance gate. Otherwise it is deleted. If CUDA
is selected, its CUDA-native output must stop being wrapped back into Warp for
the product path.

### P12.4a1 — Frozen gate and logical cases

- add the pre-result `gate.json` and compact schema documentation;
- define retained workload/capability rows and immutable logical inputs;
- define L0/L1/L2 boundaries and primary metrics exactly;
- keep bulky raw samples outside git when appropriate, with tracked hashes and
  stable artifact locations.

### P12.4a2 — Minimal CUDA-native measurement spine

Before judging CUDA performance:

- provide Torch/CUDA scene and BVH inputs for the comparator without a
  timed-region Warp view or host wait;
- preallocate the buffers needed for L0;
- record a native Torch CUDA completion event;
- remove global synchronization from the measured internal path;
- keep an explicit terminal wait only at the L1 synchronous measurement
  boundary.

This is benchmark-enabling work, not permission to switch the product default.

### P12.4a3 — Isolated native lanes

- construct Warp-native and CUDA-native resources outside measured regions;
- add lane-specific execute, completion, and delivery hooks;
- normalize only host observations and measurement records;
- forbid product code from importing the comparator package;
- add dependency tests or static checks that prevent cross-lane imports.

### P12.4a4 — Baseline and validity review

- run L0/L1 on the current common functional subset;
- run the existing Warp-native L2 rows as a baseline anchor;
- demonstrate comparator calibration and direct output parity;
- obtain a Claude review of benchmark validity before using results to choose
  optimization work.

## Workstream B — Retained Capability Closure

Preserve v2's complete capability matrix:

- plane and triangle shadow occlusion;
- directional/point lights, multiple lights, roles, bias, and tie semantics;
- Go2 static rendering;
- dynamic rigid and physics-published changing-frame workflows;
- all three output profiles and required diagnostics/counters;
- none/RGB/full/RGB8 delivery modes;
- synchronous and ordered Torch async delivery;
- target-device, readiness, buffer lifetime, overflow, CSV, and artifact
  contracts.

CUDA RGB8/readback must be CUDA-native. Dynamic CUDA scene/BVH updates must not
route through a Warp executor or stream bridge.

## Workstream C — Measurement-Driven CUDA Closure

The current justified candidates remain:

1. profile-aware output allocation and work elision;
2. one explicit Torch stream, non-blocking internal raygen/first-hit/shade, and
   one native completion event;
3. CUDA-native scene/LBVH ownership;
4. executor/session-owned buffer arenas with explicit lease/reclaim semantics;
5. fusion only after L0/L1 evidence shows launch/intermediate-buffer cost is
   material.

Each optimization must report which layer changed:

```text
L0 change -> kernel or device-memory work
L1 change -> allocation, synchronization, completion, or lifetime
L2 change -> scene update, packing, readback, delivery, or product orchestration
```

An improvement in one layer cannot be used to claim improvement in another.

C2-C4 require one joint ownership/lifetime design review before their first
implementation commit. They may land as small reviewed slices, but they must
not create temporary stream bridges or buffer lifetimes that a later slice is
expected to repair. In particular:

- CUDA-native completion and the buffer-arena lease/reclaim rules must exist
  before B3 enables ordered async delivery;
- CUDA-native scene/LBVH ownership must exist before B4 enables dynamic and
  physics-published CUDA product paths;
- B1/B2 may proceed independently after P12.4a4 because they do not require
  those temporary bridges.

## Workstream D — Full Decision Gate

Run the mandatory paired workload families from v2:

1. asymmetric synthetic static correctness scene;
2. Go2 static 960x640 historical anchor;
3. Go2 static 1920x1080 target;
4. dynamic rigid changing-transform sequence;
5. physics-published-frame sequence.

Cover shadows off/on, render-only, RGB preview sync, RGB8 Torch async ring2,
direct-light full at diagnostic resolution, GPU raygen, and the applicable
artifact smoke.

For every mandatory row:

```text
functional/correctness/stability checks pass
median_repeat_p90(CUDA L1 primary) <= median_repeat_p90(Warp L1 primary)
median_repeat_p90(CUDA L2 primary) <= median_repeat_p90(Warp L2 primary)
absolute product and memory budgets pass
```

L0 phase ratios are mandatory diagnostic evidence but are not an independent
replacement veto if the complete L1 and L2 contracts pass. Any L0 regression
must still be explained so a product win is not based on accidental omitted
work.

An unresolved result inside the predeclared noise band is `inconclusive`, not a
pass.

## Workstream E — Mandatory Consolidation

P12.4 does not finish at the benchmark decision. It finishes after one of the
following cleanup branches lands and is reviewed.

### Terminal state 1 — CUDA selected

- make CUDA the only supported production optical implementation;
- remove the Warp optical executor, Warp optical pack/readback wiring, optical
  implementation selectors/fallback branches, and Warp-only optical tests;
- remove Warp conversion from optical device-channel helpers where no remaining
  production optical path needs it;
- make scene/LBVH/output ownership CUDA-native and remove temporary parity
  adapters from product code;
- remove the Warp comparator lane and paired runner unless the owner explicitly
  funds continuing dual-backend support; retain compact evidence and git
  history instead;
- keep Warp physics unchanged;
- update exports, CLI help, docs, examples, environment manifests,
  `PROGRESS.md`, `REFLECTIONS.md`, `MANIFEST.md` when milestone criteria apply,
  and close or rewrite affected `OPEN_QUESTIONS.md` entries.

### Terminal state 2 — Warp selected

- restore Warp as the only supported production optical implementation;
- remove the CUDA optical executor/raygen/shade code, CUDA-only product guards
  and flags, native CUDA delivery code, CUDA-only tests, and unused extension
  build/loading surface;
- remove shared API fields or branches introduced solely for the failed CUDA
  experiment;
- decide `cuda_lbvh.py` by the independent-production-use rule from P12.4a0;
  document and test a retained use, or delete it;
- remove the CUDA comparator lane and paired runner after preserving compact
  evidence, unless another approved initiative actively consumes them;
- update the same project docs to state that Warp remains the optical backend
  and why CUDA was rejected.

### No third terminal state

`CONTINUE CUDA` is an active, bounded P12.4 state, not completion. If work is
paused or abandoned before the gates pass, the owner must choose a new bounded
continuation slice or execute the Warp-selected cleanup branch. Indefinite
dual-backend product maintenance, silent fallback, and abandoned experimental
flags are not accepted outcomes.

## Cleanup Acceptance Checks

The final consolidation review must include:

- `git status --short --branch` is clean at the reviewed commit;
- `git diff --check` passes before commit;
- full non-GPU and relevant GPU suites pass in the verified environment;
- CPU-only imports remain safe;
- package exports, CLI choices, docs, examples, and manifests mention only the
  supported production optical implementation;
- repository search finds no losing-executor factory, fallback, selector,
  packer, event adapter, stale test marker, or dead extension loader;
- repository search finds no product import of
  `benchmarks/optical_backend_compare`;
- a dependency audit confirms no surviving optical module imports the losing
  lane;
- every remaining backend-specific file is reachable from a production or
  explicitly retained regression entry point;
- tracked benchmark artifacts are compact and reproducible from their manifest;
- collaboration history is retained and may be moved to `collab/archive/` after
  the initiative is complete.

## Slice and Review Boundaries

```text
P12.4a0 inventory + cleanup ledger
P12.4a1 frozen gate + logical case schema
P12.4a2 minimal CUDA-native measurement spine
P12.4a3 neutral runner + isolated native lanes
P12.4a4 comparator validity + common-subset L0/L1 baseline
P12.4b1 plane-shadow and lighting parity
P12.4b2 Go2 static CUDA acceptance
P12.4b3 CUDA-native RGB8/readiness/async delivery
P12.4b4 dynamic and physics-published CUDA path
P12.4c1 profile-aware work
P12.4c2 native asynchronous pipeline
P12.4c3 CUDA-native scene/LBVH ownership
P12.4c4 buffer arena and leases
P12.4c5 conditional fusion, only if triggered
P12.4d full paired F/C/R/A gate
P12.4e selected-backend cleanup and documentation
```

Every slice gets an implementation note and Claude review. P12.4a4 must be
approved before comparator results drive optimization decisions. P12.4e is a
blocking part of completion, not follow-up housekeeping.

## Test and Evidence Additions

In addition to v2's parity matrix:

- comparator schema/unit tests must run CPU-only;
- lane-isolation tests must fail on a cross-lane runtime import;
- logical-case tests must prove both lanes receive identical scene/camera/light
  values and, for L0, identical normalized BVH topology and primitive order;
- buffer-arena tests must cover in-flight readback, backpressure, lease release,
  reuse, resize, profile change, exception cleanup, and wrong-device calls;
- completion tests must cover caller-device restoration and ensure timing does
  not insert an unrequested global synchronization;
- L2 ordered delivery must cover ring depths 1 and 2 and final drain behavior;
- primary unprofiled metrics and diagnostic profiled metrics must be separate
  artifacts;
- a cleanup-focused final test/repository-search checklist is required for
  whichever terminal branch is selected.

## Durable Documentation Policy

This proposal is not yet a settled architecture decision, so it does not
rewrite `REFLECTIONS.md`, `PROGRESS.md`, or `OPEN_QUESTIONS.md` now. After owner
acceptance:

- record the shared-semantics/native-data-lane decision in `REFLECTIONS.md`;
- record P12.4 as active in `PROGRESS.md`;
- add only genuinely long-lived unresolved questions to `OPEN_QUESTIONS.md`;
- update `MANIFEST.md` only when the selected backend and optical architecture
  become a milestone, not during temporary dual-lane implementation.

## References Consulted

- The repo's Newton/MuJoCo Warp notes in `REFERENCES.md`: hot GPU phases use
  backend-native flat SoA storage and fixed-phase execution rather than
  per-component interoperability in the hot loop.
- The existing CPU-safe request/delivery vocabulary in `optics/render_api.py`:
  semantic contracts can remain independent from Warp and Torch.
- The existing published-ring design in `physics/gpu_engine.py`: device buffer
  reuse requires explicit borrow, completion, and reclaim semantics; its
  physical Warp resources are not an optical CUDA implementation.
- Existing P12.3e evidence: CUDA target-device guards and native current-stream
  execution work, while blocking synchronization and Warp-owned inputs remain
  measured limitations for P12.4.

## Review Questions for Claude

1. Does the shared-control/native-data split avoid benchmarking CUDA through a
   Warp-oriented path without duplicating product semantics?
2. Are L0/L1/L2 boundaries sufficient to distinguish kernel weakness from
   ownership/synchronization/delivery costs?
3. Is L1 synchronous terminal completion plus L2 asynchronous submit-to-drain
   wall timing fair for both native lanes?
4. Does the normalized L0 BVH fixture preserve equivalent traversal work while
   allowing backend-native storage?
5. Are any current Warp optical capabilities or delivery semantics still
   missing from the retained matrix?
6. Are the CUDA-selected and Warp-selected cleanup branches exhaustive,
   especially exports, CLI flags, tests, pack/readback, event helpers, and
   extension loading?
7. Is the special-case decision for `cuda_lbvh.py` strict enough to prevent an
   unowned interop artifact while preserving an independently valuable
   production accelerator?
8. Should any proposed shared contract be benchmark-local instead of entering
   `optics/` so that the losing lane can be deleted cleanly?
9. Are P12.4a0-a4 small enough to review independently before functional and
   performance changes begin?
10. Do any existing imports or lifecycle assumptions contradict the required
    no-cross-lane rule?

## Recommendation

Approve v3 before implementation. Begin with P12.4a0 and P12.4a1, build the
minimal CUDA-native measurement spine, then assemble the two isolated lanes and
obtain a comparator-validity review at P12.4a4. Do not infer CUDA capability
from the current mixed product path, and do not consider P12.4 complete until
the selected-backend cleanup commit has removed the losing runtime and all
experiment-only residue.
