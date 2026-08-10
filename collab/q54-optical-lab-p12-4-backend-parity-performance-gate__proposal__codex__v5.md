Initiative: q54-optical-lab-p12-4-backend-parity-performance-gate
Stage: proposal
Author: codex
Version: v5
Date: 2026-08-10
Status: in_review
Related Files: collab/q54-optical-lab-p12-4-backend-parity-performance-gate__proposal__codex__v4.md; collab/q54-optical-lab-p12-4-backend-parity-performance-gate__decision__owner__v1.md; collab/q54-optical-lab-p12-3e-cuda-camera-raygen__implementation-note__codex__v1.md; PLAN.md; PROGRESS.md; physics/gpu_engine.py; physics/publish.py; optics/render_api.py; optics/device_channel.py; optics/device_scene.py; optics/cuda_direct_light.py; optics/cuda_lbvh.py; optics/warp_execution.py; tools/optical_pipeline_lab/physics_source.py; tools/optical_pipeline_lab/render_session.py; tools/optical_pipeline_lab/delivery.py; tools/optical_pipeline_lab/async_readback.py; tools/optical_pipeline_lab/rgb_pack.py
Owner Summary: Supersede proposal v4. Retain independent backend-native R0-R2 inputs and the real physics-to-image shipping gate. Close the remaining lifecycle gap with an explicit public CUDA consumer-stream binding owned by physics publication: publication waits its Warp/CUDA publish event and records/retains its own reclaim event on the bound external CUDA stream; CUDA optical never submits a Torch event directly to ring internals. Define R2 as a fresh end-to-end execution of all R1 work followed by payload-parameterized delivery, with separate non-gate delivery diagnostics. Track host payload readiness, consumer completion registration, logical slot reclaim, and final batch reclaim as distinct endpoints. The durable benchmark core, bounded continuation, quantitative `cuda_lbvh.py` gate, and mandatory selected-backend cleanup remain unchanged.

---

## Revision Decision

This v5 supersedes proposal v4 for review. It retains v4's accepted direction
and closes four remaining contract gaps:

1. defines a public CUDA consumer-stream publication binding rather than
   passing Torch events into private ring state or reviving a general optical
   stream bridge;
2. defines R2 as the complete R1 native transaction followed by delivery, not
   as an ambiguous delivery-only measurement;
3. parameterizes R2 completion by the retained payload/policy combination;
4. separates host payload readiness, completion registration, logical slot
   reclaim, and final batch reclaim in shipping measurements.

V5 retains the accepted requirements that CUDA preserve the current product-
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

median_repeat_p90(CUDA sim_step_to_slot_reclaimable)
    <= median_repeat_p90(Warp sim_step_to_slot_reclaimable)
```

For asynchronous shipping rows, the batch throughput gate ends at final
reclaim, not only final payload drain:

```text
median_repeat(CUDA batch_first_step_to_final_reclaim)
    <= median_repeat(Warp batch_first_step_to_final_reclaim)
```

Device-only and full-channel diagnostic product rows use their predeclared
row-specific completion endpoint (`sim_step_to_device_output` or completed full
host payload) rather than pretending every result is an RGB image. The durable
product SLO centers on host RGB8 image readiness; other profiles remain covered
by F/C and their native R0-R2 endpoints. Host readiness never substitutes for
the separate slot-reclaim gate.

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

### R2 — Payload-parameterized native transaction and delivery

R2 begins at exactly the same native-ready body-transform/static-scene input as
R1 and executes all R1 work again in the measured pass:

```text
R2(payload) path
  = world-space scene update
  -> native acceleration refit/rebuild
  -> render and requested native output completion
  -> payload-specific pack/staging
  -> ordered delivery when requested
  -> payload-specific completion endpoint
  -> output lease reclaim
  -> final drain for asynchronous batches
```

This is a fresh end-to-end measurement. Its result is not calculated by adding
separately measured R1 and delivery phase medians because those phases may
overlap.

Required retained combinations and endpoints:

| Payload/policy | Payload-ready endpoint | R2 terminal endpoint |
|---|---|---|
| none / device-only | requested native output complete; no D2H | native output lease completion/reclaim state recorded |
| RGB / sync-host | contiguous host float RGB ready | relevant native/staging leases reclaimable |
| RGB / Torch async ordered | ordered host float RGB delivered | all output leases reclaimed and final drain complete |
| full / sync-host | frozen complete host channel set ready | relevant native/staging leases reclaimable |
| RGB8 / sync-host | contiguous host uint8 RGB ready | relevant native/staging leases reclaimable |
| RGB8 / Torch async ordered | ordered host uint8 RGB delivered | all output leases reclaimed and final drain complete |

Do not add full async or another unsupported payload/policy combination merely
for matrix symmetry. Async rows cover only combinations retained from the Warp
product contract unless the owner separately expands product scope.

Every R2 row freezes in `gate.json`:

```text
payload
delivery_policy
payload_ready_endpoint
transaction_terminal_endpoint
expected_channels
expected_bytes
ring_depth
sync_or_async
native_output_reclaim_endpoint
final_drain_required
```

Both lanes use the same channel semantics, payload bytes, ring depth, ordering,
backpressure/drop policy, and host consumer work for a matched row. PNG/video
encoding and disk write are excluded from the primary metric and covered by an
artifact smoke.

### D — Delivery-only diagnostic

When diagnosis is needed, measure a separate non-gate path:

```text
completed native render output
-> payload-specific pack/staging
-> completed host/device delivery endpoint
```

D never replaces R2 and is not used for the backend replacement ratio. It only
attributes packing, D2H, ordering, and drain costs after a completed render.

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

The CUDA optical executor itself consumes Torch-native scene/BVH inputs. It does
not pass a Torch event into `PublishedRing`, and the adapter must not call
`mark_device_consumer_complete()` or mutate consumer/ring state directly.

Physics publication gains one explicit public CUDA consumer-stream contract.
The recommended API shape is:

```python
binding = engine.bind_cuda_device_consumer(
    consumer_id,
    device_ordinal=target_device_index,
    cuda_stream_handle=torch_stream.cuda_stream,
)

with binding.borrow(frame_id) as lease:
    frame = lease.frame
    torch_inputs = make_zero_copy_torch_transform_views(frame)
    # enqueue the last CUDA optical operation that reads physics-owned buffers
    lease.complete()
```

The final names may follow existing publish conventions, but the public
contract and ownership may not be weakened. `bind_cuda_device_consumer(...)`
creates a stable binding for the consumer/session lifetime. `borrow(...)`
enqueues the publish-event wait on that bound external CUDA stream.
`lease.complete()` records a physics-publication-owned event on the same stream
after all previously enqueued reads of physics-owned buffers, then advances the
existing ring's logical consumer completion.

The implementation may use Warp 1.12's public non-owning external-stream
wrapper (`wp.Stream(device, cuda_stream=handle)`) inside physics publication.
That wrapper is the explicitly allowed producer/consumer handoff; it does not
enter CUDA optical executor, scene, BVH, render, pack, or delivery code.

Rules:

- the adapter is the only product location allowed to know both Warp physics
  publication objects and Torch CUDA optical inputs;
- no Warp optical scene-update, BVH, packer, renderer, or readiness helper is
  called by the CUDA product lane;
- physics kernels and numerical state remain unchanged;
- the existing publish-ring ownership and lossless consumer semantics remain
  authoritative;
- the published slot/meta owns and retains the publish event;
- physics publication creates and retains the consumer done event used by ring
  reclaim; a Torch event is not stored in the ring;
- the bound external CUDA stream remains owned by the CUDA optical session;
  physics holds only a non-owning binding and it must be closed before the
  underlying Torch stream is destroyed;
- device ordinal and stream handle are fixed for the bound consumer; every
  borrow/complete validates them and restores the caller's ambient device;
- completions are in frame order on the stable stream. A later completion may
  cover earlier work only under that enforced single-stream ordering;
- the reclaim event remains alive until the physics producer has established
  the slot-reuse dependency by enqueuing its device-side wait on the producer
  stream before overwrite; it may not be destroyed or re-recorded early;
- zero-copy is the planned path; a D2D-copy fallback or host synchronization
  requires a new owner decision and is not silently introduced;
- no host polling, host event synchronization, or private ring-state update is
  allowed in borrow, completion, or reclaim;
- C0 and C2-C4 receive one joint ownership/stream/lifetime review before
  dynamic CUDA product enablement.

### Lease completion and failure rules

The physics-input lease ends after the last operation that reads physics-owned
transform buffers, not automatically at image delivery:

- when CUDA scene update writes independent optical-owned world buffers, record
  physics consumer completion immediately after that scene update is enqueued;
- if a fused kernel later reads physics transforms directly, hold the lease
  through that fused kernel's completion point;
- the optical result/output lease remains independent and may live through
  RGB8 packing and asynchronous readback.

On an ordinary exception after borrow, the context-managed lease records a done
event after all successfully enqueued work on the bound stream, then advances
logical completion. If CUDA launch, stream access, or event recording itself
fails, the consumer/slot is poisoned and the run aborts; it is never marked
complete without a valid reclaim event. Shutdown owns recovery of poisoned
resources. Unbinding with active leases is forbidden.

### Integration diagnostic I

Measure separately:

```text
published frame/event available
-> native optical frame input bound
-> required stream dependency enqueued
```

Report view setup, event wait/coordination, lease pressure, and any copy bytes.
I is diagnostic and excluded from R-native, but included in R-ship/A-ship.
The diagnostic distinguishes binding setup, zero-copy view construction, stream
dependency enqueue, and any unexpected host wait; a nonzero host-wait field is
a gate failure for the approved zero-copy path.

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
-> optical output/readback leases reclaimed
-> physics consumer completion recorded
-> physics slot logically reclaimable
```

Record distinct endpoints:

```text
sim_step_to_device_image_ms
sim_step_to_host_image_ms
publish_to_host_image_ms
sim_step_to_consumer_completion_recorded_ms
sim_step_to_slot_reclaimable_ms
batch_first_step_to_final_payload_drain_ms
batch_first_step_to_final_reclaim_ms
```

Definitions:

- `consumer_completion_recorded` means the physics-owned done event has been
  recorded on the bound consumer stream and registered with the ring;
- `slot_reclaimable` means every required lossless consumer has advanced far
  enough that the producer may select the slot and enqueue its device-side wait
  before overwrite. It does not claim that the done event is already physically
  complete on the GPU;
- no host synchronization is added merely to observe either endpoint.

The synchronous latency gate checks both host payload/image readiness and slot
reclaimability. The asynchronous throughput gate uses
`batch_first_step_to_final_reclaim_ms`; final payload drain is also reported but
cannot end the gate while a prior frame still pins a slot. Do not sum overlapping
phase timings to obtain either asynchronous total.

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

### C0 — Public CUDA publication binding and joint ownership

C0 is an implementation gate, not only a design note. Before C2-C4 begin,
physics publication must expose and test a public CUDA-consumer contract
equivalent to:

```python
binding = engine.bind_cuda_device_consumer(
    consumer_id,
    device_ordinal=target_device_index,
    cuda_stream_handle=torch_stream.cuda_stream,
)
with binding.borrow(frame_id) as lease:
    frame = lease.frame
    torch_inputs = make_zero_copy_torch_transform_views(frame)
    # Enqueue the final operation that reads physics-owned inputs.
    lease.complete()
```

The accepted contract must freeze:

- Torch-native static scene and per-frame transform input ownership;
- CUDA scene/LBVH/result buffer arenas;
- stable consumer device ordinal and CUDA stream handle, with device mismatch
  and rebinding rejected explicitly;
- publication-owned publish/done events and their lifetime through producer
  reclaim; the CUDA optical lane does not insert a Torch event into ring state;
- device-side publish wait, in-order completion recording, physics published-
  frame lease duration, and producer-side slot reclaim;
- the distinction between physics-input leases and independently owned optical
  result/readback leases;
- ambient CUDA device restoration on success and failure;
- ordinary exception completion after already-enqueued readers, and poison/
  shutdown behavior when CUDA is fatal or a completion event cannot be recorded;
- resize, device-change, binding close, consumer removal, and final-drain
  cleanup.

The implementation may internally create a non-owning Warp stream from the
external CUDA stream handle solely inside physics publication. It must not use
host polling, target-wide synchronization, private ring-state mutation, or a
Warp optical adapter. If this lifecycle cannot be closed through a reviewed
public publication contract, dynamic CUDA product enablement is blocked; the
implementation must extend that contract before proceeding rather than bypass
`complete_device_consumer()` internals.

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
- return a Torch CUDA completion event for optical output readiness; this event
  is not submitted directly to the physics publication ring;
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
- keep optical result/readback leases separate from the shorter physics-input
  lease, which may complete after the final scene-update read is enqueued;
- cover ring pressure, resize, profile changes, exceptions, shutdown, and final
  reclaim of both lease domains.

### C5 — Conditional fusion

Fuse only if R0/R1 evidence shows launch or intermediate-buffer cost remains
material after C1-C4. Preserve a directly testable non-camera geometry path and
the required output/lifetime contract.

Dependencies:

- B3 CUDA RGB8/async delivery waits for C0, C2, and C4;
- B4 dynamic/physics product enablement waits for the implemented and reviewed
  C0 public contract plus C2, C3, and C4;
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
- RGB preview ordered async where currently supported;
- direct-light full synchronous host-channel payload;
- RGB8 ordered async ring depth 2;
- RGB8 synchronous host image;
- direct-light full at diagnostic resolution;
- GPU camera raygen;
- device-only native completion;
- payload-specific host readiness, consumer-completion registration, and slot
  reclaimability;
- asynchronous final payload drain and final reclaim;
- separate artifact smoke.

Every performance row requires prior direct correctness, counter, overflow, and
payload checks.

## Comparator and Statistics Controls

Freeze `benchmarks/results/q54_p12_4/gate.json` before inspecting results. It
contains:

- exact case hashes and mandatory row IDs;
- R0/R1/R2/I/S boundary versions;
- for every R2 row: payload, delivery policy, completion endpoint, expected
  channels/bytes, ring depth, sync/async mode, native-output reclaim endpoint,
  and final-drain requirement;
- for every shipping row: publication CUDA-binding contract version, physics-
  slot reclaim endpoint, and the exact host-ready, consumer-completion-recorded,
  slot-reclaimable, final-payload-drain, and final-reclaim metrics;
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
- dynamic CUDA enablement would require bypassing the public physics-publication
  contract, mutating private ring state, or storing a Torch-owned event in it;
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
P12.4c0a public CUDA publication binding design and lifecycle review
P12.4c0b public binding implementation, failure semantics, and GPU tests
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
approved before benchmark results drive optimization. C0a must be approved and
C0b must pass before C2-C4 implementation or dynamic CUDA product work.
P12.4e is a blocking part of completion.

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
- public CUDA publication-binding tests for publish wait, in-order completion,
  fixed device/stream validation, ambient-device restoration, event ownership
  and lifetime, binding close, absence of host polling/synchronization, and two-
  device mismatch where available;
- injected ordinary exceptions prove completion is recorded after enqueued
  readers; injected event-record/CUDA-fatal failures prove the consumer/slot is
  poisoned and shutdown-safe rather than falsely marked complete;
- tests or instrumented traces prove every R2 row executes the complete R1
  scene-update/refit/render transaction before its payload-specific delivery;
- shipping checks distinguish host-ready, consumer completion recorded, slot
  reclaimable, final payload drain, and final batch reclaim; throughput samples
  cannot stop while reclaim pressure remains outstanding;
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
2. Is R2 unambiguously a fresh execution of all R1 work followed by the frozen
   payload-specific delivery, rather than a delivery-only sample or arithmetic
   sum of separately sampled phases?
3. Does the acceleration axis prevent CUDA LBVH gains or costs from being
   misattributed to the Warp renderer?
4. Is the verified Warp-on-CUDA physics description accurate, including the
   fact that P12.4 retires Warp optical rather than Warp physics?
5. Does the public CUDA consumer-stream binding close the existing publish-ring
   lifecycle without private-state access, host synchronization, Torch-event
   insertion, or accidental Warp-optical coupling?
6. Are event ownership/lifetime, fixed device/stream binding, ambient-device
   restoration, exception poisoning, and binding shutdown semantics complete?
7. Does each supported payload/policy row have a legal R2 endpoint and frozen
   byte/channel/reclaim contract without inventing unsupported combinations?
8. Are S start/end points correct for both synchronous latency and asynchronous
   batch throughput, with host-ready, completion-recorded, slot-reclaimable,
   final-drain, and final-reclaim endpoints kept distinct?
9. Does the durable-core/temporary-lane split keep long-term regression value
   without retaining the losing backend or creating a product registry?
10. Are the continuation deadline, two rerun campaigns, five optimization
   slices, hard failures, and Warp default terminal branch executable and hard
   to evade?
11. Is the quantitative `cuda_lbvh.py` retention gate strict and complete enough
   for a Warp-selected outcome?
12. Do either terminal cleanup branch or the permanent selected-backend
    physics-to-image monitor leave an unowned dependency or dead abstraction?

## Recommendation

Approve v5 before implementation. Build the durable benchmark core and native
input preparation first, validate R0-R2 before using results, then implement and
review the public CUDA publication binding before dynamic CUDA product work.
Close remaining capability and ownership gaps within the frozen continuation
budget, and decide on both native optical strength and the real physics-to-image
shipping path. P12.4 is complete only after the losing lane and product
implementation are removed and the selected backend's native plus physics-to-
image regressions are active.
