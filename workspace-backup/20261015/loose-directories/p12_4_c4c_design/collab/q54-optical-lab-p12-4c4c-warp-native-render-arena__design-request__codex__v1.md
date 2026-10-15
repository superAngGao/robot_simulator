# Q54 Optical Lab P12.4 C4c Warp Native Render Arena

- Initiative: Q54 optical computation workflow / P12.4 backend selection
- Stage: C4c design request
- Author: Codex
- Version: v1
- Date: 2026-09-09
- Base: `c61c9e81b9719929f724a229e015ba630f7fdc84`
- Status: design request for Claude review

This document does not authorize C4c implementation, C4d, B3, a3b result
submission, baseline access, backend selection, or a product-default change.
The existing result-activation gates remain closed.

## Outcome

C4c will give the Warp optical candidate a real native arena path with the same
ownership and lifecycle standard already implemented for CUDA C4b. It will:

- replace the allocating and blocking Warp render transaction with
  caller-owned buffers and pre-recorded commands;
- compose the approved C3 scene-update, Warp-refit, and static CUDA-LBVH work
  with that renderer on one native CUDA stream;
- use the shared C4a slot/generation/lease state machine;
- record exactly one preallocated Warp terminal event after the last stage;
- bind every execution to exact a2 allocations, C3 provenance, slot, frame,
  generation, and a single-use arena authority;
- expose only generation-scoped device output after terminal resolution; and
- remain non-result-capable until C4d, B3, and the joint activation review.

C4c is not a wrapper around `GpuDeviceBvhDirectLightOpticalExecutor`. That
executor remains the correctness oracle during selection. Its current
allocation, host materialization, intermediate event, and host wait make it
benchmark-ineligible.

## Approved inputs

C4c is subordinate to the already approved contracts:

- P12.4 backend parity/performance gate v5;
- P12.4a3 native-lane dependency order v3;
- a2 native input/readiness and a3a runtime instrumentation;
- C3a exact provenance, C3b Warp scene/refit commands, and C3e static
  Warp-owned/CUDA-LBVH storage;
- C4 arena design v1 plus the external-hold addendum v2;
- C4a control/evidence implementation; and
- C4b CUDA arena, ending at the approved single-use authority commit
  `c61c9e81b9719929f724a229e015ba630f7fdc84`.

No gate row, performance threshold, payload contract, or acceleration identity
is changed by C4c.

## Current code facts

The current Warp render path cannot be measured at the native-ready boundary:

- `execute_camera()` materializes camera rotation/translation through NumPy and
  allocates ray buffers for every call;
- first-hit allocates all geometry outputs and scalar diagnostics for every
  call;
- the FULL path initializes miss values by constructing and uploading NumPy
  arrays;
- the reduced profiles launch a separate hit-mask initializer but leave the
  other miss fields unspecified;
- first-hit records an event and synchronizes it before shade;
- shade allocates RGB, intensity, and diagnostics, then records and synchronizes
  another event; and
- diagnostics-off aliases nine unused counter arguments to the primary shadow
  overflow counter.

C3b already supplies immutable pre-recorded Warp scene-update and refit command
bundles. C3e proves that a Warp-owned scene can create stable setup-time Torch
views and run the CUDA LBVH extension on a `torch.cuda.ExternalStream` with the
same native handle. C3e is still a one-shot comparison object in the benchmark
package and is not yet a production-owned reusable stage.

C4b supplies useful control-plane precedent, but C4c will not generalize CUDA
and Warp into a permanent backend registry. The shared abstraction remains
`NativeOpticalArena`; runtime allocations, commands, completion objects, and
transaction adapters stay explicit per backend.

## 1. One Warp renderer implementation, two entry styles

There must not be a second copy of the raygen, first-hit, tie-breaking, shadow,
or lighting math.

C4c will move the reusable Warp kernels/functions into a production module
with an implementation-neutral-to-Warp name, such as
`optics/warp_native_render.py`. Both the compatibility executor and the new
command factory will call those same kernels. If moving all definitions in one
slice is mechanically unsafe, the first implementation may import the existing
kernel objects from `optics.warp_execution`, but it may not fork their math.
The final C4c state has one source of truth.

The two entry styles are intentionally different:

- compatibility executor: allocating, blocking, returns
  `OpticalComputeResult`, benchmark-ineligible, retained as oracle;
- native command factory: caller-owned storage, pre-recorded commands, no wait,
  arena-controlled, benchmark candidate.

Oracle compatibility does not authorize the executor to appear in native lane
receipts.

## 2. Exact caller-owned render storage

`WarpDirectLightBuffers` is an immutable typed view over one slot's Warp-owned
render allocations. Its ordered ABI is:

| Name | Warp dtype | Shape |
|---|---|---|
| `origins` | `wp.float32` | `[num_rays, 3]` |
| `directions` | `wp.float32` | `[num_rays, 3]` |
| `hit_mask` | `wp.int32` | `[num_rays]` |
| `range_m` | `wp.float32` | `[num_rays]` |
| `position_world` | `wp.float32` | `[num_rays, 3]` |
| `normal_world` | `wp.float32` | `[num_rays, 3]` |
| `numeric_instance_id` | `wp.int32` | `[num_rays]` |
| `material_index` | `wp.int32` | `[num_rays]` |
| `bvh_stack_overflow_count` | `wp.int32` | `[1]` |
| `bvh_max_stack_depth` | `wp.int32` | `[1]` |
| `rgb` | `wp.float32` | `[num_rays, 3]` |
| `intensity` | `wp.float32` | `[num_rays]` |
| `shadow_stack_overflow_count` | `wp.int32` | `[1]` |
| `shadow_max_stack_depth` | `wp.int32` | `[1]` |

A diagnostic arena additionally owns the nine distinct `[1]`, `wp.int32`
arrays in `SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES`, in that canonical order.
A primary arena owns none of those nine arrays. Diagnostics-off never allocates
dummy storage and never aliases a primary counter.

All listed physical buffers are allocated per slot even when a requested
profile exposes only a subset. This matches the current C4 layout decision and
keeps one fixed render graph per specification. A generation-scoped result
view exposes only `OpticalOutputProfile.guaranteed_channels`, plus the optional
diagnostic overlay in a diagnostic run. Unexposed scratch/output storage cannot
be retrieved through the public result view.

Validation freezes exact object identity, base allocation pointer, active byte
range, dtype, rank, shape, device ordinal, and owner chain. Distinct logical
outputs cannot overlap. The only allowed aliases in a Warp arena are the
explicit setup-time Warp/Torch views used by the static CUDA-LBVH stage; those
aliases are recorded as two logical views of one physical allocation and their
bytes are counted once.

## 3. Deterministic generation reset

Every generation must be correct after arbitrary sentinel values are written
to every slot buffer. Correctness may not depend on fresh allocation.

The frozen render command order is:

```text
reset scalar diagnostics
-> pinhole raygen
-> initialized BVH first-hit
-> direct-light shade
```

The initialized first-hit kernel writes all six per-ray geometry outputs for
every ray before intersection:

```text
hit_mask            = 0
range_m              = +inf
position_world       = NaN
normal_world         = NaN
numeric_instance_id  = 0
material_index       = 0
```

It then overwrites hits with the existing traversal result. This brings the
Warp miss contract into line with the CUDA C2 kernel and removes any need for
host-created initialization arrays. Raygen writes every origin and direction.
Shade writes RGB and intensity for every ray, including the configured
background for misses.

Two fixed reset kernels are permitted:

- primary reset: the four mandatory BVH/shadow scalar diagnostics;
- diagnostic reset: the same four plus the nine distinct traversal counters.

The exact reset variant is selected and recorded at seal time. Empty-ray work
still launches or otherwise records the scalar reset according to the frozen
dispatch plan, so all `[1]` outputs are deterministic. No `wp.zero_`, dynamic
array collection, host write, or lazy allocation may replace the recorded
reset command in timed execution.

Initialization is part of the timed render stage for both candidates. C4c does
not precompute a frame's output outside the native boundary.

## 4. Fixed render command ABI

`WarpDirectLightRenderCommandFactory.prepare(...)` runs outside timing and
freezes:

- exact indexed Warp CUDA device and non-default stream owner/handle;
- exact prepared local scene and a2 readiness authority;
- camera width, height, intrinsics, pose scalars, maximum distance, role mask,
  ambient/background RGB, and shadow bias;
- output profile, shadow flag, diagnostic mode, ray count, and scene counts;
- fixed fixture or replay provenance; and
- the shared Warp kernel/module identity after compilation and warmup.

`seal(...)` binds one exact slot's world/BVH/render arrays and records the
command tuple outside timing with `record_cmd=True`. The command tuple is fixed
for one `(slot_id, transform-or-camera frame)` plan and is never rebound.

Four explicit shade dispatch identities are frozen:

```text
no_shadow_primary
no_shadow_diagnostics
with_shadows_primary
with_shadows_diagnostics
```

The primary variants have no traversal-counter ABI positions. The diagnostic
variants accept exactly nine distinct counters in canonical order. No dynamic
list, fallback tensor/array, optional dummy argument, or runtime entry selection
is allowed in enqueue.

`WarpDirectLightRenderCommandEvidence` contains only independent stage facts:

```text
schema and operation ID = warp_bvh_direct_light
device ordinal and native stream handle
native-input/readiness, scene, replay-or-fixture, and camera digests
output profile, shadows, diagnostic mode, num_rays
ordered input-storage and output-storage manifests
reset/raygen/first-hit/shade entry identities
exact dispatch count and dispatch-plan digest
kernel/module/source identity
```

It does not reference an arena, C4 plan, lease, outer binding, or its own hash.
The outer C3 binding and later C4 plan reference its digest, preserving the
existing one-way provenance DAG.

The bundle has a one-way seal. `validate_bound()` reconstructs the full ABI,
storage manifest, command identity/order, evidence, stream, and owner graph
outside timing. Enqueue performs only a bounded lightweight identity check and
launches the already recorded commands via `wp.Launch.launch(stream=...)`.
Ordinary assignment cannot reopen the seal or replace its signature baseline;
`object.__setattr__()` drift is detected before the first recorded launch.

## 5. Warp transaction and terminal event

C4c adds an explicit Warp transaction layer rather than using a ready event
from the compatibility executor.

`WarpOpticalTransactionTemplate` freezes:

- exact ordered stage objects and their independent evidence;
- one exact Warp stream owner and native stream handle;
- one preallocated `wp.Event(device=...)` terminal event for the slot;
- optional, separately typed preallocated timing-event pairs for a diagnostic
  arena; and
- the C4 slot/frame/terminal-event plan identity before dormant activation.

`WarpOpticalTransaction` is fresh host state for one generation. Its legal
sequence is:

```text
validated_open
-> enqueue each registered stage exactly once and in order
-> finish exactly once
-> finished
```

`finish()` calls `stream.record_event(preallocated_event)`. Calling
`record_event()` without the supplied event is forbidden because it allocates.
No earlier stage records or exports an optical completion event. C3e's shared
Warp/Torch acceleration handoff uses stream order and no event. C0 physics-done
events and diagnostic timing events retain their separate meanings.

The terminal event can be recorded for a later generation only after C4 exact
reclaim has retired the prior authority and advanced the slot generation.
Resolution uses `wp.synchronize_event()` outside the timed contamination guard,
then commits the C4 terminal-resolution transition. Event query/poll loops are
not introduced.

Any stage or terminal-record failure after enqueue begins poisons the
transaction and C4 control arena. It creates no false completion ticket and
cannot return the slot to `FREE`. C4d will add the complete drain,
abort/quarantine, and guardian teardown paths; C4c keeps all owners strongly
held and offers no unsafe partial cleanup.

## 6. Arena-controlled execution is a capability, not a private attribute

C4c adopts the final C4b authority model from the first implementation. Merely
hiding a template, stage, command bundle, or event under an underscore is not
accepted as execution control.

The exact sequence is:

1. Arena seal registers the controlled template, its stage bundles, and the
   terminal event against a private construction capability.
2. The template receives a one-way plan identity containing slot, frame,
   terminal-event object/token, ordered runtime-stage evidence, and plan digest.
3. Untimed `prepare_sample()` creates the exact dormant transaction and exact
   active owner, then obtains a single-use prepared execution capability. The
   binding is derived from the sealed plan and verified C3 provenance; callers
   do not pass a collection of binding fields.
4. Measured `NativeOpticalArena.acquire()` returns the real C4 lease.
5. `WarpNativeOpticalArena.acquire()` atomically consumes the prepared
   capability and registers:

   ```text
   exact lease -> exact authority -> exact transaction -> exact active owner
   ```

6. Every stage enqueue and `finish()` rechecks that registration and the
   current C4 lifecycle phase.
7. cancel and reclaim retire the registration; poison retains it and forbids
   remint.

Construction capability can create dormant transactions only. It cannot mint
live authority. A lease can be registered once only. Authority types are exact
sealed types minted in the Warp timeline module; module/name strings,
duck-typed methods, copied objects, and serialized evidence are not authority.

Once a C3 scene/refit/render bundle becomes arena-controlled, its standalone
enqueue entry rejects. Only the exact authorized transaction can call its
controlled enqueue. This rule also covers the static CUDA-LBVH stage. Reaching
an arena-private executable graph cannot bypass the C4 lease or record the slot
event while the lifecycle still says `FREE`.

Required attack tests mirror and extend the C4b findings:

- direct template activation before acquire;
- direct scene/refit/render/static-LBVH stage enqueue before acquire;
- direct terminal event record through the transaction finish path;
- duck-typed or same-name authority;
- copied/unregistered authority;
- one real lease minted twice;
- frame-0 lease applied to a frame-1 transaction;
- empty/partial/reordered execution binding;
- authority reuse after cancel, reclaim, or poison; and
- a reached private executable attempting a complete transaction while the
  C4 slot remains `FREE`.

Every rejection must precede the first command launch and leave the terminal
event record count unchanged.

## 7. Exact a2 and C3 provenance

The public `WarpArenaPlanTarget` is non-executable. It exposes immutable plan
evidence and resource descriptions needed for review, not command objects,
templates, stages, events, or activation capabilities.

Like the approved CUDA target, it may retain private read-only references to
the actual runtime scene, fixture/replay, stage evidence, and storage objects
so the benchmark-to-product adapter can build exact `NativeStorageSignature`
records. Those references contain no executable command/template/event. The
adapter's issued binding is accepted only when its exact target, prepared
input, and readiness identities are registered together; a caller cannot use
the inspection fields to mint C3 provenance for another target.

`bind_c3_provenance(...)` accepts all of:

- the arena-owned target;
- the exact live `PreparedNativeInput` object;
- the exact live native readiness authority; and
- a benchmark-adapter-issued canonical C3 one-shot binding reconstructed from
  the target's actual stage graph.

The arena validates object identity and approved views, not just matching
hashes. R0 binds the exact a2 packed-scene allocations plus exact derived fixed
fixture allocations. R1/R2 bind the exact packed-scene allocations, exact
replay allocation/frame view, exact per-slot scene outputs, acceleration
outputs, and render outputs. Same-content reuploads, correct hashes with wrong
objects, a replay frame from another prepared input, or a substituted fixed
fixture fail before preparation or acquire.

The Warp benchmark adapter remains the only dependency on gate rows and C3
receipt classes. Production `optics` modules never import `benchmarks`.
Serialized C3 or C4 evidence can be validated and archived but cannot recreate
the live a2/readiness/arena authority.

The provenance DAG is strictly:

```text
a2 prepared input + readiness
-> independent scene/acceleration/render/delivery stage evidence
-> canonical C3 one-shot binding
-> immutable Warp slot plan and resource manifest
-> C4 prepared plan
-> arena-issued active lease binding
-> terminal completion and C4 lifecycle evidence
```

No lower node references a later digest.

## 8. The three Warp native transaction graphs

The gate identities and exact stage order are unchanged.

### R0 fixed renderer

```text
prepared fixed world scene + normalized fixed BVH
-> warp_bvh_direct_light
-> finish
```

There is no timed scene update or acceleration operation. The render stage
reads the exact union of a2 packed-scene metadata/material/light arrays and a2
derived fixed world/BVH fixture arrays. The primary metric begins immediately
before render enqueue; measured C4 acquire remains outside R0's renderer-only
metric while lifecycle safety is still enforced.

### R1 dynamic Warp refit

```text
warp_device_scene_update_v1
-> cpu_median_leaf4_topology_warp_refit_v1
-> warp_bvh_direct_light
-> finish
```

The immutable CPU-origin leaf-4/sort topology is session-owned. Each slot owns
its mutable world geometry/AABBs and BVH bounds; topology arrays cannot be
replaced. Scene update, refit, renderer, and completion are on the same Warp
stream with no intermediate event or host wait.

### R2 static CUDA-LBVH with Warp renderer

```text
warp_device_scene_update_v1
-> cuda_morton_karras_lbvh_rebuild_v1
-> warp_bvh_direct_light
-> warp_device_only_delivery_v1
-> finish
```

The only C4c R2 payload is the already allowed `none/device_only` row. The
delivery stage is an explicit completion-neutral no-copy stage bound to the
exact profile/output storage. It records no event and exposes no host payload.

The R2 CUDA LBVH consumes setup-time Torch views of Warp-owned AABBs and writes
setup-time Torch views of Warp-owned LBVH outputs. The Torch external stream
has exactly the Warp stream's native handle. The following Warp renderer reads
the same Warp allocations by stream order. There is no per-frame view
creation, buffer copy, cross-stream wait, or intermediate event.

## 9. Production boundary for the C3e static stage

The reusable storage/view/stream mechanism currently lives in
`benchmarks/optical/lanes/warp.py` under comparison-specific names. A
production Warp arena cannot import it from the benchmark package.

C4c will extract the runtime-neutral-to-benchmark portion into a narrow
production module, tentatively `optics/warp_cuda_lbvh.py`:

- Warp-owned LBVH outputs and scratch allocation;
- exact Warp/Torch stable-view records;
- same-native-stream binding;
- Warp traversal view over the CUDA-written arrays;
- immutable stage/factory validation; and
- resource/storage manifest helpers.

`benchmarks/optical/lanes/warp.py` retains only row/a2/C3 receipt construction,
eligibility instrumentation, and comparison naming. Existing C3e tests must
continue to prove the same 20 stable views, pointer equality, zero timed bridge,
and ambient-device restoration.

This setup-time interop is internal to the Warp candidate's selected
acceleration implementation. It is not a Warp-optical-to-CUDA-optical adapter
and does not permit sharing mutable storage between candidates. Its creation
cost is outside R1/R2 timing but its allocations and owners remain in the Warp
single-lane memory observation.

## 10. Warp arena ownership

`WarpNativeOpticalArena` composes, rather than subclasses, the C4a
`NativeOpticalArena` control plane.

Session-owned resources include:

- exact Warp device and execution stream owner;
- a2 prepared input/readiness and local scene owners;
- fixed fixture or replay allocation;
- compiled Warp kernels/modules and prepared factories;
- immutable Warp refit topology for R1;
- the Torch external stream and stable-view owners for R2;
- the single shared CUDA-LBVH workspace for R2; and
- all per-slot/per-frame immutable command templates.

Per-slot resources include:

- world geometry/AABB outputs for R1/R2;
- mutable Warp BVH bounds for R1, or all Warp-owned CUDA-LBVH outputs for R2;
- one `WarpDirectLightBuffers` graph;
- exactly one primary terminal `wp.Event`;
- optional diagnostic timing events; and
- the immutable target/executable mappings for each legal frame.

R2 workspace is session-owned and serialized by the arena's one stream, just
as approved in the parent design. Render and delivery never access it. A
submitted failure poisons the whole arena, so scratch cannot be speculatively
reused after timeline loss.

The Warp resource record represents physical allocation once and lists every
logical Warp/Torch view, dtype, shape, stride/contiguity fact, active byte
offset/range, device, pointer token, and strong owner token. Events and Python
owners are separate non-storage records. The generic C4 manifest totals the
backend records without assuming Torch tensor APIs.

The sealed configuration cannot attach, resize, replace, or individually
destroy resources. C4c implements only clean depth-one native-only/device-only
reuse needed for R0-R2 proof. Full depth-two host payload, final drain,
poisoned abort/quarantine, and C0 guardian handoff remain C4d/B3 review scope.

## 11. Preparation, acquire, output, and reclaim

Outside timing:

```text
plan_target(frame, offset)
-> exact full-graph validation
-> bind live a2/readiness and canonical C3 provenance
-> prepare_sample()
-> fresh dormant transaction + single-use prepared authority
```

Inside the R1/R2 timed boundary, `acquire()` performs only a fixed number of
identity/state checks and the C4a slot transition. It cannot walk slots,
frames, commands, owners, or resource records; rebuild tuples/hashes; allocate
host objects proportional to scene size; or call a GPU/runtime API. Full graph
validation stays in seal/plan/prepare.

The active sample owns the only allowed sequence:

```text
ACQUIRED -> enqueue -> ENQUEUED -> finish -> COMPLETION_RECORDED
```

Outside the timed contamination guard:

```text
resolve exact terminal event
-> PAYLOAD_READY
-> expose generation-scoped WarpDeviceResultView
-> release device result
-> RECLAIMABLE
-> reclaim
```

Result access before payload readiness or after release/reclaim fails. Raw slot
arrays, command objects, and terminal events are not returned. Cancellation is
allowed only before the first enqueue and burns the generation. A stale plan
or backpressure result performs no enqueue and no event record.

## 12. Device and stream discipline

Every factory, command, transaction, event, and allocation is bound to an
indexed CUDA device and one exact non-default Warp stream.

For Warp-only commands, execution uses `wp.ScopedDevice`/`wp.ScopedStream` as
needed and restores the caller's ambient Warp state. R2 CUDA-LBVH calls run
inside the retained Torch external-stream/device context and restore ambient
Torch state. The two stream handles must be equal before seal and at every
lightweight enqueue check.

No default stream fallback is allowed. A destroyed/replaced stream owner,
changed handle, wrong-device event/array/view, or ambient-device mismatch fails
before the first command whenever the timeline is still untouched. Once a
stage may have submitted, the same failure poisons instead of cancelling.

Multi-GPU tests deliberately construct the arena on `cuda:1`, reset ambient
Torch and Warp devices to `cuda:0` immediately before acquire, every stage,
finish, resolve, release, and reclaim, and verify restoration afterward.

## 13. Runtime instrumentation

C4c extends the a3a runtime audit with the actual production entry points. A
source manifest remains a coverage list, not runtime evidence.

The timed allowed operations are only:

- C4 measured acquire where required by the row;
- controlled `wp.Launch.launch(stream=frozen_stream)` for recorded Warp
  commands;
- the already audited bound CUDA-LBVH extension call on the same native stream;
- explicit device-only delivery stage entry; and
- `stream.record_event(preallocated_terminal_event)` at finish.

The instrumentation must reject inside timing:

- `wp.empty`, `wp.zeros`, `wp.array`, NumPy materialization, or host upload;
- `wp.launch(..., record_cmd=True)` or any dynamic `wp.launch`;
- `wp.to_torch`, `wp.from_torch`, DLPack, or other view creation;
- `record_event()` without the exact preallocated event;
- any intermediate optical event record;
- `wp.synchronize*`, Torch synchronize/query/poll, or a host wait;
- extension/module load or compilation; and
- a bridge or second-stream wait.

Negative instrumentation tests invoke the real stage paths without relying on
high-level self-report. The submission trace records the gate operation names;
reclaim remains only in arena-produced lifecycle evidence.

## 14. Required tests

### CPU-safe unit tests

- importing arena/contract modules initializes neither Warp nor Torch CUDA;
- exact buffer/profile/diagnostic ABI and per-slot manifest derivation;
- stage order for every mandatory Warp R0/R1/R2 row;
- immutable factory, bundle, template, target, prepared sample, and authority
  state;
- canonical one-way stage/C3/C4 provenance DAG;
- same-hash wrong a2 object, same-content reupload, wrong replay view, wrong
  fixture owner, storage reordering, overlap, and alias rejection;
- O(1) acquire check instrumentation independent of frame/resource count;
- all capability-bypass attacks listed in section 6;
- result activation and host-payload arena seal remain closed.

### Single-GPU tests

- non-square, non-centered camera with `fx != fy` and non-identity rotation;
- direct origins/directions parity and row-major `py * width + px` boundaries;
- asymmetric plane/triangle/material/light/role scene with hit and miss pixels;
- FULL, RGB_PREVIEW, and RENDER_ONLY; shadows on/off; diagnostics on/off;
- direct channel parity against the existing Warp oracle and retained host/CUDA
  tolerances;
- arbitrary sentinel fill before at least three successful reuses of one slot;
- all exposed miss values and all 4/13 scalar counters reset every generation;
- exact R0, R1 refit, and R2 CUDA-LBVH stage order on a non-default stream;
- one terminal event record, no intermediate optical event, no rerecord before
  reclaim, and safe reuse of the same event only after reclaim;
- no timed allocation, upload, view creation, extension load, sync, or bridge;
- direct controlled-stage/template/event bypass rejection before first launch;
- mutation before enqueue rejects; mutation after submission begins poisons;
- injected command and terminal-event record failures produce no false ticket
  or reclaim; and
- unavailable Go2 assets continue to fail closed without a substitute result.

### Multi-GPU tests, when available

- target `cuda:1` with ambient Warp/Torch device deliberately set to `cuda:0`
  at each lifecycle boundary;
- caller ambient device/stream restoration after success, validation failure,
  cancellation, and submitted failure; and
- wrong-device stream, event, storage, stable view, or plan rejection before
  first enqueue.

C4d will add the complete depth-two backpressure, final-drain,
external-host-hold, poison abort/quarantine, and guardian tests. C4c may reuse
the C4a planning helper but must not weaken the B3 host-payload sealing gate to
manufacture a depth-two test.

## 15. Implementation and review slices

### C4c1 — native Warp render ABI

- establish the single kernel source;
- add `WarpDirectLightBuffers`, deterministic initialization, four fixed shade
  variants, render evidence/factory/bundle, and oracle parity tests;
- no arena activation or result capability.

### C4c2 — Warp timeline and R0/R1 arena

- add controlled Warp transaction template, preallocated terminal event, exact
  single-use authority, resource records, device result view, and C4a control
  composition;
- bind exact a2/C3 provenance for R0 and dynamic-refit R1;
- review the C4b attack matrix against the Warp implementation.

### C4c3 — production static CUDA-LBVH binding and R2

- extract the C3e runtime storage/view/stage mechanism from the benchmark
  package without changing its semantics;
- bind the R2 scene -> CUDA LBVH -> Warp render -> device-only graph;
- rerun C3e zero-bridge/same-stream tests and C4c parity/instrumentation tests.

### C4c4 — C4c closure

- run the full R0/R1/R2 construction matrix, sentinel reuse, single/multi-GPU,
  regression, lint, and repository test suites;
- update resource/cleanup manifests and implementation notes;
- obtain Claude approval before starting C4d or B3.

Each slice remains non-result-capable. No slice produces a benchmark sample or
reads a baseline.

## 16. Clean repository terminal branches

C4c adds no permanent generic renderer registry, cross-backend buffer class,
or migration fallback. The cleanup ledger must classify every new module,
export, audit, test, and adapter before implementation.

If CUDA is selected, P12.4e removes the Warp native arena/render candidate,
its static Warp/CUDA-LBVH binding, lane adapter, selection-only tests, and
flags after the retained oracle obligation ends.

If Warp is selected, P12.4e removes the CUDA optical renderer/arena candidate
and paired selector. CUDA LBVH plus the narrow Warp stable-view module may
remain only if the already frozen independent acceleration benefit and
maintenance gate passes; otherwise that acceleration choice requires a new
versioned lane contract and measurement, not a silent fallback. All surviving
modules lose `comparison`, `spike`, and migration naming and move under a clear
production owner.

Whichever renderer wins, the durable survivors are limited to canonical
workloads/schema/validation, selected-backend R0/R1/R2 monitoring, selected
physics-to-image regression, and baseline/artifact manifests. The temporary
paired coordinator and losing lane are mandatory cleanup, not later
housekeeping.

## Explicitly out of scope

- C4d drain/abort/quarantine/guardian completion;
- B3 RGB/full/RGB8 conversion, pinned host storage, D2H, async payload leases;
- S physics-published execution;
- host-payload arena activation;
- a3b result records, baseline inspection, paired comparison, or selection;
- renderer math changes, fusion, CUDA refit, BVH tuning, async copy streams, or
  performance-specific specialization; and
- product-default or public compatibility-executor removal.

## Review request

Please approve or request changes on:

1. the exact caller-owned Warp buffer and deterministic reset contract;
2. one shared renderer-math source and four non-dummy shade variants;
3. pre-recorded commands plus one preallocated terminal Warp event;
4. the C4b-strength single-use authority applied to every controlled Warp
   stage and finish path;
5. exact live a2/C3 provenance rather than digest-only binding;
6. the R0, R1 refit, and R2 static CUDA-LBVH transaction graphs;
7. extracting only the production runtime portion of C3e while retaining the
   benchmark adapter boundary;
8. O(1) measured acquire, single-stream instrumentation, and parity coverage;
9. C4c1-C4c4 review slices and continued closure of a3b/baseline; and
10. the two clean terminal repository branches.
