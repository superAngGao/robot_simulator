# Q54 Optical Lab P12.4c4 Native Output Arena and Leases

Author: Codex

Date: 2026-09-07

Status: design request for Claude review; this commit does not authorize C4
implementation, C4/B3 activation, a3b, a benchmark sample, or baseline access

Base: `3408524b4b6e24a6f1167cdb30af99a0db8e4304`

Approved parent contracts:

- P12.4 backend parity/performance gate v5;
- P12.4a3 native-lane dependency order v3;
- P12.4a2 durable benchmark core;
- P12.4a3a runtime instrumentation;
- P12.4c0a v3 and the approved C0b publication binding;
- P12.4c2 native CUDA execution timeline v2 and implementation;
- P12.4c3a through C3e, ending at
  `3408524b4b6e24a6f1167cdb30af99a0db8e4304`.

## Outcome

C4 will provide the reusable native storage and lifecycle layer required by
both candidate optical lanes. It will own exact per-row arena specifications,
preallocated mutable device/output storage, terminal event slots, immutable
per-slot execution plans, generation-safe leases, ordered backpressure,
completion association, final drain, reclaim, close, and poisoned teardown.

C4 does not choose a backend and does not introduce a general backend
registry. Warp and CUDA have explicit constructors and independent runtime
allocations. The shared portion is the small control-plane lease state machine,
not a cross-runtime data bridge.

C4 also closes a currently missing prerequisite for a fair comparison: the
Warp renderer must gain a caller-owned, pre-recorded, allocation-free command
binding over arena storage. The current allocating/blocking Warp executor
cannot be labeled a native lane merely because C4 preallocated unrelated
buffers.

C4 does not implement payload conversion, RGB8 packing, D2H submission, or
host delivery. B3 owns those commands. C4 owns the physical delivery storage
and its lease lifetime after B3 supplies a reviewed immutable storage plan.
The final arena is result-capable only after the C4/B3 joint binding is sealed
and reviewed.

No result activation changes in this design or in an isolated C4
implementation. The existing values remain:

```text
a3b_result_capable_before_C4_B3_review = false
native_result_activation.result_capable = false
native_result_activation.activation_commit = null
```

## Current code and unresolved gaps

The repository now has the required native kernels and stage identities, but
not a reusable product transaction:

- C2 allocation helpers create one-off CUDA ray/geometry/shade buffers and
  caller-owned completion events. A poisoned transaction has no owner that can
  safely dispose of or quarantine those allocations.
- C3 CUDA scene/LBVH helpers allocate one-off world, AABB, BVH, and CUB
  storage. C3 Warp factories seal exact scene/refit commands, while C3e seals a
  one-shot static Warp/CUDA-LBVH comparison graph.
- C3 evidence is intentionally non-result-capable. It contains no arena,
  slot, generation, ring, or active lease authority.
- the current Warp renderer allocates ray, geometry, shade, and diagnostics
  arrays per call; initializes some full-profile miss values through host
  materialization; records an intermediate first-hit event; and synchronizes
  before shade. That is not an eligible R0/R1/R2 implementation.
- the Lab async readback ring creates views during submission, uses modulo slot
  selection without an output lease, and owns a separate lifecycle from the
  render outputs. It remains a product behavior reference, not the B3 native
  implementation.
- `required_runtime_operations()` currently lists `reclaim` in the timed
  submission trace even though a real reclaim can only occur after terminal
  completion is resolved or a downstream dependency is established. Claiming
  reclaim during enqueue would be false evidence.

C4 must close these gaps instead of wrapping the current compatibility paths.

## Frozen terms

- **arena specification**: immutable logical capacity and policy for one lane,
  row shape/profile, execution stream, payload attachment, and exact ring
  depth. It contains no runtime allocation supplied by another candidate.
- **arena**: the sealed owner of every mutable slot, terminal event, prepared
  plan, and resource required by that specification.
- **slot**: one stable physical storage graph with a stable integer ID. A slot
  is never resized, rebound, or moved to another device/stream/profile.
- **generation**: a strictly increasing positive integer assigned to a slot on
  successful acquire. A generation is never reused, including after a safe
  pre-enqueue cancel.
- **prepared slot plan**: fully validated, non-active execution binding for an
  exact slot plus replay/camera frame. It is created outside timed submission
  and cannot itself authorize output access or result submission.
- **active output lease**: the arena-issued authority for one exact
  `(arena, slot_id, generation)` tuple. It is created by the measured acquire
  transition and is the only authority that can activate, finish, resolve, or
  release that generation.
- **terminal completion**: the single backend-native event recorded after the
  final registered transaction stage. It means recorded, not physically
  complete.
- **payload lease**: a B3-created consumer hold over the slot's preallocated
  host staging view. Host storage cannot be reused while that lease is active.
- **logical reclaimable**: all required device/delivery/host holds have been
  discharged and the arena may advance the slot to `FREE`. It is distinct from
  a claim that an arbitrary host poll observed event completion.
- **final drain**: ordered resolution and release of every submitted generation
  followed by verification that every slot is free and no host payload lease
  remains.

## Exact arena specification

The production arena module must remain independent of the benchmark package.
The benchmark adapter derives and validates a production `NativeOpticalArenaSpec`
from a frozen row. At minimum the spec freezes:

```text
schema_version
lane                            # warp | cuda
device_ordinal
execution_stream_token
input_contract
native_input_manifest_sha256
native_readiness_sha256
scene_contract_sha256
fixed_render_fixture_sha256 | replay contract/frame domain
renderer_id
acceleration_class
acceleration_id
width, height, num_rays
primitive_count, node_capacity
output_profile
shadows
diagnostic_mode                 # primary_unprofiled | diagnostic_profiled
payload                         # native_only | none | rgb | full | rgb8
delivery_policy                 # native_only | device_only | sync_host | torch_async_ordered
payload_contract_id | null
ring_depth
ordered_delivery
arena_layout_schema_version
```

The adapter, not the production module, compares those fields with
`gate.json`. R0 and R1 use `payload=native_only`,
`delivery_policy=native_only`, and an effective ring depth of one. Every R2
row uses its exact frozen payload, policy, and ring depth; the current matrix
therefore permits only depth one or two. No caller-selected third depth or
arbitrary resize policy is added.

Primary and diagnostic runs use different arena specifications. A primary
arena has the four required scalar diagnostics but no optional nine-counter
traversal overlay and no phase timing events. A diagnostic arena owns all nine
distinct overlay counters and preinitialized timing-event pairs. Diagnostic
storage and events never appear in an unprofiled primary binding.

Device, resolution, ray count, primitive capacity, profile, shadows, payload,
delivery policy, ring depth, input manifest, or stream changes require closing
the old arena and constructing a new one outside timing. `resize`, `rebind`,
and lazy capacity growth are not supported.

No numerical memory budget is introduced. Allocation bytes and clean
single-lane NVML peak remain observations; OOM on a mandatory workload remains
a hard failure.

## Physical resource ownership

Each candidate owns independent CUDA allocations. Even when topology or
content is logically identical, no mutable scene, acceleration, renderer,
event, or delivery allocation is shared between candidates.

### Session-scoped immutable resources

An arena strongly retains:

- the indexed device and exact native execution stream owner;
- prepared native input, local scene, replay/fixed fixture, role/material/light
  arrays, and readiness evidence;
- compiled module/kernel owners and prepared factories;
- immutable Warp CPU-origin leaf-4 topology for the Warp dynamic lane;
- every setup-time Warp/Torch same-allocation view required by the Warp static
  CUDA-LBVH or later B3 delivery path;
- immutable per-slot/per-frame command or stage plans;
- the complete allocation/resource manifest and its digest.

All resources are created before arena seal. A sealed arena cannot attach,
replace, resize, remove, or individually close a resource.

### Per-slot mutable resources

Every slot owns the mutable graph required by its arena specification:

- world triangle/plane geometry and triangle AABBs when the boundary performs
  scene update;
- candidate acceleration outputs: slot bounds for Warp refit, or all CUDA
  Morton/Karras topology/bounds outputs for a rebuild;
- ray origins/directions;
- hit mask, range, position, normal, instance ID, and material index;
- RGB, intensity, and the four distinct primary scalar diagnostics;
- nine distinct traversal overlay counters only for a diagnostic arena;
- RGB8 device output when the reviewed B3 plan requires it;
- payload-specific pinned host tensors when the reviewed B3 plan requires
  host delivery;
- exactly one primary terminal event;
- optional diagnostic timing events only in a diagnostic arena.

The full profile still uses the native device dtypes required by the renderer.
Canonical host dtype conversion belongs to B3. C4 must not guess that a direct
D2H copy is sufficient for `full_canonical_host_v2`; the B3 storage plan must
declare any additional preallocated conversion output before final arena seal.

### Scratch

All runtime scratch, including CUDA LBVH CUB workspace, is preallocated and
bound before seal. The initial design uses one session-owned CUB workspace per
arena rather than duplicating transient scratch in every output slot. Every
acceleration plan binds that exact workspace. This is safe only because the
arena has one immutable native execution stream and every rebuild using the
workspace is submitted in strict stream order; render/delivery never reads the
workspace. Its storage identity and the single-stream dependency are included
in every affected plan and in the arena resource manifest.

Scratch is not exported through an output lease and cannot be used from a
second stream. Any submitted failure poisons the whole arena, so a lost stream
timeline can never lead to speculative scratch reuse. Changing to per-slot
scratch, multiple workspaces, or multiple execution streams requires a new
arena layout identity and ownership review.

## B3 attachment boundary

C4 and B3 use a two-phase builder, not mutable post-seal attachment:

```text
C4 arena builder
  -> add exact scene/acceleration/render requirements
  -> accept one reviewed immutable B3 DeliveryStoragePlan
  -> allocate every slot and setup-time stable view
  -> validate full disjointness and ownership
  -> seal once
```

The only C4-native delivery plans available before B3 are:

- `native_only` for R0/R1; and
- `none/device_only` for the R2 device-only row.

RGB, full, RGB8, sync-host, and ordered-async arenas cannot seal until B3
defines the exact device/host buffers, channel order, dtype/shape, command
binding, and completion endpoint. C4 never exposes an `attach_delivery()` on a
sealed arena.

C4 owns allocation and reclaim. B3 owns:

- payload/channel selection and canonical dtype conversion;
- RGB8 pack and any full-payload conversion commands;
- D2H enqueue semantics;
- host-ready validation and payload receipt;
- the payload lease presented to a host consumer.

B3 receives typed views only from an active C4 lease. It cannot select a slot,
advance a generation, record the optical terminal event, mark itself
reclaimable, or mutate arena state directly. It returns a sealed delivery stage
and transition facts through C4's capability-checked methods.

The first B3 design must use the exact transaction stream already frozen by
C2/C3/C4. No separate copy stream or intermediate render-ready event is added
in the first implementation. `torch_async_ordered` means non-blocking D2H plus
deferred ordered host resolution; it does not silently claim copy/compute
overlap. A later copy-stream DAG requires a reviewed C2 timeline revision and
new event/lease evidence before it can affect a baseline.

## Native Warp render prerequisite

C4 includes a bounded production Warp render binding because the current Warp
executor cannot consume arena storage allocation-free.

The implementation will reuse the existing Warp raygen, BVH first-hit,
direct-light, tie-breaking, shadow, and lighting math. It adds no alternate
renderer. It will add:

- a caller-owned `WarpDirectLightBuffers` contract mirroring the required C2
  ray/geometry/shade/diagnostic storage;
- explicit output initialization commands so every generation overwrites all
  exposed hit/miss values and resets every scalar counter;
- pre-recorded exact raygen, first-hit, and shade commands for the frozen
  camera/profile/shadow/diagnostic binding;
- shadow/diagnostic entry variants that do not rely on aliased dummy counters;
- immutable evidence, exact command-tuple identity/order, storage manifests,
  stream identity, and one-way validation matching the C3 command standards;
- no allocation, NumPy materialization, upload, view creation, intermediate
  event, host wait, or global synchronization during enqueue.

The commands are sealed for every required `(slot_id, replay/camera frame)`
combination outside timing. They are never rebound after seal. R0 binds only
its fixed fixture plus renderer commands. Warp R1 binds scene update, refit,
and render. Warp R2 static binds scene update, the C3e CUDA-LBVH command over
stable views, and render.

The existing blocking Warp executor remains a correctness oracle during
selection and remains benchmark-ineligible.

## CUDA slot plans

CUDA plans reuse the reviewed C2 and C3 implementations:

```text
R0: fixed fixture -> C2 render
R1: C3 CUDA scene -> C3 CUDA LBVH rebuild -> C2 render
R2: C3 CUDA scene -> C3 CUDA LBVH rebuild -> C2 render -> B3 delivery
```

For every slot/frame combination, all C2/C3 stage objects, fixed ABI tuples,
workspace, render buffers, and terminal event identity are bound and fully
validated outside timed submission. The plan contains no callable that can
allocate or load an extension.

C2's per-generation transaction remains single-use. C4 adds a reusable,
immutable transaction **template** that owns the validated stage tuple and
event slot. Timed activation creates only the host-side single-generation
state associated with an active C4 lease; it performs no device/host buffer or
event allocation and no full manifest rebuild. A successful generation can
never be activated twice. A poisoned generation is never reset. The template
may activate a later generation only after C4 has completed exact reclaim of
the previous generation.

This is not a resettable `CudaOpticalTransaction`: each activated generation
still has the C2 ordering rules and can enqueue/finish exactly once.

## Two-phase preflight and measured acquire

The existing C3 requirement for full validation before first enqueue and the
R1/R2 requirement to include lease bookkeeping are both retained through a
two-phase protocol.

Outside timed submission:

```text
lane.bind_sample(...)
-> choose the deterministic next slot plan without changing arena state
-> validate row/input/readiness/stage/storage/event/ring identity
-> return C4PreparedSamplePlan (not an active lease and not result-capable)
```

Inside timed submission, before the first GPU enqueue:

```text
lease = arena.acquire(prepared_plan)
-> require exact next FIFO slot and expected next generation
-> require FREE state and unchanged arena/plan signatures
-> transition to LEASED
-> produce module-owned C4ActiveLeaseBinding
-> lightweight pre-enqueue validation
```

`acquire()` performs no device allocation, host materialization, upload,
bridge/view creation, extension load, event construction, host wait, or global
synchronization. It is O(1) state/identity bookkeeping and is part of R1/R2
measurement. A stale plan fails before any GPU enqueue and leaves state
unchanged.

The live binding is not represented by a caller-provided capability string.
Only the arena can create it, using an internal authority tied to the exact
arena object, slot record, and generation. The runner validates the live
object and its canonical evidence before allowing the first stage. A serialized
dataclass alone cannot authorize submission.

The primary metric does not stop when the Python submission guard exits. The
pending sample retains the acquisition start marker and exact terminal/lifecycle
endpoints. `resolve()` remains outside the contamination guard, as required by
a2, but computes the actual interval through completion, payload readiness,
and row-required reclaim. Async batch time runs from first measured acquire to
final reclaim and is never constructed by summing phase medians.

Metric start remains row-specific: R0 starts immediately before its render
enqueue and excludes arena bookkeeping, matching `native_renderer_completion_ms`;
R1 and R2 start immediately before measured acquire and include the required
lease/reclaim bookkeeping. R0 still resolves and releases its slot safely
before reuse even though that later lifecycle is outside the R0 primary metric.

## Transaction and event rules

Each generation has one ordered registered stage sequence and exactly one
primary terminal event record:

```text
R0: render -> finish
R1: scene -> acceleration -> render -> finish
R2 none: scene -> acceleration -> render -> device-only delivery -> finish
R2 host: scene -> acceleration -> render -> pack/convert/D2H -> finish
```

No scene, acceleration, render, pack, or delivery stage records or exports an
optical completion event. C0's physics done event and diagnostic timing events
remain separately typed and cannot satisfy the C4 terminal event requirement.

The event object is created and initialized on the exact device before arena
seal. It is recorded once for a generation and cannot be re-recorded until the
slot has passed the complete reclaim transition. All consumers must either:

- resolve that exact event at the row's permitted host endpoint; or
- register a reviewed downstream same-device stream dependency and return a
  completion dependency that C4 can retain.

The initial R0-R2 benchmark implementation uses terminal host resolution, with
async resolution deferred and ordered. No host event query/poll loop is used.

## Arena and slot state machines

Arena state:

```text
OPEN -> DRAINING -> CLOSED
OPEN/DRAINING --unsafe submitted failure--> POISONED
POISONED --emergency sync succeeds--> ABORTED_SYNCED
POISONED --emergency sync fails--> QUARANTINED
```

Slot state for one generation:

```text
FREE
  -> LEASED
  -> ENQUEUING
  -> COMPLETION_RECORDED
  -> PAYLOAD_READY
  -> RECLAIMABLE
  -> FREE (generation retained; next acquire increments it)
```

Permitted exceptional transition before first enqueue:

```text
LEASED --cancel--> FREE
```

The generation is burned even on that safe cancel. After the first stage begins
enqueue, any exception, illegal mutation, skipped/reordered stage, delivery
failure, or terminal event-record failure poisons the slot and arena. It never
creates a completion ticket or advances reclaim.

Rules:

- slots are selected strict round-robin/FIFO; no opportunistic later-slot
  selection and no modulo overwrite;
- completion exposure, host payload delivery, and reclaim are FIFO;
- when the next slot is not free, acquire returns a typed backpressure result
  without mutating state or polling CUDA. The async coordinator must resolve
  and release the oldest generation, then retry;
- no frame is dropped. Backpressure count and wait are measured;
- duplicate acquire/finish/resolve/release, release without acquire, wrong
  lease, wrong slot/generation, out-of-order release, and stale lease access are
  rejected without state advancement;
- raw mutable tensors/arrays are not a public result. B3 and lane code receive
  capability-scoped typed views; access after release is rejected;
- normal `close()` rejects any leased, enqueuing, completion-recorded,
  payload-ready, or poisoned slot. It never performs an implicit drain;
- `drain()` resolves in FIFO order and verifies every slot and payload hold is
  released. It is part of the async batch terminal endpoint;
- arena close is idempotent only after a successful close. An active close is
  an error, not a partial cleanup.

R0's metric ends at native renderer completion, as frozen, although its slot is
still safely resolved/released before reuse. R1 includes measured acquire,
native completion, and output reclaim bookkeeping. R2 uses each row's exact
payload-ready, staging, final-drain, and final-reclaim endpoint.

## Device and host holds

One active generation tracks separate holds:

- **transaction hold**: covers all device writes through terminal event
  record;
- **device-result hold**: covers any validation or device consumer after
  finish;
- **delivery-device hold**: covers pack/conversion/D2H reads from renderer
  outputs;
- **host-payload hold**: covers consumer access to pinned host storage.

Recording a terminal event does not release any hold by itself. Host payload
readiness does not imply that the host consumer released its slot. A device
output cannot be reused while a downstream device/delivery hold remains, and a
host staging allocation cannot be reused while a payload lease remains.

For the bounded implementation, device and host storage share the same slot ID
and generation. A slot becomes `RECLAIMABLE` only after the row-required event
resolution/dependency and every applicable hold release. This conservative
coupling makes ring pressure visible rather than shifting it into a hidden
second ring.

## Submission trace versus lifecycle evidence

The a3a `RuntimeOperationTrace` remains a trace of the guarded enqueue portion.
It must not claim that a future reclaim already happened. Before a3b result
activation, gate/schema versioning will split requirements into:

```text
submission operations:
  measured acquire, scene/acceleration/render, optional pack,
  delivery enqueue, terminal completion record

arena lifecycle transitions:
  acquired, completion_recorded, payload_ready,
  device holds released, host payload released,
  slot_reclaimable, slot_reclaimed,
  optional final_drain_complete/final_reclaim_complete
```

`reclaim` is removed from `required_runtime_operations()` for the submission
trace. It becomes mandatory arena-produced lifecycle evidence. This is a
pre-result contract correction, not a relaxation after observing performance.
The reclaim implementation remains source-bound/runtime-audited in preflight,
but its actual transition evidence is produced by the arena state machine
during `resolve()`/drain rather than self-reported by the coordinator.

The submission trace and lifecycle transition trace have separate digests and
are both bound into the final lane receipt. The coordinator may attach only
run/repeat/sample identity and timing; it cannot construct either trace.

## Provenance DAG and result authority

The hash dependency remains strictly one-way:

```text
a2 native input/readiness
  -> independent C2/C3/Warp-render/B3 stage evidence
  -> immutable per-slot storage/event manifest
  -> C4 prepared slot plan evidence
  -> arena-issued active lease binding evidence
  -> terminal completion and arena lifecycle evidence
  -> lane ExecutionReceipt
```

No stage evidence references an outer arena, lease, binding, or receipt hash.

The new serialized C4 binding evidence includes at least:

```text
schema_version
arena_spec_sha256
arena_resource_manifest_sha256
arena_token
ring_depth
slot_id
slot_storage_manifest_sha256
generation
lease_token
prepared_slot_plan_sha256
row/case/boundary/lane identities
renderer/acceleration/profile/shadow identities
input/readiness/scene/replay-or-fixture provenance
ordered stage evidence and exact dataflow binding
terminal_event_token
```

Opaque run-local tokens are evidence identities, never raw CUDA handles. The
arena lifecycle receipt separately records actual transitions, terminal event
record count, completion resolution/dependency, holds, backpressure, payload
release, reclaim, final drain, and poison disposition.

`C3OneShotNativeExecutionBindingEvidence` remains non-result-capable and is not
upgraded by adding strings. C4 uses a distinct type and constructor, and the
runner also requires the live arena-issued authority. C4 tests may canonicalize
the new evidence, but `ExecutionReceipt`, result validation, and a3b remain
closed until the joint C4/B3 activation commit reviews the complete payload and
lifecycle chain.

## C0 guardian integration and resource manifest

The sealed C4 arena exposes one immutable guardian resource manifest containing
every stream, event, module, stable view, prepared stage/command, device
allocation, pinned host allocation, and session owner that could access or
destroy the graph. The S session passes the exact owner tuple to
`CudaConsumerResourceGuardian.seal()` before binding the physics consumer.

After guardian bind:

- the arena cannot close or destroy individual members without the approved
  guardian disposal check;
- C0 holds the complete C4/B3 resource graph through physics publication;
- temporary zero-copy physics transform views remain C0 lease-owned and are
  not inserted into C4 output storage;
- C0 physics completion is recorded immediately after the final scene-update
  reader of physics transforms, while the C4 output lease continues through
  acceleration, render, delivery, and reclaim.

C4 alone does not activate S. The complete resource tuple and normal/poisoned
C0+C4+B3 teardown require the later joint GPU review.

## Poisoned teardown

Normal hot paths never call a target-wide or host/global synchronization.
After any submitted generation loses trustworthy completion evidence, the
arena is fail-closed and accepts no new acquire, enqueue, release, reset, or
ordinary close.

Explicit `abort_poisoned()` is the only shutdown path and has two terminal
outcomes matching the approved C0 safety model:

1. target-device emergency synchronization succeeds; all in-flight reads and
   writes are physically complete, after which the arena may release owners
   and enter `ABORTED_SYNCED`;
2. synchronization fails; an append-only process quarantine atomically takes
   the complete arena/session resource graph before the call returns. The
   arena enters `QUARANTINED`, individual cleanup stays forbidden, and the
   process/device is terminal for new optical arenas.

If a C0 guardian is active, it remains the outer quarantine owner and receives
the entire C4 arena owner graph; C4 must not run an independent cleanup race.
Explicit session shutdown coordinates one disposition. `__del__` never
synchronizes and transfers an unsafe unclosed graph to quarantine.

Pre-enqueue validation failure or cancellation is not poison because no GPU
work has begun. Ordinary exceptions after enqueue are poison even when the
Python exception itself is not CUDA-specific; the arena cannot prove which
recorded command reached the stream without a valid terminal completion.

## Fairness rules

- both lanes derive the same logical arena specification, ray count, profile,
  shadows, payload contract, ring depth, ordering, backpressure, host consumer
  work, and lifecycle endpoints for a matched row;
- internal storage may differ only where the frozen candidate algorithm
  requires it, such as Warp refit bounds versus CUDA rebuild outputs;
- no candidate may precompute per-frame world geometry, acceleration output,
  renderer output, pack result, or host payload before measured acquire;
- no allocation, upload, view creation, extension load, host/global sync, or
  hidden bridge occurs during acquire or native enqueue;
- Warp/Torch stable views used by the Warp candidate are created once before
  seal, included in its memory/resource manifest, and never recreated per
  frame;
- primary and profiling arenas/runs remain separate;
- arena allocation and allocator-reserved memory are included in later clean
  single-lane process memory observation. No baseline subtraction or
  `empty_cache()` is allowed;
- a candidate OOM, poison, stale lease, dropped frame, overflow, or final-drain
  failure is a hard failure, not an omitted sample.

## Acceptance tests

### CPU/unit

- CPU-only imports do not initialize Torch CUDA or Warp;
- exact spec validation rejects unsupported lane, unindexed/wrong device,
  wrong stream, row/profile/payload/ring mismatch, arbitrary resize, and
  mutable post-seal attachment;
- arena/slot/generation/lease constructors cannot be forged through a public
  token or direct serialized dataclass;
- all legal state transitions and every illegal duplicate/out-of-order/stale
  transition are covered;
- a prepared plan does not reserve a slot and stale plans fail without state
  mutation;
- generation increments and burns on safe cancel; wrap/invalid generation is
  rejected;
- exact storage names, dtypes, shapes, owner identities, disjointness, event
  uniqueness, and resource-manifest hashes are validated;
- C3 one-shot evidence remains rejected as C4 authority;
- submission trace cannot contain a false reclaim; lifecycle evidence cannot
  be coordinator-populated;
- close with active/pending/payload-held slots rejects; clean close and repeated
  clean close are safe;
- a3b/S activation remains false and no result record is accepted.

### Single-GPU

- CUDA R0 and R1 reuse a depth-one slot for more generations than the measured
  frame count without allocation, view creation, event creation, or extension
  load inside acquire/enqueue;
- Warp R0/R1 native command plans use caller-owned buffers and contain no
  internal event or host wait;
- Warp and CUDA every-generation sentinel tests prove all exposed hit/miss,
  shade, primary diagnostic, and optional overlay outputs are overwritten or
  reset correctly;
- Warp render commands match the existing Warp oracle and CUDA/host parity for
  non-square cameras, full/RGB/render-only profiles, shadows on/off, planes,
  triangles, lights, roles, and miss pixels;
- R1 exact stage order, one primary terminal event, and no event rerecord before
  reclaim are observed on a non-default stream;
- depth-two acquire/finish/resolve/release preserves FIFO, blocks overwrite,
  reports backpressure, and drains every generation with no drop;
- delayed host payload release prevents slot reuse after device completion;
- wrong slot/generation/event/stage/storage/profile/stream fails before first
  enqueue;
- mutation after acquire poisons rather than returning the slot to free;
- stage and terminal-event record failure never produce a false ticket or
  reclaim;
- ambient device/stream restore on success, validation failure, poison, drain,
  and close;
- emergency-sync success releases only after synchronization; injected sync
  failure in an isolated subprocess proves the complete resource graph is
  quarantined and ordinary cleanup cannot run.

### Multi-GPU, when available

- arena target remains on `cuda:1` while ambient Torch/Warp current device is
  deliberately reset to `cuda:0` immediately before acquire, every enqueue,
  finish, resolve, and release;
- ambient device/stream is restored after every transition;
- wrong-device storage/event/stream/plan is rejected before first enqueue;
- no event or slot from one device can enter another arena's generation.

### Joint C0/C4/B3 before S

- the guardian manifest covers every arena and delivery owner with no mutable
  post-bind attachment;
- physics borrow waits on the same consumer stream, scene update is the last
  physics-buffer reader, and C0 completion occurs there;
- the output lease remains active through B3 delivery independently of the
  physics lease;
- normal close, C0 poison, C4 poison, event-record failure, emergency-sync
  success, and process quarantine retain/release both ownership graphs safely;
- final payload drain and final arena reclaim agree with
  `ShippingPublicationReceipt` and never use replay provenance.

The unavailable Go2 asset continues to fail closed. No substitute asset,
sample, or baseline is created.

## Implementation slices and review boundaries

### C4a — control contract and evidence

- add CPU-safe arena spec, slot/generation/lease state machine, prepared/live
  binding authority, lifecycle evidence, and canonical hashes;
- version the gate/schema distinction between submission trace and arena
  lifecycle evidence;
- keep all result phases and validators closed.

### C4b — CUDA arena and transaction templates

- allocate exact CUDA scene/AABB/LBVH/workspace/render/event slot graphs;
- bind C2/C3 stages per slot/frame and add generation activation over the C2
  single-use transaction contract;
- cover R0/R1 and R2 none/device-only only.

### C4c — Warp native render and arena

- add caller-owned/pre-recorded Warp render commands and reset semantics;
- bind Warp R0, dynamic refit R1, and static CUDA-LBVH R2 slot plans;
- prove exact storage, single stream, no intermediate event, and oracle parity.

### C4d — lifecycle, poison, drain, and guardian handoff

- complete reuse/backpressure/final-drain/poison GPU tests;
- freeze the complete arena resource manifest and C0 guardian handoff;
- do not activate host-payload rows without B3.

### B3 and joint activation

B3 receives a separate design/review for pack, canonical conversion, staging,
ordered D2H, host payload leases, and payload receipts. Only after C4 and B3
implementations pass their joint ownership/lifecycle review may a versioned
activation commit:

- allow final host-payload arena seal;
- replace the runner's C3 one-shot rejection with live C4/B3 authority checks;
- update `ExecutionReceipt` to the reviewed C4 binding/lifecycle types;
- enable P12.4a3b for R0-R2.

S remains closed until its later C0+C4+B3 shipping review.

## Explicitly out of scope

- arbitrary resolution/capacity growth or a general backend registry;
- separate async copy streams, multi-stream event DAGs, or claimed
  copy/compute overlap;
- full async payload, encoding, disk write, or product-default changes;
- renderer math changes, kernel fusion, CUDA refit, SAH/builder tuning, or
  profile-specific performance optimization;
- a shipping CUDA optical adapter or physics numerical changes;
- paired coordinator, performance values, baseline inspection, backend
  selection, fallback, or losing-lane cleanup.

## Acceptance requested

Please approve or request changes on:

1. C4 owning all physical slot/event/delivery storage while B3 owns payload
   commands and host payload semantics;
2. the two-phase prepared-plan/measured-acquire protocol, which preserves full
   pre-enqueue validation while including actual lease bookkeeping;
3. a distinct arena-issued live authority plus serialized C4 evidence rather
   than another caller-provided capability string;
4. session-owned single-stream scratch and strict FIFO depth-one/depth-two reuse;
5. one backend-native terminal event per generation and same-stream B3 in the
   first implementation;
6. the separate submission trace and arena lifecycle receipt, with reclaim no
   longer falsely reported during enqueue;
7. the Warp caller-owned/pre-recorded render binding as a required C4 fairness
   prerequisite;
8. fail-closed poison, emergency synchronization, quarantine, and exact C0
   guardian ownership;
9. C4a-d remaining non-result-capable until the separate B3 design and joint
   activation review.
