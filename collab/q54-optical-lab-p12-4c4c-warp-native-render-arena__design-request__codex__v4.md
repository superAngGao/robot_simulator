# Q54 Optical Lab P12.4 C4c Warp Native Render Arena

- Initiative: Q54 optical computation workflow / P12.4 backend selection
- Stage: C4c design request
- Author: Codex
- Version: v4
- Date: 2026-09-10
- Base: `b4d016609c52f624600860417cb4a111ede51c9c`
- Status: design request for Claude re-review

Supersedes v3. This revision removes the false claim that Python wrappers form
an unforgeable executable-leaf capability. It freezes a closed, trusted worker
execution domain with one fixed IPC protocol and treats in-process C4 authority
as a supported-path lifecycle invariant.

This document does not authorize C4c implementation, C4d, B3, a3b result
submission, baseline access, backend selection, or a product-default change.
The existing result-activation gates remain closed.

## Outcome

C4c will give the Warp optical candidate a real native arena path with the same
ownership and lifecycle standard already implemented for CUDA C4b. It will:

- replace the allocating and blocking Warp render transaction with
  caller-owned buffers and pre-recorded commands;
- execute the candidate only inside a frozen single-lane worker reached through
  the fixed canonical IPC state machine;
- compose the approved C3 scene-update, Warp-refit, and static CUDA-LBVH work
  with that renderer on one native CUDA stream;
- use the shared C4a slot/generation/lease state machine;
- record exactly one preallocated Warp terminal event after the last stage;
- bind every execution to exact a2 allocations, C3 provenance, slot, frame,
  generation, and a single-use arena authority;
- expose generation-scoped device output only to the trusted in-worker
  delivery/correctness path after terminal resolution; and
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

## v4 review corrections

- The security/integrity boundary is the isolated worker process plus its fixed
  canonical IPC schema, not a Python class wrapper or private closure. The
  worker runs only frozen repository code and exposes no callback, plugin,
  import, callable, pointer, stream, event, or arbitrary-object channel.
- Raw Warp CDLL and CUDA-LBVH callables are acknowledged as reachable inside
  that trusted process. Internal wrappers remain validation/instrumentation,
  but are not described as protection against arbitrary same-process Python.
- Diagnostic phase sets now match CUDA exactly: R0 owns three pairs/six records;
  R1 and R2 own five pairs/ten records. They are completion-neutral;
  diagnostics-off has none; every boundary still has one terminal event.
- C4c positively constructs only the one device-only R2 row. The five host R2
  rows must reject before allocation/seal until B3, whose joint review owns the
  complete positive R2 matrix.

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
pinhole raygen
-> reset BVH scalar diagnostics
-> initialized BVH first-hit
-> reset shadow scalar/overlay diagnostics
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

Three fixed reset identities are permitted:

- BVH reset: the two mandatory first-hit scalar diagnostics;
- primary shadow reset: the two mandatory shade scalar diagnostics; and
- diagnostic shadow reset: the same two shade scalars plus the nine distinct
  traversal counters.

The exact shadow-reset variant is selected and recorded at seal time. Splitting
the reset this way matches the CUDA C2 phase semantics: the BVH reset is part of
the `first_hit_kernel` diagnostic interval, while the shadow/overlay reset is
part of `shade_kernel`. Empty-ray work still launches or otherwise records both
required scalar resets according to the frozen dispatch plan, so all `[1]`
outputs are deterministic. No `wp.zero_`, dynamic array collection, host write,
or lazy allocation may replace a recorded reset command in timed execution.

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
- optional, separately typed preallocated timing-event pairs frozen by
  boundary: three pairs for diagnostic R0, five for diagnostic R1/R2, and none
  when diagnostics are off; and
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

A diagnostic arena freezes its phase set by boundary, matching the approved
CUDA arena contract exactly:

```text
R0:
  raygen_kernel:       start -> raygen -> end
  first_hit_kernel:    start -> BVH scalar reset -> first-hit -> end
  shade_kernel:        start -> shadow/overlay reset -> shade -> end

R1/R2:
  scene_update_kernel: start -> scene update -> end
  acceleration_kernel: start -> Warp refit or CUDA-LBVH rebuild -> end
  raygen_kernel:       start -> raygen -> end
  first_hit_kernel:    start -> BVH scalar reset -> first-hit -> end
  shade_kernel:        start -> shadow/overlay reset -> shade -> end
```

R0 therefore owns three pairs and records exactly six diagnostic events. R1
and R2 own five pairs and record exactly ten. Those records are explicitly
permitted in their diagnostic timed paths and are completion-neutral: they do
not increment the terminal event count, create a completion ticket, satisfy
`finish()`, authorize resolve or reclaim, or terminate a device/output hold.
They use distinct preallocated `wp.Event(enable_timing=True)` objects and are
consumed for elapsed-time diagnostics only after the primary terminal event
has resolved. A primary diagnostics-off arena owns no phase events and has no
phase-event ABI or record branches.

The transaction freezes the boundary, exact phase names, event identities,
stage positions, and start/end ordering. A phase event cannot be substituted
for the terminal event, recorded twice, skipped in a diagnostic run,
introduced in a primary run, or recorded outside its registered stage scope.
An R0 transaction cannot own scene/acceleration events; an R1/R2 transaction
cannot omit them.

The terminal event can be recorded for a later generation only after C4 exact
reclaim has retired the prior authority and advanced the slot generation.
Resolution uses `wp.synchronize_event()` outside the timed contamination guard,
then commits the C4 terminal-resolution transition. Event query/poll loops are
not introduced.

`WarpOpticalCompletion` carries the opaque event token, slot/generation,
device/stream identity, and terminal record count, but not the raw `wp.Event`.
Inside the worker, `WarpActiveArenaSample.resolve()` asks the execution driver
to synchronize the registered event for that exact live completion. Neither
the active sample nor its raw event crosses IPC, and no general event accessor
is exported by the worker protocol.

Any stage or terminal-record failure after enqueue begins poisons the
transaction and C4 control arena. It creates no false completion ticket and
cannot return the slot to `FREE`. C4d will add the complete drain,
abort/quarantine, and guardian teardown paths; C4c keeps all owners strongly
held and offers no unsafe partial cleanup.

## 6. Closed worker boundary and supported-path lifecycle authority

Warp 1.12 exposes `runtime.core.wp_cuda_launch_kernel` and
`runtime.core.wp_cuda_event_record` as directly callable CDLL functions after
initialization. Python closures, defaults, globals, `__wrapped__`, and raw
extension modules are also inspectable. C4c therefore makes no claim that a
Python wrapper, underscore, registry, or closure is an unforgeable capability
against arbitrary Python already executing in the same process.

The selected design is the closed-worker alternative. The integrity boundary
is the worker process and its fixed IPC protocol. Code admitted to that process
is trusted, frozen product code; the coordinator, benchmark harness, tests,
plugins, and user code remain outside. C4 authority inside the worker enforces
the supported transaction's lifecycle and catches product bugs, but it is not a
security sandbox against arbitrary in-process code.

### 6.1 Process and code boundary

Both candidate lanes run in separate, clean, single-lane workers created with
spawn semantics. The launcher invokes the frozen interpreter as:

```text
python -I -S <closed-worker-script> <runtime-manifest> <inherited-ipc-handle>
```

The stdlib-only worker script validates the exact interpreter, environment,
repository revision, tracked-file hashes, worker-module allowlist, IPC schema,
and selected lane before adding the frozen repository/site-package paths. The
parent supplies a one-shot inherited pipe/handle bound to the child process and
manifest; it is consumed before the request loop and is not a CLI/environment
bearer token.

The production worker then imports only the allowlisted physics, scene,
acceleration, renderer, arena, delivery, and receipt modules required by that
one lane. It must not provide or execute:

- Python callback, plugin, hook, entry-point discovery, or test injection;
- caller-selected module, attribute, class, function, kernel, extension, or
  callable name;
- `eval`, `exec`, `runpy`, dynamic source, notebook cell, shell command, or
  arbitrary import path;
- pickle, cloudpickle, Python object deserialization, `__reduce__`, or object
  proxying;
- CUDA pointer, allocation, stream, event, module, kernel, graph, DLPack, or
  IPC-memory/event handle supplied by the caller; or
- a debug/reflection command capable of walking globals, defaults, closures,
  `__wrapped__`, `sys.modules`, object attributes, or the Warp runtime.

The worker is single-purpose and does not load both optical backends. It starts
no general REPL/RPC server and accepts no arbitrary benchmark/test callback.
Tests drive the same production protocol from the parent process. A source or
allowlist change requires a new versioned manifest before any result is legal.

### 6.2 Fixed canonical IPC

The transport is a length-delimited canonical-JSON protocol. Control requests
are limited to 64 KiB, control responses to 4 MiB, nesting depth to eight,
object fields to 64, and scalar strings to 4096 UTF-8 bytes. The decoder rejects
invalid UTF-8, duplicate keys, non-canonical numbers, values outside each
field's enum/range/regex, unknown or missing fields, and polymorphic type tags.
It never uses pickle or Python object transport. These are protocol limits, not
memory/performance gates; changing one changes the protocol version. The only
request transitions are:

```text
HELLO(manifest_sha256, lane_id, device_ordinal)
OPEN_FROZEN_SESSION(gate_sha256, workload_set_sha256)
PREPARE_FROZEN_ROW(row_id, case_id, repeat_index, sample_count)
EXECUTE_PREPARED_SAMPLE(prepared_token)
RESOLVE_PREPARED_SAMPLE(pending_token)
RELEASE_RESOLVED_SAMPLE(resolved_token)
CLOSE_SESSION
```

Tokens are worker-issued 128-bit CSPRNG values encoded as exactly 32 lowercase
hexadecimal characters, single-use where the state transition is single-use,
and scoped to the connection and state machine. They are not Python object IDs,
device pointers, native handles, or serialized C4 authorities. Row/case details
are loaded and revalidated from the worker's frozen gate; the coordinator
cannot provide an alternative scene, replay, camera, stage list, output buffer,
callable, delivery endpoint, or acceleration implementation.

All sample tokens, plans, response envelopes, and bounded trace storage are
created by `PREPARE_FROZEN_ROW` outside timing. IPC decode/validation finishes
before the measured call. For R1/R2, the timed interval then begins immediately
before C4 acquire; for R0 it begins at the already frozen render boundary.
Encoding or transmitting the response occurs after the interval. IPC transport
cost is therefore not silently mixed into R-native, while later R-ship/S rows
retain their separately approved product boundaries.

The protocol exposes no stage-level enqueue, event-record, raw-launch,
CUDA-LBVH-call, allocation, synchronization, or introspection operation.
Unknown operations, unknown or extra fields, non-canonical field values,
pointer/handle fields, invalid tokens, wrong sequence, and oversized/deep
messages reject before acquire and before entering the execution driver. A
protocol failure cannot be retried as a different operation in the same
session; the worker closes the connection fail-closed.

For C4c, responses contain only canonical readiness/provenance/lifecycle
receipts, timing scalars after permitted resolution, error codes, and explicitly
approved correctness snapshots copied after terminal resolution. They never
contain a Python object, raw array/view, pointer, stream, event, command,
callable, module name, or reflection token. B3 must separately review any host
payload transport; C4c cannot smuggle it through this protocol.

Correctness snapshots exist only for named, source-frozen C4c validation
fixtures and contain at most 4096 gate-selected ray indices/channels. The worker,
not the caller, chooses that frozen index set. Mandatory performance rows return
no pixel/channel payload. A full image or caller-selected readback is not a
snapshot and remains blocked until B3.

The parent-side `ClosedWorkerLaneProxy` is an orchestration client, not a native
lane object. It can validate canonical response records and correlate issued
tokens, but cannot construct a2 allocations, C3 bindings, C4 leases, or lane
receipts. `PREPARE_FROZEN_ROW` performs a2 preparation and all exact-object
provenance checks inside the worker from the frozen logical case. The current
in-process `SingleLaneSampleRunner` cannot treat this proxy as result-capable;
its eventual worker-aware submission path and activation remain a3b/joint-review
work, not an implicit C4c result outlet.

### 6.3 Trusted execution driver

Inside the closed worker, a production `WarpArenaExecutionDriver` owns the raw
`wp.Launch` objects, Warp events, stream, and R2 CUDA-LBVH callable/arguments.
They are not sent over IPC or returned in receipts. This containment reduces
accidental misuse, but is not described as making Python reflection impossible.

The driver follows the approved C4b mapping:

```text
exact prepared plan
-> exact C4 acquire/lease
-> exact transaction and active owner
-> ordered scene/acceleration/render/delivery stages
-> one terminal record
-> resolve/release/reclaim
```

Construction state can create dormant transactions only. Each real lease binds
once to one transaction, slot, generation, frame, active owner, and C3 evidence.
Every supported stage call rechecks that mapping and its expected ordinal.
`finish()` records the exact preallocated terminal event once and advances C4
lifecycle only after the Warp record call succeeds. cancel/reclaim retires the
mapping; poison forbids reuse and retains all owners for C4d disposition.

The driver may use narrow Python wrappers around `Launch.launch`,
`Stream.record_event`, and the fixed CUDA-LBVH call for structural validation,
operation tracing, and fault injection. Those wrappers freeze command/event/
argument identities, device, stream, stage position, and expected record counts.
They are supported-path checks, not an adversarial native-leaf gate. The exact
trusted worker sources are forbidden from directly calling
`runtime.core.wp_cuda_launch_kernel`, `runtime.core.wp_cuda_event_record`, a
recovered closure/default/global/`__wrapped__` callable, or an unwrapped
CUDA-LBVH entry. This rule is enforced by reviewed code plus the frozen source
manifest/AST policy, not by a claim that CDLL functions are inaccessible.
The policy also rejects reflective `getattr`/module walking on these runtime
objects, `ctypes` access to the Warp core, and bytecode-only repository modules.

A public standalone C3 bundle never becomes an arena plan source. The worker
builds its C4 command graph internally from exact a2/C3 storage and retains no
protocol transition that accepts a caller-provided bundle, stage, event, or
command. Existing standalone C3 APIs remain outside the worker comparison
surface.

### 6.4 Worker evidence and paired fairness

`ClosedNativeLaneWorkerEvidence` freezes:

- protocol and trust-model versions;
- process/interpreter/environment and selected-lane identities;
- repository revision, clean-state proof, allowlisted source hashes, and AST
  policy hash;
- gate/workload/runtime manifest hashes;
- allowed request/response schemas and state-machine hash;
- Warp, Torch, CUDA, compiler, and CUDA-LBVH ABI identities when applicable;
- process-start nonce and exact live worker-session identity; and
- proof that no callback/plugin/object-deserialization surface is enabled.

The exact live worker session, not serialized evidence alone, owns all C4 arena
objects. Stage/template/arena/result evidence references the worker-evidence
hash. For later shipping comparison, Warp-optical and CUDA-optical candidates
use the same supervisor, IPC protocol, physics code, and worker trust boundary;
each process loads only its chosen optical lane. Worker startup and untimed
preparation remain outside R0/R1/R2 and S timing, while the execution requested
by `EXECUTE_PREPARED_SAMPLE` retains the already frozen timed boundaries. C4c
still does not authorize S or benchmark results.

### 6.5 Required boundary and lifecycle attacks

The negative tests distinguish attempts across the supported boundary from
arbitrary code execution inside the trusted worker:

- the parent may pre-import Warp and cache class functions, bound methods,
  `runtime.core` CDLL callables, raw events, raw launches, closure/default/
  global/`__wrapped__` references, and the CUDA-LBVH callable; spawn semantics
  must not transfer any of them into the worker;
- attempts to serialize or proxy any such object fail in the canonical encoder
  before transmission;
- handcrafted JSON requesting raw launch, raw event record, CUDA-LBVH call,
  dynamic import, callback/plugin registration, reflection, pointer/handle use,
  pickle, or an unknown operation/field fails schema validation before acquire;
- numeric values shaped like captured device pointers, event handles, stream
  handles, object IDs, or worker tokens are rejected in every field that does
  not explicitly admit that semantic type;
- a changed worker source, allowlist, protocol, trust-model, gate, or extension
  ABI fails manifest validation before device initialization;
- responses and error envelopes are recursively validated to contain no raw
  object/handle/callable escape; and
- direct template/stage/finish calls through the supported internal driver
  before acquire, with a wrong lease, after cancel/reclaim, or after poison keep
  the existing C4b negative matrix.

Each execution-bypass case that can legally reach preparation starts from a
fresh prepared-but-unsubmitted worker session. Bootstrap/HELLO/manifest attacks
remain in the no-session state. Every rejected IPC attempt must report, from
the worker's canonical error/state envelope:

```text
gpu_submission_count = 0
command_launch_count = 0
diagnostic_event_record_count = 0
terminal_event_record_count = 0
completion_ticket_count = 0
slot_state_sha256 = pre_request_slot_state_sha256
```

The corresponding per-request deltas are also zero and all cumulative counters
equal their pre-request values. This avoids treating an earlier legitimate
sample in another test as activity caused by the rejected request.

Before a session/slot exists, both state hashes use the canonical
`NO_SESSION_NO_SLOT` sentinel. After a prepared slot exists, the worker derives
the hash from its control-plane state before and after dispatch rejection; the
coordinator does not supply either value.

The worker closes the failed session without executing a GPU callable. Tests
also instrument the fixed dispatcher itself and prove that schema rejection
does not reach prepare, acquire, driver, Warp, Torch, or extension entry points.
No test-only callback or arbitrary-code hook is added to the production worker
to manufacture this proof.

An intentionally modified trusted worker that directly calls `runtime.core` is
not treated as a runtime attack that Python wrappers can stop; it is a different
program and must fail the reviewed source/manifest gate. OS debugger injection,
same-user process-memory modification, native-library replacement, and a
compromised interpreter are outside this correctness boundary and are not
misrepresented as covered.

## 7. Exact a2 and C3 provenance

The in-worker product type `WarpArenaPlanTarget` is non-executable. It exposes
immutable plan evidence and resource descriptions to the trusted worker
adapter, not command objects, templates, stages, events, or activation
capabilities. It is never serialized or returned through IPC.

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

The narrow Warp benchmark adapter remains the only dependency on gate rows and
C3 receipt classes. It is an explicitly allowlisted, source-frozen worker
module, not a coordinator-supplied callback. Production `optics` modules never
import `benchmarks`.
Serialized C3 or C4 evidence can be validated and archived but cannot recreate
the live a2/readiness/arena authority.

The provenance DAG is strictly:

```text
closed-worker evidence + exact live worker session
+ a2 prepared input + readiness
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
stream with no intermediate completion event or host wait. A diagnostic arena
may record only its fixed completion-neutral phase pairs.

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
creation, buffer copy, cross-stream wait, or intermediate completion event.
Diagnostic mode may add only the fixed completion-neutral phase records.

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
-> expose generation-scoped WarpDeviceResultView inside the worker
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
- the trusted execution driver's `wp.Launch.launch(stream=frozen_stream)` for
  the exact registered command at the exact stage/resource ordinal;
- the driver's already audited CUDA-LBVH fixed-ABI call on the same native
  stream;
- explicit device-only delivery stage entry;
- in a diagnostic arena, the boundary-specific completion-neutral records at
  their registered positions: six records for R0 and ten records for R1/R2;
  and
- `stream.record_event(preallocated_terminal_event)` exactly once at finish.

The instrumentation must reject inside timing:

- `wp.empty`, `wp.zeros`, `wp.array`, NumPy materialization, or host upload;
- `wp.launch(..., record_cmd=True)` or any dynamic `wp.launch`;
- `wp.to_torch`, `wp.from_torch`, DLPack, or other view creation;
- `record_event()` without the exact preallocated event;
- any unregistered event record, any phase-event record in diagnostics-off,
  any diagnostic record at the wrong phase/position, and any intermediate
  completion event;
- `wp.synchronize*`, Torch synchronize/query/poll, or a host wait;
- extension/module load or compilation; and
- a bridge or second-stream wait.

Negative instrumentation tests invoke the real trusted driver paths without
relying only on high-level self-report. Runtime hooks observe the actual
`Launch.launch`, `Stream.record_event`, CUDA-LBVH, allocation, synchronization,
bridge, and staging entries used by the frozen worker sources. The source/AST
manifest separately proves that those sources contain no direct `runtime.core`
or recovered-callable bypass; it is part of the closed-domain definition, not
runtime evidence. The submission trace records driver operation names;
diagnostic event evidence is separately typed, and reclaim remains only in
arena-produced lifecycle evidence.

## 14. Required tests

### CPU-safe unit tests

- importing arena/contract modules initializes neither Warp nor Torch CUDA;
- the isolated `-I -S` worker, frozen source/import allowlist, exact live
  session evidence, and strict canonical IPC state machine;
- callable/object/pickle/pointer/handle/reflection/plugin/dynamic-import and
  unknown-field/opcode attempts reject before acquire and driver dispatch;
- exact buffer/profile/diagnostic ABI and per-slot manifest derivation;
- positive stage-order construction for every mandatory Warp R0/R1 row and the
  sole C4c-eligible R2 row, `r2_go2_device_only_shadow`;
- pre-allocation/seal rejection for the other five mandatory host-payload R2
  rows while B3 is closed;
- immutable factory, bundle, template, target, prepared sample, and authority
  state;
- canonical one-way stage/C3/C4 provenance DAG;
- same-hash wrong a2 object, same-content reupload, wrong replay view, wrong
  fixture owner, storage reordering, overlap, and alias rejection;
- O(1) acquire check instrumentation independent of frame/resource count;
- all IPC-boundary and supported lifecycle attacks listed in section 6;
- result activation and host-payload arena seal remain closed.

### Single-GPU tests

Positive end-to-end GPU tests launch the exact production worker and use only
its canonical IPC. Focused C4 lifecycle unit/negative tests may exercise the
in-process product classes directly, but cannot emit a worker receipt or
benchmark result and do not redefine the worker trust boundary.

- non-square, non-centered camera with `fx != fy` and non-identity rotation;
- direct origins/directions parity and row-major `py * width + px` boundaries;
- asymmetric plane/triangle/material/light/role scene with hit and miss pixels;
- FULL, RGB_PREVIEW, and RENDER_ONLY; shadows on/off; diagnostics on/off;
- direct channel parity against the existing Warp oracle and retained host/CUDA
  tolerances;
- arbitrary sentinel fill before at least three successful reuses of one slot;
- all exposed miss values and all 4/13 scalar counters reset every generation;
- exact R0, R1 refit, and R2 CUDA-LBVH stage order on a non-default stream;
- one terminal event record, no intermediate completion event, no terminal
  rerecord before reclaim, and safe reuse of the same event only after reclaim;
- diagnostic R0 arenas record exactly six phase events plus one terminal event;
  diagnostic R1/R2 arenas record exactly ten phase events plus one terminal
  event; primary arenas record no phase events; phase events never satisfy
  completion, resolve, release, or reclaim;
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

The R2 scope is intentionally asymmetric while B3 is closed:

| Mandatory R2 row | C4c treatment |
|---|---|
| `r2_go2_device_only_shadow` | positive Warp arena construction and GPU path |
| `r2_go2_rgb_sync_shadow` | seal rejection before allocation |
| `r2_go2_rgb_async_shadow` | seal rejection before allocation |
| `r2_go2_full_sync_shadow` | seal rejection before allocation |
| `r2_go2_rgb8_sync_shadow` | seal rejection before allocation |
| `r2_go2_rgb8_async_shadow` | seal rejection before allocation |

The positive device-only case must use the real frozen Go2 asset. An
asset-missing machine continues to fail closed and cannot substitute another
scene or emit coverage/result evidence. Full positive construction of all six
R2 rows belongs to B3 plus the joint C4/B3 review.

## 15. Implementation and review slices

### C4c0 — closed native-lane worker and IPC

- add the stdlib-only `-I -S` launcher, one-shot inherited IPC channel, strict
  canonical codec/state machine, frozen code/import/AST manifest, and exact
  live worker-session evidence;
- add the trusted Warp execution driver and CUDA-LBVH call path without
  claiming Python/private-callable isolation is a native security boundary;
- prove parent-cached raw functions/objects cannot cross spawn/IPC and every
  callback/plugin/pickle/pointer/handle/reflection/dynamic-operation request
  rejects before acquire and driver dispatch;
- keep arena construction unavailable until C4c0 receives its own review.

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

- run the full C4c-eligible R0/R1 plus device-only R2 construction matrix, the
  five host-R2 seal-rejection cases, sentinel reuse, single/multi-GPU,
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
its Warp execution driver, static Warp/CUDA-LBVH binding, lane adapter,
selection-only tests, and flags after the retained oracle obligation ends. The
durable closed-worker core is specialized to the selected CUDA lane rather than
retaining a dormant Warp path.

If Warp is selected, P12.4e removes the CUDA optical renderer/arena candidate
and paired selector. CUDA LBVH plus the narrow Warp stable-view module may
remain only if the already frozen independent acceleration benefit and
maintenance gate passes; otherwise that acceleration choice requires a new
versioned lane contract and measurement, not a silent fallback. All surviving
modules lose `comparison`, `spike`, and migration naming and move under a clear
production owner. The durable closed-worker core is specialized to the selected
Warp lane and retains no CUDA-renderer dispatch or fallback.

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
3. the closed trusted-worker boundary, strict IPC/state machine, frozen
   code/import policy, honest supported-path authority semantics, and
   raw-callable smuggling attack coverage;
4. exactly six R0 or ten R1/R2 completion-neutral diagnostic records when
   enabled, none when disabled, and exactly one terminal event in every mode;
5. the C4b-strength single-use authority applied through every controlled Warp
   stage, raw executable leaf, and finish path;
6. exact live a2/C3 provenance rather than digest-only binding;
7. the R0, R1 refit, and device-only R2 static CUDA-LBVH transaction graphs,
   with all five host R2 rows rejected before B3;
8. extracting only the production runtime portion of C3e while retaining the
   benchmark adapter boundary;
9. O(1) measured acquire, single-stream instrumentation, and parity coverage;
10. C4c0-C4c4 review slices and continued closure of a3b/baseline; and
11. the two clean terminal repository branches.
