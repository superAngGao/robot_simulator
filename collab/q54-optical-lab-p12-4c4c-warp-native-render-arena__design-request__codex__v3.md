# Q54 Optical Lab P12.4 C4c Warp Native Render Arena

- Initiative: Q54 optical computation workflow / P12.4 backend selection
- Stage: C4c design request
- Author: Codex
- Version: v3
- Date: 2026-09-10
- Base: `28f70bbfc209694618d93d6ffae314444f63a6bd`
- Status: design request for Claude re-review

Supersedes v2. This revision aligns diagnostic phase coverage with CUDA for
each boundary and replaces late class monkeypatching with a mandatory,
process-first, permanent Warp worker bootstrap.

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

## v3 review corrections

- A clean Warp lane process must enter the permanent gate bootstrap before any
  physics, C3, optical, benchmark, test, or extension consumer imports Warp or
  obtains a Warp/CUDA-LBVH callable. Pre-import or cached-entry evidence makes
  the lane ineligible before arena allocation; the gate is never restored.
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
`WarpActiveArenaSample.resolve()` asks the executable vault to synchronize the
registered event for that exact live completion; no general event accessor is
exported.

Any stage or terminal-record failure after enqueue begins poisons the
transaction and C4 control arena. It creates no false completion ticket and
cannot return the slot to `FREE`. C4d will add the complete drain,
abort/quarantine, and guardian teardown paths; C4c keeps all owners strongly
held and offers no unsafe partial cleanup.

## 6. Arena-controlled execution is a capability, not a private attribute

C4c adopts the final C4b authority model from the first implementation. Merely
hiding a template, stage, command bundle, or event under an underscore is not
accepted as execution control.

That rule extends to the actual Warp executable leaves. The controlled resource
set includes:

- every raw `wp.Launch` used for scene update, refit, render reset, raygen,
  first-hit, or shade;
- the exact bound CUDA-LBVH callable/argument graph used by Warp R2;
- every diagnostic start/end `wp.Event`;
- the primary terminal `wp.Event`; and
- the exact Warp stream/event-record entry used by those events.

### Process-first bootstrap, executable vault, and leaf gate

An arena-controlled factory does not return or retain the existing public C3
bundle shape with `_commands`. Scene/refit/render command recording is factored
into one internal helper with two sinks:

- standalone C3 sink: returns the existing one-shot bundle and remains useful
  for C3 proof tests, but can never be attached to a C4 arena;
- controlled C4 sink: registers each raw launch in a production-owned
  executable vault and returns only opaque, exact-type stage/leaf identities.

Raw `wp.Launch`, raw terminal/diagnostic `wp.Event`, CUDA-LBVH callable/argument
objects, and raw CUDA event handles do not appear in the object graph reachable
from a public target, template, stage, completion, active sample, or the
arena-private executable descriptor. Those objects expose tokens and immutable
evidence only. The strong raw owners live in a module-private registry retained
for the arena session. C4d's guardian receives one opaque vault-owner capsule
that keeps those entries alive; it does not receive an enumerable tuple of raw
launch/event handles.

Hiding the raw object is defense in depth, not the authority check. Nor can C4c
install a class monkeypatch after physics or C3 has already imported Warp: an
old unbound function or bound method would remain callable. The only legal C4c
process is therefore a fresh, isolated worker started through a production
`warp_runtime_bootstrap_v1` entry point.

The launcher uses a new interpreter with spawn semantics, not fork inheritance,
and invokes the frozen interpreter as:

```text
python -I -S <bootstrap-script> <runtime-manifest>
```

The stdlib-only bootstrap therefore runs before importing any repository module
and without `PYTHONPATH`, user-site, `sitecustomize`, or an arbitrary benchmark
callback. It establishes the approved repository and environment package paths
from the frozen runtime manifest, verifies that Warp, the CUDA-LBVH Python
module, and its generated extension are absent from the import ledger, then
performs the first Warp import itself. It immediately installs permanent
production-owned gates over the actual `wp.Launch.launch` and
`wp.Stream.record_event` class entries before returning the Warp module or
importing physics, C3, optics, lab, benchmark, or test code.

The parent launcher also gives the child a one-shot inherited pipe/handle
capability bound to the child process identity and runtime-manifest hash. The
top-level bootstrap consumes and closes it before adding package paths. It is
not a command-line or environment bearer token and is never exposed to the
application request loop. Consequently, importing or executing the bootstrap
module later inside an arbitrary process cannot mint a valid startup authority.

The bootstrap has no uninstall, restore, late-attach, or in-process upgrade
operation. An already-running process that imported Warp cannot acquire C4c
capability and must be replaced by a clean spawned worker. A process that
pre-imported Warp, removed it from `sys.modules`, loaded the CUDA-LBVH extension,
or otherwise lacks the bootstrap's first-import attestation rejects before
device initialization, arena allocation, command recording, or result
capability. C4c does not claim it can discover arbitrary Python references;
instead it proves that no consumer code ran in the worker before the permanent
gate existed.

Bootstrap is a top-level worker state machine, not an importable `bootstrap()`
function:

```text
FRESH_ISOLATED_PROCESS
-> RUNTIME_PATHS_VERIFIED
-> WARP_FIRST_IMPORT_GATED
-> CUDA_LBVH_FIRST_LOAD_GATED (only when R2 is prepared)
-> APPLICATION_RELEASED
```

The exact live `WarpRuntimeBootstrapAuthority` can be minted only while the
top-level script is in the pre-application states and is sealed before handing
off to the worker request loop. There is no transition from
`APPLICATION_RELEASED` back to a bootstrap state, and serialized evidence,
module re-import, or a caller-created object cannot mint that authority.

CUDA-LBVH extension loading follows the same ordering. Only the bootstrapped
production loader may perform the first load, during a single-threaded untimed
prewarm phase before the worker opens consumer callbacks or starts helper
threads. While the import lock is held and before the generated module is
returned to any consumer, it replaces the exact fixed-ABI R2 entry with the
permanent controlled entry and retains the raw pybind callable only in the
bootstrap-owned private closure. The worker refuses a pre-existing module,
concurrent first load, or later entry replacement. A call whose exact argument
graph is registered to a C4c arena requires the live execution scope;
non-arena C3 calls remain pass-through. Direct extension import or a cached raw
callable predating bootstrap invalidates the worker rather than being tolerated
as a legacy path.

After bootstrap, every consumer can obtain only the installed gate entries.
The original class/extension callables are retained solely by the permanent
bootstrap capsule, never by an arena/vault object or exported restoration API.
For an unregistered Warp resource the class wrappers are pass-through, so
physics and standalone C3 behavior do not change. For a registered resource the
wrapper performs an O(1) exact-identity lookup and requires the active execution
scope to contain the registered transaction, live authority, active owner, C4
lease, stage ordinal, resource ordinal, device, and stream.

Module import remains CPU-safe for ordinary contract modules. Starting a Warp
worker is the explicit operation that imports Warp and initializes the gate,
and it is outside every timed boundary. The gate remains installed until
process teardown, including while a controlled session or quarantined vault
capsule is live.

`WarpRuntimeBootstrapEvidence` freezes the isolated-worker policy, interpreter
and environment identity, process-start nonce, first-import attestation, Warp
version, exact wrapped class/method identities, CUDA-LBVH ABI/entry identity,
wrapper source hash, and installation manifest digest. The exact live bootstrap
authority and evidence are required at arena seal and preflight. Stage,
template, and arena evidence reference the independent evidence hash. The hash
is provenance only; the permanent gates plus live registry/scope identity are
the execution authority. Any bootstrap/gate/schema change before results
requires a versioned artifact update.

For the later paired shipping comparison, both clean single-lane workers use
this same bootstrap, including the CUDA-optical candidate whose physics producer
still uses Warp. This keeps any pass-through class-entry overhead inside the
same physics-to-image boundary on both sides. Worker/bootstrap startup remains
outside R0/R1/R2 and the S boundary, but its exact attestation is receipt-bound.
C4c itself still does not authorize S execution or results.

The gate also freezes controlled launches against mutation. `Launch.set_dim`,
`set_param_at_index`, `set_param_by_name`, `set_params`, and their ctypes
variants reject for a registered launch after seal. A caller that captured the
raw launch while setup recorded it therefore cannot execute or rewrite it
outside the live scope. The arena verifies the installed class-wrapper
identities at seal, full preflight, and the bounded per-stage check. Replacement
before first enqueue rejects; replacement after submission may have begun
poisons the arena.

Each launch registration also freezes the raw object's kernel, module
executable, hooks, parameter-container/address-container, bounds, block policy,
adjoint flag, device, and expected-stream identities. The leaf gate compares
that fixed structural record before dispatch. Replacing a raw attribute is
therefore rejected; supported parameter mutation through any public `set_*`
entry is independently gated after registration.

For a registered event, the `Stream.record_event` leaf gate additionally
requires:

- the exact registered stream object and native handle;
- the exact preallocated event object;
- no `event=None` allocation and no caller-selected `external` mode;
- the registered terminal or diagnostic role; and
- the exact finish or diagnostic phase position.

Within a controlled execution scope, recording an unregistered event is also
rejected. Thus a captured raw terminal event cannot be recorded or re-recorded
directly, and a new intermediate event cannot be inserted. The terminal
lifecycle callback runs only after the original Warp event-record call returns
successfully.

The leaf registry and active scope are production mechanisms and import no
benchmark code. Registration/mutation uses a lock outside timing; the hot-path
lookup is fixed-size under the Python GIL and does not walk the arena graph.
The active authority is carried in a `ContextVar` so another thread or nested
submission cannot inherit it accidentally. Re-entrant controlled execution is
rejected.

C4c never unregisters a controlled raw resource as part of cancel, reclaim, or
ordinary object finalization. Those operations retire the generation authority,
not the session resource identity. C4d may remove registry entries only after
its approved device-quiescence and zero-external-hold release proof; quarantine
retains the opaque vault capsule and all entries until process teardown.

The exact sequence is:

1. Arena seal registers the controlled template, all opaque stage identities,
   every raw executable leaf, all phase events, and the terminal event against
   a private construction capability.
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

6. Every stage enqueue and `finish()` opens the exact execution scope. Each raw
   Warp launch/event-record leaf independently rechecks that scope and the
   current C4 lifecycle phase. The R2 CUDA-LBVH entry is retained only in the
   production-controlled callable's private closure; that callable checks the
   same live scope immediately before invoking the approved fixed-ABI
   extension entry.
7. cancel and reclaim retire the registration; poison retains it and forbids
   remint.

Construction capability can create dormant transactions only. It cannot mint
live authority. A lease can be registered once only. Authority types are exact
sealed types minted in the Warp timeline module; module/name strings,
duck-typed methods, copied objects, and serialized evidence are not authority.

A public standalone C3 bundle never becomes arena-controlled and is rejected as
an arena plan source. The corresponding controlled sink instead returns an
opaque command capsule whose enqueue requires the exact authorized transaction;
there is no standalone enqueue entry on that capsule. This rule also covers
render and the static CUDA-LBVH stage. Reaching an arena-private executable
graph, or retaining a raw command/event during setup, cannot bypass the C4 lease
or record the slot event while the lifecycle still says `FREE`.

Required attack tests mirror and extend the C4b findings:

- in an unbootstrapped subprocess, import Warp and cache the unbound
  `Launch.launch`/`Stream.record_event` functions, bound launch/record methods,
  and the raw CUDA-LBVH callable before invoking bootstrap; bootstrap must
  reject the process before device/arena allocation, even if module entries are
  subsequently removed from `sys.modules`;
- reject fork-inherited Warp state and any worker whose first-import
  attestation, permanent wrapper identity, or generated-extension load order is
  missing or changed;
- in a clean bootstrapped worker, prove that class and bound-method references
  obtained by every later physics/C3/optical consumer are the permanent gated
  entries;
- direct template activation before acquire;
- direct scene/refit/render/static-LBVH stage enqueue before acquire;
- capture a registered raw `wp.Launch` during setup, then call
  `raw_launch.launch(...)` before acquire, after cancel/reclaim, and from a
  wrong transaction scope; repeat through `type(raw_launch).launch(...)`;
- capture the registered raw terminal event during setup, then call
  `stream.record_event(raw_event)` before acquire, before finish, after reclaim,
  and from the wrong stream;
- mutate a captured registered launch through every supported `set_*` entry
  and through direct raw-attribute replacement;
- invoke the cached post-bootstrap CUDA-LBVH entry with a controlled argument
  graph before acquire, after cancel/reclaim, and from the wrong live scope;
- direct terminal event record through the transaction finish path;
- duck-typed or same-name authority;
- copied/unregistered authority;
- one real lease minted twice;
- frame-0 lease applied to a frame-1 transaction;
- empty/partial/reordered execution binding;
- authority reuse after cancel, reclaim, or poison; and
- a reached private executable attempting a complete transaction while the
  C4 slot remains `FREE`.

Every pre-submit rejection must precede the first command launch and leave the
terminal and diagnostic event record counts unchanged. Tests obtain raw objects
by spying on setup-time Warp constructors/recording calls; no production debug
accessor is added merely to make the attack possible. A separate test confirms
that a standalone C3 bundle remains executable only against its own non-arena
storage and is rejected as a C4 plan source.

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
runtime bootstrap evidence + exact live bootstrap authority
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
- leaf-gated `wp.Launch.launch(stream=frozen_stream)` for the exact registered
  command at the exact registered stage/resource ordinal;
- the permanently gated, already audited CUDA-LBVH fixed-ABI call on the same
  native stream;
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

Negative instrumentation tests invoke the real stage paths without relying on
high-level self-report. They also verify that the permanent bootstrap gate was
the first Warp/CUDA-LBVH consumer in the isolated worker and remains installed
for its entire lifetime; benchmark instrumentation may observe that wrapper but
may not replace it with an authority-free path. The submission trace records
the gate operation names; diagnostic event evidence is separately typed, and
reclaim remains only in arena-produced lifecycle evidence.

## 14. Required tests

### CPU-safe unit tests

- importing arena/contract modules initializes neither Warp nor Torch CUDA;
- the isolated `-I -S` worker bootstrap is the first Warp/CUDA-LBVH importer,
  produces exact live attestation, and is permanent for the process;
- every pre-import, cached-original, removed-`sys.modules`, fork-inheritance,
  late-attach, restoration, and wrapper-replacement case rejects before arena
  allocation;
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

### C4c0 — isolated Warp worker bootstrap

- add the stdlib-only `-I -S` launcher, frozen path/runtime manifest,
  first-import attestation, permanent Warp class-entry gates, and exact live
  bootstrap authority;
- add the controlled CUDA-LBVH first-load hook and permanent fixed-ABI entry
  gate without loading or compiling the extension in a timed boundary;
- prove clean-worker startup and every pre-import/cached-entry/fork/restore
  rejection in isolated subprocess tests;
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
its process-bootstrap leaf gates, static Warp/CUDA-LBVH binding, lane adapter,
selection-only tests, and flags after the retained oracle obligation ends.

If Warp is selected, P12.4e removes the CUDA optical renderer/arena candidate
and paired selector. CUDA LBVH plus the narrow Warp stable-view module may
remain only if the already frozen independent acceleration benefit and
maintenance gate passes; otherwise that acceleration choice requires a new
versioned lane contract and measurement, not a silent fallback. All surviving
modules lose `comparison`, `spike`, and migration naming and move under a clear
production owner. The permanent Warp process bootstrap remains only as the
selected product lane's launch and actual-leaf authority boundary.

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
3. the isolated, process-first, permanently gated Warp/CUDA-LBVH bootstrap,
   executable vault, live registry, and cached-entry attack coverage;
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
