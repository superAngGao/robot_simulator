# Native state observations in runtime custody, and Warp readback restoration

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: bounded design-request
Author: codex
Version: v1
Date: 2026-10-08
Status: in_review
Related Files: native-scope-aggregate gap-review v1; cuda-ambient design v1;
runtime-adoption design v2; runtime-custody implementation-plan v3;
optical_worker/runtime_custody.py; native_bootstrap.py; native_readback.py
Owner Summary: Close the existing native observation requirement using the
current descriptive journal. Repair Warp readback with its audited runtime
scope so the pristine native context is actually restored. No renderer,
performance gate, ownership authority or new wait protocol is introduced.

## Candidate and decision

Code base is 14accac (the packet supplies full commit). Production remains
identical to 9b5702b. Gap report is archived unchanged at b0d433e, SHA256
5e264476a233c42aa55929f70d4735e2258059df2481b32d4ee87f92f9d8dc66.
This is an exact design request, not implementation or positive GPU permission.

F1 is addressed by a new aggregate version with one typed journal event.
F2 is addressed by preserving both Warp's default metadata device AND prior
native current context across the existing validation call. We do not extend
CUDA's logical/raw exception to Warp. Interior observations explicitly describe
the target context selected by an enclosing scope; ambient restoration is
checked at its return boundary, never asserted at every interior helper.

The new observer is confined to the fixed sacrificial test driver. Production
collection remains CPU-only and cannot invoke getters, callbacks, imports,
storage walks, synchronization or lifecycle operations. The observer wraps
the actual existing calls, calls each original exactly once and records their
primitive results. It is not a replacement uploader, loader, guard or authority.

## Warp correction and operation accounting

In native_readback._readback_prepared_native_input, after existing source/
prepared/runtime/target/stream identity checks and host-retention setup, enter
one explicit target leaf before entering Warp's ScopedDevice(target):
operation_id indexed_validation_scope, kind runtime_import_or_init,
runtime_identity warp_scoped_device_v1, existing target descriptor rules.
The scope encloses the existing array loop, completion-event create/record/wait,
storage check and CPU validation. Return the actual report only after scope
exit. CUDA retains its current helper-level indexed scopes and call sequence.
Use a small internal body factoring if needed, without duplicating either
lane's array loop, custody policy or validator. The new Warp leaf accounts for
the scope's native context selection; it does not create an ARM or renew time.
Its one extra setup-work count is intentional and tests must assert it.

The actual existing ScopedDevice stores/restores the previous default-device
metadata and native context. Inspect the pinned __enter__/__exit__ and
ContextGuard source in the implementation preflight. No global Warp restore
policy setter, direct context setter in test code, device0 fixture, reset or
extra synchronization is permitted. Exception unwinding uses that same runtime
scope, followed by the original sticky session/controller failure policy.
Never catch a restore failure to pretend success. Returned host/device owners,
events and report remain in their original retained graph before fallible work.
The native dependency exception-safety obligation remains binding.

## Version and ownership

New emitted records use q54.p12.4.runtime-custody.v2, with exactly the existing
top-level fields, summary fields, object table and component payloads. Add only
the native_scope journal kind described below. Retain v1 decoding with its
exact old grammar: native_scope is forbidden in v1. V1 records keep their old
bytes and coverage and cannot pass native-scope acceptance. V2 records without
observations remain useful early/failure custody records, but cannot pass that
acceptance either. A version alone is never a positive result.

The selection sidecar and trusted launch inputs stay v1: their semantics are
unchanged. Their CPU durable reader accepts either explicitly supported record
version using the shared record codec, not by rewriting version strings.
No schema is copied into a second authority. Old record fixtures and real
archived evidence remain regression inputs.

The already-held SourceSession.runtime_observation owns every appended event
and observed object token. The fixed driver strongly holds its observer,
original functions, loaded library handles/functions and returned owners
through session.runtime_owners BEFORE any fallible observation or adoption.
Production returned-owner recording remains before stream metadata queries.
The driver adds no observer callback to a production API and no runtime token.
Fixed monkeypatches remain entirely outside the captured production closure.

## Exact native_scope payload

The existing envelope {index, kind, guard, lifecycle_event_index, details}
is unchanged. details has exactly:

{site, edge, before_index, operation_event_index, worker_pid, thread_id,
 module_token, runtime_token, target_token, stream_token, binding_token,
 libraries, state}.

All integers are exact int (never bool); pointers/handles are nonnegative
integers, ordinals 0..1023, PID/TID positive. No arbitrary JSON extension.
worker_pid is actual os.getpid(); thread_id is actual Linux /proc/thread-self
resolved TID. The driver requires PID==TID, and every event in this record must
repeat them. Acceptance compares them to the same run's externally observed
worker PID/start/ancestry, never merely to another file's self-report.

site is one of runtime_stream, runtime_binding, upload_scope, readback_scope,
empty, copy, event_create, event_record, event_wait, terminal_sync.
edge is before, after, or point. Only runtime_binding uses point; all others
use before/after. before_index is null for before/point; after references the
exact preceding unmatched before journal index of the same site and operation.
Pairs are properly nested, not overlapping/crossing. A thrown/hung call may
leave its before unmatched; that is structural partial evidence, never success.
No duplicate before/after/point or second pairing of an occurrence is allowed.

operation_event_index names an actual lifecycle event at or before the envelope
index, under the exact same original guard. For runtime_stream it names
low_level_leaf_entered for indexed_runtime_stream; runtime_binding names its
actual runtime_setup_binding owner_adopted event. For upload/readback scope,
it names the indexed_upload_scope/indexed_validation_scope entered leaf.
Helper sites name their actual allocation/copy/create/record/wait entered leaf,
matching the existing descriptor's leaf_kind and runtime_identity. A helper
pair has no intervening different entered leaf. The enclosing scope pair may
contain its actual nested helper leaves. terminal_sync names the actual
emergency_sync_started event, not a setup leaf or future completion event.
The envelope itself always references the last actual lifecycle event present
at append, including actual intervening adoptions. No synthetic counters.

module_token is the exact held torch or warp external file-module token.
runtime_token is exact torch._C file-module or adopted warp_runtime_root token.
target_token is the actual adopted Torch descriptor/Warp target. stream_token
is null only for runtime_stream-before; its after uses the exact stream already
held and recorded by returned_owner. Later stream_token equals runtime_binding.
binding_token is null through runtime_stream, otherwise equals the preceding
runtime_binding journal fact. The runtime_binding point is appended immediately
after that actual journal append, still in the same original setup ticket.
All applicable module/runtime/target/stream identities must agree with the
actual binding when it exists; failed prebinding cuts still require prior
module/root/target custody and matching returned-owner facts when non-null.
Do not create a canonical-source token before its existing binding occurrence.

libraries is an ordered list of exact {root, path, sha256} rows referencing
native files in the separately trusted RuntimeEnvironmentContract. CUDA has
one libcudart row; Warp has one libcuda row, at the pinned paths below. The
driver verifies hashes AND that the same resolved library is already mapped
in /proc/self/maps; it acquires only RTLD_NOLOAD|RTLD_LOCAL, with no fallback.
The descriptor alone is not proof of dlopen identity; retained handles and
actual fixed producer/source/environment are the trusted execution boundary.

CUDA state has exactly:
{initialized, logical_device, logical_basis, raw_device, primary_contexts,
 target_current_stream, held_stream}.
initialized is exact bool. logical_basis is cold_source_derived or native_return.
Before first stream only: initialized=false, logical_basis=cold_source_derived,
logical_device=0, target_current_stream=null, held_stream=null. The derivation
requires the separately observed raw0, no primary contexts, exact inspected
queued callbacks/lazy seed state and the pinned C10 source; do not call a
warming logical/current-stream getter. Other points have initialized=true and
logical_basis=native_return. Actual contradictory values are recorded, not
normalized; acceptance below rejects them. primary_contexts is an ordinal-
ordered complete list of {ordinal, active}, active exact bool. held_stream is
null only before creation, otherwise {ordinal, handle} from the actual stream.
target_current_stream is null before creation and actual nonnegative native
current raw stream afterward. It need not equal the explicit held upload stream;
restoration compares it to its own postbinding baseline.

Warp state has exactly:
{default_device, current_context, current_context_device, primary_contexts,
 target_context, target_current_stream, held_stream}.
default_device is the actual CPU metadata selected CUDA ordinal (exact int).
current_context is null for a successful null query, else positive pointer;
current_context_device is null iff current_context is null, otherwise the
actual current driver's CUdevice mapped back to an observed ordinal.
primary_contexts is complete ordinal-ordered {ordinal, flags, active}, exact
nonnegative flags and exact bool active. target_context and stream fields are
null only before real stream initialization. Thereafter target_context is the
already-created target's exact held _context pointer, positive; held_stream is
{ordinal, handle}; target_current_stream is the actual already-initialized
target current stream handle. Pointer equality with target_context plus ordinal
mapping is required for an interior target-context claim, not pointer alone.

Ordinal lists have exactly the actual enumerated count; count is frozen as8 in
this183 positive deployment. The fixed launch saves exact ordinal->UUID->driver
mapping (GPU0 a4b256c9..., GPU1 3a3112ff..., driver595.71.05), verifies no remapping
environment, and binds it to the actual environment/process evidence. Generic
CPU schema validates contiguous ordinal lists, not hard-coded host identities.

## Native observer ABI and audited ordering

CUDA reuses the independently reviewed historical captured observer call chain:
libcudart cudaGetDevice(int*) -> int, exact return0; cold _cuda_getDeviceCount
and _cuda_hasPrimaryContext for every ordinal; after real initialization,
_cuda_getDevice and _cuda_getCurrentRawStream(explicit target). Its libcudart
file is nvidia/cuda_runtime/lib/libcudart.so.12 under pinned site root, SHA
c3a75b33af334a3486d197dbd1584a2985183ba4688d237a2be5f2f679329920.
Historical outputs are source/preflight evidence only; new rows must come from
the current execution. Same callback/seed/Triton checks remain required.

Warp observer initializes only after actual quiet wp.init(), in the existing
indexed_runtime_stream leaf before original get_stream(target). It uses
os_native/libcuda.so.595.71.05 SHA
76e0d9678d41cf6b6ae71d18549d88a963d3d434f08108baa12457eb53ca88d6.
Freeze ctypes ABI: CUresult/CUdevice=c_int, CUcontext=c_void_p, flags=c_uint;
cuDeviceGetCount(POINTER(c_int)); cuDeviceGet(POINTER(c_int),c_int);
cuDevicePrimaryCtxGetState(c_int,POINTER(c_uint),POINTER(c_int));
cuCtxGetCurrent(POINTER(c_void_p)); cuCtxGetDevice(POINTER(c_int)).
All restype c_int; require return0 before consuming output. Resolve each device
handle by cuDeviceGet, never assume ordinal encoding. Call cuCtxGetDevice only
after successful nonnull current-context query; map its handle using the
already observed one-to-one mapping. Query state in fixed order: current
context, conditional current device, primary states; record default metadata
without any make_current or context property call. No primary retain/release,
setter, memory query, cudaMalloc workaround or extra sync is an observer call.

After original stream creation, use the exact target already-created _context
and existing current-stream cache through the audited target stream getter.
Before approving that getter for execution, code preflight must show its
get_context_info cache is already initialized by that same real stream path;
otherwise change the proposed getter through fresh review, not a fallback.
Do not use wp_cuda_context_get_current (it conflates errors and null) or
wp_cuda_context_is_primary (it can temporarily retain a primary context).

## Sites, scopes and normal acceptance

Install fixed wrappers after trusted baseline issuance and before runtime
initialization. Observe runtime_stream before only AFTER original _runtime_leaf
has entered, and after only AFTER original returned_owner observation. Observe
runtime_binding only after the actual binding journal entry. Helper wrappers
hold any real returned owner immediately before the after observation, then
return it to the original caller's adoption. They never synthesize success on
throw, reexecute calls, alter current streams, or use their own uploader.

For CUDA, observe both edges of all five existing helper kinds across upload
and readback (including zero-size call occurrences) plus initialization/binding
and terminal. Its helper scopes restore logical0 and the postbinding current
target stream, while raw remains1 and only primary1 is active.

For Warp, additionally wrap the actual existing ScopedDevice object used by
upload and the new readback scope, delegating its original enter/exit once.
Observe before actual enter and after successful actual exit. Interior helper
edges must see default1, native context equal to held target_context, only
primary1 active, and unchanged actual target current-stream selection. Before
and after each outer scope must see default0, null native current context and
only primary1 active. Initial prestream must see default0, null current context
and all primary states inactive. Poststream/binding must see default0, null
current context and only primary1 active. No device0 setter or stream fixture.

Terminal observations wrap the one existing explicit target sync callback;
observe before and after the actual original call while its original emergency
ticket is live, before guardian release. Warp's existing ScopedDevice sync
restores default0/null context; CUDA restores logical0/raw1. Neither observation
is a synchronization completion fact: the existing lifecycle records that.

CPU record validity checks types, occurrence order, source/environment/tokens,
same phase/guard, pair nesting and allowed sites. It does not reject a completed
snapshot merely because a primary context is foreign or selection differs.
The fixed live acceptance checker appends the factual row FIRST, then checks
expected state. Contradiction poisons via existing UNKNOWN/session failure,
and cannot be cleared by subsequent good rows. Required accepted positives
have all expected pairs and outer scope returns, complete input validation,
current full terminal record/selection, one target sync, authorized release,
and actual worker0/reap with no owned OS/GPU residual. Count comparison alone
does not replace matching every actual helper/scope occurrence and descriptor.

## Emergency, partial observations and immutable cuts

The existing setup rule remains unchanged for every other journal kind.
native_scope terminal_sync alone may be appended under the original armed
emergency guard after emergency_sync_started and before its completed/failed
event. There must be the preceding actual runtime binding and validated input,
no prior UNKNOWN/foreign history, and no cleanup/release/disarm at that index.
No native observer runs following sticky failure, failed sync or ordinary
release. A successful sync followed by observer failure must not release or
retry sync: propagate through the original emergency failure/quarantine path.

A getter error does not yield a partially fabricated state. Keep an ordered
raw diagnostic of successful getter returns and exact error, held by session;
append no complete native_scope row for that attempt, then fail the original
segment. An unmatched before remains truthful partial coverage. If a complete
row append itself fails, preserve the raw snapshot/returned owner and follow
the existing immutable validated-cut fallback without trimming/repairing tails.
The CPU collector does not query again during failure or terminal projection.

Selection retains the current/prior_validated_cut/unavailable semantics and
record-before-sidecar write order. A selected cut contains only its issued
rows; raw newer diagnostics never extend it. Acceptance rejects stale pairs
against an available full terminal worker trace, and reports incomplete scope
coverage on killed prefixes. A later write failure does not undo an authorized
release, but cannot satisfy aggregate acceptance. No success inferred from
missing after, null getter output on error, old records, or matching counts.

## Verification and review gates

After design approval, implement the bounded production/readback/schema and
fixed driver changes, repin captured source closure/catalog/policy, then freeze
for independent code/native-scope preflight before positive Warp GPU attempts.
Review the actual new ABI/getter paths, original delegation and custody graph.
Use CPU regressions plus actual v1 archive replay to preserve old meaning.

New CPU mutation proof must reject wrong versions/fields/types, token/root/
binding substitutions, stale guards, wrong leaf/runtime identity, crossed/
duplicate/reordered pairs, wrong phase, illegal emergency sites, absent return
ownership and observed/derived category misuse. Preserve contradictory valid
rows and reject them at acceptance. Exercise getter failure versus genuine
null, returned owner before observer/adoption failure, append/serialization/
persistence cuts, stale durable pair versus available full terminal trace.

After exact code preflight, run actual both-lane asymmetric/replay/both-Go2
positives; actual nonempty Warp D2H distinguishes F2 repair. Run fixed negative
observer failures before/after real return, real foreign stream and UNKNOWN
pre-return, CUDA C1 stream-only copy, and Warp missing-scope counterexample
using original unscoped body only in the sacrificial negative driver. All
contradictions must preserve actual values, poison/quarantine, and reap under
unchanged ACK/deadline rules. Existing full native/ACK/EOF/hang/cleanup/exit
matrix and C0/source regressions still apply; no test subset is acceptance.
Then request fresh independent whole-C2d3 review and fix/review until accepted.

This observation fulfills a frozen prerequisite for reliable selected-backend
preparation and fair native-ready inputs; it is not new product infrastructure.
Retain selected-lane descriptive schema/evidence under P12.4 cleanup rules;
delete temporary wrapping/paired validation with their already-defined gate.
C2e/f,C4c3/c4,C4d/B3,product GPU preparation,a3b,result/baseline,host,R2,
shipping remain closed. C4 remains active; no performance or backend decision.
