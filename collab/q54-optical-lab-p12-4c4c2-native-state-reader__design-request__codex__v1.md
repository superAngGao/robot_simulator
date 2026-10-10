# One native-state reader for captured worker preparation

Initiative: Q54 / P12.4 / C4c2
Stage: bounded design-request
Author: codex
Version: v1
Date: 2026-10-10
Status: in_review
Related Files: worker-intake gap-review v1; runtime-integration design v3;
native-scope-aggregate design v2; native-scope-cache design v1;
fixed-CUDA-artifacts design v2; runtime-custody implementation-plan v3
Owner Summary: Remove the production CUDA intake dependency on test wrappers.
Move the already reviewed read-only native queries into one captured internal
component, called explicitly by preparation. Keep the CPU journal/replay,
resource owners, native helper bodies and failure rules. Remove duplicate test
query/state logic. This is a prerequisite repair within the complete C4 goal.

## Evidence and scope

Code base: 7c8c3ec (the review packet supplies its full SHA). The frozen gap
candidate was26542543d067d20c64670d375065535420ede203. Independent review
session01a124d3-a32b-7d43-b6bb-1262e8c90d69 returned Request changes. Its full
report is archived as worker-intake gap-review v1, SHA256
b78661032b29b7e84e03a59c5f6a5c5642bd155fc63c7f5265669730f1853c34.

Real CUDA loading already retains the created modules before context unwind,
but its production custody replay requires six native before/after pairs which
only the test observer supplies. Removing those checks would weaken the
accepted artifact contract. Warp's two-sample smoke proves lifecycle progress,
but its zero native-scope rows do not prove native restoration. The reply-depth
defect is an independent bounded fix at7c8c3ec, not resolution of either gap.

This amendment replaces the native-scope design's test-only restriction for
the existing preparation observation sites. It narrowly extends the source
policy's ctypes exception. It does NOT move native queries into
_RuntimeObservation or runtime_custody.py; both remain CPU-only consumers.
No schema version string is relabeled to bypass an existing join.

Scope is the already specified runtime initialization, CUDA artifact create/
exec, upload and input-validation native observations, plus the existing
eligible input-stage emergency-sync observation. Later guarded preparation
tests may use this same reader for their existing raw scope diagnostics.
There is no second reader implementation in tests. This repair does not claim
to finish gap-review F2's final-sample/resolve/close physical-scope coverage:
that remains an explicit required part of the unfinished C2f integration.
In particular it does not introduce queries, extra guard handshakes or new
containers into EXECUTE/acquire/enqueue/finish/resolve/reclaim/normal-close.

## Internal component and ownership

Add optical_worker/native_scope.py as the sole query/state implementation.
It defines a fixed internal reader constructed only by the captured source
session, from its actual controller, admitted environment, held runtime module,
runtime root and target. There is no observer/callback/path/symbol argument on
worker IPC or a public preparation API. No injected function executes on behalf
of production collection. Production does not import tests or monkeypatch
runtime modules, source methods, helpers or journal append.

SourceSession owns exactly one reader. Retain the reader and its mutable owner/
raw-diagnostic containers in the existing session graph before initialization
can acquire a handle or make a native query. Retain every library/function
immediately after its return, before configuring metadata or calling it.
The source retains the exact module/runtime/target/stream/binding identities;
the reader validates those same identities and owning thread. It is neither
readiness, a resource guardian, a new wait authority nor a source of ownership.
All reader owners are covered before the existing single guardian seal. The
reader cannot attach resources or issue native calls after ordinary release.

Connect initialization and artifact calls directly in native_bootstrap.py.
For upload/readback, carry the original session through the existing internal
registered upload context and the source-owned readback entry. Verify exact
source/controller/runtime/target/stream identities before using its reader.
The source-bound path requires that registered reader; omission, substitution
or a copy fails before the next native call. Old C0 preparation keeps its old
context and behavior and cannot acquire a captured-source reader by supplying
metadata. No global current-observer variable or callback-enabled helper is
introduced. Low-level native helpers keep one implementation; observations
surround their actual calls in the source-bound orchestration.

Every query requires the original live setup ticket and original owner thread,
or the existing eligible emergency ticket while the original sync is active.
No new ARM, timeout renewal, synchronize, retry or context selection is added.
After a sticky failure, queries stop; CPU persistence of already held facts may
continue. A reader failure cannot promote a previously failed guard to success.

## Admitted query ABI and environment

Reuse the exact sources/ABI and ordering from native-scope-aggregate v2 and
native-scope-cache v1. The initial supported validation deployment remains the
pinned Linux environment with eight visible devices, target1 and ambient0.
Check the actual enumerated count, exact environment file rows and no device
remapping/preload/library-path overrides. This is not a claim of generic runtime
or multi-threaded support; product activation remains a separate gate.

Before acquiring a library, require its exact trusted environment descriptor,
regular resolved path and bytes, and that same resolved path in /proc/self/maps.
Acquire only RTLD_NOLOAD | RTLD_LOCAL. Missing/mismatched/unmapped libraries
fail; no searching, fallback load, RTLD_GLOBAL or alternate ABI is permitted.
The loaded handle and function objects remain strongly held. PID and native TID
are read from the actual process and /proc/thread-self; require PID == TID.

| Lane | Actual source | Fixed calls |
| --- | --- | --- |
| CUDA | site/nvidia/cuda_runtime/lib/libcudart.so.12, SHA256 c3a75b33af334a3486d197dbd1584a2985183ba4688d237a2be5f2f679329920 | cudaGetDevice(POINTER(c_int)), restype c_int |
| CUDA | admitted torch._C object | _cuda_getDeviceCount(), _cuda_hasPrimaryContext(ordinal), and after initialization only _cuda_getDevice(), _cuda_getCurrentRawStream(target_ordinal) |
| Warp | os_native/libcuda.so.595.71.05, SHA256 76e0d9678d41cf6b6ae71d18549d88a963d3d434f08108baa12457eb53ca88d6 | cuDeviceGetCount(POINTER(c_int)); cuDeviceGet(POINTER(c_int), c_int); cuDevicePrimaryCtxGetState(c_int, POINTER(c_uint), POINTER(c_int)); cuCtxGetCurrent(POINTER(c_void_p)); cuCtxGetDevice(POINTER(c_int)); all restype c_int |
| Warp | admitted held runtime.core | wp_get_error_string(), argtypes=[], restype=c_char_p; wp_cuda_context_get_stream, argtypes=[c_void_p], restype=c_void_p, only after the cache proof below |

Resolve a fixed literal symbol set; no caller-selected getattr/dlsym or raw
handle invocation. Validate the Warp getters' existing declarations rather
than rewriting them. The stream declaration was re-read at lines4842-4843 of
the actual pinned context.py, SHA256
eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee;
code preflight must also verify the implementation and changed placement.
No setter, primary-context retain/release, context property that creates a
context, memory query, stream/event creation or GPU allocation is a query.

CUDA initialization: after runtime/target adoption and entry into the existing
indexed_runtime_stream leaf, initialize reader queries. Check the original
three queued callbacks and empty lazy-seed tracker against the pinned sources.
Record raw device and all primary states before deriving cold logical0. Only
raw0, uninitialized runtime and all-inactive primary contexts permit that
derivation; no warming logical/current-stream getter is called before the real
stream creation. After it, retain/journal the real stream first, then query the
actual logical device and explicit-target current raw stream. Preserve the
existing deferred-Triton nonexecution check at the actual binding boundary.

Warp initialization: after quiet wp.init and actual target adoption, initialize
queries inside indexed_runtime_stream. Enumerate actual CUdevice handles; never
assume handle == ordinal. Check raw target._context and target._stream are None
and exact empty wp_get_error_string() before calling original get_stream once.
After production retains/journals that return, require identity with the raw
target._stream, positive raw target._context and returned handle, and a second
exact-empty native error buffer. Only then enable the target stream getter.
No error clearing, suffix matching, retry, cache initializer or metadata-only
substitute. The first returned native stream must equal the held stream.

State query order and full raw results remain as specified in aggregate v2:
CUDA raw device, all primary contexts, initialization state, actual logical
device and explicit-target current stream; Warp current context, conditional
current device, all primary states, then the proved target-cache stream query.
Check CUresult before consuming output. Successful null context is distinct
from query failure. Preserve getter names, arguments, successful values or exact
errors in order, including the prefix before a failed getter. No call retries.

## Existing observation sites and retention ordering

Use runtime-custody.v3 and its existing native_scope payload unchanged. Before/
after joins use actual journal and lifecycle indices, original guard, same
thread and exact held owners. Runtime-custody v1/v2 retain their old meanings.
The artifact parser's six-pair requirement, exact file frames, entered leaves,
create-return join and extension binding remain unchanged.

1. runtime_stream before follows the actual entered leaf. After follows the
   original returned-owner journal and reader cache prerequisite. Binding point
   follows the actual runtime_binding append, before artifact loading.
2. Artifact frame start -> entered target leaf -> before snapshot -> original
   scoped delegate exactly once -> immediate existing created-module retention
   and returned-owner journal -> original scope exit -> after snapshot -> file
   frame completion. A body/exit failure leaves an unmatched truthful before;
   do not emit successful after or retry the call.
3. At every existing upload/readback empty/copy/event-create/record/wait call,
   emit before after the original entered leaf. Retain each returned object in
   an already-held production owner container immediately, before metadata,
   after-query, append, or adoption can fail. Then emit after. Existing device
   ordinal validation/adoption still runs and retains its authority. Returned
   retention must not depend on the test probe's held list. Zero-size call
   occurrences remain covered, not inferred from nonempty arrays.
4. Warp upload/readback outer scopes get before prior to the original scope
   enter and after only after its successful exit. Interior helper facts use
   the actual entered target context; outer facts use restored ambient state.
   Track nesting by the direct orchestration boundaries, not wrapping the Warp
   class globally. Scope unwinding on failure occurs once; no later success
   query runs just because __exit__ returned.
5. The existing eligible input-stage terminal sync gets before/after under its
   original emergency guard, while owners remain held. Do not extend the v3
   terminal_sync grammar to later setup phases or normal close. Old tests that
   terminate at the validated-input phase retain their required one-sync path.

Append complete factual snapshots before checking expected scope, so a wrong
device/context/stream remains evidence. Pristine has no active primary context;
subsequent sites require only target1 active. CUDA outer helper/artifact edges
restore logical0/raw1 and the captured postbinding current stream. Warp outer
edges restore default0/null current context; interior edges require default1,
the exact target context and target ordinal1. Compare physical state, not just
descriptor fields. Contradictions mark UNKNOWN before any next normal leaf or
COMPLETE. Getter failure retains its raw prefix but emits no complete state.

On query/append/cache/identity failure, retain the error and any returned native
owner first, follow the existing sticky source/guard failure and quarantine
rules, and preserve the original exception if diagnostic persistence fails.
No emergency synchronization after failed normal guard; no release after
unproved scope. The same failure rules cover before/after and scope-exit errors.

## Consolidation and acceptance

Delete query setup, cache proof, state construction, native_scope production
and successful-call monkeypatch installation from the test observer. Existing
tests may keep a small fault/diagnostic facade which calls the actual production
reader, injects named failures, and saves raw records. It cannot synthesize or
append successful native_scope facts. Later setup tests reuse the reader's
checked raw-state query under their existing live guards; their graph/adoption/
warmup assertions remain test assertions, not production authority. Preserve
the older native evidence files without rewriting their producer identity.

Use one CPU success-coverage validator over v3 facts for initialization/upload/
input validation. It joins every actual eligible lifecycle occurrence, rather
than accepting a version, count or initial prefix. Share its normal input-cut
checks between the old sacrificial terminal validator and the integrated-worker
acceptance driver. Keep their different terminal requirements explicit: old
eligible abort has one sync; integrated clean close has zero emergency sync,
actual reclaimed samples, ACK-before-release and observed/reaped exit. Neither
mode grants final-sample physical proof from setup-only facts.

Source release advances native AST policy/catalog to v18. Pin the new reader's
exact full LF source bytes and the literal readonly symbol/ABI use; admit ctypes
only in its reviewed initialization/query implementation. Keep the existing
packed-render snapshot exception unchanged. No blanket ctypes/importlib,
dynamic lookup, test import or arbitrary library exception. Update bootstrap
pins/catalog and the frozen native runtime contract's catalog path/digest as
one release. Source verification tests must reject changed symbols, setters,
paths and transferred exceptions even if a caller supplies a recomputed hash.

Before implementation, obtain fresh independent design review. Before any new
native reader execution, obtain fresh exact-code/native-call preflight against
the actual pinned runtime/library sources, including cold queries, readonly
semantics, Warp cache proof and void-exit restoration. Local CPU doubles alone
cannot establish those properties. Reuse the existing source-preflight evidence
where exact identities match; verify the changed call placement itself.

Required implementation evidence:

- Actual production reader CPU tests for every getter failure position, wrong
  cache/error/null/stream, foreign contexts and owner/guard/thread substitution.
  Retained return survives after-query, append, adoption and scope-exit failure.
- Unchanged historical codecs and artifact join attacks: missing/crossed/stale
  pairs, replaced returned owners/file frames and transplanted guard cuts still
  reject. Source/AST and C0 regressions pass.
- Both lanes through the same real framed loop, no test-installed successful
  fact producer. Reproduce CUDA intake, require all six true pairs and physical
  target1/ambient0 restoration. Require complete setup native facts on Warp.
- Repeat affected real scope/failure cases, including raw contradictory after
  void scope exit, foreign-touch/unknown scope and actual process closure. A
  failed before/after query cannot create an authorized ordinary release.
- Remove the duplicate implementation and inspect the complete diff for a
  second state reader, callback path or hidden new guard/wait.

Then continue the full C2f workload/failure/numerical/physical-scope matrix,
bounded preallocated control traces, normal-close supervision and independent
whole integration review. F2's final samples, real Go2, profiles/diagnostics,
generation/sentinel behavior, post-submit faults, disconnects, ACK/cleanup/hang
and process closure remain required; setup success is not their acceptance.
The complete C4 systematic refactor and C4c3/c4/C4d/B3 obligations remain.
This amendment adds only preparation checks, outside future R0/R1 measurement
boundaries for both lanes. No performance result, timing threshold or backend
decision is produced. Product activation, R2/S/host delivery, result/baseline,
a3b and shipping remain at their existing closed gates.
