Initiative: Q54 / P12.4 / C4c2 integrated worker acceptance
Stage: Narrow design supplement D1
Author: Codex
Version: v1
Date: 2026-10-10
Status: draft
Related Files: runtime-integration design-request v3; native-state-reader design-request v1; worker-acceptance gap-design-boundary-review v1; native_scope.py; guarded_lifecycle.py; native_readback.py; tests/support/native_runtime_process.py; tests/support/guarded_custody_process.py
Owner Summary: Complete final-operation correctness observations with the existing reader, worker, native leaves and supervisor. Permit bounded fixed-driver diagnostic observation at original execution edges and inside the original resolve guard. Production timing remains query-free. This is the sole policy supplement in one remaining worker acceptance/refactor packet, not a new product capability or framework.

## Scope and predecessor

Production base is cf7f5a6d9ff9b3971144410a7c3ad156a806646c; author
documentation base is 3521c4d. Independent assessment session
01a12584-473c-7593-a8e3-968b8632e8c6 identifies G1–G5. Existing approvals
and their exact scopes remain intact. This supplement resolves D1/G1 only;
G2 output staging, G3 fixed/prepared work, G4 exit/fault coverage and G5 journal
recovery remain mandatory implementation under the approved parent. It is
not permission to waive any of them. No new threshold, timing result, backend
selection, source allowlist relaxation or product payload is proposed.

The reader's setup/emergency policy currently excludes normal resolve and all
final execute observation. Post-EXECUTE observation alone cannot detect an
enqueue exit error repaired by finish. Correctness tests must see those
individual edges without wrapping original execution in a fabricated setup
guard. This supplement explicitly replaces that exclusion ONLY at the fixed
validation sites below. The existing production query ABI, sole reader and
runtime/source ownership stay authoritative.

## Two precise observation contexts

1. **Original RESOLVE.** A final raw query is admitted inside the actual live
   resolve ticket of observe_terminal_completion, while its exact worker
   record is pending and its result hold remains retained. Require the
   original source/controller/reader/runtime/binding, admitted final graph,
   active record/transaction/lease/generation, event, stream and target. Check
   the existing CUDA context target value against the retained target using
   the approved terminal-target rule. Guardian is SEALED, source phase is
   fixed_final_graph_admitted, thread is the original owner, and the source,
   guard and lifecycle are healthy. Admission is before DISARMED; a completed,
   copied, prior-generation, emergency or setup ticket cannot substitute.
   Only the existing completed-ticket check and arena resolution commit grant
   resolution. Raw observations never grant it.

2. **Fixed correctness driver during EXECUTE.** Extend the existing fixed
   diagnostic facade, which already surrounds actual native helpers, to query
   the same reader at original final enqueue/finish edges. The worker still
   requires IDLE and executes the original transaction/lease implementation.
   No guard wraps EXECUTE; no ARM, sync or context selection is added by the
   observer. Reader admission additionally requires an exact driver-local
   registration established before seal, original worker/source/controller,
   admitted graph, owning thread, active operation/sample/transaction and live
   generation. The registration is diagnostic access only; it cannot execute,
   wait, adopt, resolve, reclaim or release. It is neither a caller-supplied
   Boolean nor a public/general callback, and cannot be minted by parsing an
   artifact. Only the fixed sacrificial driver installs it. It expires at the
   end of its exact observation extent, and permanently on failure/termination.

Both contexts reuse require_reader and the same raw getter implementation.
Identity/phase checks must be factored without bypassing the current ticket
check with monkeypatches, a blanket IDLE exception, or a second reader.
Production does not import tests. Fixed test instrumentation may wrap original
methods/scopes as existing diagnostic instrumentation does, but delegates each
original entry/exit/leaf exactly once and cannot supply successful query values.
Every scope is the actual precreated scope used by the original transaction.
An unrelated scope with matching metadata does not establish an occurrence.

Driver registration and diagnostic containers are retained before seal in the
existing source graph. The test driver establishes the exact callable/location
bindings before execution and records them; implementation/code review checks
those against pinned production sources. There is no request argument, config
file path, environment variable or product selector accepting observers or
reader access. Changing source pins for implementation does not admit arbitrary
new call targets. A future timing worker never installs the facade/registration
or diagnostic staging; correctness diagnostics are independent of the frozen
primary/diagnostic_profiled pass Boolean.

## Placement and raw evidence

Observe each original enqueue and finish extent separately: before original
scope entry, after actual entry before a native leaf, after the leaf while in
scope, and immediately after the original exit before any later scope entry.
Multiple R1 scene/acceleration/render scopes receive distinct ordered occurrence
indices. Shared scopes may cover multiple leaves; retain that actual nesting
rather than inventing one scope per leaf. First and last leaves and diagnostic
event records cannot be omitted. On return, retain any actual returned native
owner in its original existing slot/graph before an after-query can fail.
Final execution should acquire no new resource; a surprise return is retained
as failure evidence and cannot reopen the sealed guardian.

Inside original RESOLVE observe outer state before terminal wait and immediately
after it, then the actual diagnostic copy scope entry/interior/exit, and the
copy completion wait. Successful terminal wait alone never marks diagnostic
copy complete. The last successful guarded outer observation is the last native
state fact; CLOSE must not query a released reader. Later CPU-only transitions
and process exit have their own evidence, not invented physical snapshots.

Use the existing pinned raw getters/libraries and preserve their actual ordered
values and failures. Check all visible primary contexts and held/current target
stream. Warp outer state is the admitted null ambient context/default0 and its
established target cache; inner state is target1/default1. CUDA outer state is
its admitted logical0/raw1 state, not invented raw0; actual indexed device/stream
scope interior must be logical1/raw1 and the held target stream, and exit must
restore the recorded outer state. Getter cache initialization is forbidden
here. No library/symbol/device/context/stream discovery is added after seal.

Keep final diagnostics separate from input runtime-custody v3 rows. Freeze a
fixed test artifact schema q54.closed-worker.final-native-observation.v1 with
source/manifest/row/pass identities, worker PID/start identity, source-local
reader/controller/graph tokens, sample index/frame/camera/replay/generation,
operation (enqueue, finish, terminal_wait, diagnostic_copy, diagnostic_wait),
ordered occurrence and edge, original native owner/scope tokens, nullable
original resolve sequence, raw getter prefix/state and nullable failure.
Joins use retained live objects in-process and recorded occurrence tokens in
artifacts. Serialized tokens are descriptive and never reconstitute authority.
Include the corresponding original request sequence and lifecycle/arena trace
positions. A partial row is explicitly incomplete; no missing after-row is
filled from metadata. Final schema parser/checker stays CPU-only. No duplication
of input custody semantics or mathematical renderer/CPU oracle is introduced.

## Blocked queries and failure transfer

Fixed EXECUTE diagnostics run under a bounded request observation in the
existing parent/supervisor process. Before sending that request, the fixed
client commits a monotonic deadline using the existing GuardConfiguration
operation_timeout_ms. No per-getter refresh or progress extension is allowed.
The supervisor loop watches this shared request deadline independently of the
client thread and native worker. A blocked getter, leaf or exit cannot prevent
the supervisor from detecting expiry. Expiry closes the actual command
transport and enters the existing supervisor terminal path using its existing
process_observation_failed reason. This is an actual failed bounded process
observation, not a fabricated worker guard event. The existing core's terminal
deadline is the minimum of its prior active deadline and its bounded terminal
grace; neither an earlier active deadline nor an emergency deadline is renewed.
The core alone requests termination and the same supervisor observes/reaps exit.
Retain request-start/deadline/expiry, transport closure, core terminal trace and
actual exit; lack of worker diagnostics cannot become successful verification.
This proposes no new watchdog service, guard role or product execution deadline.

RESOLVE getter/copy/wait hangs remain under the original armed normal guard and
its original operation deadline. Parent request failure/EOF also reaches the
same terminal path; it cannot supersede an earlier guard deadline. No additional
ARM is used for diagnostic queries or copy work inside that guard. Normal close
exit supervision is the separate already-required G4 implementation using this
same parent/core, not a new diagnostic lifetime extending beyond close.

Any getter, validation, append, original leaf or scope-exit failure preserves
the first exception and actual returned owner/raw prefix. Before any later
native operation, mark observation scope unknown and poison the original
controller through its existing failure path. Known foreign ordinals remain
in the dirty set in addition to unknown; do not reduce them to target-only.
After an original scope entry or body fails, no new query is made. Execute the
original exit/finally exactly as its original exception semantics require,
without fabricating successful after-observation. Diagnostic persistence must
not replace the original error. After any sticky failure no further query,
normal ARM, emergency target sync or ordinary release is admitted. Whole-graph
quarantine and process exit retain the existing disposition. A blocked query
may leave no in-process failure record; supervisor-enforced exit is then the
only completed terminal fact.

## Precreated final output staging

Implement G2 using one reusable fixed staging set per worker, adopted before
the existing single seal/barrier. Prepare exact output references, host arrays,
native host views, explicit-stream copy scopes/operands and one diagnostic
completion event. Split preparation from existing native_readback exact-copy/
event bodies; never allocate/adopt after seal or invoke warmup .numpy()/.cpu()
conveniences there. No new product host result/lease or image IPC is introduced.

After original terminal wait, while the original resolve guard is still live,
copy the actual generation's required channels, record/wait its precreated
copy event, observe restored state, and then complete that original guard.
Only then permit existing arena commit. Persist/compare staging before RELEASE
and before staging reuse; failures retain unresolved/poisoned custody. CPU
comparison uses the existing canonical materializer, binary32-bound arguments,
CPU executor and frozen tolerance/profile/miss rules. Sentinel commands and
operands are precreated and exercise original reset/render bodies. Sentinel
completion/reuse must use the same fixed correctness execution and original
resolve protection; no uncontrolled asynchronous host mutation is permitted.

## Acceptance, consolidation and removal

Code/native preflight must verify the exact placement, pointer/owner joins,
request-deadline wiring and absence from production timing. CPU negatives cover
copied/unregistered/expired/wrong-thread/wrong-source/wrong-generation admission,
wrong ticket/event/stream, released graph, caught failure, capacity exhaustion,
callback injection and attempted use from ordinary IPC. Actual OS tests block a
getter before/inside/after enqueue and finish, and during resolve, and prove
bounded original-supervisor exit/reap without deadline renewal or target sync.

Native tests use both original lanes, pinned target1/ambient0, all required
rows/profiles/pass modes, real Go2, all120 camera/replay values and first/last/wrap
generations. Check applicable final channels against CPU reference with exact
discrete equality and frozen floating rules, plus sentinel/reset proof. Inject
enqueue exit and finish exit omission independently: an intervening later scope
must not hide either error. Include raw getter/append failure, known foreign
touch/return, unknown pre-return, copy/wait failure and the inherited quarantine
conditions. No partial worker smoke or reader subset is whole acceptance.

Implementation is one coherent G1–G5 packet: shared atomic reducer and prepared
hot-path storage/capacity, fixed final validation, bounded close-exit/fault flow,
and complete journal recovery. Preserve one implementation of each existing
state machine, native leaf and oracle. Remove superseded diagnostic wrappers
when equivalent original-path coverage exists. Retain one case-to-requirement
coverage index; do not add another audit framework. Fresh independent review
assesses the complete promised packet after required evidence, then the original
later C4 stages continue. Formal speed comparison/result/baseline, R2/S/B3 host
delivery, shipping and backend selection remain closed. Temporary lane-specific
comparison infrastructure follows the existing eventual selected-backend
cleanup decision; this supplement does not make it permanent.
