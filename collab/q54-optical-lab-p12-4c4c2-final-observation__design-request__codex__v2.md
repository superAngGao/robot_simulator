Initiative: Q54 / P12.4 / C4c2 integrated worker acceptance
Stage: Narrow design supplement D1
Author: Codex
Version: v2
Date: 2026-10-10
Status: draft
Related Files: runtime-integration design-request v3; native-state-reader design-request v1; worker-acceptance gap-design-boundary-review v1; native_scope.py; guarded_lifecycle.py; native_readback.py; tests/support/native_runtime_process.py; tests/support/guarded_custody_process.py
Owner Summary: Complete final-operation correctness observations with the existing reader, worker, native leaves and supervisor. Permit bounded fixed-driver diagnostic observation at original execution edges and inside the original resolve guard. Production timing remains query-free. This is the sole policy supplement in one remaining worker acceptance/refactor packet, not a new product capability or framework.

## Scope and predecessor

Production base is cf7f5a6d9ff9b3971144410a7c3ad156a806646c; author
documentation base is 819a26f10ba32d8ce479c71acd45c8df41e6c027. Independent assessment session
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

## V1 independent findings and closure

Fresh CLI session01a12595-6cbb-7491-b941-40f8e5cc4555 reviewed v1 at819a26f.
The complete report is archived verbatim as final-observation design-review v1;
its generated local review identifier is separately recorded there. Verdict:
Request changes, F1/F2. V2 distinguishes CPU camera coverage from native numeric
coverage and defines sequence-bound publication/completion/retirement races.
The earlier gap report's exhaustive-native interpretation does not amend the
parent. Its weaker two-generation sentinel wording likewise does not replace
C4c v7's at-least-three successful reuses. C1/C2 implementation comments below
are retained: initialize actual event storage before seal; order sentinels before
original reset on the same stream; preserve the real resolution state on a later
CPU comparison failure. These are corrections and dependencies, not new gates.

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
Use one existing-client/supervisor request-observation record, bound to exact
worker/session and protocol request sequence, with states EMPTY, PUBLISHED,
RESPONSE_OBSERVED, RETIRED or TERMINAL. A short lock protects only this CPU state;
no socket I/O, artifact write, native operation or blocking wait occurs under it.
Only one request may be outstanding. The client publishes q/start/deadline under
that lock before sending any bytes; it cannot replace a nonretired record.

The successful observation is the complete framed response received and checked
by the original codec for version, exact sequence and successful status. Capture
its monotonic observation time immediately at that point, before artifact/proof
persistence. The client publishes RESPONSE_OBSERVED for that exact q only if its
observation is strictly before D and the record is still PUBLISHED. Observation
at or after D publishes a terminal request-failure candidate, never success.
A response error, partial frame, EOF or send failure likewise requests terminal
handling. A stale/mismatched observation cannot clear or replace any record.

The core-owning supervisor is the sole committer of RETIRED or TERMINAL and the
sole caller of core fault/kill APIs. Under the short lock it arbitrates:
- a still-PUBLISHED request at time >= D becomes TERMINAL;
- a matching RESPONSE_OBSERVED with timestamp < D may retire successfully unless
  terminal disposition was already committed or the core has already failed;
- a late/error observation becomes TERMINAL, regardless of supervisor scheduling;
- TERMINAL is permanent; a late completion cannot change it or retire q.
Expiry wins at equality. A supervisor delayed past D cannot lose a late response
by clearing/replacing its obligation. A timely response already observed before
D is allowed to retire when the supervisor resumes, provided expiry/termination
had not already committed. This distinguishes delayed supervisor scheduling from
late worker completion. No next request is sent until the supervisor has retired
q successfully; the client waits outside the lock, with stop/terminal checks.
Only then may q+1 replace RETIRED. There is no deadline refresh within q.

The supervisor loop watches this shared request deadline independently of the
client thread and native worker. A blocked getter, leaf or exit cannot prevent
the supervisor from detecting expiry. Expiry first latches TERMINAL and the client's stop condition, then shuts down
both directions of the actual command endpoint outside the lock. On shutdown
failure, close that endpoint; retain the first observation/transport error even
if closure also fails. No terminal outcome is published as fenced until that
shutdown/close attempt is recorded. Already accepted kernel bytes cannot be
recalled; no subsequent send is permitted and late client work cannot retire
or open another request. A finally path commits the existing supervisor fault
regardless of a transport-closure error. Join the client after the original
worker is disposed; a client that does not join is incomplete acceptance.
Enter the existing supervisor terminal path using its existing
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
completion event. Initialize its actual native storage by its required first
record/wait during guarded preparation, before the final barrier; merely creating
a lazy event wrapper does not satisfy this. Split preparation from existing native_readback exact-copy/
event bodies; never allocate/adopt after seal or invoke warmup .numpy()/.cpu()
conveniences there. No new product host result/lease or image IPC is introduced.

After original terminal wait, while the original resolve guard is still live,
copy the actual generation's required channels, record/wait its precreated
copy event, observe restored state, and then complete that original guard.
Only then permit existing arena commit. Persist/compare staging before RELEASE
and before staging reuse. Copy/wait failure leaves the sample unresolved; a CPU
comparison/persistence failure after arena commit poisons its actual resolved-held
state and forbids release/reuse, without pretending resolution never occurred. CPU
comparison uses the existing canonical materializer, binary32-bound arguments,
CPU executor and frozen tolerance/profile/miss rules. Sentinel commands and
operands are precreated and exercise original reset/render bodies. Establish
possible-submission custody before the fixed sentinel fill, submit the fill on
the exact same ordered stream before original reset/render, and protect completion
through that sample's original resolve before reuse. Never fill output while its
D2H copy or another generation may still use it. At least three successful slot
reuses retain the parent sentinel requirement; no uncontrolled asynchronous host
mutation is permitted.

## Acceptance, consolidation and removal

Code/native preflight must verify the exact placement, pointer/owner joins,
request-deadline wiring and absence from production timing. CPU negatives cover
copied/unregistered/expired/wrong-thread/wrong-source/wrong-generation admission,
wrong ticket/event/stream, released graph, caught failure, capacity exhaustion,
callback injection and attempted use from ordinary IPC. Actual OS tests block a
getter before/inside/after enqueue and finish, and during resolve, and prove
bounded original-supervisor exit/reap without deadline renewal or target sync.

Coverage distinguishes the following inherited obligations. This supplement
does not introduce mandatory all120 full-resolution numerical GPU/CPU comparison.
No reduced resolution, surrogate asset or weaker tolerance is authorized.

| Accepted requirement | Required observation/coverage |
| --- | --- |
| Runtime-integration v3 section9 item1 and section3 | CPU materializer/sequence tests check every120 frozen camera value, replay/ABI selection and first/last/wrap metadata, including unchanged native binary32 arguments; a new scalar label alone is not evidence. |
| Section9 item2: asymmetric and dynamic | Compare actual final-generation origins/directions, hit/miss/geometry, IDs/material and RGB/intensity for the frozen cases against the existing CPU oracle, using applicable profile channels and existing tolerances. Preserve asymmetric rotation and dynamic transformed geometry. |
| Section9 items2/4: real Go2 and all rows | Actual final correctness for all six R0 and two R1 rows in both lanes and primary/diagnostic passes, real Go2 at each frozen row resolution, applicable outputs, existing fixtures and tolerances. Reuse the existing oracle/fixture algorithms. |
| Section9 items1/4 and C4c v7 sentinel requirement | First/last/wrap plan and generation joins; at least three successful slot reuses with distinguishable prepared sentinels, correct per-generation output/reset, shadow on/off and exact lane-appropriate counters. Verify corresponding selected final generations before release; no requirement to compare full-resolution images for every intervening orbit/replay index. |
| Section9 device-scope and failure coverage | Pinned target1/ambient0 on both lanes; original enqueue and finish entry/interior/exit observed separately, including first/last/wrap generations where used; no after-close native query. |

The single integrated case-to-requirement index must name the exact final
samples/channels and artifact paths satisfying those rows, rather than only
reporting test totals. Run original intermediate generations where FIFO/wrap
requires them; do not fabricate a frame selection or label warmup output final.
All120 CPU value checks cannot replace the bounded final numerical cases, and
the numerical cases need not expand into every120 full-resolution CPU images.

Inject enqueue exit and finish exit omission independently: an intervening later
scope must not hide either error. Include raw getter/append failure, known foreign
touch/return, unknown pre-return, copy/wait failure and the inherited quarantine
conditions. Preserve the separate healthy-IDLE, known-target post-submit fault
that legally enters one emergency sync; the diagnostic failure policy must not
force every unrelated fault into unknown scope to avoid testing that path.
No partial worker smoke or reader subset is whole acceptance.

CPU deadline scheduling tests cover successful observation strictly before D,
exactly at D and after D with the supervisor delayed, expiry winning before a
late publication, stale q completion after q+1 is considered, partial reply,
EOF/send failure and shutdown/close failure. Assert no next request before
successful supervisor retirement and no request after terminal latch. Original
OS getter-hang tests still prove worker disposal and joined client through the
same supervisor, not only a controlled-clock model or outer harness timeout.

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
