# C2d1: adopt guarded failure custody at the actual setup exception boundary

Initiative: q54-optical-lab-p12-4c4c2d-guarded-custody
Stage: design-request
Author: codex
Version: v2
Date: 2026-10-03
Status: in_review
Related Files: runtime-integration design codex v3 sections4/5/8/9;
source-consumption design/code reviews; optical_worker/lifecycle.py;
optical_worker/native_setup.py; optical_worker/wait_protocol.py;
optical_worker/wait_supervision.py
Owner Summary: Freeze the concrete C2d custody adapter where real setup already
catches exceptions. Do not wrap the old retry policy or copy the upload loops.
No dispatcher, GPU preparation activation, arena/result authority or C4 closure.

Base: `11603cd2772aea79c07579704285767f6c5ac622`.
Supersedes the v1 proposal, preserving its historical text. Independent v1
findings F1/F2 are addressed in sections 4, 5 and 6 below. This revision adds
no implementation or activation.
This is an implementation-interface supplement to approved runtime-integration
v3, not a replacement of its ordered failure rules or a new performance gate.
The complete C2d/e/f work remains necessary. C2d0 CPU code is approved at
`1f0576b`; its success alone never authorizes CUDA-capable imports.

## 1. Code facts requiring an adoption interface

`native_setup._abort()` catches inside both real a2 upload functions, adopts
the exception graph where possible, and calls `lifecycle.terminate()` before
returning `NativeWorkerPreparationAborted`. An outer context manager alone
cannot stop an emergency retry: the old termination already happened inside.

`NativeWorkerSetupLifecycle.terminate()` is the C0 v2 implementation. It may
invoke its A/K notifier then sync whenever target-only work is dirty. It has no
guard client, pending handshake or failed-normal-segment facts. Reusing it
unchanged inside an ARMED setup segment violates runtime-integration v3 section8.

`GuardWaitClient` already owns exact live tickets and actual ARM/COMPLETE ACK
state. Its `complete()` explicitly rejects emergency tickets. The supervisor
already keeps ARMED_TERMINAL until observed exit/deadline; no new wire protocol
or second timer is needed. Its replay validator is the sole supervisor truth.

The existing final-barrier helpers consume a setup permit and seal, call the
runtime sync directly, then commit C0 quiescence before an outer DISARMED ACK.
They are not admissible C2d final-barrier adapters. C2d1 must reject accidental
use of those legacy commit methods, not silently generate final readiness.

## 2. Bounded implementation and unchanged components

Deliver `optical_worker/guarded_lifecycle.py::NativeGuardedSetupLifecycle`, a
production-owned subclass of `NativeWorkerSetupLifecycle`, and fixed CPU
integration tests. Inherit the one existing device descriptor/permit, scope
tracking, register-before-next-leaf and guardian storage implementation; do not
copy either Warp/Torch upload loop or create a second resource guardian.

Its constructor creates one exact `GuardWaitClient` from an inherited private
socket and exact `GuardConfiguration`, before any CUDA-capable operation. The
controller owns that client and original socket strongly; callers do not supply
duck-typed clients, notifier functions, policy booleans or parsed evidence.
Session identity and target derive from this configuration; lane is fixed to
the reviewed literal warp/cuda value. One controller/thread/session only.

The inherited C0 constructor's mandatory notifier is a fixed internal rejecting
function, never an alternate A/K fallback. This is a tripwire, not the way a
guard failure selects quarantine. The overridden v3 `terminate()` is the actual
method called by unmodified `native_setup._abort()`; it never calls the legacy
`terminate()` or `commit_clean_close()` implementation.

C0 lifecycle/native_setup source files, worker, launchers, wire policy and
existing behavior stay unchanged in this slice. The new subtype reuses their
protected state as an internal implementation extension, not by mutating ring
internals or by monkeypatching per-instance methods. No supported guarded path
may explicitly call a base-class termination method to bypass its override.
The closed trusted-domain assumption remains; this is not a hostile Python
subclass/reflection sandbox.

Permit a narrow C2b client API extension to inspect exact live/completed ticket
identity from its actual stored transition history. It must reject copied,
foreign, stale, emergency-as-normal and unregistered tickets. No new ARM/
COMPLETE message, wire version, ACK meaning or supervisor state is introduced.
No caller-filled `guard_succeeded` flag is authority.

## 3. Controller states and the real exception path

Construction is CPU-only: guardian/client/socket/configuration are retained
before runtime entry. Each normal segment has fixed role setup or resolve:

```text
IDLE -> ARM_PENDING -> ARMED -> operation returned
     -> COMPLETE_PENDING -> DISARMED observed -> IDLE
```

An exact controller-owned segment context performs ARM before yielding to its
trusted fixed driver. It records the actual ticket and client state. Nested,
cross-thread, duplicate exit and wrong-role use reject before another ARM or
leaf. A sticky `guarded_segment_failed` records any operation/handshake failure.

The setup permit methods require the exact active setup ticket before entering
the inherited scope/permit logic. Every leaf is possible before its real API
call; successful-return counts remain diagnostic. A returned owner is adopted
before ordinal checking exactly as in C0. Foreign/unknown scope is irreversible.

If `native_setup._abort()` calls terminate while that context is still ARMED,
the controller marks segment failure, revokes operations, freezes/quarantines
the original guardian and builds v3 terminal evidence immediately, before the
exception reaches the context exit. The context observes this terminal state
and does not attempt COMPLETE, a second terminate, a new ARM or sync.

If a leaf outside native_setup raises, the segment exit commits the same
failure/custody transition before propagating. It retains the original exception
and traceback graph in a terminal hold even if the guardian is already sealed;
it must not try to reopen sealed resource coverage to add that exception.
No normal successful completion record is created on either path.

Explicit terminal operations are single-use. A repeated terminate/close rejects
without an additional sync, release or transfer. Reentrant failure cannot finish
an earlier transition successfully. Finalizers never authorize release.

## 4. Ordered termination and retained resource graph

Apply runtime-integration v3 section8 in order, from actual state:

1. Foreign/unknown scope or unprovable ownership => quarantine, no sync.
2. Pending/armed/failed normal guard or sticky failed segment => quarantine,
   no COMPLETE/new ARM/sync, even with zero measured sample launches.
3. Already entered emergency => quarantine on any further error, no retry.
4. Healthy actual IDLE and no possible GPU setup (no started descriptor, dirty
   ordinal or unknown scope) => terminal ARM/ARMED, zero GPU sync, then clean
   cancel/release/exit. No cleanup starts while the supervisor is still IDLE.
5. Healthy actual IDLE with complete target-only possible work => one emergency
   ARM/ARMED followed by at most one exact target sync.

For step 4, use the existing role `emergency` as a terminal supervision
envelope, with local purpose `cpu_cancel`. For step 5 its purpose is
`target_sync`. This fixes the previously unspecified terminal observation for
CPU-only cancellation; it does not add a wire role/message or a second timer.
The role means non-completable terminal deadline, not proof that a GPU sync ran.
Freeze purpose before ARM; no normal guard may be relabelled terminal.

In both branches, supervisor commits the same configured operation deadline
before its ARMED ACK; controller requires the exact ACK/ticket before any
release authorization or potentially blocking cleanup. No COMPLETE is sent.
The original deadline bounds reference release, destructors and pre-hard-exit
code while the parent remains alive. No use of response-write timeout or test
outer timeout as the watchdog. CPU cancellation never calls/imports a runtime.
ARM/ACK failure retains the entire graph, closes the private channel to signal
EOF, and hard exits without cleanup/sync; supervisor's existing EOF/ACK/operation
terminal rules apply, with no deadline restart. Notification failure cannot
fall back to unguarded cleanup. Fixed driver never waits for a post-terminal
response before hard exit. A deliberately stalled supervisor is not proved
healthy by a successful write; the worker's ACK timeout ends the attempt.

The exact sync callable is selected by the trusted fixed driver from the same
lane/runtime/target already adopted by setup, not from IPC or a parsed record.
No descriptor check may import/init a runtime merely to discover its ordinal.
The controller checks the exact emergency ticket/session/target before calling
that leaf. ACK failure means zero sync calls and quarantine.

On terminal cpu_cancel ACK, or target_sync ACK plus successful exact sync,
with zero external holds, authorize whole-graph
release and terminal exit. Keep the exact emergency ticket ARMED; never send
COMPLETE/DISARMED or restart/clear its deadline. Cleanup and exit remain inside
that deadline. The guardian release routine is the existing sole resource
release implementation, not a parallel list of selected objects to free.

Install the append-only terminal custody entry before ARM, sync or release.
It strongly retains the controller/guardian, client/socket and original failure
context; record transfer only after the entry exists. Quarantine before release
also transfers the frozen guardian via its existing API. During/after release,
that API is NOT called from RELEASE_AUTHORIZED/RELEASED: terminal custody alone
retains remaining reachable resources and new cleanup exception/traceback.

On cleanup failure, retain all remaining owners plus exception/context/client
in process-terminal custody, no retry. Do not claim already released owners
remain present. Failure before release retains the complete graph, including
partial setup objects, host upload owners, runtime modules and traceback owners.

A process-wide append-only terminal custody registry must retain the controller,
its guardian, socket/client, active ticket and failure graph before a terminal
disposition can escape to the fixed driver. Quarantine never returns to a
request loop. The fixed driver consumes the disposition by `os._exit`; its
supervisor separately observes/reaps exit. Emergency-success terminal custody
may include released guardian metadata, not a false claim of live allocations.

CPU tests use owners and real sockets; they cannot approve GPU release safety.
Later real drivers must prove complete storage coverage, including external
session/module owners, before any setup leaf. An unknown pre-return ownership
case remains unknown/quarantined, not repaired by a successful target sync.

## 5. Guard observations and versioned evidence

Use v3 for the guarded lifecycle/termination records, not C0 v2 records with
extra fields. Reuse the existing guardian coverage schema and the unchanged
guarded-wait protocol/supervisor evidence v1. The old validators retain their
old accepted domain; no automatic evidence upgrade or live fallback.

### 5.1 Exact record envelope and value domains

Freeze `q54.p12.4.setup-lifecycle.v3` and
`q54.p12.4.transport-termination.v3`. A lifecycle snapshot has exactly:
`schema_version, configuration, lane_id, record_kind, transition_trace,
guardian_coverage, summary`. Schema version is setup-lifecycle.v3;
configuration is unchanged `GuardConfiguration.to_payload()`; lane is
warp/cuda; record_kind is prefix/terminal. A termination record has exactly
`schema_version, lifecycle`, with transport-termination.v3 and an embedded
terminal lifecycle. No self-hash field: SHA-256 is over canonical JSON of this
payload, external to it. A parser produces inert evidence, never authority.

All new objects reject extra/missing keys; enums are exact strings, booleans
exact bool, counters exact int (never bool), nonnegative <= I64_MAX; ordinals
0..1023; identifiers use lifecycle.py's existing identifier grammar; digests
64 lowercase hex. Sequence bounds use wait_protocol.MAX_SEQUENCE. Lists keep
order; device sets are sorted unique lists. Nullable fields explicitly use
JSON null, not omitted keys. Each transition has exactly:
`index, epoch, operation, guard, details`. Index is contiguous from zero;
epoch is 0 until first setup descriptor, then exactly 1 (no new setup epoch).
Guard is null or exactly `{sequence, role}`; configuration supplies its session
and target. It names an ATTEMPT during ARM_PENDING, not a claimed live ticket.
A returned ticket becomes authoritative only at the corresponding ACK event.

Summary has exactly these replay-derived keys:
`controller_state, guardian_state, guard_state, guard_ref, next_sequence,
terminal_purpose, guarded_segment_failed, possible_dirty_device_ordinals,
unknown_device_scope, setup_operation_count, entered_leaf_count,
setup_work_counts, normal_complete_count, terminal_arm_attempt_count,
emergency_sync_attempt_count, emergency_sync_outcome, custody,
release_status, disposition`.

Domains: controller_state = RUNNING/TERMINATING/TERMINAL_INTENT;
guard_state = IDLE/ARM_PENDING/ARMED/COMPLETE_PENDING/FAILED;
guard_ref = null or guard shape above; next_sequence = valid sequence or null
on exhaustion. terminal_purpose = null/cpu_cancel/target_sync.
emergency_sync_outcome = not_attempted/started/succeeded/failed;
custody = none/retained/full_quarantine/remaining_quarantine;
release_status = not_authorized/authorized/started/completed/failed;
disposition = null/cancelled_before_submission_exit/aborted_synced_exit/
quarantined_process_exit/multi_or_unknown_device_quarantine_exit/
cleanup_failed_exit. Guardian state/coverage use unchanged coverage v1.
All counts/maps derive from trace, including leaf-kind map and coverage's
records/per-device counts. No sample latch, quiescence or observed-process-exit
field exists in this slice; it cannot represent those later capabilities.

### 5.2 Inherited events and scope grammar

Use the exact C0 definitions at approved base `11603cd` for
DeviceScopeDescriptor, GuardianResourceRecord and coverage v1, not its loose
termination replay as the v3 state machine. Allowed inherited events/details:

| Operation | Exact details | Precondition/effect |
| --- | --- | --- |
| guardian_created | lane_id, target_device_ordinal | Index0, epoch0, guard null; CONSTRUCTING; must match root config/lane. |
| owner_adopted | Full GuardianResourceRecord fields | CONSTRUCTING only; contiguous record sequence; retained before event; device owner only under active setup guard; host owner may be CPU/pre-guard. |
| setup_operation_started | descriptor_sha256, operation_id, leaf_kind | RUNNING, ARMED setup, CONSTRUCTING, no pending permit; new descriptor -> operation count+1, possible work already latched before leaf. |
| dirty_ordinal_added | device_ordinal, source, reason | Unique ordinal; source/reason identifiers; justified by current descriptor or just-adopted device owner. Never remove. |
| unknown_scope_set | source, reason | Once, identifiers; descriptor/returned owner or explicit unknowable failure. Never clear. |
| low_level_leaf_entered | descriptor_sha256, operation_sequence | Exact most recent unused permit and active setup guard; consume once BEFORE real call; never enters foreign/unknown scope. |
| guardian_sealed | coverage_sha256 | CONSTRUCTING -> SEALED once; no pending permit, known scope. Current records/state reconstruct this digest. |
| guardian_partial_frozen | coverage_sha256 | CONSTRUCTING/SEALED -> PARTIAL_FROZEN once during termination; digest replays new state. |
| guardian_quarantine_transferred | coverage_sha256 | PARTIAL_FROZEN/SEALED -> TRANSFERRED_TO_QUARANTINE once, before full_quarantine custody event. |
 
Descriptor validation adds `descriptor_observed {descriptor}` immediately
before its corresponding setup_operation_started, same active guard/epoch.
The full canonical descriptor is recorded, not just its supplied hash; the
two identities must agree. Inherited dirty events must exactly cover newly
declared ordinals and unknown flag before leaf entry. Returned-owner device
scope must likewise produce any new dirty/unknown event before next leaf.
An exception between these steps admits only a prefix or conservative
termination; summary includes descriptor/owner-declared possible scope even
before its notification event. It may not use a missing notification to select
cpu_cancel or target-only release. Explicit unknown_scope_set always tightens.

Every inherited event is attached to the actual guard/attempt at occurrence by
the controller's trace hook. Initialize CPU trace context before base __init__
calls that hook. No zipped independent counters. C0 may emit repeated operation
names for distinct resources/leaves; reject duplicate identities and illegal
predecessors, NOT all repeated names. No other C0 event (barrier, clean close,
sample admission, old notifier or termination) is admitted.

### 5.3 New events: exact details and transition grammar

`failure_kind` below is one of
operation/arm_ack/complete_ack/protocol/thread/reentrant/sync/cleanup.
`fault` reuses C0 _FAULT_KINDS; detection_phase is an identifier.
A failure value is an enum, not an arbitrary exception string. Retain actual
exception/traceback locally; no serialized object substitutes for that hold.

| Operation | Exact details | Required predecessor/effect |
| --- | --- | --- |
| normal_arm_started | {} | RUNNING, IDLE, role setup/resolve, exact next sequence; ARM_PENDING before calling client.arm. |
| guard_armed | {} | Exact pending reference and received ACK; ARMED; no duplicate. |
| normal_body_returned | {} | ARMED normal, body returned successfully, no pending setup permit/failure. |
| normal_complete_started | {} | Body returned once; COMPLETE_PENDING before client.complete. |
| normal_disarmed | {} | Exact successful DISARMED ACK; IDLE/ref null; normal_complete_count+1. |
| guard_failed | failure_kind | Pending/active normal attempt or guarded misuse; FAILED and sticky segment failure. Never COMPLETE or another ARM afterward. |
| termination_started | fault, detection_phase | Once from RUNNING; TERMINATING; pending permit revoked. Original guard state retained for ordered decision. |
| terminal_custody_retained | {} | Once after termination_started, registry append completed; custody=retained BEFORE further blocking work. |
| terminal_arm_started | purpose | After custody+freeze, original healthy IDLE, no sticky failure; purpose cpu_cancel iff no possible work/scope; target_sync iff complete known exact target. Role emergency; ARM_PENDING; count<=1. |
| terminal_arm_failed | failure_kind | Pending terminal attempt; FAILED; no sync/release. |
| emergency_sync_started | device_ordinal | ARMED emergency, target_sync only, exact target, once BEFORE sync call; outcome started. |
| emergency_sync_succeeded | device_ordinal | Matching started sync returned; outcome succeeded; no COMPLETE. |
| emergency_sync_failed | device_ordinal | Matching started sync raised; outcome failed, no retry/release. |
| resource_release_authorized | reason | reason cpu_cancel/emergency_sync; terminal ACK and correct purpose, zero holds and clean scope; emergency_sync also requires succeeded; guardian PARTIAL_FROZEN/SEALED -> RELEASE_AUTHORIZED. |
| ordinary_cleanup_started | {} | Authorized once; set release_status started BEFORE release_whole_graph. |
| guardian_released | {} | Release call returned and live guardian RELEASED; status completed. |
| cleanup_failed | guardian_state | After authorized/start/completed, observed state RELEASE_AUTHORIZED or RELEASED; status failed. No rewind, second release or guardian quarantine API. |
| quarantine_custody_committed | extent | extent full/remaining; full requires guardian TRANSFERRED_TO_QUARANTINE and no release start; remaining requires cleanup_failed and actual RELEASE_AUTHORIZED/RELEASED. Registry already retains context/remaining graph. |
| terminal_intent | disposition | Once, last terminal event; exact legal ending below. No claim of process exit. |

For normal_disarmed, guard field names the completed ticket even though the
resulting summary guard_ref is null; otherwise an event inside a pending/live
guard uses that reference, including inherited and termination events.
Pre-guard CPU events use null. On failure preserve the last attempt reference
for evidence even if GuardWaitClient's failed snapshot no longer exposes its
ticket; stored origin is the previously observed real attempt, not a replacement.
Sequence advances when guard_armed is observed, using next_sequence(); failed
pending attempts never authorize another attempt.

Termination starts after an observed failure or explicit shutdown request;
native_setup inner abort marks guard_failed BEFORE termination_started while
the normal guard is armed. Context unwinding cannot add a second failure/
termination. Foreign/unknown always selects full quarantine even if the guard
also failed. A normal guard active/pending at explicit termination is failed,
never completed to manufacture healthy IDLE. No new ordinary work follows
termination_started. Reentrant misuse invalidates success; it adds at most one
failure transition and follows existing custody, never reenters release/sync.

Legal terminal_intent endings (bidirectional, not one-way implications):

- cancelled_before_submission_exit: purpose cpu_cancel, ARMED terminal,
  zero descriptors/entered leaves/dirty/unknown, zero sync, release completed,
  custody retained, guardian RELEASED.
- aborted_synced_exit: target_sync, ARMED terminal, exactly one succeeded sync,
  known exact target, release completed, custody retained, guardian RELEASED.
- multi_or_unknown_device_quarantine_exit: compromised scope, no sync/release,
  full_quarantine and guardian TRANSFERRED_TO_QUARANTINE.
- quarantined_process_exit: failed normal guard/handshake, terminal ARM failure,
  sync failure, or unprovable ownership; no release, full_quarantine and
  guardian TRANSFERRED_TO_QUARANTINE. An unprovable owner must have made scope
  unknown before selection and therefore uses the multi/unknown disposition.
- cleanup_failed_exit: valid earlier release authorization, cleanup_failed,
  remaining_quarantine, guardian RELEASE_AUTHORIZED/RELEASED. It never asserts
  the complete original allocation graph survived reference release.

Repeated terminate/close reject without adding another event or side effect.
CPU/no-GPU cancellation is not ABORTED_SYNCED and does not manufacture device
quiescence. A retained guardian coverage manifest is historical ownership
metadata, not a claim that all corresponding live refs remain after cleanup.

### 5.4 Prefixes, supervisor compatibility and authenticity limit

Prefix snapshots end at any real committed event, with record_kind=prefix,
disposition null and no terminal_intent. Their summary replays exactly; pending
ACK/sync/cleanup are unfinished, not failures or successes inferred from time.
Terminal snapshots require exactly one final terminal_intent and record_kind=
terminal. No event follows it. A captured prefix cannot be upgraded by the
coordinator, and an absent worker tail remains null.

The fixed test artifact has exactly `schema_version, worker, supervisor`,
version `q54.p12.4.guarded-custody-validation.v1`. Worker is null or a v3
lifecycle snapshot; supervisor is unchanged validated guarded-wait evidence v1,
complete only when its actual process observation says EXITED. Configuration
must be identical. Match every worker ACK to supervisor ack_written and every
successful normal completion to matching COMPLETE/DISARMED history, ordered
by sequence/role. Pending worker attempts may correspond to no ARM received,
ARM committed or ACK written but not received. Worker may omit later facts if
killed; it may never claim facts absent from supervisor. Prefix normal-complete
count may lag, never exceed supervisor count; terminal successful release
permits no unmatched extra guard/COMPLETE. Terminal emergency remains without
COMPLETE, including cpu_cancel. Returncode is solely supervisor-observed;
worker terminal intent neither guarantees a particular returncode nor exit.

Worker records alone replay consistency, not authenticity against an actor
rewriting an entire legal trace. Runtime exact client/guardian/custody identities
provide local authority; source-frozen execution and independent supervisor
observations support test artifacts. No record/hash can mint live authority.
Tests may not expect an arbitrary self-consistent invented history to be
cryptographically rejected by public SHA-256 alone.

The complete immutable sealed-resource snapshot and final graph/quiescence
proof stay with the later final-setup composition. C2d1 forbids the inherited
`commit_target_barrier`, measured-submission admission and clean-close methods;
no synthetic graph/readiness hash or upload-only proof stands in for C4 setup.
Successful a2 upload can be inspected only by a fixed test driver, then undergo
terminal cleanup through these rules. It creates no prepared sample token.

## 6. Validation and checkpoint dependencies

C2d1 code acceptance requires CPU tests through the real controller, client,
supervisor transport/core and real subprocesses, not a copied mock state machine:

- normal setup segments: ACK-before-leaf, COMPLETE/DISARMED ordering, exact
  ticket identity and truthful normal-complete count;
- call the actual `native_setup._abort()` with this subtype during ARMED setup:
  zero emergency ARM/sync/COMPLETE, original graph and exception retained;
- failed leaf before return and submit-then-raise modeled at a fixed test leaf:
  sticky possibility/failure, no fallback; restored fields cannot revive it;
- pending ACK, no/wrong/EOF ACK, failed normal completion, duplicated/out-of-order
  complete, foreign client/ticket/configuration and thread/reentrant misuse;
- idle target-dirty emergency success: exactly one ARM/sync, no COMPLETE,
  release authorization distinct from cleanup and actual exit;
- no-GPU cpu_cancel: ordinary successful exit, CPU-owner cleanup hang and
  post-cleanup/pre-hard-exit hang with parent alive; actual terminal ARM/ACK
  precedes release, sync count zero, same deadline kills/reaps stalled child;
  no/wrong/EOF/stalled terminal ACK means no release/sync and retained graph;
- emergency sync/cleanup/exit hang with live parent: the same real supervisor
  deadline kills/reaps child; no second synchronization;
- foreign/unknown scope: target sync sentinel remains untouched; graph survives
  until hard exit, ordinary destructors do not run on quarantined paths;
- complete trace mutation/re-hash attacks: invented/duplicate transitions,
  false IDLE, false zero dirty scope, falsely successful ACK/sync/release, wrong prefix/terminal tails and
  fabricated worker exit all reject; hand-authored legal fixtures follow the
  above grammar, not implementation-generated expectations; copy/rehash cannot
  become local authority; legitimate independent fully rewritten histories are
  not falsely advertised as detectable by hashes alone;
- source policy/catalog repins and C0 A/K, C1c S/V/E, C2b core/process and C2d0
  source-only regressions; explicit no-diff check for protected C0 source files.

The fixed test launcher may select only enumerated fault cases. It is not a
product role or arbitrary callback entry. CPU fixtures are declared as such;
they do not satisfy any real-device mandatory gate.

Then continue C2d, not declare it complete:

1. Adopt this same controller in the captured-source bootstrap's guarded role;
   guardian precedes CUDA-capable import/init, all module owners are retained
   before body execution, and source/actual-import audits remain active.
2. Exercise the existing real Warp/Torch upload functions with register-before-
   copy under the same controller in source-frozen supervised sacrificial
   processes. Mandatory both-lane target1/ambient0, foreign pre-return touch,
   unknown scope, upload/ready-event/record faults require real GPUs, not mocks.
3. Compose final resource seal/barrier evidence, warmup and the future C2e graph
   in dependency order from runtime-integration v3. A GPU-complete upload is
   not full graph readiness. C2f still tests the actual same integrated drivers.

Each code checkpoint has fresh independent review and scoped PID/parent/exit,
residual-worker and GPU-process checks. Do not start a new GPU round on an
unresolved owned residual. No timeout is treated as an observed process exit.

## 7. Scope and removal

No product request surface, protocol v2 negotiation, GPU preparation gate,
R0/R1 activation, arena authority, a3b, result/baseline, R2/S/host or B3 is opened
here. Shared collab rules and historical approvals are not edited. C4 remains
incomplete until all existing parent requirements are actually met.

The new custody controller is intended to become the sole selected worker's
runtime lifecycle; it is not a permanent alternative selectable by requests.
At P12.4e remove unused C0/native mode, losing lane adapters and transitional
sacrificial programs after equivalent retained coverage. Keep shared resource
ownership, descriptor/permit and wait protocol implementations single-sourced.
The cleanup ledger must record this adoption/deletion obligation in its code
candidate. No duplicate allocator, kernel, upload loop or broad generic bridge
is authorized by this supplement.
