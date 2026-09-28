# C4c2b: supervised normal-wait protocol core

Initiative: q54-optical-lab-p12-4c4c2b-supervised-waits
Stage: design-request
Author: codex
Version: v3
Date: 2026-09-29
Status: in_review
Related Files: C4c arena design v7; C4c1 ABI design v3; C4c2a extraction
design/request and code-review v1; optical_worker/lifecycle.py;
tools/optical_closed_worker_supervisor.py; tests/support/sacrificial_watchdog.py
Owner Summary: Implement the CPU-only protocol core needed to supervise repeated
setup/resolve waits. Preserve sealed C4c0 A/K behavior and all GPU/result gates.
This is a dependency slice toward C4c2, not a reduced definition of C4 completion.

Base: `3e78b481ab27db36330a9b7379c61b4e2bcecfb5` (unapproved v2).
V3 retains the independently confirmed F1 ACK closure and repairs v2 R1 with
an encoding-derived session capacity covering sequence, counters and trace tail.
The approved code ancestor is C4c2a `67b401878bb92371590885af7835be36430d3009`,
archived at `0da32db`. Earlier designs/reviews remain history; only v3 is proposed.

## 1. Code-fact gap scan

The product supervisor accepts one `A`, sets a deadline, sends `K`, and never
disarms that terminal deadline. It correctly supervises emergency shutdown.
It has no repeated normal-wait arm/complete protocol. `native_setup.py` calls
ready-event synchronization directly, and its target barrier also blocks.
`NativeWorkerSetupLifecycle.terminate()` can retry a target synchronization
after a failed setup barrier; it does not know that a guarded normal wait has
already failed. Those behaviors must not be silently reused for C4c2.

The C1c S/V/E test-only watchdog proves the relevant ACK-before-call mechanics,
but its V/E roles are terminal and its source is deliberately not product code.
Importing that test driver into the worker would violate the frozen closure.

C4c2a now supplies exact CPU-safe input/readiness/provenance definitions, but
does not solve supervision, final readiness order, the native Warp timeline or
the live R0/R1 lease graph. This protocol core closes only the first dependency.
It cannot authorize GPU preparation or recorded execution on its own.

## 2. Delivery, boundaries and authorization

Deliver two stdlib-only production modules:

- `optical_worker/wait_protocol.py`: strict versioned control records, bounded
  framed I/O and client ticket state;
- `optical_worker/wait_supervision.py`: deterministic supervisor transition
  core, deadline ownership and replayable CPU evidence.

Deliver unit tests and a fixed standalone subprocess test harness that runs the
actual core and client. No arbitrary callback, source, pickle, module path,
native handle or executable object may cross its control protocol. Fixed test
fault names select compiled-in test behavior; this harness is not a product
worker or a product manifest mode.

Do not edit C4c0 runtime-contract/gate JSON, its worker/session/launcher/
supervisor, native_setup or its lifecycle. Do not change C1c test driver or its
S/V/E semantics. Do not install a second watchdog in an existing running worker.
No GPU module is imported by these new modules, and this slice runs no GPU
preparation or sample. The future worker integration must explicitly adopt the
new protocol through a separate versioned manifest/code review.

This scope is CPU protocol infrastructure, not the subsequent runtime
implementation requiring the C4c2a section8 integration design. Section9 below
records that integration's actual dependencies and hard gates; none is silently
considered implemented or approved by this core's tests.

## 3. Trust and configuration

The control link is a private inherited socketpair between the trusted worker
and its supervisor. It is separate from the parent's request channel. The
parent cannot issue ARM/COMPLETE by sending ordinary worker requests. The
supervisor and worker receive the same immutable configuration before startup:
protocol version, session identity, target ordinal, ACK budget and operation
budget. Configuration is plain CPU metadata: validating an ordinal must never
initialize a runtime or inspect the default GPU.

Use protocol ID `q54.closed-worker.guarded-wait.v1`. Session identity is exactly
32 lowercase hex characters. Ordinal is a non-bool integer in0..1023, matching
the existing HELLO domain. Both timeout values are explicit positive non-bool
integer milliseconds in1..2^31-1; this is an encoding/arithmetic limit, not a
product budget. This core provides no default product time budget.
Future adoption will source values from the reviewed runtime manifest, not a
parent request or measured result. No new performance or memory gate is added.

Only one thread owns a client and one event-loop thread owns a supervisor core;
capture thread identity at construction and reject cross-thread entry before
sending bytes/changing state. Each session has its own socket, sequence, tickets
and deadlines. Reentrancy and nested waits fail closed, not silently serialize.
This is supported-path control in a closed trusted process, not a Python/native
memory security boundary. `__module__`, hashes and private names are not proofs
of authority. Parsed evidence never creates a live ticket or reopens a session.

## 4. Fixed wire contract

Use the existing production canonical JSON codec and framing rules, with a
guard-specific maximum frame body of1024bytes. This is a bounded control-record
size, not a sample limit. The record has exactly these fields:

```text
protocol_version, session, sequence, device_ordinal, role, operation
```

`sequence` is a non-bool integer in0..MAX_SEQUENCE (section7). ARM consumes the next sequence;
COMPLETE references that same exact sequence. No wraparound. Role is exactly
`setup`, `resolve` or `emergency`; operation is exactly `ARM`, `ARMED`,
`COMPLETE` or `DISARMED`. Responses echo every non-operation field exactly.
There are no optional extension fields or caller-selected deadlines.

Setup and resolve are normal, repeatable guarded waits. Emergency is terminal:
it can be armed once only from idle and never completed/disarmed. Failure of an
already armed setup/resolve wait does NOT arm a second emergency wait. The
original deadline continues to protect its terminal exit.

Reject unknown/missing/extra/duplicate fields, booleans as integers, wrong
version/session/device, role drift, noncanonical JSON, invalid lengths,
non-finite/floating sequences, partial EOF and oversized input. A stream parser
must retain partial frames without blocking the supervisor's timer/liveness
loop. Decode errors after any phase make the session terminal.

## 5. Worker client state and exact ticket

Client states: IDLE, ARM_PENDING, ARMED, COMPLETE_PENDING and FAILED.
Construction begins IDLE. A ticket is created only after receiving the precise
ARMED record. It is immutable and bound to client identity, session, ordinal,
role and sequence. COMPLETE requires the exact outstanding object (`is`), not a
copy, reconstructed dataclass, matching fields or a matching hash.

`arm(role)` is allowed only in IDLE. It changes to ARM_PENDING before any write,
sends ARM and reads ARMED within one shared absolute monotonic ACK deadline
covering both send and receive, not two resettable budgets. Only a matching ACK
changes to ARMED and permits the caller's blocking leaf to begin. Timeout, EOF,
wrong ACK, write failure or exception makes FAILED irreversibly; no ticket is
returned. The driver must then quarantine and exit without invoking the leaf.

`complete(ticket)` is allowed only after a normal blocking leaf has returned
successfully, with the exact outstanding normal-role ticket. It enters
COMPLETE_PENDING before sending. Only exact DISARMED returns it to IDLE and
retires the ticket. The same total send+receive ACK deadline rule applies.
Wrong/stale/cross-client/retired tickets reject before writing and make the
client FAILED. Emergency completion, nesting, duplicate complete and reused
sequence also make it FAILED. Any COMPLETE exchange failure requires the fixed
driver to retain its graph and hard-exit without ordinary cleanup, without
retrying COMPLETE or arming emergency. Client shutdown closes its private link;
it never fabricates successful disarm. No asynchronous destructor runs this policy.

The protocol client does not accept or execute the blocking callable. Fixed
trusted driver code owns that call, catches its exception while retaining its
guardian and never calls complete on failure. Unit tests cannot pretend that
an ARMED record alone proves a GPU call happened or succeeded. Success and
failure are distinguished by the driver's actual control path in later runtime
integration. A `fail()` operation makes the client irreversibly FAILED locally
without a wire operation or deadline reset; it cannot disarm anything.

## 6. Supervisor state, pending ACKs and deadlines

States are exactly IDLE, ARM_ACK_PENDING, ARMED_NORMAL, ARMED_TERMINAL,
DISARM_ACK_PENDING, TERMINATING and EXITED. State begins IDLE. A successful
ARM commits its active identity, original operation deadline and a separate
pending ARMED response BEFORE any response byte is sent. The ACK-send deadline
is `commit_time + ack_budget`; the operation deadline is `arm_time + operation_budget`.
Only complete transport write of that exact ACK may commit ARMED_NORMAL or
ARMED_TERMINAL. Pending is not equivalent to delivered.

Normal COMPLETE may be accepted only in ARMED_NORMAL with exact identity,
before every current deadline. It commits DISARM_ACK_PENDING, not IDLE.
The original operation deadline remains armed. Set a new absolute DISARMED
send deadline to `complete_time + ack_budget`. Clear neither deadline until
the entire exact DISARMED response has been written and the core records that
outcome. At that point retire the active identity and both normal deadlines
and enter IDLE. Partial writes never change deadlines. A returned native wait
is not permission to leave an unbounded ACK-output interval.

At most one1024-byte response body plus its4-byte header may be pending.
The supervisor transport is nonblocking. It watches process status, parent
liveness, input, pending output and the earliest deadline on every iteration.
While either ACK is pending, it may retain at most one partial input frame
(up to1028bytes); a complete input frame before ACK-write commit is a protocol
fault, not a queued next operation. Extra bytes beyond that bound are a fault.
It never accepts a second request or creates a second response while pending.
On terminal entry, discard unsent response bytes and cease protocol output;
a prefix already written cannot be recalled and cannot become a valid full ACK.
If an ACK was fully written but recording its outcome fails, retain the
original guard and terminate; do not process buffered input as if committed.

ARMED_TERMINAL rejects every further request. An emergency operation ends with
worker exit, never DISARMED. Failed normal waits also leave their original
guard armed; there is no second emergency attempt. Unexpected EOF, stale or
malformed input, failed output, parent loss or expired deadline enters
TERMINATING irreversibly. Failed normal ACK exchanges at the client separately
trigger the hard-exit policy in section5.

Maintain operation, pending-ACK and sticky terminal deadlines. The effective
deadline is their minimum. On non-expiry terminal entry use the minimum of all
existing deadlines and `now + operation_budget`; cancel pending output only
AFTER folding its deadline into the terminal deadline. Parent loss therefore
cannot extend an earlier ACK/operation deadline. Repeated fault observations
retain the first reason and earliest deadline and create no duplicate event.
COMPLETE never clears terminal state. Deadlines at `now` are already expired;
check expiry before handling input/output success, including buffered COMPLETE.
No partial read/write or new fault restarts a budget.

Expiry records a single deadline-expired observation. The event loop commits a
single kill-request event BEFORE calling the process kill operation. This is
an attempt, not proof of exit. It keeps observing/reaping until a real return
code is available; kill error is recorded, never interpreted as successful
reaping. Only an actual process poll/wait return code permits EXITED. There is
no timeout that fabricates exit. If process status remains unavailable, only
incomplete TERMINATING evidence is possible and the outer process owner must
retain the handle/custody. No reset, restart or new session is available.

## 7. Normative schema and replay

All representations below are exact field sets, with no extension fields.
Times are integer monotonic **nanoseconds**, never floats or wall-clock time.
`I64` means a non-bool integer in0..2^63-1; `I32` means a non-bool signed
32-bit integer. Millisecond budgets are non-bool integers in1..2^31-1 and
conversion is exactly `budget_ms * 1_000_000`. Every timestamp addition is
checked against I64 BEFORE committing/issuing ACK. Equal consecutive
timestamps are legal; decreasing/out-of-domain clock readings are not.

Configuration:
`{protocol_version, session, device_ordinal, ack_timeout_ms, operation_timeout_ms}`
with the domains in sections3–4. All evidence uses the same config object
content, not a second source of timeout/identity values.

`Seq` is a non-bool integer in0..MAX_SEQUENCE, a strict subset of I64.
Active wait identity is either null or `{sequence:Seq, role}`, with the three
frozen roles. The session/device are inherited from config and checked on
every wire record. Pending ACK is either null or
`{operation, sequence:Seq, role, deadline_ns:I64}`; operation is ARMED or DISARMED.

Evidence root:

```text
schema_version = "q54.closed-worker.guarded-wait-evidence.v1"
configuration = exact configuration above
trace = ordered list of exact events below
summary = exact derived summary below
```

Event: `{index:I64, at_ns:I64, operation, details}`. Index starts0 and increases
by1; the first and only initialization event has operation `initialized`,
details `{}`, and initializes IDLE/next_sequence0/all deadlines null.
The remaining event grammar and transitions are:

| operation | exact details | precondition and effect |
| --- | --- | --- |
| arm_committed | {sequence:Seq, role} | IDLE, exact next sequence; set active/operation deadline and pending ARMED deadline; ARM_ACK_PENDING |
| ack_written | {operation, sequence:Seq, role} | exact pending ACK before expiry; ARMED commits its corresponding armed state and clears only ACK deadline; DISARMED retires active/operation/ACK deadlines, increments normal-complete count and enters IDLE |
| normal_complete_received | {sequence:Seq, role} | ARMED_NORMAL exact active normal role; enter DISARM_ACK_PENDING with ACK deadline; retain original operation deadline |
| terminal_entered | {reason} | any live nonterminal state; preserve first reason; fold all deadlines as section6; drop pending output; TERMINATING |
| deadline_expired | {source} | an effective deadline exists and at_ns >= it, no previous expiry; source is selected as below; terminal reason becomes deadline_expired only if none existed; drop pending output after retaining deadline; TERMINATING |
| kill_requested | {} | TERMINATING, deadline_expired exists, no prior kill request; record attempt only |
| kill_failed | {error_kind} | exactly one preceding kill_requested, no prior kill_failed/exit; record OS failure only, retain TERMINATING |
| process_exited | {returncode:I32} | any non-EXITED state after actual poll/wait observation; clear active/pending/deadlines, set returncode; EXITED |
| clock_rejected | {reason} | live state, at most once; event at last valid at_ns; terminal deadline is min(existing deadlines, that at_ns); retain custody and prohibit output; first terminal reason wins, otherwise clock_fault |

The exact `terminal_entered.reason` enum is `parent_lost`, `input_eof`,
`invalid_message`, `request_out_of_order`, `ack_write_failed`,
`sequence_exhausted`, `process_observation_failed`. The first observation
wins; subsequent fault notifications in TERMINATING do not append another
terminal event. The exact `kill_failed.error_kind` enum is `permission_denied`,
`process_missing`, `other_os_error`; none substitutes for a return code.
The exact `clock_rejected.reason` enum is `backward`, `out_of_domain`,
`deadline_overflow`; rejected raw values are not serialized as timestamps.
Constructor clock failure rejects construction before there is a live core.

The `deadline_expired.source` enum is `terminal`, `ack`, `operation`.
Choose the earliest deadline; ties use that listed priority. Terminal entry
folding/cancelling pending output retains its earliest deadline value even
though its source is subsequently terminal. Expiry event alone does not mean a
kill was attempted. Kill request alone does not mean the process exited.
`clock_rejected` may not reset an existing earlier terminal reason; otherwise
the derived reason is clock_fault. Its immediate deadline is then subject to
the ordinary expiry/kill observation sequence. No events follow process_exited.

The exact derived summary is:

```text
state                              one of the seven states
next_sequence                      Seq or null (exhausted)
active_wait                        active identity or null
pending_ack                        pending ACK or null
operation_deadline_ns               I64 or null
terminal_deadline_ns                I64 or null
effective_deadline_ns               I64 or null
terminal_reason                    null, terminal reason enum, deadline_expired, clock_fault
arm_count                          I64
normal_complete_count              I64
deadline_expired                    bool
kill_requested                     bool
kill_error_kind                    null or kill_failed enum
observed_returncode                I32 or null
complete                           bool, exactly state == EXITED
```

### Encoding-derived session capacity and exhaustion

The protocol's fixed structural constants are:

```text
I64_MAX = 2^63 - 1
MAX_GUARDS = (I64_MAX - 8) // 4 = 2305843009213693949
MAX_SEQUENCE = MAX_GUARDS - 1 = 2305843009213693948
```

This supersedes v1/v2's inconsistent inclusive2^63-1 sequence domain. It is a
canonical encoding capacity, not a configurable deployment/performance limit.
Reject sequence MAX_SEQUENCE+1 before ARM/ACK; do not saturate counts, truncate
trace or wrap to0. Initial next_sequence is0; consuming MAX_SEQUENCE sets it to
null. The last active ticket may complete normally. Any subsequent ARM is
sequence_exhausted and terminal, even if it supplies0. The client enforces the
same capacity before sending a further ARM; no resume/reset API exists.

A full normal guard contributes exactly four events. Emergency contributes at
most two. Initialization contributes one; after the last ordinary operation
there can be at most six additional events: terminal_entered, clock_rejected,
deadline_expired, kill_requested, kill_failed and process_exited, each at most
once. Intermediate failures stop normal guards rather than adding an unbounded
tail. Repeated faults, ticks and clock failures append no additional events or
reset deadlines. In particular clock_rejected is single-use for the session;
repeated bad clock observations retain its original immediate-expiry/kill
obligation, and cannot keep the loop waiting for a new valid clock reading.
Its immediate deadline triggers the ordinary expiry/kill sequence at the last
valid timestamp when no newer valid reading is available.

Therefore, with N <= MAX_GUARDS, total events <= 1 + 4*N + 6. At the maximum,
events <= 9223372036854775803 and final index <= 9223372036854775802, both below
I64_MAX. Both counts are <= MAX_GUARDS. Eight units were reserved before the
four-event division so the complete terminal tail fits without another codec
representation or overflow exception. No new trace operation may be added
without revisiting this proof and versioned contract.

Check every sequence/count/index increment and timestamp addition before
committing a transition or issuing an ACK. Before a new ARM reserve its worst
four-event contribution plus the entire remaining terminal tail. The fixed
session cap makes that reservation succeed for every advertised legal ARM.
Invalid/injected state that violates the reservation fails closed without
issuing ACK; it cannot be serialized as a legal full-session history. Runtime
must not reach such a state through the frozen transition grammar.

Boundary tests exercise the actual checked arithmetic for all incremented
fields, MAX_SEQUENCE and its first invalid value, exact counts and tail/index
capacity, and canonical encoding at each maximum. A reduced-capacity
mathematical test model may complement these primitive checks; it is not a
substitute implementation or an accepted high-sequence short trace. Valid
session replay always begins with initialization and sequence0.

Summary counts come exclusively from arm_committed and successful DISARMED
ack_written events. A pending DISARMED does not increment normal-complete.
No device completion, quiescence, allocation safety, arena state or guardian
release field exists here. There is deliberately no `watchdog_killed` claim:
the evidence distinguishes requested kill, OS error and observed return code.

Snapshots are allowed at every committed state, including pending ACK and
unreaped TERMINATING. They have `complete=false` and null observed_returncode.
Final evidence requires complete=true and actual process_exited as its last
event. Reject snapshots presented as final. Kill requested but status unknown
is never final; no timeout or caller-supplied exit flag changes that meaning.

The validator replays this grammar from initialization, checks identity,
sequence, arithmetic, timestamps, exact fields and summary equality, and
rejects missing/reordered ACK events, invented transitions, deadline extension,
post-terminal normal completion, forged kill/exit summaries or extra events
after exit. Root SHA256 is computed through the existing canonical codec.
It proves content consistency, not authenticity: an entirely rewritten legal
trace is not cryptographically distinguishable from another legal trace.
Parsed evidence never creates a live client ticket or supervisor process.

The trusted supervisor is assumed to remain scheduled/functioning after its
ACK. Before ACK an unresponsive supervisor is covered by the client's deadline;
after ACK a permanently blocked worker leaf is covered by the independent
supervisor. This is not a hardware-fault-tolerant security boundary.

## 8. Acceptance for this slice

Run CPU tests in fresh subprocesses and independently inspect actual core use:

1. Canonical codec/field/type/limit matrix, partial frames and EOF at every
   prefix, duplicate JSON keys, wrong ACK identity and sequence maximum.
2. Setup -> complete -> resolve -> complete -> emergency -> exit; exact ticket
   identity, concurrent instances, cross-thread/reentrant rejection, duplicate
   complete/arm, copied/foreign/retired tickets and attempted post-failure reuse.
3. Deadline committed before ACK; normal complete disarms only its own deadline;
   parent loss before/during/after complete cannot be cleared; late buffered
   COMPLETE, partial-frame trickle and stalled ACK reader cannot starve expiry.
   Separately stall DISARMED after normal COMPLETE, retain the original deadline,
   inject partial writes/parent loss, and reject a buffered next ARM before ACK
   commit. Verify no response bytes are emitted after terminal entry.
4. Real supervisor/child processes with parent alive: normal wait hangs forever,
   emergency wait hangs, supervisor holds socket but never ACKs, wrong ACK,
   ACK EOF and response-send failure. Use fixed child fault programs. Prove
   leaf-entry count zero without ACK; prove bounded kill/reap after ACK.
5. Child resource-finalizer sentinels: watchdog kill/process-terminal hard exit
   does not run ordinary destructor cleanup. This proves process behavior,
   not CUDA safety or guardian completeness. Later GPU validation remains
   mandatory and cannot be replaced by these CPU sentinels.
6. Mutate each evidence field, repeat transitions/recompute hashes, invent
   dispositions and spoof summaries. Replay must reject; legal traces roundtrip.
   Include independently specified fixtures for every table transition, pending
   ACK snapshots, missing/reordered ACK outcomes, sequence exhaustion, integer
   time/config boundaries, equal timestamps, deadline overflow, kill failure,
   unknown exit status and snapshot-as-final rejection.
7. New modules' cold import attempts include no Torch/Warp/benchmark/test/tool
   modules. Production C0 gate/runtime and existing request/AST surfaces are
   byte-identical. Rerun C0 protocol/watchdog regression and C1c watchdog tests.

## 9. C4c2 runtime integration obligations (not delivered here)

Before integration implementation, its separate complete design must close:

- **Materialization.** Move the actual workload/assets/MJCF/camera owner out of
  benchmarks/examples/tools without duplicating mathematics. Freeze every
  resolved render parameter. `non_identity_compound_xyz_v1` currently lacks a
  numeric resolver; an owner clarification has been requested. No invented
  resolution, microfixture substitution or gate hash rewrite is authorized here.
- **Admission.** Enumerate actual transitive repository modules and exact source
  hashes. Read-only `ctypes` packed-argument inspection in warp_native_render
  requires narrow function/operation admission, never a global exemption or
  raw native-launch privilege. Product runtime v2 adoption must be explicit.
- **Setup/readiness.** Guardian precedes CUDA-capable import; register-before-
  next-leaf throughout uploads/allocations/compile/record/warmup. Ten frozen
  warmup frames must be observed before final readiness. The final supervised
  target barrier binds already-created final readiness and the recorded graph,
  then permits guardian sealing. The existing warmup0 factory dependency needs
  explicit provisional/final ownership ordering, not mutated frozen evidence
  or a barrier/readiness hash cycle.
- **Timeline.** R0 render-only; R1 scene update -> leaf4/sort immutable-topology
  Warp refit -> render. R0 has3 diagnostic pairs, R1 has5; one optical terminal
  event after all stages. Diagnostics-off creates none. C2 enqueue/finish and
  C4a exact lease state authority compose without a second control state machine.
- **Authority/cost.** Exact a2 allocations/storage and one-shot C3 identities
  bind the actual stages. Lease -> transaction/owner/slot/generation is unique;
  static template identity cannot mint live execution. Full preflight remains
  before timing and measured acquire remains O(1).
- **Failure custody.** The possible-submission latch precedes every launch/event
  leaf, even submit-then-raise. A failed/timed-out guarded wait goes directly to
  whole-graph process quarantine, never a second emergency sync. Target-only
  unguarded setup failure may use one emergency guard; foreign/unknown scope
  never target-syncs its way to ordinary release.
- **Idle close.** Historical submission stays latched. New exact post-resolution
  and reclaim evidence, not clearing that latch or an old setup state, proves
  all tokens/leases/holds retired. CLOSE ACK and actual process-exit evidence
  remain distinct. Resource graph stays retained through failed response writes.
- **Same-slice proof.** Real Warp/CUDA setup and post-submit disconnect/failure
  matrices, normal wait hangs, target1/ambient0 and guardian sentinels must pass
  before GPU preparation/submission is enabled. No protocol-core approval is
  permission to bypass that prerequisite.

C4c3 static interop/R2, C4c4 closure and C4d remain required. B3, host payload,
shipping, a3b, result/baseline and backend selection remain separately closed.

## 10. Cleanup

The selected-backend worker may retain this common supervision core; it has no
renderer selector or candidate-specific math. Register it in the cleanup ledger
as shared safety infrastructure pending actual production adoption. Remove it
if no selected production/test consumer remains. C1c S/V/E stays test-only and
is retired with its sacrificial harness after equivalent production-lifecycle
coverage is approved; do not count two differing protocols as one version.
The fixed new subprocess harness remains a protocol regression test, not a
second product supervisor. No cleanup or gate change is performed in this
design-only commit.
