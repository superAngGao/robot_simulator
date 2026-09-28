# C4c2b: supervised normal-wait protocol core

Initiative: q54-optical-lab-p12-4c4c2b-supervised-waits
Stage: design-request
Author: codex
Version: v1
Date: 2026-09-29
Status: in_review
Related Files: C4c arena design v7; C4c1 ABI design v3; C4c2a extraction
design/request and code-review v1; optical_worker/lifecycle.py;
tools/optical_closed_worker_supervisor.py; tests/support/sacrificial_watchdog.py
Owner Summary: Implement the CPU-only protocol core needed to supervise repeated
setup/resolve waits. Preserve sealed C4c0 A/K behavior and all GPU/result gates.
This is a dependency slice toward C4c2, not a reduced definition of C4 completion.

Base: `0da32dbf733e7da483db867a7fcd59595b510e61`, evidence archive following
approved C4c2a code `67b401878bb92371590885af7835be36430d3009`.

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
integer milliseconds; this core provides no default product time budget.
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

`sequence` is a non-bool integer in0..2^63-1. ARM consumes the next sequence;
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

Client states: IDLE, ARM_PENDING, ARMED, COMPLETE_PENDING, FAILED, TERMINAL.
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
Wrong/stale/cross-client/retired tickets reject before writing. Emergency
completion, nesting, duplicate complete and reused sequence are rejected.

The protocol client does not accept or execute the blocking callable. Fixed
trusted driver code owns that call, catches its exception while retaining its
guardian and never calls complete on failure. Unit tests cannot pretend that
an ARMED record alone proves a GPU call happened or succeeded. Success and
failure are distinguished by the driver's actual control path in later runtime
integration. A `fail()` operation makes the client irreversibly FAILED locally
without a wire operation or deadline reset; it cannot disarm anything.

## 6. Supervisor state and deadlines

Supervisor states: IDLE, ARMED_NORMAL, ARMED_TERMINAL, TERMINATING, EXITED.
Only its monotonic clock defines deadlines. In IDLE, a valid next ARM commits
the active identity and `now + operation_budget` BEFORE producing ARMED. The
caller sends that response only after commit. If response encoding/writing
fails, it enters TERMINATING; it cannot clear or extend the existing deadline.

In ARMED_NORMAL, only exact COMPLETE before expiry may retire the active wait.
It commits IDLE before producing DISARMED. Failure to deliver DISARMED still
forces TERMINATING with a terminal-exit deadline, even though the native wait
was reported successful. This conservative outcome is not ordinary cleanup
authorization. A late COMPLETE cannot revive an expired wait.

ARMED_TERMINAL rejects every new control request. Emergency success ends by
worker exit, not disarm. Expiry, malformed/stale/out-of-order message, channel
EOF, ACK-send failure or parent liveness loss makes TERMINATING irreversibly.
The event loop independently watches process exit and parent-liveness while
handling partial reads/writes. It may never block on reading a full frame or
writing an ACK beyond the active deadline.

Track a normal-operation deadline and a sticky terminal-exit deadline. The
effective kill deadline is their minimum when both exist. Parent loss/error
sets `terminal_deadline = min(existing, now + operation_budget)`; it must never
postpone an earlier guard deadline. COMPLETE cannot clear a terminal deadline.
Normal COMPLETE at/after its deadline is expired, even if already buffered.
Before every transition, evaluate deadline expiry with the actual clock.

When the deadline expires the supervisor kills and reaps the worker process;
it does not run Python cleanup inside that worker. Keep the event loop alive
until exit status is collected. Record whether exit preceded deadline, was
watchdog-killed, or failed to be reaped. No reuse/reset/new session in the same
core. Observing process exit from any state commits EXITED without claiming
device completion or resource release.

Repeated parent-loss/transport-fault observations retain the first cause and
earliest deadline and append no duplicate state transition. Duplicate ARM or
COMPLETE is an error, not an idempotent retransmission; the protocol never
retransmits an ambiguous request.

## 7. Evidence, replay and safety meaning

Define immutable canonical supervisor evidence with version/config identity,
the ordered transition trace, final state, next sequence, active wait identity,
deadline disposition, arm/normal-complete counts, terminal reason, exit status
and watchdog-kill observation. Fixed operation enums describe initialization,
arm commit, ACK transport outcome, normal completion commit, terminal entry,
deadline expiry and observed process exit. Each transition includes its
monotonic timestamp and the exact CPU identity fields necessary for replay.

The validator replays the state machine from IDLE. It rejects invented/repeated
operations, sequence gaps, device/session drift, backward timestamps, deadline
extension, COMPLETE after terminal/expiry, forged summaries and normal
completion for emergency. It derives counts/final state/deadlines from the
trace rather than trusting duplicated fields. SHA256 is content integrity,
not authentication. Evidence does not claim arena state, GPU completion,
quiescence, allocation safety or guardian release; those are different owners.

The client and supervisor both fail closed independently. In particular an
unresponsive supervisor holding the socket open cannot permit a blocking leaf:
the client ACK deadline expires first and the leaf is never called. After ACK,
a permanently blocked leaf is bounded by the supervisor, even with the parent
still alive. The trusted supervisor itself is an assumed functioning process;
this protocol is not a hardware fault-tolerant system.

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
  warmup frames and the final supervised target barrier precede final readiness
  and guardian sealing. The existing warmup0 factory dependency needs explicit
  provisional/final ownership ordering, not mutated frozen evidence.
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
