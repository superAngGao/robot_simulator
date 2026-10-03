# C2d1: adopt guarded failure custody at the actual setup exception boundary

Initiative: q54-optical-lab-p12-4c4c2d-guarded-custody
Stage: design-request
Author: codex
Version: v1
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
4. Healthy actual IDLE and no GPU setup entered => clean cancel/release/exit.
5. Healthy actual IDLE with complete target-only possible work => one emergency
   ARM/ARMED followed by at most one exact target sync.

The exact sync callable is selected by the trusted fixed driver from the same
lane/runtime/target already adopted by setup, not from IPC or a parsed record.
No descriptor check may import/init a runtime merely to discover its ordinal.
The controller checks the exact emergency ticket/session/target before calling
that leaf. ACK failure means zero sync calls and quarantine.

On emergency sync success with zero external holds, authorize whole-graph
release and terminal exit. Keep the exact emergency ticket ARMED; never send
COMPLETE/DISARMED or restart/clear its deadline. Cleanup and exit remain inside
that deadline. The guardian release routine is the existing sole resource
release implementation, not a parallel list of selected objects to free.

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

Record normal-segment begin/ACK, operation failure/success, COMPLETE start/ACK,
emergency begin/ACK, sync start/outcome, poison, custody transfer, release
authorization/start/completion and intended terminal disposition. Each record
has a contiguous index, operation enum and exact operation-specific fields;
unknown or duplicate operations reject. Guard references are session, target,
sequence and role derived from the actual ticket/client, never supplied hashes.

Every inherited setup/owner/scope transition is associated with the active
guard identity or explicit CPU/pre-guard state at occurrence. Do not infer that
association later by zipping independently editable counters. Preserve exact
ordering with the custody and guard events in one lifecycle replay stream.

The v3 replayer derives and checks: dirty/unknown scope, permit entered count,
adopted owner coverage, active/pending/failed guard state, sticky segment failure,
one emergency attempt maximum, sync outcome, guardian state, release/custody
status and final disposition. It reuses shared existing parsing/validation
helpers where applicable; do not copy a permissive summary-only validator.

A terminal record is descriptive and cannot authorize another controller or
GPU operation. Worker disposition is not actual process exit. The supervisor's
unchanged replayer supplies observed exit, guard deadlines and normal-complete
count; matching configuration/sequence/role and compatible worker ordering are
checked when the fixed test driver assembles its validation artifact. A missing
worker tail after hard kill remains missing, never synthesized as success.

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
- emergency sync/cleanup/exit hang with live parent: the same real supervisor
  deadline kills/reaps child; no second synchronization;
- foreign/unknown scope: target sync sentinel remains untouched; graph survives
  until hard exit, ordinary destructors do not run on quarantined paths;
- complete trace mutation/re-hash attacks: invented/duplicate transitions,
  false IDLE, false zero dirty scope, falsely successful ACK/sync/release and
  fabricated exit all reject; copied evidence cannot become local authority;
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
