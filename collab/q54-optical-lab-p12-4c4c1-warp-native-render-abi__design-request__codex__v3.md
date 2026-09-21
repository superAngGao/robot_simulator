# Q54 Optical Lab P12.4 C4c1 Warp Native Render ABI — Lifecycle Closure

- Initiative: Q54 optical computation workflow / P12.4 backend selection
- Stage: C4c1 design request
- Author: Codex
- Version: v3
- Date: 2026-09-21
- Base: `cecf858dc985958a904b650a9e4a57e2782f587e`
- Status: revised design request for Claude review

This document supersedes the sacrificial-process lifecycle sections of C4c1
v2. The v2 exact readiness migration, public compatibility isolation,
single-source Warp ABI, caller-owned storage, deterministic reset, four fixed
shade entries, CPU canonical oracle, counter invariants, and activation gates
remain unchanged. If lifecycle wording conflicts, v3 controls.

The revision closes three safety gaps:

1. possible GPU submission is latched before entering every launch leaf;
2. every blocking GPU quiescence operation is armed by the supervisor before
   the worker enters it; and
3. the whole-graph guardian exists before any CUDA-capable import or setup
   operation and adopts setup resources incrementally.

No implementation, GPU preparation, result, or baseline outlet is authorized
by this document.

## 1. Safety facts are deliberately separate

The sacrificial worker records three different facts and never substitutes one
for another:

```text
possible_submission_latched
    A launch leaf is about to be entered. Work may reach the GPU even if the
    call raises and never returns normally.

confirmed_launch_return_count
    A launch leaf returned normally. This is diagnostic evidence only.

device_quiescence_proven
    A supervised target-device barrier completed successfully. Only this fact,
    together with complete ownership and zero external holds, can authorize
    ordinary release.
```

`recorded_launch_count` from v2 is renamed
`confirmed_launch_return_count`. It must not decide whether a failure is
pre-submit or post-submit.

## 2. Frozen sacrificial-worker state machine

The only legal successful progression is:

```text
PROCESS_STARTED_STDLIB_ONLY
-> GUARDIAN_CREATED
-> SETUP_EPOCH_OPEN
-> SETUP_OWNERS_INCREMENTALLY_ADOPTED
-> SETUP_BARRIER_WATCHDOG_ACKED
-> SETUP_QUIESCENCE_PROVEN
-> GUARDIAN_SEALED
-> POSSIBLE_SUBMISSION_LATCHED
-> RECORDED_LAUNCHES_ATTEMPTED
-> VALIDATION_DRAIN_WATCHDOG_ACKED
-> VALIDATION_DEVICE_QUIESCENCE_PROVEN
-> RESULT_INSPECTED
-> RESOURCE_RELEASE_AUTHORIZED
-> PROCESS_EXITED
```

The following orderings are invalid:

- importing Warp or another CUDA-capable module before `GUARDIAN_CREATED`;
- sealing the guardian before setup quiescence;
- entering a recorded launch before the possible-submission latch commits;
- entering any blocking wait before the supervisor acknowledges its deadline;
- inspecting device results before validation quiescence; or
- releasing any covered owner before release authorization.

The state authority is monotonic. No transition clears the possible-submission
latch, dirty-device set, ownership-unknown flag, poison flag, or quarantine
requirement.

## 3. Guardian creation and setup adoption

### 3.1 Creation precedes every CUDA-capable import

The sacrificial entry script starts with Python stdlib and the already
allowlisted CPU-only lifecycle contract. It constructs an empty
`SacrificialRenderGuardian` before it imports:

- `warp`;
- `optics` modules that may import Warp;
- prepared-input upload/build helpers;
- BVH or render modules; or
- any module whose import may initialize, compile, allocate, or touch a CUDA
  runtime.

The entry script has a source/AST/import-ledger test proving this order. A
CUDA-capable module observed before guardian creation is a protocol violation;
the child exits before further setup and cannot claim clean cancellation.

Guardian creation is pure CPU work. Determining the declared target ordinal is
also pure metadata and cannot initialize CUDA or query an ambient device.

### 3.2 Register-before-next-GPU-leaf adoption

Setup runs inside the approved C4c0 setup epoch. Before each CUDA-capable leaf,
the worker monotonically records its possible dirty-device scope. After a leaf
returns an owner, that owner is adopted by the guardian before any subsequent
CUDA-capable leaf may run.

The adopted graph includes, as soon as each object exists:

- imported runtime/module owners whose teardown may release CUDA state;
- indexed device and stream owners;
- partially and fully constructed prepared-input allocations;
- upload readiness events and stable views;
- scene, transform, AABB, BVH, scratch, output, and diagnostic storage;
- compiled Warp module/kernel owners;
- objects created by `record_cmd=True`, including every raw launch owner;
- validation-drain event/state; and
- any host storage whose destruction can affect registered CUDA memory.

An owner-producing helper must obey one of two contracts:

1. it returns through a scoped builder that transfers every created owner to
   the guardian before another GPU leaf; or
2. on an exception before transfer, it marks `ownership_scope_unknown` and
   transfers all reachable partial-owner capsules it does possess.

If a helper may create an allocation that cannot be returned or enumerated
when it raises, that helper is not cleanly recoverable. The guardian records
unknown ownership and the worker is process-terminal quarantine only. A later
device sync cannot upgrade this case to ordinary per-object release.

Python module globals and local variables are not accepted as an ownership
protocol. Strong guardian adoption or whole-process quarantine is required.

### 3.3 Setup quiescence and seal

After upload, compilation, warmup, and command recording finish, setup remains
open until an explicit target-device quiescence barrier completes. That
barrier uses the supervised blocking-operation protocol in section 5.

Ordinary setup success requires:

- possible dirty devices are exactly the declared target ordinal;
- `ownership_scope_unknown` is false;
- all returned owners were adopted before the next GPU leaf;
- the setup barrier completed after supervisor ACK; and
- setup evidence and the guardian resource manifest canonicalize successfully.

Only then is `SETUP_QUIESCENCE_PROVEN` recorded and the guardian sealed.
Sealing freezes the owner identities, storage intervals, command identities,
device/stream identity, readiness authority, and setup evidence. A sealed
guardian cannot add, remove, replace, resize, or individually release owners.

Any setup/import/compile/upload/recording failure before the proof is handled
by the existing C4c0 rules:

- an exact target-only dirty scope with complete known ownership may attempt a
  supervised failure-only emergency sync, provided a supervised setup barrier
  has not already been entered and failed;
- foreign device, unknown device, or unknown ownership goes directly to
  whole-process quarantine; and
- a setup barrier that raises or times out goes directly to process-terminal
  quarantine under its already armed role-`S` watchdog; and
- no failure path performs ordinary Python teardown before its terminal
  disposition.

## 4. Pre-call possible-submission latch

Immediately before every recorded `wp.Launch.launch()` or other fixed
executable leaf, the worker commits the following CPU-only transition:

```text
possible_submission_latched = true
attempted_launch_count += 1
attempted_command_position = fixed_command_position
```

The transition is immutable and must complete before control enters the launch
call. The launch is forbidden if the transition receipt cannot be recorded.

Only after `Launch.launch()` returns normally may the worker increment:

```text
confirmed_launch_return_count += 1
```

Therefore all of these are post-submit poison cases:

- the launch enqueues work and then raises;
- the launch raises with unknown submission status;
- a Python wrapper raises after the native leaf is entered;
- the process loses IPC while the launch call is active;
- a later recorded command fails; and
- the validation drain fails or hangs.

The worker never tries to prove that a raising launch did not enqueue. Once a
launch leaf has been entered, clean cancellation is permanently unavailable.

For diagnostic evidence:

```text
0 <= confirmed_launch_return_count <= attempted_launch_count
possible_submission_latched == (attempted_launch_count > 0)
```

These relations are replay-validated. Only
`possible_submission_latched`/`attempted_launch_count` control safety.

## 5. Supervisor-armed blocking operations

### 5.1 No unsupervised blocking CUDA wait

The sacrificial child must not enter any potentially blocking CUDA/Warp wait,
event synchronization, stream synchronization, or device synchronization
unless the supervisor has first committed a deadline and returned the exact
one-time ACK.

This rule covers all three blocking phases:

```text
S = normal setup-quiescence barrier
V = normal validation drain
E = failure-only emergency synchronization
```

It is not limited to exception handling. In particular, a normal validation
drain is never called directly.

### 5.2 Fixed handshake

The test supervisor protocol is extended with a fixed guarded-operation
handshake. It accepts only the three enumerated operation roles above and a
monotonic operation sequence number:

1. worker sends `ARM(role, sequence)`;
2. supervisor validates the role/sequence, commits the deadline in its own
   state, and only then replies with the one-time ACK;
3. worker validates the exact ACK and enters the fixed blocking callable;
4. if the callable returns, the worker records its local completion;
5. for setup barrier `S`, worker sends `COMPLETE(sequence)`, supervisor removes
   the deadline and ACKs the disarm before setup may continue; if the setup
   barrier raises instead, the child is immediately process-terminal and does
   not attempt a second blocking synchronization;
6. validation drain `V` and emergency sync `E` remain armed through the
   one-shot child's terminal response/exit, so a hang during final inspection,
   serialization, or teardown cannot create a reusable session.

An ACK is bound to the child session, role, and sequence and cannot be replayed
for another wait. EOF, timeout, wrong ACK, duplicate sequence, supervisor
state drift, or a blocked supervisor causes quarantine without entering the
blocking callable.

The supervisor deadline is based on its monotonic clock. The worker cannot
extend, cancel, or replace it. A response-write timeout remains unrelated and
is not used as a GPU execution timeout.

### 5.3 Timeout dispositions

If an armed `S`, `V`, or `E` operation does not finish by the supervisor
deadline, the supervisor terminates the disposable child. The child is not
allowed to run ordinary finalizers or report successful release.

For `V`, even though every recorded launch returned successfully, a raised or
hung drain is a post-submit quarantine. The already armed role-`V` watchdog
remains the terminal authority; the worker does not try a second emergency
sync with a different role. No result inspection, owner release, or normal
response occurs.

For `S`, timeout occurs before the guardian seal. The partially constructed
resource graph stays process-owned until termination.

## 6. Success and failure dispositions

### 6.1 Successful validation

After all fixed recorded commands return, the worker arms role `V` and waits
for the supervisor ACK before calling the validation drain. A successful drain
records `VALIDATION_DEVICE_QUIESCENCE_PROVEN`.

Only then may the worker:

- inspect/copy outputs;
- compare against the independent CPU oracle;
- serialize the fixed test result;
- authorize release of the sealed guardian graph; and
- exit the one-shot process.

The role-`V` watchdog remains active until process exit.

### 6.2 Failure before a recorded launch

A failure is cleanly pre-submit only when all are true:

- setup quiescence was proven;
- guardian is sealed;
- `possible_submission_latched` is false;
- `attempted_launch_count == 0`; and
- no other CUDA-capable leaf is active.

Because setup is already quiescent, the sealed graph may be cancelled and
released without another device synchronization.

### 6.3 Failure after the latch

Once the latch is set, the graph is poisoned. The only legal outcomes are:

- supervised role-`E` emergency sync succeeds, after which the disposable
  process exits under the still-armed watchdog; or
- watchdog arming fails, device/ownership scope is foreign or unknown, sync
  raises, or sync hangs, so the full graph remains quarantined and the
  supervisor terminates the process.

No caught exception may unwind into a public executor or ordinary pytest
process. A poison marker without guardian retention and a terminal disposition
is invalid evidence.

Role `E` applies to command/IPC failures that occur before validation drain is
entered. Once role `V` has been ACKed and its drain call begins, any exception
or timeout goes directly to process-terminal quarantine under the existing
role-`V` deadline; it is never followed by another blocking sync attempt.

## 7. Required lifecycle tests

The v2 failure matrix is strengthened as follows.

### 7.1 Setup/guardian tests

- source/import-ledger proof that guardian creation precedes the first Warp or
  CUDA-capable import;
- failure during Warp import/init;
- failure after the first prepared-input upload but before readiness;
- failure during compile/warmup;
- failure during `record_cmd=True` before the raw launch owner is returned;
- a helper that creates an unenumerable partial owner and raises, proving
  quarantine-only disposition; and
- setup barrier hang with the guardian unsealed and all known owners retained.

Each test asserts that no covered owner finalizer runs before successful
quiescence authorization or supervisor-confirmed process death.

### 7.2 Launch-latch tests

For every fixed command position:

- raise before the latch: no launch callable invoked and clean cancellation is
  allowed after setup quiescence;
- raise after latch but before invoking the mocked native body: post-submit
  poison is mandatory;
- native launch submits and then raises: `attempted_launch_count` advances,
  `confirmed_launch_return_count` does not, and post-submit poison is
  mandatory;
- launch returns then the next command fails: prior work remains covered; and
- IPC closes while the launch leaf is active: process-terminal handling is
  mandatory.

Tests must not use successful return count as the safety oracle.

### 7.3 Blocking-operation watchdog tests

- supervisor holds the channel but never ACKs `S`, `V`, and `E`; the blocking
  callable is never entered;
- setup barrier hangs after ACK; supervisor kills the child;
- all recorded commands succeed, then the normal validation drain hangs after
  ACK; supervisor kills the child, no result is inspected, and no finalizer
  marker fires;
- validation drain raises after ACK; graph is poisoned and no result/release
  occurs;
- emergency sync hangs or raises; quarantine/process termination occurs; and
- stale, duplicated, wrong-role, or wrong-sequence ACKs cannot authorize a
  blocking call.

The real supervisor—not a mocked ordinal or in-process timer—must cover the
hung normal-drain case.

## 8. Revised implementation slices

The v2 slices remain, with the following lifecycle ordering inside C4c1c:

1. implement and test the pure-CPU guardian/state evidence;
2. extend the fixed sacrificial supervisor protocol for roles `S`, `V`, and
   `E` with ACK-before-call semantics;
3. enforce guardian-before-import and incremental setup adoption;
4. prove supervised setup quiescence and seal;
5. implement the pre-call possible-submission latch;
6. execute the recorded bundle only in the disposable child;
7. perform the supervised normal validation drain;
8. inspect results and run the independent CPU oracle only after quiescence;
   and
9. run the full setup/launch/drain fault matrix before C4c1 review.

The product closed worker remains GPU-disabled. These test-only lifecycle
mechanisms do not create C4 slot/generation/lease authority and cannot satisfy
C4c2/C4d acceptance by themselves.

## 9. Additional acceptance conditions

C4c1 cannot be approved unless:

- [ ] the guardian exists before every CUDA-capable import/init/setup leaf;
- [ ] every setup owner is adopted before the next GPU leaf or ownership is
      marked unknown and quarantined;
- [ ] setup quiescence is supervised and proven before guardian seal;
- [ ] possible submission is irreversibly latched before every launch call;
- [ ] a submit-then-raise leaf is treated as post-submit poison;
- [ ] successful launch-return counts are diagnostic only;
- [ ] normal validation drain obtains a supervisor deadline ACK before entry;
- [ ] an all-launches-successful but permanently hung drain is killed by the
      real supervisor;
- [ ] no output is inspected or owner released before validation quiescence;
- [ ] all partial-submit paths retain the entire sealed graph until supervised
      terminal disposition; and
- [ ] v2 readiness, compatibility isolation, CPU oracle, counter invariants,
      and all activation gates remain unchanged.

## 10. Review decision requested

Please review whether v3 now freezes one conservative lifetime rule from the
first CUDA-capable import through final validation exit, including the
ambiguous case where a launch submits work and then raises and the nominal case
where a successful command sequence hangs forever in its normal drain.

Approval still authorizes only C4c1 implementation. C4c2–C4c4, C4d/B3,
product worker GPU preparation, a3b, results, and baselines remain closed.
