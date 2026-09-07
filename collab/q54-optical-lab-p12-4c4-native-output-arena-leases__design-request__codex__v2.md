# Q54 Optical Lab P12.4c4 Native Output Arena and Leases (Codex v2)

Author: Codex

Date: 2026-09-07

Status: P1 external-host-hold lifecycle revision for Claude re-review; no C4
implementation, C4/B3 activation, a3b, benchmark sample, or baseline is
authorized

Base: `7fda91495a599cca0a292051404b729fe6a4ee90`

This document is the authoritative revision of the poison/shutdown portions of
v1. All unaffected v1 ownership, arena, prepared-plan/measured-acquire, Warp
native render, transaction, provenance, fairness, scope, and activation rules
remain unchanged. Where v1 says a successful emergency device synchronization
alone authorizes owner release, this v2 replaces that rule.

## P1 finding and corrected invariant

An optical arena may contain two independently in-flight ownership domains:

1. CUDA/Warp commands reading or writing device and pinned-host storage;
2. an external CPU consumer still reading a payload from pinned-host storage.

Target-device synchronization proves only the first domain is quiescent. It
cannot prove that an external CPU consumer has released its payload lease.
Therefore resource disposal requires the conjunction:

```text
resource_release_authorized
  = device_quiesced
  AND external_hold_count == 0
  AND quarantine_transfer_not_required
```

No individual arena, slot, pinned tensor, event, stream, module, or session
owner may be released before that predicate is true.

## External hold contract

Every payload exposed to code outside the lane is represented by an arena-owned
external hold record. The corresponding `HostPayloadLease` strongly retains:

- the arena/session retention owner;
- the exact slot and generation;
- every pinned tensor and host view accessible through the lease;
- its immutable payload/channel manifest;
- one single-use external-hold authority.

Raw host arrays are not independently returned without that lease owner. A
consumer may read the payload while the lease is active and must explicitly
call `release()` or leave its context manager normally. Release is
linearizable, single-use, and rejects fabricated, stale, wrong-generation, or
duplicate authorities.

Arena poison changes what release means but does not disable it:

- no new payload lease or external hold may be created after poison;
- ordinary slot release, reclaim, reuse, generation advance, and transition to
  `FREE` remain forbidden;
- an already-active payload lease may perform exactly one
  **external hold discharge**;
- discharge removes only that external CPU hold. It does not assert GPU
  completion, validate a failed payload, reclaim a slot, or recover the arena;
- the poisoned slot remains retained and never returns to `FREE`, even when its
  last external hold is discharged.

The same rule applies to any future non-device external hold. Such a hold must
be registered before exposure and must participate in the exact count; an
unregistered raw pointer/view escape is a lifecycle contract violation and a
hard failure.

## Revised arena state machine

The normal state machine remains:

```text
OPEN -> DRAINING -> CLOSED
```

The poison/abort state machine is now:

```text
OPEN/DRAINING --unsafe submitted failure--> POISONED

POISONED --abort_poisoned()--> ABORTING_DEVICE

ABORTING_DEVICE --device sync fails-->
    QUARANTINED_UNTIL_PROCESS_EXIT

ABORTING_DEVICE --device sync succeeds, external_hold_count > 0-->
    SYNCED_WAITING_EXTERNAL_HOLDS

ABORTING_DEVICE --device sync succeeds, external_hold_count == 0-->
    ABORTED_SYNCED

SYNCED_WAITING_EXTERNAL_HOLDS --one hold discharged, holds remain-->
    SYNCED_WAITING_EXTERNAL_HOLDS

SYNCED_WAITING_EXTERNAL_HOLDS --last hold discharged-->
    ABORTED_SYNCED
```

`SYNCED_WAITING_EXTERNAL_HOLDS` is not a cleanup authorization and not a
recoverable arena state. The device timeline is known quiescent, but the full
resource graph remains strongly retained and all normal arena operations stay
disabled. Only these operations are legal:

- discharge an already-registered external hold;
- inspect immutable shutdown/lifecycle evidence;
- repeat `abort_poisoned()` to obtain the current disposition without another
  device synchronization;
- transfer the complete graph to process-lifetime quarantine during owner or
  process teardown.

`ABORTED_SYNCED` means both device safety and external ownership safety have
been established. Only this transition may create a disposal authorization.
It never changes a poisoned slot to `FREE`, never permits reuse, and never
creates a valid completion/payload receipt for the failed generation.

## Atomicity and races

Abort and external discharge share one arena lifecycle lock/state authority.
The implementation must be linearizable:

1. poison closes creation of new holds and snapshots the registered hold set;
2. `abort_poisoned()` moves to `ABORTING_DEVICE` exactly once;
3. external release may race with device synchronization, but it can only mark
   its existing hold discharged;
4. after synchronization returns, abort reacquires the lifecycle authority and
   evaluates the current—not the earlier snapshotted—external hold count;
5. owner release is authorized exactly once only if synchronization succeeded
   and the current count is zero;
6. if the last hold is discharged after synchronization, that discharge
   performs the same guarded final predicate check and transitions to
   `ABORTED_SYNCED` exactly once.

No resource is freed between the synchronization result and the guarded final
predicate check. Re-entrant discharge/finalization fails closed. Repeated
`abort_poisoned()` never synchronizes twice and returns the stable current
disposition.

## Revised slot behavior after arena poison

When any submitted generation poisons the arena, every non-free generation and
every slot reachable from a registered external hold becomes
`RETAINED_POISONED`. Each record preserves:

- original slot ID and generation;
- terminal/payload state reached before poison;
- exact active device and external holds;
- completion/event/storage owners;
- lifecycle transition history.

An external hold discharge updates only the hold set and transition history.
The slot remains `RETAINED_POISONED`; it cannot become `RECLAIMABLE` or `FREE`,
and its event cannot be re-recorded. Thus a successful emergency sync plus a
late host release authorizes destruction of the failed arena, not reuse of any
of its generations.

## abort_poisoned() dispositions

The public shutdown result is expanded from two terminal labels to the
following exact dispositions:

```text
aborted_synced_release_authorized
synced_waiting_external_holds
quarantined_until_process_exit
```

`synced_waiting_external_holds` is explicitly non-terminal. It reports:

- emergency synchronization succeeded;
- device work is quiescent;
- one or more external holds remain;
- resource release is not authorized;
- ordinary close/reclaim/reuse remains forbidden.

The call does not wait indefinitely for a CPU consumer. When the last consumer
releases normally, the arena may transition to
`aborted_synced_release_authorized`. If the session must tear down first, the
entire graph transfers to `quarantined_until_process_exit` instead.

If device synchronization fails, quarantine remains permanent regardless of
later external releases. Discharging a host hold after permanent quarantine is
allowed only to invalidate that consumer's lease; it never removes or releases
the append-only quarantined resource graph.

## Explicit close and finalizers

Normal `drain()` and `close()` still reject poison. They cannot substitute for
the abort protocol.

Explicit poisoned session shutdown performs:

1. `abort_poisoned()` once;
2. if release is authorized, dispose the complete arena graph once;
3. if waiting on external holds, retain the complete graph and return the
   non-terminal disposition without partial cleanup;
4. if device sync fails or shutdown cannot retain a live waiting owner, atomically
   transfer the complete graph to process-lifetime quarantine.

Arena/session `__del__` never synchronizes, never assumes external consumers
are gone, and never performs member-by-member cleanup. An unclosed poisoned
arena, including a device-synced arena with active external holds, is
transferred with its full resource graph to process-lifetime quarantine.

`HostPayloadLease.__del__` does not authorize arena cleanup. Explicit release
is required for normal post-sync finalization. If a lease is abandoned, the
retained graph remains safe until process exit rather than guessing that a raw
host view did not escape.

## C0 guardian and joint shutdown

C0 device safety and C4 external-host safety are separate proofs. In the later
shipping session:

- C0 emergency synchronization may establish `device_quiesced` for the shared
  device graph;
- it does not set C4's external hold count to zero and does not authorize C4
  arena/pinned-storage destruction;
- the C4 arena owner and active payload leases continue strongly retaining all
  covered storage after C0 releases a publication-only hold;
- every arena/session destructive method checks the C4 disposal authorization,
  not merely the C0 sync disposition;
- if either C0 requires process quarantine or C4 still has external holds when
  the joint session owner disappears, the complete engine + guardian + arena +
  delivery + payload graph transfers atomically to quarantine;
- joint cleanup is permitted only after device quiescence and all external
  holds are discharged.

The C0 guardian resource tuple therefore includes the C4 external-hold barrier
owner as well as the arena/session and physical allocations. Releasing the
guardian's own reference cannot bypass the barrier or call individual arena
cleanup.

## Evidence revision

The C4 arena lifecycle receipt adds exact transition-produced fields:

```text
external_hold_count_at_poison
external_hold_count_after_device_sync
external_hold_discharge_count
device_emergency_sync_attempt_count
device_emergency_sync_succeeded
device_quiesced
waiting_external_holds
external_holds_discharged
resource_release_authorized
poisoned_slot_count
retained_pinned_storage_count
shutdown_disposition
shutdown_transition_trace_sha256
```

Invariants:

- `resource_release_authorized` implies device sync succeeded, device is
  quiescent, external hold count is zero, and disposition is
  `aborted_synced_release_authorized`;
- `synced_waiting_external_holds` implies device sync succeeded, count is
  positive, pinned storage remains retained, and release authorization is
  false;
- `quarantined_until_process_exit` always has release authorization false;
- external hold discharge never records slot reclaim, `FREE`, valid payload
  completion, or arena recovery;
- lifecycle evidence is generated by the arena transitions, not populated by
  the lane or coordinator.

These fields join the existing one-way evidence DAG after active lease binding
and before the final lane receipt. They do not alter stage evidence or create a
hash cycle.

## Required P1 tests

The C4 implementation must add a depth-two GPU lifecycle test with this exact
sequence:

```text
slot 0 / generation A:
  render + D2H complete
  payload ready
  external host payload lease remains active

slot 1 / generation B:
  enqueue or terminal-event record failure
  arena becomes POISONED

abort_poisoned():
  injected/real target-device synchronization succeeds
  state becomes SYNCED_WAITING_EXTERNAL_HOLDS
```

The test directly proves:

- slot 0 pinned tensors, host views, payload lease, arena, and session owners
  remain strongly retained;
- the slot does not become free and neither generation can be reused;
- member close/free/resize and arena close remain rejected;
- the existing host consumer can still read its payload safely;
- no second device synchronization occurs;
- releasing slot 0's payload performs only external hold discharge;
- the last discharge transitions the arena to `ABORTED_SYNCED` and authorizes
  complete graph disposal exactly once.

Additional unit/GPU coverage must prove:

- two active payload leases require two discharges;
- duplicate/fabricated/wrong-generation discharge fails;
- release racing with emergency sync uses the post-sync current hold count;
- sync failure transfers all device and host resources to permanent quarantine,
  and later payload release cannot unquarantine them;
- arena/session `__del__` with a synced-but-held payload cannot free pinned
  storage;
- C0 sync success with an active C4 payload lease does not authorize joint
  cleanup;
- joint shutdown releases only after both the device and external-hold proofs
  are true.

## Revised acceptance request

Please approve or request changes on this P1 closure:

1. emergency device sync and external host hold discharge are independent,
   conjunctive release proofs;
2. poison forbids normal reclaim/reuse but preserves a single-use discharge
   path for already-active external holds;
3. `SYNCED_WAITING_EXTERNAL_HOLDS` retains the complete graph without blocking
   indefinitely or returning a false terminal disposition;
4. the last discharge may authorize destruction but never returns a poisoned
   slot to `FREE`;
5. finalizers and C0 joint shutdown obey the same barrier and quarantine the
   complete graph when safe release cannot be proven;
6. all other v1 decisions, especially result activation remaining closed until
   B3 and joint review, remain unchanged.
