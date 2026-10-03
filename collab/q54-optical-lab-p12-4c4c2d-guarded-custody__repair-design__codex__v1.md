# C2d1: terminal-failure admission and cross-history repair

Initiative: q54-optical-lab-p12-4c4c2d-guarded-custody
Stage: repair-design
Author: codex
Version: v1
Date: 2026-10-03
Status: in_review
Related Files: guarded-custody design-request v2; code-review v1;
optical_worker/guarded_lifecycle.py; optical_worker/guarded_evidence.py
Owner Summary: Repair the two reproduced failures without new worker roles,
GPU admission, sync retries or product gates. This narrow supplement specifies
the missing terminal-failure event before implementation.

Base and code counterexample revision: `4ee33ab7a09720f0dc08f3e86bb199af59e58e40`.
The full independent code-review v1 is archived unchanged alongside this file.
The earlier implementation is not approved. The rest of accepted custody v2
remains authoritative; this document tightens its sections 4 and 5.3/5.4.

## 1. Terminal fault observation and side-effect admission

Add one event to the existing not-yet-approved lifecycle-v3 grammar:
`terminal_operation_failed`, with exact details `{failure_kind}` using the
existing enum. It is allowed at most once, only while controller TERMINATING,
after terminal custody retention, before terminal intent. It preserves current
guard/sequence/purpose and guardian state; a sticky replay fact forbids later
sync entry, release authorization, cleanup entry, or successful sync/cleanup
outcome. Add replay-derived exact bool `terminal_operation_failed` to summary.
This is a schema delta within this unpublished implementation, not an upgrade
of C0 evidence or a compatibility fallback. Existing hand-authored v3 fixtures
must explicitly carry false unless the event occurs.

Record this event at actual cross-thread/reentrant error observation while
terminating. Retain the actual exception and full reachable controller graph.
The event must not falsely claim a pending handshake, already-started sync or
cleanup raised. The original watchdog remains active; no new ARM/COMPLETE.

Failure recording and admission use the controller's one RLock. Under that
lock, check sticky failure and exact ticket, then commit each side-effect's
existing before-call marker. Do not hold the lock across network ACK waits,
sync calls or reference release/destructors. The marker is the conservative
linearization point: a later failure means the admitted call may be in flight;
it cannot erase the marker. A failure committed before that point prohibits
the call. Native calls may enqueue before raising, so outcome counters never
replace before-call admission. No claim of atomicity with GPU execution.

Specifically:

- ARM may receive its real ACK after a fault: retain truthful `guard_armed`,
  but do not enter sync or authorize release. Transfer full graph to quarantine.
- Sync admission commits `emergency_sync_started` while locked; invoke outside.
  If a fault arrives during/after that call, record `emergency_sync_failed`,
  not succeeded, and quarantine. There is never a second sync.
- Release admission checks failure, then authorizes the guardian, records
  `resource_release_authorized` and `ordinary_cleanup_started` under the same
  lock, before calling release outside it. A later failure yields cleanup_failed
  and remaining custody, never completed/safe ordinary exit. If failure precedes
  authorization, full quarantine is mandatory; no fake cleanup_failed.
- Success-outcome recording and the check of sticky failure share the lock.
  A fault after a successful sync but before release still prevents release.
- Terminal intent is committed under the same lock. A fault before that commit
  invalidates a still-pending successful ending; if cleanup has already released
  refs, record cleanup_failed and remaining custody, never claim the original
  full graph. Calls after terminal intent only reject, with no second effect.

The reducer permits a matching failed outcome of an operation genuinely
admitted before the fault; it forbids successful outcomes or new side effects
after it. `quarantined_process_exit` includes this terminal failure with no
release; cleanup_failed_exit still requires real prior authorization.

## 2. Ordered cross-history compatibility

Continue validating worker and supervisor individually first. Construct ordered
per-sequence attempt identities, ACKs, and COMPLETE attempts from their actual
traces. Sequence identity includes role (session/target are root configuration).

For every worker ARM attempt, if the supervisor has committed that sequence,
its role must match, including when no worker ACK was received. Every worker
ACK must exist in supervisor ACK history. A worker normal_disarmed must match
supervisor COMPLETE and DISARMED. A worker pending attempt may have no received
ARM, or matching committed ARM/ACK; lack of ACK alone proves neither outcome.

For a terminal worker record, supervisor committed ARM identities must be a
subset of worker attempted ARM identities, and received COMPLETE identities
must be a subset of worker normal_complete_started identities. Thus failed
COMPLETE ACK may coexist with a real supervisor completion, but failure before
COMPLETE cannot. No additional guard can follow a definitive terminal record.

For a prefix, supervisor may legitimately be ahead. Known identities still
must match, and earlier worker-observed ACK/completion facts must be present;
unobserved later work is not invented in the worker record. If prefix already
contains sticky normal/terminal failure or termination_started, it has crossed
the irreversible no-new-normal-work boundary: enforce the same subset bounds
except that an as-yet-unobserved single terminal emergency attempt is allowed
only after a healthy-IDLE termination prefix with no failure/compromised scope.
If this last permitted attempt appears, it must be exactly the next sequence
and emergency role. It cannot contain COMPLETE. Prefix truncation never permits
new work after known failure. Absent worker evidence remains allowed, with no
invented tail or local authority.

This is consistency of two observed histories, not cryptographic authentication
against a fully rewritten consistent history. No coordinator-generated hash
or parsed record becomes an execution capability.

## 3. Proof and unchanged scope

Use the review's real socket/core/transport delayed-ACK repro. Parameterize
cpu_cancel/target_sync and wrong-thread/context reentry: fault before admission
must mean zero sync and release calls, full owner retention/quarantine, truthful
replay, no COMPLETE/new ARM and rejected repeated termination. Also retain
existing in-sync/destructor failure tests and add fault-at-success-boundary
cases; verify remaining versus full custody based on actual release entry.

Reject both review F2 examples. Add negative role, forbidden COMPLETE and extra
ARM cases for terminal and irreversible-failure prefixes; positive pending/no
ARM, matching unreceived ACK, failed COMPLETE ACK, absent worker, lagging healthy
prefix and permitted terminal-prefix continuation. Run existing real process
artifact cases unchanged, source catalog/policy and C0 regression. Every round
records actual exit/reap plus residual/GPU checks; CPU mocks prove no GPU safety.

No changes to C0 lifecycle/native_setup, worker protocols, launchers, source
bootstrap admission, collab workflow, arena authority, product GPU preparation,
a3b/result/baseline or later C4/B3 gates. Fresh independent design review precedes
this event-grammar change; implementation then receives fresh code review.
