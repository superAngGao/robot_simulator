# C4c1c CPU lifecycle and disposable-supervisor checkpoint

Initiative: q54-optical-lab-p12-4c4c1c-supervision
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-09-28
Status: in_review
Related Files: C4c1 ABI design v3; tests/support/sacrificial_lifecycle.py; tests/support/sacrificial_watchdog.py; tools/optical_sacrificial_test_supervisor.py
Owner Summary: Implements only v3's CPU lifetime authority and separate test-only S/V/E supervisor. No recorded GPU execution or product activation.

## Exact boundary

Base c8c675a records independent approval of the recording/preflight candidate
266fc821. This checkpoint implements C4c1c steps 1–2, not the complete C4c1c
driver. All newly executable subprocess scenarios are fixed CPU probes. No
Warp import, upload, kernel launch, native completion event or benchmark result
is reachable through this test supervisor. Its finite CPU fixture scenarios
are not a generic script/callback/plugin runner.

The production C4c0 worker, allowlist, A/K supervisor, manifest, protocols and
activation gate are unchanged. Test-only modules are not imported by product
code. C4c1c's later GPU driver must separately prove guardian-before-runtime,
source/import coverage, incremental ownership of actual allocations and raw
recorded commands, independent CPU numeric oracle, real partial-submit faults,
and complete storage provenance. This CPU checkpoint cannot stand in for any
of those tests, real device scope or C4 arena lease authority.

## Behavior

- The CPU lifetime state authority composes the production whole-resource-graph
  guardian and CPU device-scope descriptor. It does not duplicate CUDA cleanup
  or change C4c0's existing setup ordering. Here supervised quiescence precedes
  guardian seal, as required by C4c1 v3.
- Each setup leaf has an identity-consumed permit. Another leaf cannot begin
  until returned owners transfer. Owner-producing exceptions conservatively
  mark unknown ownership; foreign/unknown device scopes can only quarantine.
- Launch attempt/latch advances before the driver may enter the command. A
  normal return advances only a separate diagnostic count. Fixed order and
  duplicate/reentrant attempts are rejected.
- The S/V/E channel carries only a session, role, monotonic integer sequence
  and fixed operation. Canonical length-bounded frames reject wrong/replayed
  ACKs. The supervisor commits its monotonic deadline before replying. Only S
  can disarm, and setup cannot progress until the disarm ACK arrives. V/E stay
  armed through inspection, release and one-shot exit.
- ACK failure never enters the wait. S/V/E wait failure quarantines the entire
  retained graph without a second sync. The disposable child uses hard exit,
  not normal exception unwinding. Real subprocess tests verify no owner
  finalizer on ACK rejection, raised waits or watchdog termination.
- A separate test supervisor watches the parent-liveness descriptor and a
  bounded nonblocking protocol parser. Parent loss/protocol failure kills the
  disposable child; no restart is possible. Harness timeouts are safety test
  deadlines, not performance gates or product budgets.

## Evidence and limitations

Initial focused CPU/protocol execution on 101 robot_sim: 46 passed. Additional
scope/sequence/retention tests subsequently ran together with the existing C0
unit protocol suite: 64 passed; seven clean-worker integration tests rejected
the intentionally dirty scratch worktree at manifest construction. This is
not a product-code failure or a passing integration claim. Those tests must
be repeated on the frozen clean candidate before approval. One further
CPU-denied-import test was added before freezing. The clean e271ed7 candidate
subsequently passed all four files: **72 passed in 10.70s**, including the
seven previously environment-blocked integration tests. A line-wrap-only
follow-up fixes the final import-test Ruff E501 before independent review.

No arbitrary evidence payload is accepted as lifecycle authority. Snapshots
are detached diagnostic observations, not result-capable serialized receipts.
The eventual GPU driver must bind actual resources and existing render bundle
preflight; this checkpoint does not assert physical storage freeze from the
CPU guardian alone. The sole CPU fixture exposes no external host payload
holds. Host leases and joint C0/C4 shutdown remain later-stage obligations.

No C4c2–C4c4, C4d/B3, host payload, product preparation, shipping, a3b, result
or baseline gate changes. Final cleanup keeps this bounded test mechanism only
while its selected backend needs these regression tests; no product comparator
or migration fallback is introduced.
