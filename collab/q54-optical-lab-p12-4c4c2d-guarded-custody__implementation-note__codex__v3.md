# C2d1: enforce compromised scope and observe public operation failures

Initiative: q54-optical-lab-p12-4c4c2d-guarded-custody
Stage: implementation-note
Author: codex
Version: v3
Date: 2026-10-03
Status: in_review
Related Files: custody design v2; approved repair-design v2; code-review v2
Owner Summary: Repair F3/F4 within the approved CPU custody contract. Candidate
requires fresh independent review; no runtime or result admission is added.

Base: `90f50bb3ef67dd99182e4da0e5bc8479cd76e53b`. The complete independent v2
report is archived verbatim (SHA256
`6c555b14a883bb342df9bc1632d55638ba2794ef48b25ac163742f1c50346289`). Its
Request changes verdict remains unchanged. Prior candidate's actual broad run
is separately recorded in test-evidence v2, not claimed as new-candidate proof.

## Changes and scope

- Live normal-work admission checks replay-derived compromised device scope.
  Normal ARM, body success, COMPLETE and DISARMED reducer predicates reject
  it too. Early descriptor/owner facts still permit their truthful bookkeeping
  and conservative termination; the single inherited guardian adopts returned
  owners before rejecting their foreign/unknown scope.
- Closed legacy APIs observe owner/terminal/active-operation errors through
  the existing sticky fault/custody mechanism; they never call C0 admission.
  Other public setup/ownership/seal validation errors share observation so
  catching them cannot manufacture a healthy guard completion. Invalid
  termination entry and segment roles are covered as analogous rejection paths.
- No new failure boolean, schema version, callback, watchdog, release code,
  runtime import, IPC surface, activation or gate. Existing RLock ordering and
  no-lock-across-ACK/sync/destructor rules are retained. Post-intent rejection
  leaves the committed evidence unchanged. Healthy pre-guard validation remains
  distinct from a failed in-flight operation.
- Only the two changed production module digests are repinned; policy and
  inventory are unchanged. Packaging hashes use Git LF bytes, without changing
  production byte-exact verification.

Tests use actual controller/client/core/socket paths: explicit unknown scope,
caught foreign/unknown returned owners and descriptors; no COMPLETE/new ARM/
leaf after compromise; worker-trace rejection after early facts; all three
closed APIs during delayed terminal ACK, admitted sync and real cleanup; other
swallowed public validation errors. Sentinel ownership and failed/full versus
remaining custody are checked, not merely exception types.

## Author validation before remote candidate run

Local Python3.14, no Torch/Warp: lifecycle/evidence157 passed; expanded with
wait-core after analogous-entry tests279 passed; those three plus source-policy
444 passed in19.51s, actual exit0. Ruff check/format and git diff --check pass.
An earlier new-test run had three fixture failures: two reused a host sentinel
as the returned device owner (correctly triggering the earlier ownership-role
rejection), one omitted GuardProtocolError from the intentionally invalid
metadata rejection. Fixtures now use distinct owners and formal exception
classes, without weakening validator behavior. New tests verify retention of
both distinct owners. No matching local test Python process remains.

Primary183:40031 still timed out; authorized fallback101 connected. Exact clean
prior candidate90f50bb verified; original dirty checkout's four paths preserved.
101 GPU2 is H200 NVL; GPUs0/1 are A100. Pre-round scoped pytest/worker process
query was empty. Unrelated training GPU PID2386045 was observed and preserved.
No GPU tests or installation are authorized by this CPU slice. Broad real-
process/source/C0 tests will run in a new clean detached candidate worktree;
their results and actual exit/PID checks are not preclaimed here.

## Still closed

Integrated C2d guarded bootstrap/runtime-module custody and real Warp/Torch
multi-GPU setup-fault matrices; full graph/barrier/warmup/readiness; C2e/f arenas;
product GPU preparation; a3b/result/baseline; R2/host/S; later C4/B3 and shipping
activation remain closed. CPU fixtures are not GPU completion/release proof.
Cleanup/adoption ledger and shared collab rules are unchanged. C4 is incomplete.
