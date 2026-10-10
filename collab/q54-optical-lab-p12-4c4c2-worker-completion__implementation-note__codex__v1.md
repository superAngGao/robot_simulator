Initiative: q54-optical-lab-p12-4c4c2-worker-completion
Stage: C2f integrated worker implementation, in progress
Author: Codex (implementation)
Version: 1
Date: 2026-10-10
Status: draft
Related Files: runtime-integration design-request/review codex v3; systematic-refactor gap-review v1; systematic-refactor code-review v7
Owner Summary: R1 preparation/final composition is independently approved and integrated into this development branch. The next deliverable is the complete worker request/token/submission/resolve/reclaim/close path. Current prework is not C2f approval or activation, and provides no speed result.

## Baseline and current state

Independent R1 approval binds fb49e7eba847d2168a0de81d791357c64c3f9e1c,
with native production/test revision13d17dd293f9c79028260b856e99e39d5f344da5.
Its unchanged report is systematic-refactor code-review v7, session
01a1248e-3bb3-7850-9565-295f071e3b77, SHA256
b8a3c1893edbd7a29e63476288c3c47cb35fe88a2e193e94e8e60117fea69ebd.
Evidence-only archive commit4ad9f0715be1429b85ac83a350a2e97cab03e257 records
the actual bounded verdict. It does not approve this branch's subsequent work.

The development worktree is c4_worker_completion, branch
implementation/c4-worker-completion. Merge296b8b0558deb5da3b8cd73a3e2536242a13a2a7
integrates that approved baseline with the following unreviewed C2f prework:

- Worker timeout handling retains the caught TimeoutError beyond its except
  block, so the existing fault journal and terminal disposition execute instead
  of raising UnboundLocalError. CPU tests reproduce idle-read and CLOSE_ACK
  write timeouts. The real Linux idle timeout keeps its parent alive and uses
  the frozen120000ms deadline: worker exit64, supervisor exit0 and original
  timeout/termination evidence. The first test attempt with1000ms was rejected
  by the manifest and is retained as a failed test, not a product failure.
- Warp now delegates begin_drain/close to NativeOpticalArena, as CUDA already
  does. The existing drain invalidates unacquired plans without burning slot
  generations and permits existing pending/resolved work to finish/reclaim.
  No new cancellation or slot state machine is introduced. CPU tests cover
  prepared-only, pending and resolved states, future/raw-submit rejection and
  close before/after reclaim. This is an interface prerequisite for the worker's
  CANCEL_ONLY semantics, not implementation of the IPC token lifecycle.
- The existing request/response validators accept an explicit expected v2
  protocol for the integrated session. Their default remains legacy v1, with
  unchanged old error text and request keys. Cross-version messages and unknown
  configured versions reject. This does not negotiate or activate v2 by changing
  a parent-supplied field; the v3 manifest/source-bound dispatcher remains to be
  implemented. An initial unit failure caught the normalized request retaining
  v1; it was corrected before commit330037f.

The merged catalog v17 has90 entries, SHA256
8b72add231f6f6cec8471c401d75a9943b04987368f3e280e659f3b0c4df4489.
The one catalog merge conflict was resolved from the actual staged Git LF source
bodies, preserving both the approved R1 repair and worker prework. All90 digests
and existing AST checks passed. Updating the protocol's existing exact source
pin did not add an AST exception. Catalog/source contracts will be frozen again
for the complete C2f review candidate.

## Verification recorded so far

- Timeout CPU protocol regressions:14 passed in1.68s.
- Linux protocol/worker run at a5cab09:24 passed,1 failed at manifest validation
  for the test's incorrect deadline. Corrected idle-timeout test atcd83fca:
  1 passed,10 deselected in122.01s; launched worker and parent confirmed absent.
  Raw evidence: /data2/shared/ang.gao/c4-worker-timeout-cd83fca-tmp.
- Warp drain targeted cases:3 passed,19 deselected in1.62s.
- Warp and common arena regression suites:63 passed in12.31s; Ruff passed.
- Protocol v1/v2 separation plus existing worker protocol tests:16 passed
  in1.68s after repair; Ruff passed. No native execution is claimed for these.

Scratch ledgers remain in c4_assets/worker_timeout_evidence_v1.json and
worker_drain_prework_evidence_v1.json. Tests were terminal before the current
implementation continued. No author GPU jobs are outstanding.

## Complete next packet

Follow runtime-integration design v3 sections4,7,8,9. Reuse the existing
ClosedWorkerSession parser/serial loop, captured source setup orchestration,
NativeGuardedSetupLifecycle, both native arenas and their guarded resolvers.
Implement one token owner; do not clone the slot or wait state machines.

Required remaining work includes versioned v3 manifest/state/evidence and v2
requests with the old C0 A/K mode separate; row/payload and exact response/trace
capacity rejection before CUDA-capable work; single PREPARE with actual canonical
setup; prebuilt one-use128-bit tokens; FIFO EXECUTE acquire/enqueue/finish with
sticky possible-submission state; pending issuance only after terminal record;
RESOLVE under the exact wait and DISARMED; RELEASE and authority retirement;
CANCEL_ONLY without renumbering plans; actual PostReclaimIdleEvidence over token,
arena, hold and guard state; ACK-before-cleanup and parent-observed/reaped exit.
Preserve the historical submission latch and the existing ordered failure
disposition, including no retry after failed normal guards.

The current worker still rejects product PREPARE and the guarded lifecycle still
rejects final sample submission. Those are incomplete implementation points,
not evidence of C2f completion. Validate the actual integrated worker for all six
R0 and two R1 rows in both lanes, original CPU oracle channels and tolerances,
real Go2, dual-GPU scope, all required setup/post-submit/transport/ACK/cleanup
failures, hangs and process closure. Only then freeze one coherent candidate for
a fresh independent review. Do not micro-accept the prework above.

Whole-C4 systematic refactor, C4c3/c4 and C4d/B3 remain mandatory. Product
activation, R2/S/host delivery, result/baseline/a3b, performance conclusions,
backend selection and shipping remain closed at their existing gates.
