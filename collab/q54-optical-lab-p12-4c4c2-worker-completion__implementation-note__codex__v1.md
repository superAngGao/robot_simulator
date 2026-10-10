Initiative: q54-optical-lab-p12-4c4c2-worker-completion
Stage: C2f integrated worker implementation, in progress
Author: Codex (implementation)
Version: 1
Date: 2026-10-10
Status: draft
Related Files: runtime-integration design-request/review codex v3; systematic-refactor gap-review v1; systematic-refactor code-review v7
Owner Summary: R1 preparation/final composition is independently approved. The shared worker dispatcher now drives native arena submit/resolve/reclaim and guarded close in CPU integration tests. Full native PREPARE/transport/process acceptance and independent review are still incomplete. This is an implementation checkpoint, not C2f approval, activation or a speed result.

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

The merged pre-implementation catalog v17 had90 entries, SHA256
8b72add231f6f6cec8471c401d75a9943b04987368f3e280e659f3b0c4df4489.
The one catalog merge conflict was resolved from the actual staged Git LF source
bodies, preserving both the approved R1 repair and worker prework. All90 digests
and existing AST checks passed. Updating the protocol's existing exact source
pin did not add an AST exception. Catalog/source contracts will be frozen again
for the complete C2f review candidate.

## Current worker integration, not yet an acceptance candidate

Work continues from40679f0402a1ee5fe369ed6d478cf3ed586aefc0 under the approved
runtime-integration designv3. The v3 manifest and worker evidence have explicit
v2 request dispatch; old C0 mode remains separate. The captured source owns
the one worker/controller binding before native setup. Worker/row CPU definitions
load in the existing supervised definition-import segment, so PREPARE does not
attempt an unguarded late repository import.

The existing serial dispatcher now connects PREPARE to the existing source
setup methods; EXECUTE to acquire/enqueue/finish; RESOLVE to the exact guarded
terminal wait; RELEASE to device-result release and reclaim; CANCEL to the
common arena drain operation. Possible-submission facts become sticky before
the first native leaf. Normal close cross-checks actual arena counts, holds,
tokens, setup origin and idle guard, keeps the graph through ACK writing, then
uses the existing guardian release. Reparsed idle evidence cannot authorize
release. A cleanup failure retains remaining custody and requires hard exit.

The parsed resolution record binds the specific next resolve guard sequence;
a prior sample's completed wait cannot resolve a later sample. Idle evidence
also binds the actual preceding guarded trace. These descriptive records do not
replace the registered source, active arena object or actual wait ticket.

Integration exposed a real lane difference: CUDA burned a generation on
pre-enqueue failure, while Warp only poisoned its active wrapper. The existing
CUDA cancel/retain helpers now live in the common native arena module and both
lanes call them. No second slot cancellation or GPU cleanup algorithm was added.
Tests inject failures before the first enqueue and after a launch/event call;
the former frees the burned generation, the latter preserves possible submission
and selects supervised failure cleanup. Product renderer/kernel code is unchanged.

The new CPU fixture binds before real setup construction and uses the production
worker, graph, arena and guard implementations. It doubles native allocation,
launch and event calls, and manually publishes the prepared test batch after the
actual final graph/barrier. It therefore does NOT validate the full PREPARE
handler, cold native manifest/bootstrap launch, real transport or GPU numerics.
It checks FIFO/consumed tokens, cancellation while pending, ACK-before-release,
copied proof rejection, stale wait replay, failed wait/lost DISARMED, cleanup
failure, pre-leaf generation cancellation, submit-then-raise, every R2/S row's
early rejection, and oversized response rejection before token allocation.

Latest CPU regression command (Python3.14, no Warp installation):

```text
python -m pytest tests/unit/optics/test_integrated_worker_lifecycle.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/optics/test_guarded_lifecycle.py tests/unit/optics/test_warp_arena.py tests/unit/optics/test_native_optical_arena.py tests/unit/optics/test_cuda_arena_resolution.py -q --disable-warnings --maxfail=2
235 passed in67.31s
```

Earlier in this checkpoint, the protocol/guard/final-graph regression run passed
193 in190.75s; the intermediate integrated/common-arena run passed106 in56.71s.
These overlap and must not be added as distinct coverage. Initial integration
fixture failures were corrected: its generator scopes did not model reusable
Warp scope objects, and its cleanup fault tried to replace a read-only instance
method instead of the guardian class method. A collection error in the new
preflight tests was fixed by declaring their diagnostic fixture dependency.
Ruff passes the changed worker/protocol/guard and arena implementation plus
the new integration test. All90 staged LF source bodies pass the existing AST
policy with refreshed exact source pins; no new AST exception was introduced.

Current catalogv17 SHA256:
7535eb47f6c9a2fa746eddd3337a0da698b915d85aa5ee9b0bc030ce145ec643.
Policy SHA256:
08a940137df8d0cf82ccd287f1fbf1aa4ad8201d59d5bb25c7ff6c7f0f83a466.
These are author checkpoint identities, not independently reviewed releases.

Code checkpoint4c33b87d00b9133dbb59eb5647ab997b9b0d0593 records this WIP.
Post-checkpoint source-policy and actual captured-source-custody regressions:

```text
python -m pytest tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_source_custody.py -q --disable-warnings --maxfail=2
195 passed in137.76s
python -m pytest tests/unit/benchmarks/test_optical_closed_worker.py -q --disable-warnings --maxfail=1
2 passed,9 skipped in0.71s
```

The nine legacy process cases require Linux and remain unexecuted in this
Windows run; previous Linux evidence does not validate this new code revision.
The first broader source-policy/bootstrap/custody run had243 passed,9 skipped,
2 failed in141.12s: one test intentionally reads HEAD bytes while the source pin
was still an uncommitted change; the other still expected only the two prior
CPU definition modules. The new exact five-module expectation also verifies
guarded adoption and absence of numpy/torch/warp imports. The195-pass rerun
checks the committed pins and that updated cold-process test, including its
failure injections. The bootstrap cases passed in the initial broader run;
its two failures are retained here rather than reported as a green full run.

All owned pytest sessions ended. Sampled source-test PIDs41320 and37456 are
absent; a scoped process query found no remaining pytest for this checkpoint's
source/worker commands. Cold-process helpers assert subprocess termination and
collect child output before returning. No GPU job was launched in this checkpoint.

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

## Integrated native probe and intake gap, 2026-10-10

Candidate26542543d067d20c64670d375065535420ede203 added the native v3
manifest builder/validator and fixed client through the actual framed worker
loop. The remote clean candidate is
/data2/shared/ang.gao/robot_sim_worker_2654254. Raw probe evidence is
/data2/shared/ang.gao/c4-worker-26542543d0-probe. Two asymmetric samples in
Warp completed EXECUTE/RESOLVE/RELEASE and normal close (worker1984161,
supervisor1984157, exit0/reaped). CUDA reached extension loading but custody
checkpoint rejected missing native scope pairs (worker1989673,
supervisor1989666, exit74/reaped). No final-sample numerical or speed claim.

The fresh independent gap review is archived verbatim in
q54-optical-lab-p12-4c4c2-worker-intake__gap-review__codex__v1.md:
session01a124d3-a32b-7d43-b6bb-1262e8c90d69, Request changes,
SHA256 b78661032b29b7e84e03a59c5f6a5c5642bd155fc63c7f5265669730f1853c34.
The reviewer confirmed that production already retains created CUDA modules
before scope unwind; the absent facts are produced only by test wrappers.
Warp's successful lifecycle has zero native-scope rows and does not prove
physical scope. F1 requires a bounded design/source-policy amendment before
moving the observation mechanism into production. F2 remains full integrated
scope-coverage work. Neither finding is closed by the Warp smoke.

The current bounded error-path fix keeps the full termination record in the
separate journal and sends its digest/disposition in the native error reply.
The old C0 reply and depth8 codec bound are unchanged. The fixed supervisor
reads and verifies the actual terminal journal when hard exit prevents a
worker.json; the acceptance driver reports request-flow failure before trying
to read an arena that was never constructed. The original CUDA failure remains
preserved. These fixes do not close the intake gap or have independent code
approval yet.

Local Python3.14 worker/protocol regressions:60 passed,9 Linux-only skips in
53.89s. A separate targeted run reproduced the original nested terminal reply
and verified the compact replacement in both lanes:2 passed. Ruff on changed
production/test files and py_compile on both support drivers passed. One
initial pytest command used an incorrect optics path for benchmark tests and
collected no tests; the60-pass command used the correct benchmark paths.
All90 staged LF sources pass source-policy validation; updated catalog v17
SHA25692aefc3fde3ccdb48abb78b1aac768cc314527be74a28a8df77b12eee306c01c.
At2654254 the remote Linux legacy protocol/worker suite passed27 in126.11s;
that run precedes this error-path change. The reviewer verified campaign and
replay PIDs absent and preserved64 raw files with matching hashes. The CLI
review exited0; its original report and launch/exit records remain in c4_assets.

## Remaining integrated acceptance

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

The product launcher still rejects GPU PREPARE. Final submission now requires
the exact source-bound native worker and active serial EXECUTE; standalone
captured setup graphs still cannot submit. Before a review candidate, complete
the native v3 runtime-contract/manifest construction and validation, fixed native
driver through the SAME framed loop, pre-GPU bounded trace-capacity checks,
preallocated sample/ticket/trace work and fixed acquire work-count checks.
Current guarded recording still reconstructs descriptive trace containers; it
does not yet satisfy the approved no-per-sample-container requirement. Complete
request/response/arena/guard/parent-exit evidence binding and post-ACK bounded
process supervision. Validate the actual integrated worker for all six
R0 and two R1 rows in both lanes, original CPU oracle channels and tolerances,
real Go2, dual-GPU scope, all required setup/post-submit/transport/ACK/cleanup
failures, hangs and process closure. Only then freeze one coherent candidate for
a fresh independent review. Do not micro-accept the prework above.

Whole-C4 systematic refactor, C4c3/c4 and C4d/B3 remain mandatory. Product
activation, R2/S/host delivery, result/baseline/a3b, performance conclusions,
backend selection and shipping remain closed at their existing gates.
