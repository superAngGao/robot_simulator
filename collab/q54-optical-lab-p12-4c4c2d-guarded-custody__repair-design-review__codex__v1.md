# C2d1 terminal-failure repair — independent design review v1

Initiative: q54-optical-lab-p12-4c4c2d-guarded-custody
Stage: repair-design-review
Author: codex (fresh independent reviewer)
Version: v1
Date: 2026-10-03
Status: in_review
Related Files: guarded-custody repair-design codex v1; accepted design-request codex v2; runtime-integration design-request codex v3; guarded-custody code-review codex v1
Owner Summary: Request changes. The proposed delayed-ACK admission and terminal-history rules address the original counterexamples, but two boundary cases remain unspecified: already-compromised scope in a prefix, and terminal fault observation before custody retention. No existing implementation or broader gate is approved.

## Identity and verdict

Reviewer session: `/root/c4_c2d1_repair_design_review_v1`, independent design-review session, no delegation and no candidate edits.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Base: `4ee33ab7a09720f0dc08f3e86bb199af59e58e40`.
- Frozen HEAD: `4897fdc18d9df6658084c6c12018787be51ad6c6`.
- Candidate: `collab/q54-optical-lab-p12-4c4c2d-guarded-custody__repair-design__codex__v1.md`.
- Independently verified SHA-256: `5d063efc1511ce0023a9ba44b1d57b05a3fb9ef7a6c426f13b275b97261b5c7e`.
- Archived original code-review SHA-256 independently verified: `a91fdbab5a94d67f04cc08abaa0c6b9dff44ca2daca8b81b0e7fbf1b11db8a4e`.
- Working tree clean at initial and final inspection. Base-to-HEAD changes are only the archived review and this repair-design candidate.
- Verdict: **Request changes**, for RD1 and RD2 below. These are design findings, not a claim that a repair has been implemented and failed testing.

Read `AGENTS.md`, `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md` fully. Inspected the candidate, accepted custody v2, parent runtime-integration v3, complete archived code review and original external repro. Reviewed the live controller/reducer/client, supervisor observation paths, guardian freeze/transfer/release, inherited descriptor scope handling, actual native_setup._abort dispatch, and relevant handwritten fixtures/tests. Author assertions and old test counts were not approval evidence.

## Findings

### RD1 — P2, blocking: a compromised-scope prefix is still treated as freely lagging

Location: repair-design v1 lines 89–98, especially the boundary predicate at lines 92–95; accepted custody v2 section 5.2's descriptor/owner-derived scope rule.

Concrete case: a worker prefix ends immediately after `descriptor_observed` for foreign ordinal 0 with configured target 1, or for an unknown-scope descriptor. Its replay already says compromised scope, but controller_state remains RUNNING and guarded_segment_failed remains false. No `termination_started` or `guard_failed` event has occurred yet. This is a valid prefix under the accepted design: scope is deliberately derived from the descriptor before subsequent dirty/unknown notifications, so truncation cannot erase the declaration.

Pair it with an individually valid supervisor history: ARM/ACK setup0, COMPLETE/DISARMED setup0, ARM/ACK resolve1, observed exit74. The known sequence/role and ACK facts match. The proposed subset restriction is triggered only by sticky normal/terminal failure or termination_started, so this prefix does not trigger it. The extra COMPLETE and next normal ARM consequently remain allowed by the stated rules, even though known compromised scope already makes that continuation impossible for the fixed controller.

Independent bounded CPU reproduction constructed both foreign and unknown cases using the production replay, production combined validator, handwritten fixture prefixes, and actual supervisor core. Both component pairs validate in the unchanged implementation. More importantly for this design review, neither pair violates the new section 2 predicate as written. The added terminal-failure summary field would be false and does not resolve the issue.

Actual code supports the impossibility: inherited `NativeWorkerSetupLifecycle.begin_device_leaf` at lifecycle.py:535 records scope and calls `_raise_if_compromised()` before issuing a permit. The guarded context then fails/terminates rather than completing that segment. Accepted custody v2 explicitly requires conservative scope derivation even before notification events and permanent quarantine for foreign/unknown scope.

Consequence: the repair would still accept contradictory observed histories at a legitimate truncation boundary. This is not cryptographic authenticity or detection of wholly rewritten matching histories.

Why listed checks miss it: section 3 calls for terminal and irreversible-failure prefixes, but does not explicitly include descriptor/owner-declared compromised scope before the failure event. Existing negative fixtures primarily mutate completed worker stories.

Required direction: include replay-derived compromised scope in the irreversible prefix boundary, regardless of whether `guard_failed`, scope notification, or termination has yet been appended. Enforce the same attempted-ARM/started-COMPLETE subset bounds, with no emergency-continuation exception for compromised scope. Apply this to descriptor declarations, returned-owner declarations, and explicit unknown-scope facts, not only summary flags copied from input.

Required proof: negative cross-artifact tests at those prefix cuts for both foreign and unknown scope; reject extra COMPLETE, normal ARM, and emergency ARM. Preserve positive matching original ARM/ACK histories and valid healthy lagging/healthy-IDLE termination-prefix continuations. Scope may become known before its notification; test that exact early cut.

### RD2 — P2, blocking: terminal entry exposes a fault interval in which the new event is illegal

Location: repair-design v1 lines 24–25, 33–44 and 107–112; actual starting implementation guarded_lifecycle.py:134–143 and :338–343.

The new event is legal only after terminal custody retention, but must be recorded at actual cross-thread/reentrant error observation while terminating. The starting implementation separately commits `termination_started`, sets `_terminating`, retains the controller, and records `terminal_custody_retained`. Its `_record` lock protects an individual event, not this entire transition. A wrong-thread public call can therefore be observed after termination_started and before terminal_custody_retained. There is also an intermediate mismatch where replay says TERMINATING but `_terminating` is still false.

At that point, `guard_failed` is illegal because it is a RUNNING event, while the newly proposed `terminal_operation_failed` is illegal because custody has not yet been recorded. Merely putting failure recording and sync/release admission under the same RLock does not close this interval. The proposal explicitly freezes sync admission, release admission, success outcomes, and terminal intent as critical sections, but does not freeze terminal entry/state publication/custody as one critical section or provide another legal failure transition for that window.

Consequence: an implementer must invent a correctness-critical rule: defer/latch the fault without its prescribed event, emit an illegal event, or make terminal entry atomic. The first two can lose truthful failure evidence or leave subsequent ARM admission dependent on an unrepresented fault. The existing code already has the exposed event/flag interval; this is not a hypothetical new architecture.

This finding is a source-level interleaving/design counterexample, not a claimed execution of a future repair. The bounded original delayed-ACK repro starts later and therefore cannot settle it.

Required direction: explicitly define a single controller-lock critical section for terminal-entry admission, publication of TERMINATING/_terminating, strong custody installation and terminal_custody_retained, with no externally callable/blocking operation in that section; or specify an equally conservative, fully representable pre-custody fault path. Make the ordering relative to normal failure observation unambiguous. A fault observed after custody but before terminal_arm_started must also reject that new ARM and select full quarantine; no existing watchdog should be claimed if one has not yet been armed. The entry rule must not expand lock coverage across ACK waits, native sync or destructors.

Required proof: deterministic cross-thread entry-boundary tests, in addition to delayed-ACK cases, for both cpu_cancel and target_sync. Observe failures around termination_started/custody/terminal-ARM admission; assert no illegal replay event, no lost sticky fault, no new ARM/sync/release after a committed pre-admission fault, retained full graph, correct single terminal disposition, and unchanged repeated-termination rejection. Include the existing in-sync and destructor-reentry cases so entry protection does not accidentally hold the controller lock over blocking/destructive work.

## What the proposal does settle

- A standalone terminal-failure event is feasible within the unpublished v3 reducer and avoids fabricating handshake, sync, or cleanup failures when those operations never began.
- The real client already owns exact registered ticket identity. The proposed lock/check/marker order can use that capability without creating authority from parsed evidence.
- Delayed terminal ACK can be truthfully retained as guard_armed while the sticky terminal failure prevents subsequent sync/release. The controller need not mutate the thread-owned client from a foreign thread.
- Before-call admission outside a held lock across native calls is a clear conservative linearization choice. Late failure cannot erase a genuinely admitted call. Full versus remaining custody and outcome/terminal-intent separation are compatible with the inherited guardian.
- Requiring terminal supervisor ARM/COMPLETE identities to be subsets of actual worker attempts directly rejects original code-review F2. Role comparison even for an unacknowledged worker attempt closes its other counterexample. Failed COMPLETE-ACK uncertainty and absent-worker evidence remain legitimately permitted.
- There is no new wire role, sync retry, authority factory, C0 fallback or opened GPU/product gate. Existing watchdog supervision remains the mechanism after real ARM admission.

These conclusions support the direction, not approval of the gaps or of existing code. The parent author independently raised the same scope-prefix and entry-atomicity concerns during this session after this reviewer had identified them; findings are grounded in the inspected rules/code and the independent reproduction above.

## Verification, process hygiene, and limitations

Environment: local Windows PowerShell and `C:/Python314/python.exe`. No SSH, installation, runtime GPU work, implementation edit, shared-workflow edit, or gate change. Scratch files and this report are outside the candidate.

Executed:

1. `git rev-parse HEAD`, `git status --short/--porcelain`, `git diff --stat 4ee33ab7a09720f0dc08f3e86bb199af59e58e40 HEAD`, `git diff --check 4ee33ab7a09720f0dc08f3e86bb199af59e58e40 HEAD`, and SHA-256 checks. Exact frozen revision/digests; clean tree; diff check passed.
2. `C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/c2d1-review-runner.py`. Existing external runner imposes a 15-second child bound and exact-child kill/reap on timeout. Runner PID8096, parent17764; repro child4848, parent8096. Actual child exit0; command completed normally. Both original F1 variants and both original F2 examples reproduced. Exit0 here means successful bug reproduction, not correctness.
3. `C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/c2d1-repair-design-scope-repro.py`. New external self-bounded runner, 15-second child wait and exact-child kill/reap on timeout. Runner5624, parent17764; child14192, parent5624. Actual exit0. Both foreign and unknown descriptor prefixes reported `known_compromised=true`, `guarded_segment_failed=false`, controller RUNNING, and acceptance of extra supervisor COMPLETE/ARM by the unchanged validator. Summaries for the shortened prefix are produced by the actual reducer, not advertised as independently authored expected summaries or future-repair execution.
4. Exact `Get-Process` queries for PIDs8096,4848,5624,14192: all absent after completion. The original real-socket fixture joins its peer threads and closes both sockets. The new scope repro launches no descendants beyond its bounded child and creates no socket/GPU workload.
5. Local `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader`: no reviewed runner/child PID appeared. Output contained unrelated desktop applications (SearchApp, TextInputHost, TeamViewer, Omniverse Launcher, Feishu, EdgeWebView, Teams, and Code), which were untouched. This is scoped evidence, not an assertion of global GPU idleness.
6. Final HEAD, candidate hash, archived review hash, and clean-tree checks repeated successfully.

An initial read used the nonexistent root REVIEW_PRINCIPLES.md path; the complete actual `collab/REVIEW_PRINCIPLES.md` was then read before task inspection. This was a read-path error, not a validation failure or repository change.

No broad regression or GPU suite was run: this is a document-only design delta, the repair does not exist yet, and those suites cannot resolve the two missing design rules. All required real-process/source-policy/C0 and repair-specific regression evidence remains necessary for the later implementation review. CPU repros prove control/evidence behavior only, not CUDA/Warp synchronization or allocation safety.

## Exact scope

Request changes applies only to this CPU guarded-custody repair-design supplement. Revise and freeze the contract, then obtain a fresh independent design review before implementation; implementation requires its own fresh code review. The original implementation remains unapproved.

Still closed: integrated C2d setup/source-frozen guarded bootstrap; real Warp/Torch upload and dual-GPU target/ambient/foreign/unknown tests; module/session custody before CUDA-capable import; final graph/barrier/warmup/readiness composition; arenas/C2e/C2f; product GPU preparation; a3b, result/baseline authority; R2/host/S, later C4 and B3; shipping/activation and C4 completion. No prior approval is extended or rewritten.
