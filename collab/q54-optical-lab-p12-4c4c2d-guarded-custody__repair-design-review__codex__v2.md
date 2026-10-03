# C2d1 terminal-failure repair — independent design review v2

Initiative: q54-optical-lab-p12-4c4c2d-guarded-custody
Stage: repair-design-review
Author: codex (fresh independent reviewer)
Version: v2
Date: 2026-10-03
Status: accepted
Related Files: guarded-custody repair-design codex v2; original custody design-request codex v2; runtime-integration design-request codex v3; guarded-custody code-review codex v1; repair-design-review codex v1
Owner Summary: Approve the exact CPU/test-only repair-design supplement. RD1 and RD2 are closed by explicit scope-derived prefix restrictions and atomic terminal-entry/custody publication plus pre-ARM fault admission. The unchanged implementation still reproduces F1/F2 and is not approved. Implementation and fresh independent code review remain required; no integrated or GPU/product gate opens.

## Identity, revision, and verdict

Reviewer session: `/root/c4_c2d1_repair_design_review_v2`, fresh independent design review, no delegated work, no implementation edits, and no inherited author reasoning used as approval evidence.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Base: `4897fdc18d9df6658084c6c12018787be51ad6c6`.
- Reviewed frozen HEAD: `2ea88c842a6eefd5714e02cee6ad15400b33737c`.
- Candidate: `collab/q54-optical-lab-p12-4c4c2d-guarded-custody__repair-design__codex__v2.md`.
- Independently verified candidate SHA-256: `01a464293f4474f1665ee0ae3d2eda5477df43cec2f2ebb7374dcf87195656e2`.
- Archived code-review v1 SHA-256: `a91fdbab5a94d67f04cc08abaa0c6b9dff44ca2daca8b81b0e7fbf1b11db8a4e`.
- Archived repair-design-review v1 SHA-256: `4b0a4e87be2c90fd2114613e37f49642a78e4fd2851b9d2b213269eb25120ed9`.
- Initial and final working tree: clean. Base-to-HEAD changes are exactly the archived repair-design-review v1 and repair-design v2. Production/test/tool code is unchanged from original failing code revision `4ee33ab7a09720f0dc08f3e86bb199af59e58e40`.
- Verdict: **Approve**, design only, for this exact candidate digest/revision and bounded scope.

Read `AGENTS.md`, `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md` in full; read the original custody v2 requirements, complete archived code review and repair-design review, and relevant parent runtime-integration v3 requirements (source supervision, setup/coverage, ordered failure custody, validation and completion boundaries). Inspected actual controller/reducer/client, guardian and inherited scope/owner paths, `native_setup._abort`, relevant test fixtures and both external reproductions. Author assertions and prior verdicts were treated as claims to verify.

## Findings and closure

No substantive unresolved design findings.

### RD1 closed — compromised scope is authoritative before notification/failure

Candidate section 2, especially lines 110–123, now includes replay-derived compromised scope in the irreversible prefix boundary. It explicitly derives that scope from the full descriptor, returned-owner declaration and explicit unknown-scope facts, before later notification or `guard_failed`. It applies attempted-ARM/started-COMPLETE subset bounds and forbids the emergency-continuation exception for compromised prefixes.

This matches actual code facts: `guarded_evidence._Replay` declares descriptor/owner scope as those events are replayed; `compromised` does not depend on `guarded_segment_failed`. Inherited `begin_device_leaf` and `adopt_returned_device_owner` reject compromised scope before a subsequent valid leaf, with returned owners first retained. Thus the two reproduced early-descriptor prefixes are no longer freely lagging under the proposed contract. Section 3 requires negative cuts for descriptors, returned owners and explicit unknown facts, rejecting extra COMPLETE, normal ARM and emergency ARM, while preserving matching original ARM/ACK and genuinely healthy lagging prefixes.

### RD2 closed — entry, custody and fault observation have one representable order

Candidate section 1, lines 40–57, explicitly makes normal-state inspection/failure, `termination_started`/`_terminating` publication, strong terminal custody installation and `terminal_custody_retained` a single controller-RLock critical section. Foreign-thread failure uses the same lock. It therefore commits either a normal failure before entry or a terminal failure after custody, never an event between those legal domains. Trusted bookkeeping is explicitly non-reentrant; callbacks, waits, native calls and resource release are excluded from this section.

A committed fault before `terminal_arm_started` now prohibits that ARM and selects full quarantine; no unarmed terminal watchdog is claimed. ARM admission itself checks/marks under the same lock, with the socket exchange outside it. Section 3 explicitly requires deterministic entry/publication and pre-ARM schedules in CPU-only and target-dirty fixtures, truthful replay, retained full graph, single terminal disposition, and repeated-termination rejection.

The proposed implementation is feasible with the existing controller RLock and append-only custody list. Guardian partial freeze/authorization are state-only operations; actual reference release is a separate existing method. Exact live-ticket validation already exists in the thread-owned real client, so a foreign error can latch controller failure without mutating that client from the wrong thread. The current code's separate entry operations are not claimed to already meet this design.

### Original F1/F2 receive explicit repair contracts, not implementation approval

- F1: the new once-only terminal failure event preserves truthful pending/active guard state. Failure checks and before-call markers use one lock for terminal ARM, sync and release; success outcomes and terminal intent are also ordered against fault observation. A late real ACK can remain evidence without authorizing sync/release. A previously admitted sync/cleanup can produce a matching failed outcome, while failure before admission cannot be mislabeled as an operation that ran. Post-release faults use remaining custody, not a false full-graph claim. No lock is held across ACK wait, sync or destructors, preserving supported reentry/failure observation.
- F2: pending attempts must match supervisor role even without a worker ACK. Terminal or irreversible-failure/compromised records restrict supervisor ARM/COMPLETE identities to worker attempts/started completions. This rejects both original contradictory pairs while allowing a real COMPLETE with lost ACK, matching unreceived ARM ACK, absent worker evidence and legitimate healthy-prefix lag. The one future emergency allowed after a healthy-IDLE termination prefix is exactly the next sequence and is never completable.
- The one reducer remains the source of descriptive summaries; the new bool is replay-derived, not authority supplied by a caller. Parsed histories/hashes cannot create client tickets or custody authority. Whole consistently rewritten histories remain outside the authenticity claim.
- No extra wire role, fallback, timer restart, second sync, copied upload/guardian implementation or runtime activation is introduced. The unpublished v3 schema delta is explicit and requires fixture updates; C0 evidence is not silently upgraded.

## Independent verification and observations

Environment: Windows PowerShell, local `C:/Python314/python.exe`; no SSH, installation, GPU execution, candidate edit, shared-policy edit, reset, push or process kill was performed. Existing external runners/repros were read before execution. `-B` prevented bytecode writes.

Executed checks:

1. `git rev-parse HEAD`, `git status --short`/`--porcelain`, `git diff --stat` and `git diff --name-status 4897fdc18d9df6658084c6c12018787be51ad6c6 HEAD`, plus `Get-FileHash -Algorithm SHA256` for candidate and archives: exact revision/digests and clean tree. Final revision/hash/status checks repeated.
2. `git diff --check 4897fdc18d9df6658084c6c12018787be51ad6c6 HEAD`: exit 0. `git diff --quiet 4ee33ab7a09720f0dc08f3e86bb199af59e58e40 HEAD -- optical_worker tests tools`: exit 0, confirming unchanged failing implementation. Read-only v1/v2 `git diff --no-index` confirmed the precise design delta; Git emitted normal LF/CRLF advisories, not a mutation or validation failure.
3. `C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/c2d1-review-runner.py`: runner PID 12100, parent 2800; repro child PID 21292, parent 12100. Actual child exit 0 and command exit 0. CPU-cancel delayed-ACK error still reaches RELEASE_AUTHORIZED and remaining quarantine; target-dirty error still invokes one sync after terminal fault; both F2 incompatible artifacts still validate. Exit 0 means successful bug reproduction, not correctness or repair execution.
4. `C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/c2d1-repair-design-scope-repro.py`: runner PID 2644, parent 16756; child PID 30504, parent 2644. Actual child exit 0 and command exit 0. Both foreign and unknown descriptor prefixes end at `descriptor_observed`, replay compromised=true and failure=false/controller RUNNING, and the unchanged validator accepts extra COMPLETE/ARM. The revised design explicitly forbids that acceptance; this is not a test of a future implementation.
5. Both external runners have a 15-second child bound and exact-child kill/reap on timeout; neither timeout fired. The original socket fixture closes both endpoints and joins its peer thread. The scope repro creates no socket/GPU workload or additional descendants. No new scratch executable was needed.
6. Exact `Get-Process -Id 12100,21292,2644,30504 -ErrorAction SilentlyContinue` returned no processes after completion. `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` contained no reviewed runner/child PID. It listed unrelated desktop applications (SearchApp, TextInputHost, TeamViewer, Omniverse Launcher, Feishu, EdgeWebView, Teams and Code), all untouched. This is scoped residual evidence, not global GPU idleness.

The initial PowerShell read rendered Chinese review principles incorrectly using its default encoding; the entire file was reread with explicit UTF-8 before task inspection. Large combined reads were subsequently split to read the relevant content completely. These were read/output issues, not code or environment test failures.

## Coverage limits and required implementation proof

This is a document-only design checkpoint. No repaired event grammar or locking implementation exists in the reviewed diff. Entry-boundary scheduling, success-boundary failures, returned-owner/explicit-unknown cross-history negatives, updated handwritten v3 fixtures, and regression acceptance remain mandatory proof for the implementation, not results claimed here. Existing in-sync and destructor-reentry tests were inspected, not counted as newly executed suite evidence.

No broad GPU/regression suite was run for this documentation delta. Later code acceptance must run the specified original real socket repros, new pre-admission/entry/success-boundary matrix, allowed and forbidden history combinations, unchanged real-process artifact cases, source catalog/policy and C0 regressions, with actual exit/reap and residual/GPU checks. CPU mocks/fixtures prove control and evidence behavior only, not CUDA/Warp completion or resource safety. No hostile in-process Python/native sandbox or cryptographic authentication is claimed.

## Exact approval scope

Approval permits implementation of this C2d1 CPU/test-only repair-design supplement only, followed by a frozen candidate and fresh independent code review. It does not approve original code F1/F2, complete C2d, or extend any historical approval.

Still closed: integrated C2d setup and source-frozen guarded-role/bootstrap adoption; module/session custody before CUDA-capable import; real Warp/Torch upload and mandatory dual-GPU target/ambient/foreign/pre-return/unknown matrices; final graph/barrier/warmup/readiness composition; arenas/C2e/C2f; product GPU preparation; a3b and result/baseline authority; R2/host/S; later C4/B3; shipping/activation and C4 completion. Existing cleanup/adoption/removal obligations remain unchanged.
