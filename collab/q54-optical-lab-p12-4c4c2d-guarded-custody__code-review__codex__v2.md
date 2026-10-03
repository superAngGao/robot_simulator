# C4c2d1 guarded custody — independent code review v2

Initiative: q54-optical-lab-p12-4c4c2d-guarded-custody
Stage: code-review
Author: codex (fresh independent reviewer)
Version: v2
Date: 2026-10-03
Status: in_review
Related Files: guarded-custody design-request v2; repair-design v2; code-review v1; repair-design-review v1/v2; runtime-integration design-request v3
Owner Summary: Request changes. Original delayed-ACK F1 and both original F2 pairs now fail closed, but two independently reproduced gaps remain: compromised scope can still COMPLETE and start new normal work, and disabled legacy API rejection during termination bypasses sticky terminal fault observation. Existing passing suites do not close these cases. No GPU, integrated C2d or C4/product gate is approved.

## Identity, revision, and verdict

Reviewer session: `/root/c4_c2d1_guarded_custody_code_review_v2`, fresh independent code review, no delegation, no candidate edits.

- Immediate base: `be3a4616fb2e5507bd9074e7ecc271b3eb739ae1`.
- Reviewed frozen HEAD: `90f50bb3ef67dd99182e4da0e5bc8479cd76e53b`.
- Original failing code: `4ee33ab7a09720f0dc08f3e86bb199af59e58e40`; overall C2d1 code base: `8cbf8645123a3d7e3f6cf4bbc1434fa3a6612472`.
- Local candidate: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Remote candidate: `/data3/ang.gao/robot_sim_c2d1_code_v2`, authorized fallback `ang.gao@192.168.80.101`.
- Both candidate HEADs and clean working trees independently checked initially and finally. The original dirty remote checkout was not touched.
- Accepted repair contract: repair-design v2 at `2ea88c842a6eefd5714e02cee6ad15400b33737c`; independently verified SHA-256 `01a464293f4474f1665ee0ae3d2eda5477df43cec2f2ebb7374dcf87195656e2`.
- Verdict: **Request changes**, for F3 and F4 below. These are bounded CPU controller/reducer correctness findings, not demands for hostile-Python security or new GPU authority.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely. Read custody requirements, the accepted repair supplement, full earlier code/design reviews, and relevant parent runtime-integration failure/custody rules. Inspected actual implementation/test diffs, controller/reducer/client call and state flows, inherited guardian/permit ownership, native_setup._abort, fixed subprocess harness, catalog repins and protected-source boundaries. Author implementation notes and reported counts were treated as claims, not approval evidence.

## Findings

### F3 — P2, blocking: compromised scope is not an irreversible boundary in the live controller or lifecycle reducer

Locations: `optical_worker/guarded_lifecycle.py:154` (`_check_running`), `:198`–`:201` (normal success/COMPLETE), `:244` (`mark_unknown_device_scope`); `optical_worker/guarded_evidence.py:241`–`:282` (normal ARM/body/COMPLETE/DISARMED predicates), compared with the new irreversible-boundary check at `:732`–`:755`.

Independent public-API reproduction, without production monkeypatching or object mutation:

1. Construct the actual controller/client with a real socket pair and actual supervisor core/transport.
2. Enter a setup segment and enter one exact-target CPU fixture leaf.
3. Call `controller.mark_unknown_device_scope(source="fixed_test", reason="unknown_return")`. This records unknown scope and returns normally.
4. Leave the segment normally, then enter a new `controller.segment("resolve")`.

Observed: the first context sends COMPLETE/DISARMED, the second sends a new ARM, its body executes, and it too completes. Termination later quarantines, but the accepted lifecycle record reports `unknown_device_scope=true`, `guarded_segment_failed=false`, and `normal_complete_count=2`. Its trace contains `unknown_scope_set -> normal_body_returned -> normal_complete_started -> normal_disarmed -> normal_arm_started -> guard_armed`. No catch is needed for the explicit-unknown case. Separately reproduced the same continuation after catching the public returned-owner scope rejection for ordinal0 or unknown ordinal; owner adoption itself occurs before rejection as required.

Cause: `_check_running` checks controller/terminal/failure state but not replay-derived compromised scope. Inherited `mark_unknown_device_scope` records rather than raises. The reducer also admits normal body success, completion and subsequent normal ARM after compromise. The repair only bounds an externally supplied supervisor tail when the worker snapshot is cut at the compromised prefix; it does not enforce the same irreversible boundary on the worker's own continuation. This is not a wholly invented matching history: the actual controller executes it.

Consequence: an unknown/foreign ownership situation can be converted to healthy IDLE and a new guarded body can run before eventual quarantine. This contradicts accepted repair section2's explicit no-new-normal-work/no-COMPLETE boundary after descriptor/owner/explicit unknown facts, and the parent ordered rule that foreign/unknown scope permits quarantine and hard exit only. Eventual quarantine does not legalize intervening COMPLETE/new work.

Why existing tests miss it: new evidence tests truncate at the compromised fact and add only supervisor continuation; returned-owner live tests let the exception escape the segment; no live test marks unknown scope and then returns normally, or catches the scope error before normal context exit. Existing reducer tests do not append a worker normal-success continuation after the compromise.

Repair direction: enforce the same sticky scope boundary in live normal-work/success admission and in the single reducer, based on declared descriptor/owner/explicit-unknown facts, not just later notification or exception propagation. Preserve exception-graph adoption/retention and actual native_setup._abort behavior; do not accidentally prevent conservative custody while preventing normal work. No changes to protected C0 implementation or new stage authority are needed.

Required proof: explicit-unknown and caught returned-owner foreign/unknown live cases must produce no normal COMPLETE or new ARM/body after the fact, retain the graph and terminate conservatively. Add worker-trace negatives for normal body/COMPLETE/new ARM following each compromised-prefix source, including early descriptor/owner cuts and explicit unknown. Keep matching original ACK/prefix uncertainty and healthy lagging histories accepted; rerun native_setup._abort and original process matrix.

### F4 — P2, blocking: disabled legacy methods reject without observing a terminal reentrant/thread fault

Locations: `optical_worker/guarded_lifecycle.py:254`–`:261` (`commit_target_barrier`, `latch_measured_sample_submission`, `commit_clean_close`), contrasted with `_check_running`/`_fail_guard` at `:134`–`:163` and terminal admission at `:291` onward.

Independent delayed-ACK reproduction uses the same supported failure-injection shape as original F1, with no production monkeypatching:

1. Create actual controller/client/socket/core/transport and retain a sentinel host owner; optionally enter/complete a target-dirty setup segment.
2. Terminate on the owner thread.
3. Once the peer has committed emergency ARM but before writing ARMED, call `controller.commit_clean_close()` on the peer thread and catch its `WorkerSetupProtocolError`.
4. Deliver the real ACK.

Observed: CPU-only termination releases the guardian and returns `cancelled_before_submission_exit`; target-dirty termination invokes the sync fixture once and releases the guardian, returning `aborted_synced_exit`. Both report `terminal_operation_failed=false`, `release_status=completed`, and the sentinel is no longer held by the guardian. The illegal public call was detected/rejected before either side effect but did not invalidate them.

Separately, from inside the admitted sync fixture, call and catch each of the three disabled methods. Each produces successful sync/release/`aborted_synced_exit` with no terminal-failure event. This is the same caught-reentry pattern already tested with `terminate()` and `begin_device_leaf()`, but these overrides bypass the common observation path. No claim is made that the disabled legacy operation itself performed GPU work or directly released resources.

Cause: these public overrides unconditionally raise without checking owner/terminal state or retaining/recording the failure. The repaired admission machinery is effective only when the error reaches `_fail_guard`; these locally detected cross-thread/reentrant errors never do.

Consequence: the accepted repair's rule to record actual cross-thread/reentrant errors while TERMINATING, and the unchanged rule that further emergency errors invalidate pending success, do not cover all controller-owned public entrypoints. A known pre-admission misuse can still be followed by sync/release; a known in-sync misuse can still be marked succeeded. Closed legacy admission is not a reason to exempt its detected errors from custody ordering. This does not require guarding arbitrary hostile reflection or an explicit base-class bypass.

Why existing tests miss it: `test_admission_stays_closed` checks only that the methods raise on a healthy pre-terminal controller. New delayed-ACK tests inject `begin_device_leaf` or context protocol misuse; success-boundary tests use the same checked path. No disabled-method call is injected during ACK/sync/cleanup.

Repair direction: route terminal/thread/reentrant rejection of these disabled public methods through the existing lock-protected fault-observation/exception-retention mechanism while keeping their underlying capabilities closed. Preserve post-terminal single-use rejection and existing healthy out-of-scope behavior where required; do not call the C0 implementations. Audit analogous controller-owned public rejection paths rather than adding an isolated boolean.

Required proof: parameterize the disabled methods at terminal ACK for cpu_cancel and target_sync (zero subsequent sync/release, full retention, one truthful terminal failure); during genuinely admitted sync (failed sync/full custody); and during actual cleanup (failed cleanup/remaining custody). Keep no-lock-across-native/destructor, post-intent rejection and repeat-termination tests passing.

The parent author independently flagged the disabled-method coverage hypothesis during this review. F4 is based on this reviewer's source inspection and independently executed repros, not solely on that claim. F3 was independently identified and reproduced before the parent confirmed the same code predicates.

## Original findings and repaired areas

- Original F1's exact delayed-terminal-ACK `begin_device_leaf` counterexample is closed for both cpu_cancel and target_sync: zero sync, no release authorization/cleanup, one `terminal_operation_failed` followed by truthful real `guard_armed`, full quarantine. This does not close F4's missed observation path.
- Both original F2 artifacts independently reject: pending role conflict rejects with `worker ARM attempt conflicts with supervisor role`; COMPLETE/new ARM after the terminal failed normal segment rejects with `supervisor ARM after irreversible worker boundary`.
- The unmodified original repro runner exits1 at the first now-rejected F2 artifact. Its earlier F1 outputs were examined; that exit1 is not treated as full F2 proof. The independent runner explicitly catches and verifies rejection of each original F2 pair.
- RD2's implemented terminal-entry critical section covers state publication/custody/freeze; fault and ARM/sync/release/outcome/intent admission share the lock. Inspected and executed scheduling tests verify no entry publication gap and no lock across ACK/native/destructor calls. Full versus remaining custody tracks actual release entry.
- RD1's cross-artifact prefix matrix passes for early descriptor, returned-owner and explicit unknown facts. Its consistency extension needs F3's live/reducer counterpart before the full irreversible-boundary claim holds.
- Actual native_setup._abort dispatches to the override with the inherited guardian/permits; no duplicate upload loop, legacy terminate delegation, copied release implementation or runtime activation was introduced. Exact client ticket identity/original ACK checks remain covered.
- Immediate catalog delta changes only the two guarded module digests. Lane inventories and AST policy hash are identical. Remote SHA-256 matches exact catalog entries: guarded_evidence `3cd523292919200b353b4376fdcc8b11fa3a144327eed33eda79b9444be4b970`; guarded_lifecycle `de1ba20ab795f9fa1af9ad3f9a40db9c1947d0b54083806b623808b84c57a2e6`. Catalog presence does not activate captured-source guarded runtime adoption.

## Independent verification

Local environment: Windows PowerShell, `C:/Python314/python.exe`, no Torch/Warp. Remote environment: `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python3.12`. No installation, GPU workload, reset, push, broad kill or candidate edit. Primary-host transport failure was supplied in the packet; fallback revision/tree were independently verified.

SSH prefix:

```text
ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101
```

Executed commands/results:

1. `git rev-parse HEAD`, clean `git status --porcelain`, actual immediate/overall diff and design SHA checks, repeated at end. `git diff --check 8cbf864 HEAD`: pass. `git diff --quiet 8cbf864 HEAD -- optical_worker/lifecycle.py optical_worker/native_setup.py optical_worker/worker.py tools/optical_closed_worker.py tools/optical_closed_worker_supervisor.py`: explicit exit0.
2. Local `python -B -m ruff check` on guarded_lifecycle.py, guarded_evidence.py, wait_protocol.py, guarded lifecycle/evidence/process tests and tests/support/guarded_custody_process.py: all checks passed.
3. Remote from the exact candidate: `timeout --signal=TERM --kill-after=5s 90s <python3.12> -B -m pytest tests/unit/optics/test_guarded_lifecycle.py tests/unit/optics/test_guarded_evidence_fixtures.py tests/unit/optics/test_guarded_custody_process.py tests/unit/optics/test_guarded_wait_core.py -q -s -p no:cacheprovider`: **260 passed in 13.57s**, actual command exit0, no skips. All22 real process cases were unchanged. Each actual supervisor returned0 and independently observed/reaped its worker; timeout sentinels/assertions were not substituted for inner process observations.
4. Remote same outer bound/interpreter with `tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_leaf_descriptors.py tests/unit/optics/test_sacrificial_sources.py -q -p no:cacheprovider`: **270 passed, 2 skipped in 28.90s**, actual exit0. The two Go2 real-asset materialization lane cases were skipped because this invocation did not set ROBOT_SIM_TEST_GO2_MODEL_DIR; they are not claimed passed. Source-policy, catalog, C0 and descriptor regressions otherwise completed. This is not the entire bootstrap/Go2/C1c acceptance matrix.
5. `C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/c2d1-review-runner.py`: original external15-second bounded child, actual child exit1 at the now-rejected first F2 pair; both repaired F1 outputs inspected as above. No timeout fired.
6. `C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/c2d1-code-v2-independent-repros.py`: new external20-second bounded child; first execution and extended final execution both actual exit0. Both original F2 pairs reject independently; F3's three live variants and F4's three in-sync plus two delayed-ACK variants reproduce. Exit0 means completed counterexample collection, not correctness. Repro uses production controller/client/core/transport/reducer and existing fixture constructors, not a copied state machine. All peer threads joined and sockets closed. No production monkeypatch or mutation is used by this repro.

The author separately reported a 15-file remote run of689 passed in169.23s, exit0/no skips. That run was not duplicated while active and is not represented as independent reviewer evidence. Broader no-skip bootstrap/Go2 acceptance remains necessary on any repaired candidate; two decisive live failures already prevent acceptance here.

## Process hygiene and actual exits

Local original runner22700(parent21436), child33596(parent22700), child exit1. First independent runner18344(parent20628), child18696(parent18344), exit0. Extended independent runner10492(parent21760), child14184(parent10492), exit0. Exact final Get-Process lookup found all six absent. Outer child bounds did not fire and no kill was needed. Ruff's short-lived OS PID was not sampled; its command completed exit0.

Remote focused launch: timeout2372418(parent2372331), pytest2372439(parent2372418), start10:18:04 remote. The process-case ledger is `case: supervisor / worker / worker-returncode`; every supervisor returncode is0. Tests check worker identity parent equals the exact supervisor and final evidence ends in process_exited.

```text
cpu_success: 2372548 / 2372549 / 0
cpu_cleanup_hang: 2372551 / 2372555 / -9
cpu_exit_hang: 2372564 / 2372568 / -9
normal_success: 2372585 / 2372586 / 0
normal_leaf_raise: 2372587 / 2372588 / 74
inner_abort: 2372589 / 2372593 / 74
pre_leaf_raise: 2372598 / 2372599 / 74
foreign: 2372601 / 2372605 / 74
unknown: 2372606 / 2372607 / 74
sync_raise: 2372608 / 2372609 / 74
sync_hang: 2372610 / 2372611 / -9
cleanup_raise_before: 2372684 / 2372685 / 74
cleanup_raise_after: 2372686 / 2372687 / 74
normal_no_ack: 2372688 / 2372689 / 74
normal_wrong_ack: 2372700 / 2372701 / 74
normal_ack_eof: 2372702 / 2372703 / 74
normal_stalled_disarm: 2372704 / 2372705 / 74
terminal_no_ack: 2372706 / 2372708 / -9
terminal_wrong_ack: 2372754 / 2372755 / 74
terminal_ack_eof: 2372756 / 2372757 / 74
target_cleanup_hang: 2372758 / 2372780 / -9
target_exit_hang: 2372805 / 2372806 / -9
```

Remote source/C0 launch: timeout2374713(parent2374627), pytest2374734(parent2374713), start10:19:24; sampled cold-source child2375164(parent2374734). Both command handles completed exit0. Final `ps -C python3.12,timeout -o pid,ppid,lstart,args` was empty after each round; exact lookup of timeout/pytest/cold-child PIDs was also empty. This is scoped process evidence, not a global all-process absence claim. Short-lived remaining source-test child IDs were not individually sampled.

Remote `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` before and after rounds showed only unrelated training PID2336351. That process and its unrelated launcher chain were preserved. Local GPU query contained unrelated desktop applications and none of the reviewed runners/children. No GPU work was authorized or performed by this review.

One initial read used nonexistent root REVIEW_PRINCIPLES.md; the complete correct collab file was then read. Large combined reads were split after output truncation. An initial remote Python-wrapper command lost quoting through PowerShell/SSH and returned a shell syntax error before launching any test; the direct bounded pytest command supplied actual evidence. These are tooling/read issues, not candidate test failures or external-state changes.

## Scope, residual risk, and required next checkpoint

Request changes applies to exact HEAD90f50bb's C2d1 CPU controller/reducer/client/C0 exception-boundary slice only. Original F1/F2 examples are repaired but F3/F4 block acceptance. Repair, freeze a new revision and obtain a fresh independent re-review with these external repros retained. Do not edit this returned report when archiving it; the separate evidence-only archive may reference this reviewed revision without changing the verdict.

Still closed: source-frozen guarded-role/runtime adoption; module/session custody before CUDA-capable imports; real Warp/Torch upload and mandatory target/ambient/foreign/pre-return/unknown multi-GPU matrices; complete setup graph/barrier/warmup/readiness composition; arenas/C2e/C2f; product GPU preparation; a3b/result/baseline authority; R2/host/S; later C4 and C4d/B3; shipping/activation and C4 completion. Existing cleanup/adoption/removal obligations are unchanged.

CPU fixtures prove control flow, ownership retention of CPU sentinels, evidence consistency and actual process supervision, not real CUDA/Warp completion or allocation-release safety. Parsed records and public hashes are not authentication or execution capabilities; no hostile in-process sandbox or full trusted-history rewrite detection is claimed. No prior design/code approval is broadened by this report.
