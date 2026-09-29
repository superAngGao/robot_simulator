# Independent design re-review: C4c2 runtime integration v3

Initiative: q54-optical-lab-p12-4c4c2-runtime-integration
Stage: review
Author: codex
Version: v3-review
Date: 2026-09-30
Status: accepted
Related Files: runtime-integration design codex v3; complete independent review codex v2; parent C4c design v7; C4c2b supervised-waits design v3; optical_worker/wait_protocol.py; optical_worker/wait_supervision.py; optical_worker/lifecycle.py
Owner Summary: Approve the exact v3 design only. Both prior blocking findings are closed against the retained guarded-wait v1 implementation. No source implementation, device acceptance, product preparation/submission, result authority or activation is approved by this report.

## Reviewer, revision and isolation

Reviewer session: `/root/c4_c2_integration_design_review_v3`, a fresh independent Codex design-review session. I received the bounded task packet and repository, not the implementation conversation. I did not author or mutate the candidate. This is role/context isolation, not a separate vendor or security boundary.

Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.

Branch: `implementation/c4-supervised-waits`.

Base and observed merge-base: `2e6f9f5ce9861ce7bf26583e825802546b36cf9d`.

Reviewed HEAD: `674af14eb55e34e3a249b2ee58b3e6ede54621a9`.

Candidate: `collab/q54-optical-lab-p12-4c4c2-runtime-integration__design-request__codex__v3.md`.

Independently verified SHA-256: `a76bbe8963064876895ba56eebe72221d941af97f0c84722189b505d6536067`.

Complete archived v2 report SHA-256: `94ab6e4c8a2bf7be95db86ba304d93e97a76695df6b4a22ba75f452e6397aa8f`.

Initial and final tracked/untracked status were clean; HEAD and candidate digest remained unchanged. The base-to-HEAD diff adds only the v3 proposal (798 lines) and unchanged complete v2 report (160 lines). No source, test, gate or shared-review-policy changes are in the candidate. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely as UTF-8. I read the candidate and prior report, compared v2/v3, and checked affected protocol, supervision, lifecycle and parent-design requirements. This follow-up does not re-execute every already-closed first-round topic.

## Verdict

**Approve**, for the exact design revision and bounded scope stated below.

No substantive blocking or non-blocking finding remains in this re-review. The revised policy has legal successful and failed traces in the retained protocol; it does not require changing guarded-wait v1. Approval is not a claim that the integrated drivers already implement these rules.

## Prior findings: independent disposition

### F1 — closed at design level

V3 section 4 and sections 8–8.1 now separate successful normal segments from terminal emergency segments:

- Normal: ARM / ARMED / operation success / COMPLETE / DISARMED, then quiescence or resolution commit.
- Emergency: true IDLE / ARM / committed deadline / exact ARMED / one target synchronization / observed sync success / release authorization / terminal worker exit / supervisor-observed exit. There is no emergency COMPLETE or DISARMED.

The emergency client ticket remains ARMED and the supervisor remains ARMED_TERMINAL until exit or fault/deadline handling. Cleanup/exit does not clear or restart the deadline. This matches GuardWaitClient.complete(), which rejects emergency tickets, and the production supervisor transitions in wait_supervision.py, which permit normal_complete_received only for ARMED_NORMAL and non-emergency role. Parent v7 section 6.6.2 permits whole-graph release after successful exact-target synchronization and zero external holds, without requiring emergency disarm.

V3 explicitly distinguishes successful synchronization and release authorization from completed cleanup and observed process exit. A cleanup exception retains remaining owners without falsely claiming already-released owners are still held. A hung/killed worker cannot acquire a fabricated successful record. Existing lifecycle events emergency_sync_started, emergency_sync_succeeded and resource_release_authorized remain the proposed facts; versioned integrated evidence additionally binds the exact session/target/sequence and actual supervisor outcome.

I reran the original production-core counterexample: emergency COMPLETE is still rejected and causes TERMINATING without DISARMED. That is the correct negative result; v3 no longer asks an implementer to take that transition. New CPU probes verify the legal normal-to-emergency trace, unchanged terminal deadline through a modeled post-sync interval, deadline expiry at equality, and refusal to treat kill-request evidence as observed exit. Section 9 item 9 separately mandates real production-driver success and cleanup/exit-hang tests. Those remain implementation acceptance work, not claimed device proof here.

### F2 — closed at design level

V3 section 8 orders failure classification before clean/no-sample disposition. Pending ARM/COMPLETE ACK, failed guard, and failed leaves inside armed setup/resolve take quarantine/exit with no COMPLETE, replacement ARM or synchronization. Sections 8.1–8.2 explicitly include allocation/copy/compile/record/warmup launch and submit-then-raise, even if no wait was reached. The sticky integration state is guarded_segment_failed, not merely normal_guard_failed.

Emergency admission now checks the exact live client/session and successful normal trace: actual IDLE, no active or pending guard, and no failed segment. It can cover known-target sample enqueue/finish failure outside a normal guard; the same failure inside guarded warmup cannot enter emergency. The rule also preserves foreign/unknown-scope and incomplete-coverage precedence.

This matches GuardWaitClient.arm()'s IDLE requirement and the supervisor's arm_committed transition. Rerunning the original active-setup counterexample still rejects a nested emergency ARM. Additional CPU probes show ARM_ACK_PENDING, ARMED_NORMAL and DISARM_ACK_PENDING faults preserve the earlier deadline and never increment normal completion. The probe does not pretend that the supervisor can independently observe an arbitrary driver leaf failure: marking that failure before generic termination and retaining custody are explicit integrated-driver obligations. Section 9 items 5, 6 and 9 require corresponding real failures and zero further ARM/COMPLETE/sync calls.

## Affected high-risk invariants checked

- The protocol-version distinction remains explicit: closed-worker protocol v2 versus unchanged guarded-wait v1. Old C0 A/K semantics and source-frozen adoption boundaries are not silently altered.
- Existing lifecycle.terminate() still lacks integrated guard-failure knowledge and may perform its old emergency path. V3 does not claim otherwise: C2d must adopt the new ordering and versioned evidence rather than directly reuse that fallback after a failed normal segment. Existing commit_clean_close() still rejects historical sample submission; the new exact post-reclaim proof is future integration, not a cleared latch.
- Parent-v7 conservative custody remains: dirty scope expands monotonically; foreign/unknown scope is never repaired by target synchronization; partial resources must be retained before unwinding. Complete coverage means all resources possibly touched so far, not imaginary completion of every planned constructor.
- The one resource seal still precedes the final normal barrier. Quiescence requires synchronization success plus DISARMED and binds the immutable graph/coverage/readiness. V3 does not add a second seal or let emergency completion create normal readiness, resolution, reclaim or reusable IDLE.
- Sample/session submission latches remain sticky. Fault handling cannot select ordinary release merely because no current request is submitting. Post-reclaim normal CLOSE still requires its exact proof and ACK discipline; fault/ACK-loss tests remain required, including reclaimed-idle and close-ACK cases. No general reusable shutdown is introduced.
- Resource release authorization, cleanup completion and actual process exit remain different facts. The actual core records kill_requested separately and allows final evidence only after process_exited; CPU replay checks consistency, not authenticity or live GPU custody.
- Section 9 item 11 adds scoped PID/start/parentage and hang checks without a numerical performance gate or permission to terminate unrelated work. The revised emergency tests preserve negative emergency-COMPLETE coverage and add the successful trace previously missing.
- V2-to-v3 changes leave prior warmup/final-readiness, camera/materialization, row-semantic migration, token/cancellation, one-slot arena, source-closure and cleanup requirements intact. They do not reopen their established design decisions or claim their future code acceptance is complete.

## Executed verification

Read-only inspection used Get-Content with explicit UTF-8, rg --files / rg -n, Get-FileHash, git status, git rev-parse, git branch --show-current, git merge-base, git diff --stat / --name-status, candidate-only git diff --check, and git diff --no-index between v2/v3. The no-index comparison reported expected document differences and Git's CRLF warning; it did not write files. The candidate-only whitespace check passed. All inspection shell tool calls completed synchronously with exit code 0 and no running-session ID.

Probe commands, from the repository:

```text
C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/c4-c2-integration-review-v2-probe.py C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design
C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/c4-c2-integration-review-v3-probe.py C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design
```

The original probe was read and rerun unchanged: both original counterexamples reject as expected, followed by 5 passed in 0.21 seconds; process exit 0.

The new external probe was created with apply_patch. SHA-256: `83e76b05c980b4d32c9ded9f9a8ac0b07d4fb6f1a9230c7d74d8e4e0759468c3`. It imports actual production core modules and checks normal/emergency sequencing, emergency deadline retention, incomplete kill evidence, and three pending/active fault states. All five printed probe cases passed. It then ran these production test selections in the same interpreter:

```text
test_guarded_wait_core.py::test_normal_snapshot_golden_and_final_evidence
test_guarded_wait_core.py::test_ordering_errors_cannot_commit_completion
test_guarded_wait_core.py::test_stalled_disarm_keeps_watchdog_and_parent_loss_cannot_clear
test_guarded_wait_core.py::test_disarm_ack_earlier_deadline_expires_at_equality
test_guarded_wait_core.py::test_pending_ack_send_failure_keeps_original_guard
test_guarded_wait_core.py::test_evidence_replay_not_self_hash_acceptance
```

Result: 17 passed in 0.23 seconds; process exit 0. These selections and probe cases use CPU state transitions, not real watchdog child processes or device synchronization. Calls to observe_exit in these tests are explicitly simulated observations, not claims of testing OS reaping. The Python driver processes themselves did exit and were checked separately below. No failure was masked or repaired in the candidate. `-B`, sys.dont_write_bytecode and pytest `-p no:cacheprovider` prevented candidate artifacts.

## Process ledger, hygiene and limitations

The test driver PowerShell was PID 38672, creation time `2026-09-30T04:15:05.4089470+08:00`, parent PID 34632. It launched Python PID 38124 for the unchanged v2 probe, observed exit 0, then Python PID 34848 for the v3 probe, observed exit 0. Neither probe launches subprocesses. After both returned, a Win32_Process direct-child query returned zero remaining children. The driver shell completed with exit 0.

The subsequent scoped check ran under PowerShell PID 38100, creation time `2026-09-30T04:15:49.1073250+08:00`, parent PID 34632. It checked recorded test PIDs plus Python command lines referencing these exact probe paths: zero residual processes. This check shell also completed synchronously with exit 0. Earlier read-only inspection shells were synchronous successful tool calls; their individual PIDs were not separately sampled. No CLI reviewer, worker, supervisor, background service, remote process or GPU process was started by this review. The reviewer is this agent session, not an independently spawned CLI process. No unresolved owned residual or hang was observed.

No process was killed. The shared tool-service parent and unrelated work were not attributed to this review or terminated. No SSH, scp, primary183/fallback101 access, remote health claim, GPU reset, GPU process query, asset download/install, benchmark, result writer or baseline generation occurred. No GPU process was launched, so remote/GPU hygiene checks were not applicable to this CPU-only round. In particular the unrelated TileOPs test/worktree was untouched.

This is design feasibility and contract verification, not GPU correctness or resource-lifetime acceptance. Real Warp/CUDA driver failure matrices, target1/ambient0, guardian sentinels, real cleanup-hang watchdog reaping, actual source closure, Go2 asset materialization, camera binary values, correctness/parity, O(1) acquire and integrated token/reclaim evidence were not executed here. They remain the mandatory later checkpoint tests. CPU replay cannot substitute for them.

Only the new probe and this report were created by this session, both outside the candidate. No candidate edit or commit was made.

## Precise approval scope

Approval authorizes implementation of the exact v3 design through the separate ordered C2c materialization, C2d lifecycle, C2e Warp timeline/arena and C2f integrated-worker checkpoints, each with independent code review and required evidence. The bounded integration is R0/R1, one slot, none/native-only device delivery, independent single-lane Warp/CUDA workers. This approval does not declare C2 or C4 complete.

A separate activation review is required before live product preparation/submission. Tests and component approvals do not supply that authority.

Still closed: every R2 row (device-only until C4c3 and host until B3), S/physics publication, host payload/external holds/depth two, full-image IPC, reusable shutdown/joint C4d lifecycle, backend comparison/selection, native result authority, a3b, baseline generation, shipping/default activation, and parent C4 completion. Existing component/CPU-camera approvals retain only their original scopes. C4c3, C4c4, C4d/B3, real-device acceptance, final cleanup and activation reviews remain required.

The unchanged full report may be recorded under collab in an evidence-only commit under the shared workflow exception. That recording cannot change the reviewed revision, verdict, requirements or closed gates.
