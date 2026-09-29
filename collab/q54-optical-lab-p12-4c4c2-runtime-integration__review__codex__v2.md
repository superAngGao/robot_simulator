# Independent design review: C4c2 runtime integration v2

Initiative: q54-optical-lab-p12-4c4c2-runtime-integration
Stage: review
Author: codex
Version: v2-review
Date: 2026-09-30
Status: in_review
Related Files: runtime-integration design v2; parent C4c design v7; C4c2a design v1; C4c2b design v3; camera-resolution design/design-review/code-review v1; optical_worker/wait_protocol.py; optical_worker/wait_supervision.py; optical_worker/lifecycle.py
Owner Summary: Request changes. The prior warmup, seal, camera migration, row identity and token clarifications are substantially closed. Two related emergency-disposition rules remain inconsistent with the explicitly retained guarded-wait v1 state machine. No runtime or product gate is approved by this report.

## Reviewer identity, exact candidate and isolation

Reviewer session: `/root/c4_c2_integration_design_review_v2`, independent Codex design-review session. I received the bounded review packet and repository, did not author the candidate, and did not edit or commit anything in it. This is context/role isolation, not a claim to a different vendor or security boundary.

Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.

Branch: `implementation/c4-supervised-waits`.

Base and independently observed merge-base: `7717556fa0614e0512973424c109df640de6b7f4`.

Reviewed HEAD: `2e6f9f5ce9861ce7bf26583e825802546b36cf9d`.

Candidate: `collab/q54-optical-lab-p12-4c4c2-runtime-integration__design-request__codex__v2.md`.

Independently checked candidate SHA-256: `9dc28eca653e9d620ecb243502dbe94b4fe64e2fb053399d268ddbd4429ccc03`.

Initial and final tracked/untracked status was clean. HEAD remained unchanged. The diff from base adds only the v2 design and the complete archived v1 Claude report. Camera code `1bcb622af82d820c7247dced82b58c459b89233c` is an ancestor. No source implementation changed in this candidate.

Read `AGENTS.md`, `collab/README.md` and `collab/REVIEW_PRINCIPLES.md` completely, in UTF-8. Read the candidate, original v1 and previous report rather than accepting the new author's closure statements or the old report's quotations as evidence. Read C2a v1, C2b v3 and the camera design/reviews; inspected the relevant parent-v7 transaction, authority, resource ownership, termination, review-slice and cleanup requirements against actual code.

## Verdict

Request changes.

The design is substantially more explicit and feasible after the v2 additions, but its emergency success condition has no implementation in the mandatory retained protocol, and the emergency-entry predicate does not distinguish an idle guard from failure inside an already-armed setup segment. These are design-policy conflicts, not failures of the current protocol code. They must be settled before integration implementation is authorized.

## Findings

### F1 — P1, blocking: emergency release requires a handshake forbidden by the adopted protocol

Location: candidate section 8, line 610 (`emergency sync AND completion handshake succeeded`), together with section 4's retained `q54.closed-worker.guarded-wait.v1` requirement.

Trigger: a target-only post-submit failure occurs while the guard client is idle. The worker legitimately arms an emergency guard, receives ARMED, and its sole emergency target synchronization returns successfully. The design then requires a completion handshake before graph release.

Observed dependency contract: C2b v3 sections 4–6 explicitly make emergency terminal and prohibit COMPLETE/DISARMED. `GuardWaitClient.complete()` at `optical_worker/wait_protocol.py:265` rejects an emergency ticket before writing. `optical_worker/wait_supervision.py:173` commits emergency ARMED to `ARMED_TERMINAL`; its `normal_complete_received` transition at line 187 requires `ARMED_NORMAL` and a non-emergency role. The emergency deadline remains armed through worker exit. Parent C4c v7 section 6.6.2 permits successful target emergency synchronization to authorize whole-graph release and process exit; it does not require an emergency disarm.

Independent CPU reproduction using the actual production supervisor core:

```text
Emergency ARM/ARMED -> ARMED_TERMINAL
Emergency COMPLETE rejected: COMPLETE does not bind the outstanding normal wait
After forbidden COMPLETE: TERMINATING, no DISARMED response
```

Consequence: implementing the apparent post-sync COMPLETE/DISARMED handshake makes the intended `aborted_synced_exit` success path impossible. Alternatively, an implementer must silently delete this requirement, reinterpret “completion handshake” as the earlier ARM acknowledgement, or change terminal emergency protocol semantics. Those are materially different lifetime policies. Disarming the watchdog would additionally remove its intended coverage of release/exit. The ambiguity cannot be left to implementation.

Why existing checks do not close it: existing core tests correctly reject emergency completion; the new design's broad emergency-success/failure matrix does not specify the exact legal successful trace and still contains the incompatible acceptance predicate. A general assertion that guards work does not resolve which policy applies.

Required direction: preserve guarded-wait v1. State explicitly that emergency release requires the exact successful ARM/ARMED exchange, one successful target synchronization, complete known target-only coverage and zero external holds. There is no emergency COMPLETE/DISARMED exchange. The emergency guard/deadline remains live through whole-graph release and actual process exit; the supervisor separately observes/reaps exit. If a new post-sync handshake is genuinely intended instead, it requires a separately versioned protocol design and deadline/cleanup analysis, not reuse under v1.

Proof required after repair: define and test a successful trace `IDLE -> emergency ARM/ARMED -> one sync success -> release authorization -> worker exit -> supervisor observed exit` with no COMPLETE/DISARMED, no second sync, no reuse and no fabricated exit. Include cleanup/exit hang after successful sync to prove the original terminal deadline remains effective. Keep the existing emergency-COMPLETE negative test.

### F2 — P2, blocking design clarification: emergency entry must require an idle guard, not merely absence of a failed wait

Location: candidate section 4 lines 305–313 (guards cover import/compile/record and nonnested setup segments), section 8 lines 609–615 (emergency admission and `normal_guard_failed`).

Trigger: a setup segment has successfully armed and is executing a known-target allocation/copy/record/compile or warmup launch. That leaf raises before any event/device wait is attempted. Coverage may be complete and the target known; there has been no previous failed guarded *wait*, but the guard client remains ARMED for setup. The section-8 emergency row literally applies, even though starting an emergency guard is prohibited in that state.

Observed dependency contract: `GuardWaitClient.arm()` at `optical_worker/wait_protocol.py:240` requires IDLE. C2b v3 section 5 requires the fixed driver to retain its guardian, not call complete on a failing leaf, and hard-exit after a protocol/wait failure; no nested/restarted guard is allowed. The production supervisor likewise accepts a new ARM only from IDLE. The original operation deadline must remain in force after a segment fails.

Independent CPU reproduction using the actual core:

```text
Emergency ARM inside active setup guard rejected: ARM is out of order
Active-setup failure cannot start a second guard
```

Consequence: an implementer can choose between attempting an invalid second ARM (then quarantining on its failure), falsely completing the failed setup segment to get back to IDLE, or quarantining directly under the original guard. The second choice would assert success for a failed segment; the first is unnecessary protocol misuse. The desired direct-quarantine policy should be normative. This is not a claim that the existing core permits nesting or that the present disabled worker is unsafe.

Why existing checks do not close it: section 9 lists setup faults, failed barriers and guard failures, but does not assert zero second-ARM/COMPLETE attempts for a non-wait leaf failure inside an already-armed segment. Marking only `normal_guard_failed` for normal wait/barrier failure does not unambiguously classify a failed import/compile/recording segment.

Required direction: make the disposition table disjoint and order it by actual guard state. Any fault while setup/resolve is ARM_PENDING, ARMED or COMPLETE_PENDING, any unsuccessful guarded segment (including non-wait leaves), or a previously failed guard must retain/quarantine the graph, preserve the original deadline and terminally exit without COMPLETE, a new ARM, second synchronization, or ordinary cleanup. Allow a single emergency ARM only when the client is genuinely IDLE with no pending ACK, no active/failed guard, complete known target-only coverage and possible work requiring an abort sync. This includes post-submit failures outside an active normal guard; it does not include the guarded setup counterexample. Normal successful setup/resolve segments still use COMPLETE/DISARMED.

Proof required after repair: distinguish (a) post-submit fault with idle guard, for which one emergency ARM/sync is permitted; (b) allocation/compile/record/launch submit-then-raise inside an armed setup segment, for which zero further ARM/COMPLETE/sync calls are permitted; and (c) ARM/COMPLETE pending-ACK failures. Assert original deadline retention, exact custody, no successful completion evidence, and bounded observed process exit. Existing real-device setup/normal-wait failure requirements remain mandatory; CPU replay alone cannot prove them.

## Previous-round findings: independent disposition

P1.1, warmup/readiness: closed at design level. Section 5.1 now defines a private setup-only authority and two immutable readiness objects. The actual registry is keyed by readiness-subject identity, not by one slot per prepared input (`native_contract.py:242–322`), and `certify_native_readiness` accepts zero or ten warmups while creating a new object. FinalReadinessOrigin refers downstream to observed warmup facts; provisional readiness does not refer back to final graph/quiescence. Actual warmup authority, warmed-variant equivalence and inability to use it for measured execution remain code-review obligations.

P1.2, seal ordering: no remaining contradiction. The original v1 already explicitly froze complete resource coverage before the final barrier; the earlier report's barrier-then-seal reading is not supported by that text. V2 section 5.2 now states one seal, exact resource snapshot fields, and separate operation/quiescence evidence. This matches `NativeWorkerResourceGuardian.coverage_payload`, `seal_guardian`, `commit_target_barrier`, and parent v7 section 6.6.3. SEALED is not completion or release authority. Preserve the historical report, but do not implement its suggested second seal.

P1.3, row/case identity: closed. The prior quoted “No existing ... row, threshold or gate is rewritten” sentence does not appear in integration v1; it appears in the separate bounded camera-resolution design. Integration v1 already proposed v13. V2 section 3.1 correctly separates unchanged CanonicalOpticalCase bytes/hash, extended catalog metadata, and deliberately changed row-semantic identity. The actual `canonical_row_semantics_sha256` includes the catalog and shared contracts, so the proposed v8 semantic preimage plus resolved-render contract must change the digest. Deferring actual sequence-byte/digest freezing to CPU checkpoint C2c is appropriate because that checkpoint requires independent complete-preimage recomputation before runtime use, not placeholder acceptance.

P2.1, Go2 camera extraction: closed at design level. The source table in v1 already named the tool owner; v2 names exact functions, dependency order and CPU adapter. Actual `build_lab_video_camera`, `build_model_bounds_camera` and `_look_at_camera_rotation` use CPU NumPy/math and the specified camera/transform types. Workloads currently call the example MJCF importer; the importer and its transitive mathematical helpers can be extracted without moving CLI/render/video. Real assets and importer byte validation remain mandatory, not replaced by a surrogate or parent evidence.

P2.2, diagnostics: closed. “Pass diagnostics setting” is a pass-specific boolean, not an instruction to enable diagnostics. V2 explicitly maps primary to false, diagnostic to true and freezes the identical variant across warmup and final samples. Diagnostics-off owns no diagnostic event/ABI path.

P2.3, S/R2/host rejection: closed. Section 7 explicitly rejects at strict row admission before camera/materialization/CUDA-capable work; an independently called CPU S resolver rejects unsupported_case separately. All R2, including device-only, remain closed in C2.

P2.4, acquisition/cancellation: closed. EXECUTE is one serial handler performing acquire/enqueue/finish before exposing pending. ACQUIRED is never exported; public CANCEL only consumes unacquired prepared tokens. CANCEL_ONLY prevents later execute but permits mandatory resolve/release of an existing pending/resolved generation. Internal pre-leaf cancellation burns its generation and terminates rather than replanning immutable future generations. This fits the current C4 cancel/resolve/release/reclaim control behavior.

P2.5, protocol baseline: closed. `protocol.py` actually declares closed-worker protocol v1 and the exact eight request names. V2 distinguishes that version from guarded-wait v1 and explicitly changes response/state adoption. F1/F2 concern its failure-policy composition, not whether those protocol identities exist.

## Other high-risk areas inspected

- Source/trust boundary: closed trusted process rather than hostile-Python sandbox; literal module/source closure and bounded AST exception for actual read-only packed-argument inspection. `_render_launch_snapshot` really uses the listed ctypes operations to validate actual parameter addresses/types/bytes. No reason was found to remove that inspection or grant general CDLL/native-call privileges. Exact future source lists and hashes still require implementation review.
- Materialization/camera chain: existing assets preflight checks exact canonical Git revision/remote, inventory, tracked/untracked state, bytes and manifest; existing workloads and MJCF source paths were inspected. Design additionally requires validated read bytes during materialization and shared frozen sequence bytes, avoiding a preflight-then-reopen gap and lane-local rounding. Dynamic frame-zero bounds include the instance transform. S remains unresolved/disabled.
- Native scalar/provenance binding: one binary64-to-binary32 conversion, row-major rotation, actual Warp descriptor/CUDA ABI checking, camera digest not substituted for sequence digest, and new C3 downstream reference are coherent. Identical hashes alone cannot create live a2/C3/C4 authority.
- Setup custody: guardian before runtime import; adopt-before-next-leaf including staging and partial owners; monotonically expanding device scope; foreign/unknown scope quarantine; one immutable coverage seal followed by guarded barrier and quiescence. Existing native_setup performs direct waits and may retry via terminate; v2 correctly identifies that integration must replace those behaviors rather than silently reuse them.
- Timeline/arena: reuse C4a control, immutable R0 graph/R1 leaf4-sort refit graph, one terminal record and three/five completion-neutral diagnostic pairs, generation-specific exact authority, fixed-count acquire and no hidden GPU preparation after seal. These are implementation/test obligations, not properties demonstrated by this document.
- Resolve/close: actual CUDA `resolve()` directly synchronizes before its C4 transition, so the proposed split into guarded wait plus exact internal state commit is necessary. Existing lifecycle clean-close rejects historical submission; the new exact post-reclaim proof correctly preserves that latch and requires zero holds/leases, FREE slots and fully accounted tokens. ACK completion and actual process exit are correctly distinct. Failed ACK remains fault custody with retained graph.
- Scope/cleanup: C2c/d/e/f plus separate activation reviews are ordered; no performance thresholds or measurement selection introduced; no R2/S/host/result/a3b/baseline/product gate opened by testing. Losing-lane/selector and transitional harness cleanup remains tied to P12.4e and retained coverage.

## Verification, commands and observed results

Read-only inspection commands used `Get-Content -Raw -Encoding UTF8` and bounded `Select-Object`/line-number reads, `rg --files` and targeted `rg -n`, `Get-FileHash -Algorithm SHA256`, `git status --short/--porcelain=v1`, `git rev-parse HEAD`, `git branch --show-current`, `git merge-base 7717556 HEAD`, `git merge-base --is-ancestor 1bcb622 HEAD`, `git diff --stat/--name-status 7717556 HEAD`, candidate-only `git diff --check`, and `git show --check HEAD`. Inspection shells returned synchronously; no shell was intentionally left running.

Inspected source paths include `optical_worker/{protocol,worker,manifest,lifecycle,native_setup,native_contract,native_input,wait_protocol,wait_supervision}.py`, `optics/{native_arena,cuda_arena,warp_native_render}.py`, `benchmarks/optical/{schema,workloads,assets}.py`, `examples/mujoco_menagerie_robot_preview.py`, `tools/optical_pipeline_lab/camera_presets.py` and `tests/unit/optics/test_guarded_wait_core.py`. Camera design/design-review/code-review evidence was checked as a bounded prerequisite, not rerun as this review's output.

Candidate-only whitespace check passed. `git show --check HEAD` reports 16 Markdown trailing-double-space hard-break lines in the verbatim archived prior report. They are not changes to the candidate's operative design or a substantive design finding, and I did not rewrite historical evidence.

A few initial discovery queries guessed absent source/test names or used unsupported Windows wildcard path syntax; the correct paths were found with `rg --files`. One attempted inline Python invocation failed at parsing because Windows shell argument quoting removed string quotes. No test or runtime ran in that attempt; its shell observed exit 1 and no surviving child. These are tool invocation/discovery failures, not candidate failures.

External scratch only, created via apply_patch:

`C:/Users/AG/新建文件夹/c4_assets/c4-c2-integration-review-v2-probe.py`

Final probe SHA-256: `9e97ea01bc840afb8aa69f783f18ee838dc9659152377f6c09ff74fd760d63d8`.

Successful command from the repository:

```text
C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/c4-c2-integration-review-v2-probe.py C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design
```

The probe imports the actual CPU protocol/core, reproduces both illegal transitions above, and invokes pytest in the same interpreter for:

```text
tests/unit/optics/test_guarded_wait_core.py::test_normal_snapshot_golden_and_final_evidence
tests/unit/optics/test_guarded_wait_core.py::test_ordering_errors_cannot_commit_completion
```

Final execution: exit 0, 5 passed in 0.18 seconds. The earlier probe revision (F1 only) also exited 0 with 5 passed in 0.25 seconds; the final probe and digest identify the complete two-counterexample run. These selected tests are deterministic CPU state tests and launch no child processes or watchdogs. `-B`, `sys.dont_write_bytecode` and `-p no:cacheprovider` avoid repository artifacts. No mock or CPU result is presented as GPU lifetime proof.

## Process hygiene and environment limits

Local execution used Python 3.14; Torch/Warp are unavailable per packet. No GPU command, SSH, scp, remote process, asset download/install, background process, benchmark, baseline, result writer or implementation mutation was launched. Report/probe are the only review-created files, both outside the candidate. No processes belonging to other work were killed.

Process ledger: all read-only inspection tool calls completed synchronously with exit code 0; their outputs were returned without a running-session ID. The failed inline-Python attempt ran under PowerShell PID 20768, exited 1 and had no remaining child. Successful first probe ran under PowerShell PID 35660 with Python PID 32512 and exit 0. Final probe ran under PowerShell PID 25500 with Python PID 25016 and exit 0. Each driver queried its remaining direct children after Python exit and found none. Subsequent `Get-Process -Id 25016,25500,32512,35660,20768,25292 -ErrorAction SilentlyContinue` found none of those recorded review processes alive. A broader read-only Win32_Process snapshot showed another preexisting PowerShell PID 33876; it was neither attributed to this review nor terminated. Process snapshots cannot establish the health of unrelated remote work, and none is claimed.

Primary 183 timeout and fallback101 H200 availability were supplied in the packet; I did not claim a new transport check or use either host. Remote/GPU validation is unnecessary to prove the two design contradictions, but remains mandatory for subsequent implementation acceptance. Real Go2 materialization, all camera binary values, real dual-device restoration/quarantine, guardian finalizers, submitted GPU failure matrices, O(1) acquire counts, native correctness/parity and actual product source closure were not executed in this review. This is a design review, not implementation acceptance.

## Precise approval scope and follow-up

No approval of v2 is issued. Fix F1 and F2 in a new version and obtain a fresh independent design re-review of the exact candidate. The prior findings' closure above does not independently authorize integration implementation while the complete design remains Request changes. No need to modify the proven guarded-wait v1 implementation to satisfy these findings; the preferred repair is explicit compatible integration policy.

Any eventual approval of this design can authorize only the described separately reviewed C2c/d/e/f implementation checkpoints for one-slot R0/R1 none/native-only device work in independent single-lane Warp/CUDA workers. It cannot itself authorize live product preparation/submission, and test execution never supplies that authorization. A separate activation review remains necessary.

Still closed: every R2 row (including device-only until C4c3); S/physics publication; B3/host payload, external host holds and depth two; full-image IPC; normal reusable shutdown/joint C4d lifecycle; backend selection/comparison; native result authority, a3b, baseline generation, shipping/default-product activation; and C4 parent completion. Existing CPU camera approval and component approvals retain only their original exact scopes. C4c3, C4c4, C4d/B3, real-device acceptance and final cleanup/reviews remain required.
