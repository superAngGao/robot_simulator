# C4c2d1 guarded custody — independent code review v1

Initiative: q54-optical-lab-p12-4c4c2d-guarded-custody
Stage: code-review
Author: codex (fresh independent reviewer)
Version: v1
Date: 2026-10-03
Status: in_review
Related Files: guarded-custody design-request codex v2; runtime-integration design-request codex v3; guarded_lifecycle.py; guarded_evidence.py; wait_protocol.py
Owner Summary: Request changes. Two independently reproduced gaps remain: a terminal-handshake failure does not prevent the subsequent sync/release-authorization path, and combined evidence accepts contradictory worker/supervisor histories. Passing CPU suites do not close these counterexamples. No GPU/product/C4 gate is approved.

## Identity, revision, and verdict

Reviewer session: `/root/c4_c2d1_guarded_custody_code_review_v1`, independently spawned with no implementation conversation, no delegation, no candidate edits.

- Base: `8cbf8645123a3d7e3f6cf4bbc1434fa3a6612472`.
- Reviewed HEAD: `4ee33ab7a09720f0dc08f3e86bb199af59e58e40`.
- Local candidate: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Remote candidate: `/data3/ang.gao/robot_sim_c2d1_code_v1`, authorized fallback `ang.gao@192.168.80.101`.
- Both HEADs independently checked; both working trees clean before/after verification. Original dirty remote checkout untouched.
- Requirements: approved guarded-custody design v2 at `f1f4394`, original digest `00aa16cfef0dfd83b7f44081f845c9a3397d79435b4b3cf319e27c467d840983`, plus parent runtime-integration v3. Candidate changes that design's trailing whitespace only.
- Verdict: **Request changes** for F1 and F2 below. No approval of this code revision or a larger stage is granted.

Read AGENTS.md, collab/README.md, and collab/REVIEW_PRINCIPLES.md completely. Inspected requirements, actual diff, controller/reducer/client and guardian/resource paths, native_setup._abort, CPU harness and tests, source inventory/bootstrap boundary, and cleanup obligation. Author counts and descriptions were not used as approval evidence.

## Findings

### F1 — P2, blocking: terminal-ACK-time failure still allows a subsequent sync or release authorization

Locations: `optical_worker/guarded_lifecycle.py:134` (`_fail_guard`), `:285` (`_terminal_arm`), `:300` (`_release`), and especially `:353`–`:365` (post-ACK dispatch).

Trigger, reproduced without mutating candidate objects or monkeypatching production methods:

1. Construct the real controller/client/socket and real supervisor core/transport.
2. For the dirty case, complete one normal setup segment with an exact target descriptor/entered CPU fixture leaf.
3. Call terminate on the owner thread. After the supervisor commits emergency ARM but before writing its ACK, call `controller.begin_device_leaf(...)` from the peer thread.
4. This correctly raises `guarded setup belongs to another thread`; `_fail_guard` observes `_terminating` and sets `_terminal_fault=True`.
5. Deliver the real ACK and let terminate continue.

Observed: dirty-target termination invokes the sync fixture once **after** this recorded local failure. Only after the callable returns does line365 inspect `_terminal_fault` and report `emergency_sync_failed`. In the CPU-only variant, no sync occurs, but `_release` first moves the guardian to RELEASE_AUTHORIZED and emits resource_release_authorized, then notices the earlier fault and returns cleanup_failed_exit/remaining_quarantine. No cleanup had started at the time of the failure.

This violates design section4's ordered rule that any further error after emergency entry selects quarantine, and the requirement that reentrant failure cannot finish an earlier transition successfully. It can initiate an unnecessary native blocking operation after the controller has already detected failure; the eventual watchdog/quarantine does not make that extra operation legal. The CPU branch also incorrectly treats a pre-authorization failure as an authorized-cleanup failure. The terminal failure itself is only a local flag and has no faithful standalone trace event here.

Existing tests cover wrong-thread use during a normal segment and reentry **inside** sync or destructors. They do not inject failure while terminal ARM/ACK is pending, before either next side effect.

Repair direction: make a detected terminal fault monotonically invalidate the remaining terminal operation before sync entry or release authorization. Retain the entire graph and produce a truthful terminal failure/quarantine history; do not fabricate a sync failure for a sync that should never run or a cleanup failure before authorization. Keep failure observation and side-effect admission ordered against the supported cross-thread/reentry checks. If the approved event grammar cannot represent the required state, make that narrow discrepancy explicit rather than silently weakening the ordered failure rule.

Required proof: deterministic real socket/core/transport tests injecting wrong-thread and context/reentry errors during terminal ACK, for both cpu_cancel and target_sync. Assert zero sync, zero release authorization/release calls, original owner retention, full quarantine, no COMPLETE/new ARM, truthful replay, and single-use subsequent termination. Retain existing sync/destructor-reentry tests to distinguish failures that occur after those operations genuinely start.

### F2 — P2, blocking: combined artifact validation accepts incompatible execution histories

Location: `optical_worker/guarded_evidence.py:669`–`:706`, especially worker-attempt collection at :689 and the release-only restriction at :702.

Two independently reproduced examples pass `validate_custody_artifact`:

1. Worker hand-authored legal `arm_failure` terminal story: attempt `(0, setup)`, no ACK, terminal failure/full quarantine. Supervisor valid final history: committed and ACKed `(0, resolve)`, observed exit74. Same configuration/session/target. The pending attempt's conflicting role is never compared because the worker did not report an ACK.
2. Worker hand-authored legal `inner_abort` terminal story: `(0, setup)` ARMED, leaf failure, full quarantine, zero normal completions. Supervisor valid final history: ACKed setup0, received COMPLETE and wrote DISARMED for setup0, then committed/ACKed resolve1, observed exit74. The worker's terminal failure forbids both that COMPLETE and the new ARM, but the combined validator accepts them.

Each component is internally valid; the pair is impossible for this fixed worker/controller. This is not a demand for SHA-256 authenticity or rejection of a wholly rewritten consistent legal trace. It is a missing consistency check between two supplied histories that explicitly contradict one another, including independent supervisor evidence. It undermines the promised artifact proof for failed normal guards and handshake identity.

Cause: sets of supervisor ACKs check only facts positively claimed by the worker. Worker pending attempts are collected but not matched against the supervisor at the same sequence. Unmatched guard/COMPLETE rejection applies only when release_status is completed, so terminal quarantine histories may improperly be treated like arbitrarily lagging prefixes. The new tests exercise worker replay mutations but contain no direct negative cross-artifact matrix.

Repair direction: compare ordered per-sequence attempt/completion histories, including known role identity and known terminal/failure bounds. Preserve legitimate uncertainty: a pending ARM may have no received ARM, or its matching ARM/ACK may exist; a COMPLETE whose ACK failed may genuinely appear in the supervisor; an unfinished prefix may lag later valid supervisor work. Do not allow conflicting role identity, COMPLETE after a worker failed before sending it, or new attempts after a definitive terminal worker record.

Required proof: reject both examples, with rehashed component payloads if desired, and test the allowed pending/no-ARM, committed-but-unreceived ACK, failed COMPLETE-ACK, absent worker, and genuinely lagging prefix combinations. Run the real process artifact suite unchanged after repair.

## Independent verification and results

Environment: local `C:/Python314/python.exe`; remote `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python3.12`. No installs, GPU work, reset, garbage collection, pushes, or broad process kills.

SSH prefix used:

```text
ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101
```

Primary transport failure was supplied in the review packet; this reviewer used authorized fallback and verified its revision/tree rather than modifying or switching the original dirty checkout.

Commands/results:

1. `git rev-parse HEAD`, `git status --short/--porcelain`, `git diff --stat 8cbf864 HEAD`, actual diff/source inspection; final checks repeated. Clean exact HEAD.
2. `git diff --check 8cbf864 HEAD`: pass.
3. `git diff --quiet 8cbf864 HEAD -- optical_worker/lifecycle.py optical_worker/native_setup.py optical_worker/worker.py tools/optical_closed_worker.py tools/optical_closed_worker_supervisor.py`: exit0; protected sources unchanged.
4. Local `python -m ruff check` on guarded_lifecycle.py, guarded_evidence.py, wait_protocol.py, tests/support/guarded_custody_process.py, and the three new test files: all checks passed.
5. Remote, from the exact candidate: `timeout --signal=TERM --kill-after=5s 90s <python3.12> -B -m pytest tests/unit/optics/test_guarded_lifecycle.py tests/unit/optics/test_guarded_evidence_fixtures.py tests/unit/optics/test_guarded_custody_process.py -q -s -p no:cacheprovider`: **79 passed in 12.97s**, actual command exit0, no skips.
6. Remote expanded independent repeat with `tests/unit/optics/test_guarded_wait_core.py` prepended to the same command: **199 passed in 13.35s**, actual command exit0, no skips.
7. `C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/c2d1-review-runner.py`: independently authored external repro runner, 15-second subprocess timeout with exact-child kill/reap on failure. Actual child exit0. F1 dirty fixture logged `sync executed after terminal fault`; CPU fixture logged RELEASE_AUTHORIZED/remaining_quarantine; both F2 examples logged `incorrectly_accepted: true`. Here exit0 means successful reproduction of the bugs, not correctness.

Scratch files outside candidate: `c4_assets/c2d1-review-repros.py` and `c4_assets/c2d1-review-runner.py`. Repros reuse test fixture construction and handwritten stories but use actual production controller/client/core/transport/validators. F1 injects a public wrong-thread call at a deterministic real peer-ACK boundary; F2 creates individually legal supervisor evidence through its core. No arbitrary reflection attacker or actual CUDA execution is claimed. Two preliminary short local repro executions also exited0 before the final externally bounded rerun.

Checks supporting the bounded design (not an approval despite findings):

- Actual native_setup._abort calls the overridden terminate, not a copied upload loop or old terminate delegate. In-process and real-process tests cover the inner exception path and retained exception graph.
- One inherited guardian/permit implementation is reused; returned owners are adopted before foreign/unknown scope rejection. Failed normal segments retain the original graph, have no emergency retry/sync/COMPLETE, and terminal custody is append-only.
- Ticket checks bind exact registered identity and original ACK fields; copied/changed/stale tickets are tested. C0 barrier/clean-close/measured-admission entrypoints remain rejected on the subtype.
- Replay includes full descriptor scope and owner facts, so missing subsequent notifications cannot erase foreign/unknown possibility. Release authorization, cleanup start, completion, remaining custody, terminal intent, and supervisor exit remain separate in normal covered paths.
- All22 real subprocess cases independently passed: CPU and target sync/cleanup/pre-exit hangs were killed/reaped by the real supervisor while their parent remained alive; no test outer timeout was credited as exit. Quarantine destructor/atexit sentinels and failed ACK paths were checked.
- Native source inventory additions are limited to the two new modules; the unchanged six-core CPU bootstrap is not expanded into guarded runtime adoption. Cleanup ledger records eventual sole-controller adoption and deletion obligations.

## Process hygiene / exact observed exits

Final bounded local repro: runner23272, parent14368; child23672, parent23272, observed exit0. Preliminary repro PIDs33376(parent9952) and34940(parent35612) also returned exit0. Final exact Get-Process lookup for all four PIDs returned no processes. Peer threads are joined by the fixture; no child process was launched by the repro itself. Ruff command handle returned exit0; its OS PID was not sampled.

Remote first run ledger below is `case: supervisor / worker / worker returncode`; all supervisor returncodes were0. The harness checked each worker identity.json parent equals its exact supervisor PID and final supervisor evidence ends in process_exited.

```text
cpu_success: 2316546 / 2316547 / 0
cpu_cleanup_hang: 2316548 / 2316549 / -9
cpu_exit_hang: 2316576 / 2316577 / -9
normal_success: 2316593 / 2316594 / 0
normal_leaf_raise: 2316595 / 2316596 / 74
inner_abort: 2316601 / 2316602 / 74
pre_leaf_raise: 2316603 / 2316604 / 74
foreign: 2316605 / 2316606 / 74
unknown: 2316611 / 2316612 / 74
sync_raise: 2316613 / 2316614 / 74
sync_hang: 2316619 / 2316620 / -9
cleanup_raise_before: 2316671 / 2316679 / 74
cleanup_raise_after: 2316685 / 2316686 / 74
normal_no_ack: 2316687 / 2316692 / 74
normal_wrong_ack: 2316699 / 2316703 / 74
normal_ack_eof: 2316704 / 2316706 / 74
normal_stalled_disarm: 2316707 / 2316712 / 74
terminal_no_ack: 2316713 / 2316714 / -9
terminal_wrong_ack: 2316719 / 2316720 / 74
terminal_ack_eof: 2316722 / 2316728 / 74
target_cleanup_hang: 2316729 / 2316730 / -9
target_exit_hang: 2316758 / 2316759 / -9
```

Expanded run launch shell PID2321063, parent2320979, started remote09:46:53. Command returned exit0. The short-lived timeout/pytest PIDs were not captured before exit; this is a launch-ledger limitation, not a claim of unidentified residuals. Expanded worker ledger, again all supervisor returncodes0 and exact worker-parent checks passed:

```text
cpu_success: 2321156 / 2321157 / 0
cpu_cleanup_hang: 2321161 / 2321162 / -9
cpu_exit_hang: 2321178 / 2321180 / -9
normal_success: 2321203 / 2321204 / 0
normal_leaf_raise: 2321205 / 2321210 / 74
inner_abort: 2321211 / 2321212 / 74
pre_leaf_raise: 2321214 / 2321215 / 74
foreign: 2321216 / 2321220 / 74
unknown: 2321221 / 2321222 / 74
sync_raise: 2321223 / 2321224 / 74
sync_hang: 2321229 / 2321230 / -9
cleanup_raise_before: 2321246 / 2321247 / 74
cleanup_raise_after: 2321248 / 2321249 / 74
normal_no_ack: 2321254 / 2321262 / 74
normal_wrong_ack: 2321272 / 2321273 / 74
normal_ack_eof: 2321277 / 2321278 / 74
normal_stalled_disarm: 2321283 / 2321284 / 74
terminal_no_ack: 2321285 / 2321286 / -9
terminal_wrong_ack: 2321291 / 2321292 / 74
terminal_ack_eof: 2321293 / 2321298 / 74
target_cleanup_hang: 2321299 / 2321300 / -9
target_exit_hang: 2321316 / 2321317 / -9
```

Final remote `ps -C python3.12,python,timeout -o pid,ppid,lstart,args` contained only pre-existing unrelated Python626988(parent204324, started Sep23), left untouched. Final `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` was empty. The earlier author broad run was visible during the first check and was not touched or counted as reviewer-owned evidence. Final tree/HEAD recheck remained clean/exact.

Two initial read-only remote process-query attempts had PowerShell/SSH quoting errors (one invalid ps PID expression, one grep expression interpreted by the remote shell); corrected simple ps commands supplied the final observations. They did not alter the candidate or launch any workload successfully. A negative local Get-Process lookup returned command status1 because none of the queried processes remained; it is not a test failure.

## Limits and exact scope

This review is the C4c2d1 **CPU guarded-custody implementation interface** checkpoint, not C2d integration or C4 completion. F1/F2 block approval even within that scope. Fix and freeze a new revision for fresh independent re-review, retaining these repros.

The remote source-policy/catalog/bootstrap/C0/C1c broad suite was not independently rerun in this session after two decisive failures were established; source boundaries and protected no-diffs were inspected, and the author broad run is not substituted as independent review evidence. Full relevant regressions remain necessary for repaired-candidate acceptance. The fixed CPU sync fixtures prove control flow only, not CUDA/Warp completion or real allocation safety. No full trusted-history rewrite authenticity, hostile Python/native sandbox, or unsampled global absence of processes is claimed.

Still closed: source-frozen guarded-role adoption and module/session custody before CUDA-capable import; real Warp/Torch upload and mandatory dual-GPU foreign/pre-return/unknown matrices; final graph/barrier/warmup/readiness composition; arenas/C2e/C2f; product GPU preparation, a3b, result/baseline, R2/host/S, C4c3/c4, C4d/B3, shipping/activation, and C4 completion. Existing design approvals are not reopened or extended by this report.
