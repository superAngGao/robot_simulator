# C4c2d1 guarded custody — independent code review v4

Initiative: q54-optical-lab-p12-4c4c2d-guarded-custody
Stage: code-review
Author: codex (fresh independent reviewer)
Version: v4
Date: 2026-10-03
Status: in_review
Related Files: custody design-request v2; repair-design v2; code-review v1/v2/v3; repair-design-review v1/v2; runtime-integration design-request v3
Owner Summary: Request changes. Original F1–F5 counterexamples now close, and the independent 15-file CPU regression run passes 751 tests without skips. One adjacent exception-composition failure remains: a previously observed thread/operation failure blocks conservative unknown-scope bookkeeping in the actual preparation handler, preventing its inner abort and losing the unknown-scope fact. No integrated runtime, GPU, product or C4 gate is approved.

## Identity and verdict

Reviewer session: `/root/c4_c2d1_guarded_custody_code_review_v4`, fresh independent code review, no delegation and no candidate edits. Author claims and prior verdicts were not treated as proof.

- Immediate base: `6f87d23a5ce2c1acefc8795caf79709fa7ea33e2`.
- Overall base: `8cbf8645123a3d7e3f6cf4bbc1434fa3a6612472`.
- Reviewed frozen HEAD: `6b5080fbfa00309c757ee3467fa6cb53b78fd57d`.
- Local candidate: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Remote isolated candidate: `/data3/ang.gao/robot_sim_c2d1_code_v4`, authorized fallback `ang.gao@192.168.80.101`.
- Exact HEAD and clean working trees verified initially and finally on both hosts. Original dirty remote checkout was not touched.
- Accepted repair-design v2 SHA-256: `01a464293f4474f1665ee0ae3d2eda5477df43cec2f2ebb7374dcf87195656e2`.
- Archived code-review v3 SHA-256: `add59aab2abab30d7da299c123b58fcba174bd224797225ed1a1da96d5c4d128`.
- Verdict: **Request changes**, for F6 below. No code approval at this revision.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely. Read custody design v2, accepted repair v2, full archived code reviews v1–v3 and repair-design reviews v1/v2, and relevant parent runtime-integration v3 source/setup/custody/validation requirements. Inspected immediate and overall boundaries, controller/reducer/client and guardian call/state flow, actual native_setup._abort and both preparation handlers, new tests, process harness, and source catalog.

## Finding

### F6 — P2, blocking: a recorded failure prevents conservative unknown-scope intake before the real preparation abort

Location: `optical_worker/guarded_lifecycle.py:279`–`:282`, where `mark_unknown_device_scope` unconditionally uses `_check_running`; failed-state rejection at `:161`–`:165`. Affected unchanged callers: `optical_worker/native_setup.py:284`–`:296` and `:464`–`:476`. Each actual prepare function first records unknown scope when its runtime/target is unavailable, then calls `_abort`.

Independent reproduction executes both actual `prepare_cuda_native_input_for_closed_worker` and `prepare_warp_native_input_for_closed_worker`, with lane-matched production controllers and real client/socket/supervisor-core/transport. The external CPU fixture intercepts only the selected runtime import using a temporary `sys.meta_path` finder. At that actual import boundary it deterministically makes a foreign-thread public `require_adopted()` call, observes its required rejection, and raises a fixed ImportError. It does not modify candidate source, production methods, controller fields or the state machine; no Torch/Warp import succeeds and no GPU call executes.

The preparation call starts healthy, successfully admits/enters its runtime-import leaf, then receives the supported cross-thread failure while that leaf is in flight. Thus this is not merely an attempt to start preparation after a known failure. The normal preparation exception handler runs with runtime/target unavailable and attempts `mark_unknown_device_scope`. `_check_running` raises `WorkerSetupProtocolError("guarded setup is failed or terminal")` before recording the conservative fact or reaching `_abort`.

Both lanes produce this state inside the still-active setup context:

```text
exception = WorkerSetupProtocolError (not NativeWorkerPreparationAborted)
termination_evidence = None
controller_state = RUNNING
guardian_state = CONSTRUCTING
guard_state = FAILED
guarded_segment_failed = true
unknown_device_scope = false
custody = none
disposition = null
```

The source owner remains strongly retained, and the replacement exception's context is the original import error. When outer context exit is subsequently permitted, it commits full quarantine, but selects `quarantined_process_exit` with unknown=false rather than the preparation handler's intended unknown-scope disposition. There is zero sync, new ARM, COMPLETE or release. This finding does **not** claim premature destruction or resumed GPU work: the immediate strong controller hold already prevents that. It concerns the promised pre-unwind terminal transition/result and truthful conservative scope evidence.

A positive control runs the same actual import-failure path without the intervening thread fault. Both lanes then raise NativeWorkerPreparationAborted, record unknown=true, and have exact terminal evidence/full quarantine/`multi_or_unknown_device_quarantine_exit` before context exit. This isolates failure-observation composition from the ordinary import-failure path.

The accepted custody design requires the real preparation exception boundary to complete before outer unwinding, and foreign/unknown facts to remain conservative and monotonic. The current reducer already allows `unknown_scope_set` while controller RUNNING even if guard failure is sticky (`guarded_evidence.py`, unknown-scope branch around :363); the live wrapper blocks this conservative tightening as though it were ordinary new work. The F5 exception-owner exception does not address the preceding handler operation.

Why checks miss it: the new 14-case matrix and three process faults call `_abort` directly after the initial failure. Their explicit-unknown case records unknown while healthy, then raises, rather than recording unknown after failure observation. Therefore they skip the actual prepare handler's mark-unknown-before-abort ordering. All 751 existing tests can pass while both actual preparation functions exhibit F6.

Repair direction: distinguish owner-thread, pre-terminal conservative scope/failure bookkeeping from ordinary work admission. Allow the existing preparation handler to record unknown scope after observed failure without clearing failure, issuing a permit, importing a runtime, reviving a normal segment, or authorizing sync/release. Preserve wrong-thread and terminating/post-intent rejection, sealed-state custody, single-use termination and the one reducer/guardian. Do not edit protected C0 code, copy the upload functions, or use legacy termination as a fallback.

Required proof: exercise both actual prepare functions with fixed CPU import/target-resolution failures and deterministic prior public-operation/thread failure while the leaf is in flight. Assert NativeWorkerPreparationAborted with exact terminal evidence before leaving the outer context, unknown=true and the correct full-quarantine disposition, original source/exception graph retention, zero sync/new ARM/COMPLETE/release, and one terminal transition whether the abort is caught or propagated. Include healthy import-failure controls and negative wrong-thread/terminal/post-intent scope calls, plus original F1–F5 and relevant process/source/C0 regressions. This is CPU exception-composition proof only, not authorization to implement or validate the still-closed GPU integration slice.

The author independently suggested this adjacent mark-unknown boundary during this session. F6 is grounded in this reviewer's source inspection and independently authored/executed reproduction, not that suggestion alone.

External reproduction: `C:/Users/AG/新建文件夹/c4_assets/c2d1-code-v4-independent-repros.py`, SHA-256 `4a4545f896a0773417b5a400d93a6ce861897b91f8f1ed08714e1547a56e06e0`. Its actual exit0 means the positive controls and expected current counterexamples were collected/asserted, not candidate correctness.

## Earlier findings and high-risk checks

- F1 remains closed: unmodified original delayed-terminal-ACK repro produces zero sync and no release authorization in both CPU and target-dirty cases. The real late ACK is retained, terminal failure is sticky, and custody is full quarantine.
- F2 remains closed: both original incompatible artifact pairs explicitly reject, including pending-role conflict and forbidden COMPLETE/new ARM after terminal normal failure. The original runner exits1 at its first correct rejection; the v2 runner independently catches both F2 rejections before its later early exit. Neither early exit is represented as a full-tail pass.
- F3 remains closed: independent explicit-unknown normal exit with no intervening public operation, plus caught foreign/unknown returned-owner failures, rejects context completion, preserves owners, and forbids later resolve entry. Normal-work reducer predicates and irreversible cross-history bounds remain intact.
- F4 remains closed across all three disabled legacy methods and require_adopted at delayed ACK (CPU/dirty), admitted sync, and real destructor cleanup. Pre-admission rejection prevents sync/release; in-sync rejection produces failed sync/full custody; cleanup reentry produces failed cleanup/remaining custody.
- F5's original four probes now report returned_abort=true and terminal_evidence_before_context_exit=true. The new candidate matrix independently passes all 14 failed/scope/sealed/caught/propagated cases. It checks exact exception identity retention, evidence identity, full guardian transfer, zero extra effects and unchanged terminal history after unwind. The three retention-negative cases reject non-exceptions/wrong role/wrong operation without adopting them. Ordinary admission is not reopened by the narrow exception-retention branch.
- The retained design's terminal-entry/custody and ARM/sync/release/outcome/intent transitions still use one RLock. Real ACK/native/destructor operations remain outside that lock. Ticket checks use exact registered ACK-created object identities, roles/session/target, not descriptive records. Whole versus remaining quarantine remains distinguished by actual release entry.
- There is one inherited guardian/permit/release implementation; no copied upload loop or C0 termination delegate. Protected C0 lifecycle/native_setup, worker and production launchers are unchanged from overall base.
- Immediate production delta is the 14-line exception-retention branch and its catalog repin. Parsed catalog comparison confirms unchanged lanes and AST policy, exactly one changed source digest: guarded_lifecycle from `f1dd11570b9c6e7638548670628f038d33c2ff3f9dc1159020784b91d3d90106` to `aa93c6a50bd8e15ecd73a998089bc4d208bffdb3269b44abc8061330b80d60d3`. Remote source bytes match. Guarded evidence stays `1c4a23d630938eb28e0aeb1da070b9f4ea8b32b7f60097af3fd020b3aaa301c5`. Catalog inclusion does not activate captured-source runtime adoption.

## Independent verification

Local environment: Windows PowerShell, `C:/Python314/python.exe`, no Torch/Warp. Remote interpreter: `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python`. No installs, GPU workloads, push/reset, broad kills or candidate edits. Scratch code/report stay outside the candidate. The author's active remote run was allowed to finish before this independent run; its output is not substituted for reviewer evidence.

SSH prefix:

```text
ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101
```

Primary 183 transport failures were supplied in the packet; this reviewer used authorized fallback after exact revision/tree checks.

Executed:

1. Initial/final HEAD and porcelain status; immediate/overall diff inspection; accepted design/archive/repro hash checks. Both candidate trees remain clean at exact HEAD.
2. `git diff --check 8cbf864 HEAD`: exit0. `git diff --quiet 8cbf864 HEAD -- optical_worker/lifecycle.py optical_worker/native_setup.py optical_worker/worker.py tools/optical_closed_worker.py tools/optical_closed_worker_supervisor.py`: explicit exit0.
3. `C:/Python314/python.exe -B -m ruff check` on guarded_lifecycle.py, guarded_evidence.py, wait_protocol.py, the three guarded lifecycle/evidence/process tests and tests/support/guarded_custody_process.py: all checks passed, exit0.
4. Unmodified local original repro runner: child exit1 at first correctly rejected F2 pair; F1 outputs independently inspected. Unmodified v2 repro: child exit1 at correct explicit-unknown context rejection, after explicitly verifying both F2 cases. Unmodified v3 repro (SHA `6b1bf9c692f0e19b0a53a2dd1933199e75a9e72aae2384c940a2948f961a7be3`): child exit0; all original F3/F4 predicates and four F5 pre-unwind outputs checked.
5. New external v4 repro above: child exit0, both lane-matched preparation functions reproduce F6 and healthy controls pass. The initial scratch run used the helper's default cuda lane for both calls; the final rerun explicitly constructs each controller with the matching lane and reproduces the same result. All import finders are removed, peer threads joined, and sockets closed.
6. Independent remote command below: **751 passed in 170.08s**, no skips, actual command exit0 (tool session81915). No outer timeout fired. Real Go2 fixture enabled. Warp deprecation warning is not a failure or proof of GPU execution.

```text
cd /data3/ang.gao/robot_sim_c2d1_code_v4
ROBOT_SIM_TEST_GO2_MODEL_DIR=/data3/ang.gao/c4_asset_frozen/out/external/mujoco_menagerie/unitree_go2 timeout --signal=TERM --kill-after=15s 420s /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -m pytest tests/unit/optics/test_guarded_lifecycle.py tests/unit/optics/test_guarded_evidence_fixtures.py tests/unit/optics/test_guarded_custody_process.py tests/unit/optics/test_guarded_wait_core.py tests/unit/optics/test_guarded_wait_process.py tests/unit/optics/test_native_bootstrap.py tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/optics/test_sacrificial_sources.py tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_leaf_descriptors.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py tests/unit/benchmarks/test_optical_benchmark_contracts.py -q -s -p no:cacheprovider
```

## Actual exits and process hygiene

Local original runner36196(parent36356), child27956(parent36196), actual child exit1. V2 runner32720(parent36356), child36496(parent32720), exit1. V3 runner30516(parent36356), child22940(parent30516), exit0. Initial v4 scratch runner34168(parent35256), child36216(parent34168), exit0. Final lane-matched v4 runner13240(parent36408), child17304(parent13240), exit0. All ten exact PIDs absent at final Get-Process check. The existing 15/20/25-second child bounds did not fire; no kill was needed. Ruff's short-lived PID was not sampled; command exit0 was observed.

Remote timeout2451309(parent2451223), pytest2451330(parent2451309), start11:05:25 remote. Sampled source-bootstrap child2453645(parent2451330). Actual command completion exit0 observed, not inferred from absence. The following 25 process cases are `case: supervisor / worker / actual worker returncode`; all supervisors returned0. The tests compare recorded worker parent identity against the exact supervisor PID, verify parent remained alive, and require final actual supervisor process_exited evidence.

```text
cpu_success: 2451563 / 2451566 / 0
cpu_cleanup_hang: 2451569 / 2451578 / -9
cpu_exit_hang: 2451605 / 2451606 / -9
normal_success: 2451631 / 2451632 / 0
normal_leaf_raise: 2451633 / 2451634 / 74
inner_abort: 2451635 / 2451636 / 74
foreign_inner_abort: 2451639 / 2451640 / 74
unknown_inner_abort: 2451645 / 2451652 / 74
invalid_permit_inner_abort: 2451693 / 2451694 / 74
pre_leaf_raise: 2451695 / 2451699 / 74
foreign: 2451700 / 2451701 / 74
unknown: 2451702 / 2451703 / 74
sync_raise: 2451708 / 2451709 / 74
sync_hang: 2451714 / 2451715 / -9
cleanup_raise_before: 2451728 / 2451729 / 74
cleanup_raise_after: 2451734 / 2451735 / 74
normal_no_ack: 2451739 / 2451741 / 74
normal_wrong_ack: 2451742 / 2451743 / 74
normal_ack_eof: 2451748 / 2451749 / 74
normal_stalled_disarm: 2451753 / 2451755 / 74
terminal_no_ack: 2451756 / 2451757 / -9
terminal_wrong_ack: 2451761 / 2451766 / 74
terminal_ack_eof: 2451770 / 2451772 / 74
target_cleanup_hang: 2451773 / 2451774 / -9
target_exit_hang: 2451786 / 2451791 / -9
```

Final exact ps query for timeout/pytest/sampled source child and all50 supervisors/workers returned only the header. Final `ps -C python,python3.12,timeout -o pid,ppid,lstart,args` showed only unrelated pre-existing Python626988 and training chain2434115/2434217/2434311, all untouched. Other short-lived broad-suite descendants were not individually sampled; this is a ledger limitation, not a claim that every OS PID was recorded.

Before/after remote GPU-process queries showed only unrelated training2434311. Local query showed unrelated desktop applications and none of the ten reviewed runner/child IDs. No GPU work was run or authorized by this review. These are scoped hygiene observations, not proof of global idleness or impossibility of hangs.

Read/diagnostic issues: initial default-encoding Chinese output was reread fully as UTF-8; oversized combined reads were split. The one-line catalog diff exceeded the output budget; focused parsed comparison supplied the actual result. Two initial parsed comparisons used nonexistent property names (`sources`, `source_inventory`) and produced PowerShell null errors; the corrected `source_digests` comparison confirms the single repin above. These did not mutate files or constitute candidate test failures.

## Scope and limitations

Request changes applies only to exact6b5080f's C2d1 CPU controller/reducer/client/native-setup exception-composition slice. Original F1–F5 cases are closed, but F6 blocks acceptance. Preserve this report/repro, repair within the accepted boundary, freeze a new revision and obtain fresh independent review. Passing test totals do not override the reproduced contract failure.

Still closed: integrated C2d source-frozen guarded runtime adoption and module/session custody before CUDA-capable imports; real Warp/Torch upload and mandatory target/ambient/foreign/pre-return/unknown multi-GPU fault matrices; full graph/barrier/warmup/readiness composition; arenas/C2e/C2f; product GPU preparation; a3b/result/baseline authority; R2/host/S; later C4 and C4d/B3; shipping/activation and C4 completion. Existing cleanup/adoption/removal obligations and historical approvals are unchanged.

CPU fixtures prove control flow, CPU-owner retention, record consistency and actual process supervision, not real CUDA/Warp completion or allocation-release safety. The new import hook is a fixed external CPU failure injector, not a claimed source-frozen runtime implementation or hostile-Python attack. Parsed records and public hashes are not authentication or live authority. No fully rewritten-history detection, hostile-Python sandbox, or unsampled global process-absence claim is made.
