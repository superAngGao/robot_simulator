# C4c2d1 guarded custody — independent code review v3

Initiative: q54-optical-lab-p12-4c4c2d-guarded-custody
Stage: code-review
Author: codex (fresh independent reviewer)
Version: v3
Date: 2026-10-03
Status: in_review
Related Files: custody design-request v2; repair-design v2; code-review v1/v2; repair-design-review v1/v2; runtime-integration design-request v3
Owner Summary: Request changes. Original F1/F2 and F3/F4 counterexamples now fail closed, and the independently run broad CPU suite passes 730 tests without skips. One exception-boundary regression remains: a previously recorded operation failure or compromised scope prevents native_setup._abort from adopting its exception and reaching terminal disposition before context unwinding. No broader runtime/GPU/product/C4 gate is approved.

## Identity and verdict

Reviewer session: `/root/c4_c2d1_guarded_custody_code_review_v3`, fresh independent code review; no delegation, implementation edits, or inherited implementation conversation.

- Immediate base: `90f50bb3ef67dd99182e4da0e5bc8479cd76e53b`.
- Overall C2d1 base: `8cbf8645123a3d7e3f6cf4bbc1434fa3a6612472`.
- Reviewed frozen HEAD: `6f87d23a5ce2c1acefc8795caf79709fa7ea33e2`.
- Local candidate: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Remote candidate: `/data3/ang.gao/robot_sim_c2d1_code_v3`, authorized fallback `ang.gao@192.168.80.101`.
- Both exact HEADs and clean working trees checked initially and finally. Original dirty remote checkout was not touched.
- Accepted repair-design v2 SHA-256 independently checked: `01a464293f4474f1665ee0ae3d2eda5477df43cec2f2ebb7374dcf87195656e2`.
- Archived code-review v2 SHA-256 independently checked: `6c555b14a883bb342df9bc1632d55638ba2794ef48b25ac163742f1c50346289`.
- Verdict: **Request changes**, for F5 below. No code approval is granted at this revision.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely; read custody requirements, accepted repair, parent runtime-integration requirements and the full archived code-review v1/v2 and repair-design-review v1/v2 reports. Inspected actual immediate and overall implementation boundaries, controller/reducer/client state and exception flow, inherited guardian/permit ownership, native_setup._abort and its real preparation callers, process harness, source catalog and cleanup obligation. Author notes and test counts were claims to check, not approval evidence.

## Finding

### F5 — P2, blocking: failure observation now blocks the real inner abort from reaching terminal custody

Locations: `optical_worker/guarded_lifecycle.py:253`–`:256` (`adopt_host_owner` unconditionally calls `_check_running`), `:154`–`:165` (compromised/failed rejection), and `:178`–`:189` (new immediate public-error observation). The affected unchanged caller is `optical_worker/native_setup.py:106`–`:128`, especially exception adoption at :112–:117 before terminate at :118.

Reproduction through actual controller/client/socket/supervisor core/transport and the actual `_abort`, without production monkeypatching or object mutation:

1. Enter a normal setup segment.
2. Trigger and catch a supported setup failure: `begin_device_leaf(descriptor(0))` for configured target1, a foreign returned owner, or an invalid permit. The new public-operation wrapper has already recorded `guard_failed`. Alternatively, explicitly mark unknown scope and then raise a leaf exception.
3. In that exception handler, call `_abort(controller, exception=original, sync=...)`, as the existing preparation functions do.
4. Inspect state inside the segment, before its context exit.

Observed in all four variants: `_abort` raises a replacement `WorkerSetupProtocolError` from its initial `adopt_host_owner` call. For compromised cases the message is `compromised scope forbids further normal work`; for the invalid-permit case it is `guarded setup is failed or terminal`. `_abort` never reaches `terminate` and never returns `NativeWorkerPreparationAborted`. At that exact pre-unwind boundary:

```text
termination_evidence = None
guardian.state = CONSTRUCTING
controller_state = RUNNING
guard_state = FAILED
guarded_segment_failed = true
custody = none
disposition = null
```

The context exit eventually quarantines when allowed to run. That does not restore the promised inner exception boundary or exception result. This report does **not** claim premature release: `_fail_guard` has already strongly retained the controller and failures; normal operations remain rejected, and no extra sync/COMPLETE was observed. The concrete failure is that the real native helper cannot complete its required terminal transition/evidence before unwinding, and callers receive the wrong exception instead of its disposition-bearing `NativeWorkerPreparationAborted`.

The accepted custody design sections1–3 make the unmodified `_abort` the actual exception boundary and require immediate freeze/quarantine/evidence before context exit. Prior F3 explicitly required preserving this behavior while forbidding ordinary work after compromise. Conservative retention of the already-raised exception is not a new ordinary setup operation. Routing it through the same healthy-running predicate now prevents that retention/termination composition for exactly the newly sticky failures.

Why existing tests miss it: `test_failed_leaf_never_retries_and_holds_exception` at test_guarded_lifecycle.py:142 invokes `_abort` after a plain RuntimeError, while the controller is still healthy/armed. It does not first trigger the public-operation observer or scope boundary. New caught-error and compromised-scope tests at :651 and :760 instead rely on outer context exit. The real process harness's `inner_abort` likewise uses plain RuntimeError; its `foreign`/`unknown` branches unwind directly without calling `_abort`. Thus all 730 tests can pass without exercising the crossed condition.

Repair direction: preserve a narrowly scoped, retention-only path for the original native-setup exception graph after failure/compromise so the existing `_abort` can reach the guarded terminate override. Do not reopen normal host/device work, clear failure, use the C0 termination fallback, change protected C0 source, copy upload loops, or rely solely on outer unwinding. Use the existing guardian and terminal holds, including their sealed-state behavior. The precise bounded implementation should make custody intake distinct from work admission without minting a new caller-controlled authority.

Required proof: inside an active setup segment, invoke the real `_abort` after foreign/unknown descriptors, returned foreign/unknown owners, explicit unknown scope, and already-observed public-operation errors. Before exiting the context, assert it returns the proper NativeWorkerPreparationAborted carrying the exact terminal evidence, guardian full quarantine, original owner/exception graph retention, zero sync/new ARM/COMPLETE/release, and exactly one termination/custody transition. Catch or propagate the returned exception and verify context unwinding adds no second transition. Retain normal RuntimeError and sealed-guardian coverage. Rerun original F1–F4 and the real process/source/C0 matrix on the repaired frozen candidate.

External reproduction: `C:/Users/AG/新建文件夹/c4_assets/c2d1-code-v3-independent-repros.py`, SHA-256 `6b1bf9c692f0e19b0a53a2dd1933199e75a9e72aae2384c940a2948f961a7be3`. Its final four `abort_probe` records include the exact pre-context-exit observations above. Exit0 means the probe completed, not that F5 is correct behavior. This reviewer identified and reproduced the issue independently; the author subsequently confirmed the inspected call-order conflict while leaving the candidate frozen.

## Closure of earlier findings and high-risk inspection

- F1 remains closed: original delayed-terminal-ACK wrong-thread calls produce zero sync, no release authorization, truthful late guard_armed, and full quarantine in CPU and target-dirty cases.
- Both original F2 pairs reject independently: unacknowledged role conflict and COMPLETE/new ARM after a definitively failed terminal worker. The original runner exits1 at the first correct rejection; that exit alone is not treated as proof of the second case. The v2 independent script explicitly verified both before its later early exit.
- F3's original live cases are closed. The new independent repro marks explicit unknown and exits normally with **no intervening public operation**; it also catches foreign/unknown returned-owner rejection then immediately exits. Each context rejects, retains owners, produces zero normal completions/full quarantine, and rejects a later resolve body. This supplements the candidate's new test, which additionally attempts a leaf before exit. The reducer independently rejects normal ARM/body/COMPLETE/DISARMED after compromise; early descriptor/owner/explicit-unknown prefix compatibility tests pass.
- F4 is closed for all three disabled APIs and require_adopted. Independent probes exercise delayed ACK for both CPU/dirty cases, admitted sync, and actual destructor cleanup. Pre-admission failures mean zero subsequent sync/release and full custody; in-sync rejection produces failed sync/full custody; cleanup rejection produces failed cleanup/remaining custody. Post-intent evidence and single-use rejection tests pass.
- Terminal-entry/custody publication, ARM/sync/release admission, outcomes and intent use the shared RLock. Executed scheduling tests cover entry and success boundaries and verify the lock is not held across ACK/native/destructor calls. Truthful late terminal ACK and full versus remaining retention remain distinct.
- Exact ticket/role/session/target and registered-original-ACK identity checks, permit single use, source-replay shape constraints, healthy lagging prefixes, pending/unreceived ACK and failed COMPLETE-ACK uncertainty were inspected and tested. One existing guardian/permit/wait implementation remains; there is no copied release/upload loop or fallback termination.
- Protected C0 lifecycle/native_setup, worker and production launchers are unchanged from overall base. Catalog immediate delta contains only two guarded-module digest changes; lane inventories, schema and AST policy remain unchanged. Remote bytes match lifecycle `f1dd11570b9c6e7638548670628f038d33c2ff3f9dc1159020784b91d3d90106` and evidence `1c4a23d630938eb28e0aeb1da070b9f4ea8b32b7f60097af3fd020b3aaa301c5`. Catalog inclusion is not captured-source runtime adoption.

## Independent verification

Local environment: Windows PowerShell, `C:/Python314/python.exe`, no Torch/Warp. Remote interpreter: `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python`. No installation, GPU workload, push, reset, broad kill or candidate edit. All scratch files are outside the candidate. The author broad run was allowed to finish before reviewer remote execution; its results are not substituted for this independent run.

SSH prefix used:

```text
ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101
```

Primary183 transport failure was supplied in the task packet. Fallback revision and tree were independently verified.

Executed commands/results:

1. `git rev-parse HEAD`, `git status --porcelain`, immediate/overall `git diff` inspection, and SHA-256 checks; final HEAD/tree checks repeated. Exact clean candidate throughout.
2. `git diff --check 8cbf864 HEAD`: pass. `git diff --quiet 8cbf864 HEAD -- optical_worker/lifecycle.py optical_worker/native_setup.py optical_worker/worker.py tools/optical_closed_worker.py tools/optical_closed_worker_supervisor.py`: explicit exit0.
3. `C:/Python314/python.exe -B -m ruff check` on guarded_lifecycle.py, guarded_evidence.py, wait_protocol.py, three guarded test files and tests/support/guarded_custody_process.py: all checks passed, exit0.
4. Remote exact candidate, command below: **730 passed in 168.42s**, no skips, actual command exit0 (tool session65561). Real Go2 fixture enabled. No outer timeout fired. A Warp deprecation warning is not a failure or GPU execution proof.

```text
ROBOT_SIM_TEST_GO2_MODEL_DIR=/data3/ang.gao/c4_asset_frozen/out/external/mujoco_menagerie/unitree_go2 timeout --signal=TERM --kill-after=15s 420s /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -m pytest tests/unit/optics/test_guarded_lifecycle.py tests/unit/optics/test_guarded_evidence_fixtures.py tests/unit/optics/test_guarded_custody_process.py tests/unit/optics/test_guarded_wait_core.py tests/unit/optics/test_guarded_wait_process.py tests/unit/optics/test_native_bootstrap.py tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/optics/test_sacrificial_sources.py tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_leaf_descriptors.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py tests/unit/benchmarks/test_optical_benchmark_contracts.py -q -s -p no:cacheprovider
```

5. `C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/c2d1-review-runner.py`: original15-second bounded child, actual exit1 at the now-rejected first F2 pair. Both prior F1 outputs inspected; no timeout.
6. `C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/c2d1-code-v2-independent-repros.py`: original20-second bounded child, actual exit1 at the now-rejected explicit-unknown context exit. Both F2 pairs explicitly rejected before that point. File SHA-256 matches supplied `d44236b25b8323201b2b854ec44cac4998162d16234f9021ca67ef56cbe0dfbd`. No claim that its unexecuted tail passed.
7. `C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/c2d1-code-v3-independent-repros.py`: new25-second bounded child, actual exit0. Independently asserts all three original F3 shapes and16 F4 method/boundary combinations, then records all four F5 counterexamples. Real peer threads joined and sockets closed; no production monkeypatch or candidate mutation.
8. PowerShell JSON comparison against `git show 90f50bb:optical_worker/data/native_sources/v2.json`: identical policy and lane inventories, exactly two source digest changes; remote sha256sum matches them.

## Actual exits and process hygiene

Local original runner2248(parent12004), child30752(parent2248), actual child exit1. V2 runner6832(parent12004), child9804(parent6832), actual child exit1. New v3 runner16424(parent31372), child33312(parent16424), actual child exit0. All six exact PIDs absent after execution. No child bound fired or kill was needed. Ruff's short-lived PID was not sampled; its command returned0.

Remote test timeout2417178(parent2417155), pytest2417179(parent2417178), started10:45:35 remote. Sampled cold-source child2418805(parent2417179). Command completion was observed as exit0, not inferred from process absence. The22 guarded-process ledger entries below are `case: supervisor / worker / actual worker returncode`; all supervisor returncodes were0. The harness checked each worker's recorded parent equals its exact supervisor PID and final supervisor evidence ends in process_exited.

```text
cpu_success: 2417418 / 2417420 / 0
cpu_cleanup_hang: 2417424 / 2417425 / -9
cpu_exit_hang: 2417441 / 2417442 / -9
normal_success: 2417456 / 2417462 / 0
normal_leaf_raise: 2417463 / 2417464 / 74
inner_abort: 2417475 / 2417476 / 74
pre_leaf_raise: 2417477 / 2417479 / 74
foreign: 2417480 / 2417484 / 74
unknown: 2417485 / 2417490 / 74
sync_raise: 2417491 / 2417492 / 74
sync_hang: 2417496 / 2417497 / -9
cleanup_raise_before: 2417523 / 2417526 / 74
cleanup_raise_after: 2417531 / 2417532 / 74
normal_no_ack: 2417561 / 2417573 / 74
normal_wrong_ack: 2417615 / 2417656 / 74
normal_ack_eof: 2417976 / 2417995 / 74
normal_stalled_disarm: 2418008 / 2418009 / 74
terminal_no_ack: 2418013 / 2418014 / -9
terminal_wrong_ack: 2418023 / 2418024 / 74
terminal_ack_eof: 2418025 / 2418026 / 74
target_cleanup_hang: 2418030 / 2418031 / -9
target_exit_hang: 2418048 / 2418049 / -9
```

Final exact `ps -p` lookup for timeout/pytest/cold-source child and all44 supervisor/worker PIDs returned only its header. A final `ps -C python,python3.12,timeout -o pid,ppid,lstart,args` showed only unrelated pre-existing Python626988 and the unrelated training chain2385821/2385914/2386045; all were preserved. Other short-lived broad-suite descendants were not individually sampled, a launch-ledger limitation rather than a claim that every OS PID was recorded.

Before/after remote `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` showed only unrelated training2386045. Local GPU query contained unrelated desktop applications and none of the reviewed runner/child IDs. No GPU work was run. This is scoped hygiene evidence, not proof of global idleness or impossibility of hangs.

Initial default-encoding reads rendered Chinese incorrectly and were reread fully with UTF-8. Large combined reads were split after truncation. One initial remote `&& ps` check returned1 because its process selection was empty and therefore skipped the following GPU query; that query was rerun independently before testing. A verbose catalog display exceeded its output budget; the subsequent focused comparison supplied the actual two-digest result. These were read/diagnostic tooling issues, not candidate test failures.

## Scope and remaining limits

Request changes applies only to exact6f87d23's C2d1 CPU controller/reducer/client/native-setup exception-boundary slice. Original F1–F4 counterexamples are closed, but F5 prevents acceptance of this revision. Repair and freeze a new candidate for fresh independent review; retain this report and external counterexample unchanged. Passing test counts do not override the reproduced contract failure.

Still closed: integrated C2d source-frozen guarded runtime adoption and module/session custody before CUDA-capable imports; real Warp/Torch upload and mandatory target/ambient/foreign/pre-return/unknown multi-GPU fault matrices; full setup graph/barrier/warmup/readiness; arenas/C2e/C2f; product GPU preparation; a3b/result/baseline authority; R2/host/S; later C4 and C4d/B3; shipping/activation and C4 completion. Existing adoption/removal obligations and historical approvals are unchanged.

CPU fixtures prove control flow, CPU-owner retention, record consistency and actual process supervision, not CUDA/Warp completion or real allocation-release safety. Parsed records/public hashes are not cryptographic history authentication or execution capabilities. No hostile-Python sandbox, fully rewritten-history detection, or unsampled global process-absence claim is made.
