# C2d2 captured-source custody — independent code review v1

Initiative: q54-optical-lab-p12-4c4c2d-source-custody
Stage: code-review
Author: codex (fresh independent reviewer)
Version: v1
Date: 2026-10-07
Status: in_review
Related Files: source-custody design-request/design-review v1; source-consumption design v1; guarded-custody design v2, repair-design v2 and code-review v5; runtime-integration design v3
Owner Summary: Request changes for one P2 defect in the mandatory real-process test oracle. Both no-ACK cases reject a legitimate supervisor-killed/reaped endpoint. The independently exercised CPU source/guardian implementation otherwise behaved conservatively; this is not approval of C2d2, runtime integration, GPU preparation or any later gate.

## Identity and verdict

Reviewer session: `/root/c4_c2d2_source_custody_code_review_v1`. Fresh independent code review, not implementation. No further delegation or candidate edits. Author results and historical approvals were not substituted for independent execution.

- Comparison base: `af509cb5e3ed4103a1035a1298cd20bb652f45e4`.
- Frozen reviewed HEAD: `f7692b4b7bfd50521956c5c83fcb2bc94049906a`.
- Tree: `9457aea929df94c594ac8462b5e4cc81699e819d`.
- Local candidate: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Independent remote clone: `/data2/shared/ang.gao/robot_sim_c2d2_code_v1`, SSH `tileops-dev`, actual hostname `hd03-gpu2-0031`.
- Local and remote HEAD/tree matched; initial/final porcelain status was empty.
- Accepted design SHA-256 independently checked: `088f245b305de66666d0d54912f517d9148f4dfd446a7c6b57eb84383a7671fa`.
- Verdict: **Request changes**. One blocking P2 test defect; no P1 finding established. No implementation bypass or unsafe release was demonstrated by these tests.

Read AGENTS.md, collab/README.md and REVIEW_PRINCIPLES.md completely. Read the accepted source-custody design/review, source-consumption design, guarded-custody design/repair and v5 report, and parent source/setup/failure/validation/cleanup requirements. Checked actual diff, source loader/finder/session flow, guardian/controller/client interaction, policy/catalog pins, test injectors and real supervisor transport/core. Earlier approvals establish their historical scope only.

## Blocking finding

### F1 — P2: no-ACK process test requires a worker-written exit that the supervisor may preempt

Location: `tests/unit/optics/test_native_source_custody_process.py:85–86`; configuration/injection at `tests/support/guarded_custody_process.py:313,385–386`; fixed child's final marker/exit at `tests/support/source_custody_process.py:145–151`.

Trigger: execute `test_captured_source_real_supervision[warp-source_no_ack]` or its CUDA counterpart on the actual POSIX H200 host. Both client ACK wait and supervisor pending-response deadline are 1000 ms. The supervisor can commit the deadline and kill/reap the child before its exception path writes `worker.json`, `no_runtime`, or calls `os._exit(74)`.

Independent full run: **2 failed, 28 passed in 112.25 s**, actual pytest exit 1. Both failures are line 85, `assert -9 == 74`. Each supervisor returned 0. Before reaching that assertion, the test successfully validated the combined artifact, final actual `process_exited` observation, child PID/parent relationship, empty stderr, and absence of atexit/destructor markers. The failing safety-bound timeout did not fire.

An additional outside-tree reproduction ran the unchanged actual release, not a repackaged mutant, and observed this exact trace in both lanes:

`initialized -> arm_committed -> deadline_expired -> kill_requested -> process_exited`

Both worker returncodes were -9; `complete=true`, `kill_requested=true`, `deadline_expired=true`, `arm_count=1`, `normal_complete_count=0`, terminal reason `deadline_expired`; parent remained alive. The fixed import-body marker did not exist. Both worker record and `no_runtime` marker were absent. The supervisor waited/reaped the actual child. Thus changing only the exit-code assertion still leaves the following marker assertion invalid for this allowed endpoint.

This is a test-oracle failure, not evidence that the production deadline should be increased, restarted, or disabled. The accepted custody design explicitly separates worker terminal intent from observed exit and permits absent worker tails. A pending ACK failure must not become normal completion or a new emergency attempt, but it need not win the race against supervisor termination.

Consequence: the required fresh POSIX acceptance suite is deterministically red in the independent run despite valid supervision. The suite cannot support code approval in this form, and a superficial returncode-only repair would remain incomplete.

Why existing checks miss it: the assertion branches only on fault names ending in `hang` or `source_parent_loss`; every other fault is classified as guaranteed self-exit 74. It ignores actual validated supervisor deadline/kill evidence when classifying `source_no_ack`.

Repair direction: recognize the two allowed no-ACK endpoints using actual evidence. A self-exit branch must retain its required worker evidence/markers; a supervisor-killed branch must require the correct committed deadline/kill/actual-reap trace and allow absent worker tail/terminal markers. Continue checking no normal COMPLETE/new ARM, no body/runtime entry, no destructor, parent liveness and actual exit. Do not count an outer timeout as success or relax unrelated fault assertions.

Proof after repair: rerun the complete 30-case both-lane real-process matrix; explicitly exercise both legal no-ACK endings with evidence-aware assertions, including an absent worker record/marker. Preserve original timer/protocol semantics. Freeze a new candidate and obtain fresh independent re-review.

## Non-blocking observation: precise meaning of spec identity

The source audit checks the retained exact loader and module objects plus `ModuleSpec` name/parent/loader/origin/package fields; it does not retain the original `ModuleSpec` object itself. A shallow equal-valued copy assigned to a completed module's `__spec__` passes `session.audit()` in both lanes. This was independently tested and is recorded rather than described as a rejected attack.

No altered executable source, foreign loader, callable, additional module or live authority was admitted in this probe. The accepted design's loader requirements emphasize exact source/loader/module identities and origin/package checks, consistent with the inherited audit. Accordingly this is not treated as an additional blocking security finding. Avoid claiming that *every* spec-object replacement is rejected; distinguish metadata drift from equal metadata replacement. If literal spec-object identity is intended as a new admission requirement, settle that explicitly and test it rather than inferring it from the current checks.

## High-risk implementation assessment

The actual cumulative loader closure is six core modules, ten after CPU custody prelude, and twelve after fixed native_setup/native_contract definition import. The directly executed bootstrap is a separate initial owner, not a thirteenth captured-loader module. `_CapturedSourceLoader` remains the only repository source execution implementation; no default repository-path/pyc fallback was added.

Binding constructs the captured controller/configuration classes itself. It retains the launch configuration separately from the client's revalidated configuration and checks the actual retained client configuration by identity. The session and snapshot have separate one-shot identity registries; the source session is adopted before post-bind execution. No caller-created controller or configurable module/callable selector was added.

For post-bind execution, loader registration as started precedes guardian module adoption, which precedes `source_to_code` and body execution. A failed explicit module adoption still leaves the exact partial module in the enclosing strongly retained session. The production guardian/release implementation is reused rather than duplicated. Normal fixed import uses the controller's actual setup segment and performs source/module audit before COMPLETE/DISARMED.

Design-review C1/C2 cautions were checked: first/later pre-guard adoption failure explicitly calls the existing failure intake and selects terminal full custody; swallowed nested-loader failures preserve the original exception rather than just a later audit error. Candidate tests assert original error identity, no retry/new ARM/sync, partial module retention and unchanged terminal evidence after an attempted flag restoration. The launch/client configuration identity distinction is exercised, including rejection of an equal-valued replacement client configuration.

Independent external probes checked both lanes against actual POSIX release bytes: replacing a loader's source with a distinct equal byte string rejects; importing a catalogued but execution-ineligible repository module rejects; audit after healthy terminal CPU cancellation rejects without a new normal operation. These use the real loader/controller and joined socket/core peer, not a copied source implementation. All eight probes, including the two descriptive equal-spec probes, exited 0 and were reaped.

The independent real-process matrix exercised prelude failure, module adoption, compilation/body/nested-caught/audit failures, import hang, parent loss, wrong/EOF ACK, failed DISARMED, cleanup hang and exit hang. Except F1's test assertion, all 28 cases passed. Hang cases were killed/reaped by supervisor evidence, not outer test timeout. No CUDA-capable body was invoked; checks of loaded NumPy/Torch/Warp roots stayed empty in completed CPU probes. Deliberate body/compile fault instrumentation is outside the admitted release and is not credited as successful frozen-source execution.

Catalog v3 has unchanged literal path and lane inventories versus v2. Only root schema/policy digest and three source digests changed: native_bootstrap, native_source_policy and native_sources. Bootstrap v4 pin is `7dfb6074c7b5dc639a0763d21217c4e294afc340a03ee89b3ba058e8d088e02c`. The policy still pins complete bootstrap and loader/finder definitions; full source-policy/rehashed-mutant and cached-code regression suites pass. Historical v2 is not an executable fallback.

Actual delta is 12 files, 690 insertions/33 deletions. No protected C0 lifecycle/native_setup/native_contract/worker/production launcher, guarded protocol/core/controller/evidence grammar, product gate/schema, arena or collaboration-rule edit appears. Explicit protected-path diff exited 0. Cleanup ledger adds the fixed test paths without removing either candidate lane.

## Independent verification

Environment: Windows PowerShell, `C:/Python314/python.exe` for read-only data inspection/Ruff; primary H200 host with `/home/ang.gao/miniforge3/envs/robot_sim/bin/python3.12` for independent CPU executions. SSH prefix: `ssh -o BatchMode=yes -o ConnectTimeout=8 tileops-dev`. No install, push, reset, broad kill or GPU workload. Scratch scripts/report are outside the candidate; test fixture writes are confined to disposable test/temp directories.

1. Initial/final `git rev-parse HEAD`, `git rev-parse 'HEAD^{tree}'`, porcelain status, full diff and protected-path diff. Exact clean frozen candidate on both hosts. `git diff --check af509cb5 HEAD`: exit 0. Full accepted-design SHA-256 matched.
2. Ruff over the three changed production Python files, three support files and two new custody test files: `All checks passed!`, exit 0.
3. `python3.12 -B -m pytest -q -s tests/unit/optics/test_native_source_custody_process.py`: **2 failed, 28 passed in 112.25 s**, exit 1, tool session 58455. Failures F1 only. Temporary packaged release `/tmp/pytest-of-ang.gao/pytest-104/native-source-release0`, revision `0e2a2a5fa901244405991f47d84643b66d8755cc`. Fixture explicitly packages LF bytes; production verifier remains byte-exact.
4. `python3.12 -B /data2/shared/ang.gao/c2d2-review-probes.py /data2/shared/ang.gao/robot_sim_c2d2_code_v1`: eight isolated `-I -S -B` children, each actual returncode 0, joined peer and actual reap, runner exit 0. Script SHA-256 `f29cb9e4a2fcdf810dbcd1a5772318653c8fca2170514e0e55e550fa27b7d0b6`.
5. Broad independent command below: **752 passed, 2 skipped in 163.12 s**, exit 0, tool session 7651. Includes all 26 new in-process source-custody cases, source-only direct entry (including actual POSIX checkout), stale pyc/metadata/reload attacks, policy/catalog/source, existing lifecycle/evidence/wait/process and C0 regressions. The two skips were real Go2 source/materializer cases with no asset env in that invocation, not claimed as passes.
6. `python3.12 -B /data2/shared/ang.gao/c2d2-noack-repro.py /data2/shared/ang.gao/robot_sim_c2d2_code_v1`: both actual-release no-ACK cases met the independent evidence-aware expectations above, runner exit 0. Script SHA-256 `0120f8ff385569173dfdfeb22482af7e6cdb840399e9dac372e65f8b11406ed0`. Temp paths `/tmp/c2d2-independent-noack-_ta7imou` and `/tmp/c2d2-independent-noack-znwmsltc`; no stale prior artifact reused.
7. Fresh rerun with `ROBOT_SIM_TEST_GO2_MODEL_DIR=/data2/shared/ang.gao/c2d2_frozen_assets/mujoco_menagerie/unitree_go2`, `timeout --signal=TERM --kill-after=15s 180s python3.12 -B -m pytest -q -s -p no:cacheprovider tests/unit/optics/test_native_sources.py -k go2`: **2 passed, 39 deselected in 45.26 s**, exit 0, session 35325. Uses the actual source suite's asset validation/materialization, not a surrogate. This is CPU asset regression only.

Broad command (working directory is the independent clone):

```text
timeout --signal=TERM --kill-after=15s 420s /home/ang.gao/miniforge3/envs/robot_sim/bin/python3.12 -B -m pytest -q -s -p no:cacheprovider tests/unit/optics/test_native_source_custody.py tests/unit/optics/test_native_bootstrap.py tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_guarded_lifecycle.py tests/unit/optics/test_guarded_evidence_fixtures.py tests/unit/optics/test_guarded_custody_process.py tests/unit/optics/test_guarded_wait_core.py tests/unit/optics/test_guarded_wait_process.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/optics/test_sacrificial_sources.py tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_leaf_descriptors.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py
```

No listed outer timeout fired. The separate optical benchmark-contract suite was not run by this reviewer; author-reported local missing-Torch failures and author's remote execution are not credited as reviewer results. Real runtime/GPU tests are intentionally unexecuted because they remain outside this checkpoint's authority.

Read/tool issues: an unquoted PowerShell `HEAD^{tree}` was parsed incorrectly on the first identity command; quoted rerun established the tree above. Initial default-encoding Chinese text was reread with explicit UTF-8; oversized reads were split. Remote `rg` was unavailable for one process inventory query; a later `grep`/exact-PID inventory replaced it. These were non-mutating tooling issues, distinct from F1.

## Process identities, actual exit/reap and round-end hygiene

All source-process supervisors below returned 0. Notation is `case: supervisor / worker / actual worker returncode`. Tests checked each child's recorded parent against its supervisor and the final supervisor `process_exited` evidence; supervisors called wait/reaped children, and pytest communicated/reaped supervisors.

```text
warp:
success: 2909448 / 2909449 / 0
prelude_raise: 2909496 / 2909497 / 74
module_adopt: 2909515 / 2909516 / 74
compile_raise: 2909533 / 2909534 / 74
body_raise: 2909599 / 2909604 / 74
caught_nested: 2909647 / 2909648 / 74
audit_raise: 2909673 / 2909674 / 74
import_hang: 2909746 / 2909747 / -9
parent_loss: 2909831 / 2909832 / -9
no_ack: 2909954 / 2909958 / -9
wrong_ack: 2910024 / 2910025 / 74
ack_eof: 2910050 / 2910051 / 74
stalled_disarm: 2910115 / 2910140 / 74
cleanup_hang: 2910199 / 2910200 / -9
exit_hang: 2910378 / 2910380 / -9
cuda:
success: 2910511 / 2910512 / 0
prelude_raise: 2910615 / 2910620 / 74
module_adopt: 2910689 / 2910691 / 74
compile_raise: 2910967 / 2910968 / 74
body_raise: 2911020 / 2911021 / 74
caught_nested: 2911051 / 2911052 / 74
audit_raise: 2911155 / 2911157 / 74
import_hang: 2911173 / 2911174 / -9
parent_loss: 2911306 / 2911308 / -9
no_ack: 2911668 / 2911669 / -9
wrong_ack: 2911733 / 2911734 / 74
ack_eof: 2911779 / 2911780 / 74
stalled_disarm: 2911864 / 2911865 / 74
cleanup_hang: 2911925 / 2911926 / -9
exit_hang: 2912242 / 2912243 / -9
```

Independent probe runner 2913349 (parent 2913348) launched/reaped children 2913354, 2913416, 2913507, 2913561, 2913587, 2913647, 2913725, 2913943, each exit 0. No-ACK runner 2919307 (parent 2919306) launched supervisors 2919309/2919346, both exit 0; their workers 2919310/2919347 exited -9 and were reaped.

Broad runner: timeout 2914301 (parent 2914297), pytest 2914303 (parent 2914301), start 2026-10-07 19:27:09 remote. Eight isolated inherited-import probes 2917497, 2917500, 2917502, 2917505, 2917508, 2917511, 2917513, 2917520 each recorded parent 2914303 and exit 0. Existing custody process supervisors/workers also recorded actual endpoints:

```text
cpu_success 2917648/2917653/0; cpu_cleanup_hang 2917718/2917720/-9;
cpu_exit_hang 2917744/2917745/-9; normal_success 2917791/2917792/0;
normal_leaf_raise 2917794/2917795/74; inner_abort 2917797/2917801/74;
foreign_inner_abort 2917803/2917806/74; unknown_inner_abort 2917814/2917815/74;
invalid_permit_inner_abort 2917816/2917817/74; cuda_import_failure 2917819/2917820/74;
warp_import_failure 2917821/2917822/74; cuda_import_thread_failure 2917823/2917825/74;
warp_import_thread_failure 2917827/2917829/74; pre_leaf_raise 2917831/2917832/74;
foreign 2917834/2917835/74; unknown 2917836/2917837/74;
sync_raise 2917838/2917839/74; sync_hang 2917843/2917844/-9;
cleanup_raise_before 2917870/2917871/74; cleanup_raise_after 2917872/2917874/74;
normal_no_ack 2917875/2917876/74; normal_wrong_ack 2917889/2917890/74;
normal_ack_eof 2917891/2917892/74; normal_stalled_disarm 2917898/2917899/74;
terminal_no_ack 2917900/2917901/-9; terminal_wrong_ack 2917941/2917942/74;
terminal_ack_eof 2917946/2917948/74; target_cleanup_hang 2917953/2917954/-9;
target_exit_hang 2917987/2917988/-9.
```

Final exact `ps -p ... -o pid,ppid,lstart,args` lookup for all listed owned PIDs, broad timeout/runner, sampled source child 2915481, and sampled asset chain 2919306/2919408/2919422 returned only the header. Final `ps -C python,python3.12,timeout` showed only unrelated Python 2587008 (parent 2586183, start 14:25:56). GPU inventory before and after showed only unrelated `2238410, nvidia-cuda-mps-server`, untouched. Eight H200 devices were enumerated; no GPU workload was run.

Local command chain sampled PowerShell 10284 -> Codex 10808 -> Code 34516 (parent 22524). Codex started 19:15:47 local; it is the continuing shared CLI, not a separate OS isolation claim. Local Python process enumeration was empty at final query; local GPU inventory contained desktop applications only. No owned residual or unresolved review hang was observed. No unrelated process was signalled or killed.

Ledger limits: the first full-30 pytest runner and every short-lived Git/read/Ruff/source-test descendant were not individually PID-sampled. Their actual command completion was observed, and relevant subprocess harnesses wait/reap their children. This report does not claim a complete OS process ledger or global idleness. Parent process monitoring messages were coordination, not substituted test evidence.

Original remote checkout was only read at the final identity check: clean `main` at `f771e612beaeea126e26926738097512cd2d809c`. It was not edited. Reviewer scratch files and disposable test evidence were retained; nothing material was deleted.

## Authority, limits and next checkpoint

No new approval is granted until F1 is repaired and independently reviewed. The assessed implementation scope is only CPU captured-source phase extension, exact session/source/module/guardian binding, fixed supervised native_setup/native_contract definition import, current policy-v4/catalog-v3 pins, and fixed CPU validation support.

Still CLOSED: third-party extension/create_module/partial-import ownership; site-package/runtime admission; actual Torch/Warp initialization/upload and mandatory target1/ambient0, foreign pre-return, unknown-scope and upload/ready-event fault matrices; complete final graph/seal/barrier/warmup/readiness; full C2d, C2e/f, C4c3/c4 and C4d/B3; product GPU preparation, a3b, results/baseline, R2/host/S, shipping and activation. C4 is not complete.

CPU owner retention, frozen-source bytes and validated supervisor histories do not establish GPU completion, actual device allocation-release safety, hostile-Python isolation, or authority from serialized records. Preserve the existing staged approval and cleanup/removal obligations.
