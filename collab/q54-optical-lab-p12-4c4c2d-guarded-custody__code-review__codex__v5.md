# C4c2d1 guarded custody — independent code review v5

Initiative: q54-optical-lab-p12-4c4c2d-guarded-custody
Stage: code-review
Author: codex (fresh independent reviewer)
Version: v5
Date: 2026-10-03
Status: accepted
Related Files: custody design-request v2; repair-design v2; code-review v1–v4; repair-design-review v1/v2; runtime-integration design-request v3
Owner Summary: Approve the exact C2d1 CPU guarded-custody implementation slice at 854c2ea. Original F1–F6 counterexamples close; an independent 24-case actual-prepare CPU exception matrix and the 769-test remote regression pass. No source-frozen runtime, real-GPU, full-setup, arena, product or C4-completion gate opens.

## Identity, revision and verdict

Reviewer session: `/root/c4_c2d1_guarded_custody_code_review_v5`, fresh independent code-review session. No delegated work, implementation edits, or inherited author reasoning used as approval evidence.

- Overall base: `8cbf8645123a3d7e3f6cf4bbc1434fa3a6612472`.
- Prior reviewed revision / immediate comparison base: `6b5080fbfa00309c757ee3467fa6cb53b78fd57d`.
- Reviewed frozen HEAD: `854c2ea6a8b6ec4021cb35b330542b56dc5a1935`, including repairs `8b1918d` and `854c2ea`.
- Local candidate: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Remote isolated candidate: `/data3/ang.gao/robot_sim_c2d1_code_v5b` on authorized fallback `ang.gao@192.168.80.101`.
- Exact HEAD and clean working trees verified initially and finally on both hosts. The original dirty remote checkout was not touched.
- Accepted repair-design v2 SHA-256: `01a464293f4474f1665ee0ae3d2eda5477df43cec2f2ebb7374dcf87195656e2`.
- Archived code-review v3 SHA-256: `add59aab2abab30d7da299c123b58fcba174bd224797225ed1a1da96d5c4d128`; v4: `23e785818ebe171f65db37bea1429d1b8ab94e70e94f145ddf5858dbd313eb53`. Independently verified.
- Verdict: **Approve**, only for this exact revision and the bounded CPU interface/control/evidence/test scope described below.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely; read custody design v2, accepted repair v2, all four prior code reviews and both repair-design reviews, and parent runtime-integration requirements relevant to source admission, setup, failure custody, validation and closed gates. Inspected actual immediate/overall diffs, controller/reducer/client state and exception flows, inherited guardian/permit paths, actual native_setup handlers and _abort, test isolation/process supervision, source inventory/catalog and unchanged activation boundaries. Author notes and reported results were claims, not substitute evidence.

## Findings and original counterexample closure

No substantive unresolved findings in the reviewed scope. No new blocking issue was found.

### F6 closed — conservative scope intake reaches the actual inner abort

`guarded_lifecycle.py:280` now separates owner-thread conservative unknown-scope intake from ordinary work admission. It validates identifiers before inherited mutation, permits this metadata tightening after normal failure/compromise while RUNNING, and continues to reject wrong-thread and terminating/post-intent calls. It does not clear failure, issue a permit, start a runtime, COMPLETE a failed segment, or authorize sync/release. The existing reducer already admits this conservative event; no parallel state machine or new schema was added.

Both unchanged actual preparation handlers can now execute their mark-unknown-before-_abort sequence after failure observed during an in-flight leaf. The existing narrowly bounded exception-owner intake then retains the original error, and _abort reaches guarded termination before outer context unwinding. Full quarantine and `multi_or_unknown_device_quarantine_exit` are recorded with the correct unknown fact.

The unmodified original v4 counterexample now reaches `NativeWorkerPreparationAborted`, unknown=true, full custody and exact terminal evidence before context exit. It exits1 at its obsolete `assert not is_abort` on the CUDA/prior-thread-failure case; that early exit is not claimed as a full two-lane pass.

A new external independent script explicitly verifies the complete repaired tail: 24 cases across CUDA/Warp, actual import failure/target-resolution failure, no prior fault/thread misuse/owner-thread invalid-permit failure, and caught/propagated abort. Each executes the production prepare function, controller, client, real socket and supervisor core/transport. Each asserts original cause identity, original source/error guardian retention, exact termination-evidence identity before leaving the context, unknown=true, sticky failure, full guardian quarantine, zero normal COMPLETE/terminal ARM/sync/release, exactly one termination/custody transition, and unchanged evidence after unwind. All24 pass.

The target-resolution cases supply only an inert test-owned import stub whose device/get_device function injects the failure. They do not import a real runtime, mutate production methods/controller fields, or simulate successful GPU preparation. Only that exact stub is removed afterward; no real runtime module is unloaded. The import-boundary cases preserve the original counterexample shape. This is CPU exception-composition proof, not source-frozen runtime adoption.

### Earlier findings remain closed

- F1: original delayed-emergency-ACK wrong-thread calls produce zero subsequent sync/release authorization for CPU-cancel and target-dirty cases. Real late ACK remains truthful; terminal failure is sticky and full quarantine is selected.
- F2: both original incompatible artifact pairs explicitly reject: pending role conflict and forbidden COMPLETE/new ARM after terminal normal failure. The original runner stops at the first correct rejection; the v2 runner verifies both before its later early exit. No unexecuted tail is credited.
- F3: independent explicit unknown followed immediately by normal context exit, and caught foreign/unknown returned-owner failures, all reject completion/new work and retain full custody. Live admission, normal-success reducer predicates, and irreversible cross-history bounds remain consistent.
- F4: all three disabled legacy methods plus require_adopted remain covered at delayed ACK for CPU/dirty cases, admitted sync, and actual destructor cleanup. Pre-admission failure prevents sync/release; admitted sync fails with full custody; actual cleanup reentry yields remaining custody rather than a false full-graph claim.
- F5: all four original probes report returned_abort=true and terminal evidence before context exit. Candidate tests also cover failed/scope/sealed/caught/propagated cases and reject improper exception-retention inputs. No ordinary admission is reopened by the retention-only path.

## High-risk implementation and scope checks

- Descriptor declaration/start/scope notifications, permit entry, and returned-owner adoption/notifications now use the same existing controller RLock. A foreign-thread fault cannot split the CPU metadata chain and leave required notifications stranded. Inspected inherited implementations contain bookkeeping/adoption only at these calls, with the actual device leaf afterward. Deterministic candidate scheduling tests at descriptor_observed and owner_adopted pass.
- Terminal entry/state/custody publication and ARM/sync/release/outcome/intent admission remain ordered by that lock. Socket ACK waits, native synchronization and whole-graph reference release remain outside it. Existing reentry/success-boundary tests pass; this is conservative before-call admission, not atomicity with a GPU.
- Wrong-thread and terminating/post-intent unknown reports still fail closed; owner-thread reports after failure can only tighten scope. Sealed-guardian handling does not reopen resource coverage; the original failure graph remains in terminal holds.
- One inherited guardian/permit/release implementation is reused. There is no copied upload loop, C0 termination delegation, alternate emergency notifier, new wire role, watchdog restart or second sync. Original exception/traceback custody and released-versus-remaining ownership remain distinguished.
- Exact ticket validation uses registered ACK-created object identity and original fields, role/session/target and actual client transition history. Descriptive records/hashes are not execution capabilities. Reducer and combined artifact checks retain pending-ACK uncertainty, healthy lagging-prefix acceptance, and irreversible terminal/failure/compromised-prefix restrictions.
- Protected C0 lifecycle/native_setup, worker and production launchers are byte-identical to overall base. Overall changed-file inventory does not include product activation, gate/schema, arena or shared workflow changes.
- Immediate catalog comparison has identical root fields except source_digests, with exactly one repin: guarded_lifecycle from `aa93c6a50bd8e15ecd73a998089bc4d208bffdb3269b44abc8061330b80d60d3` to `99bc155a4d85c36929484004cbf8a989f9b80bc3db03af73d933f7987f896bbd`. Guarded evidence remains `1c4a23d630938eb28e0aeb1da070b9f4ea8b32b7f60097af3fd020b3aaa301c5`. Remote bytes match. Lane inventories and AST policy are unchanged. Catalog inclusion is not captured-source runtime admission.
- Cold-import isolation is test-only: a fresh -I -B child invokes the fixed stdlib harness and shared real-socket peer, not pytest or a unit module. Existing assertions remain in that harness. The broad independent run proves these probes work despite collection-time runtime imports in the parent; no isolation relaxation, runtime unloading or package install was needed.

## Independent commands and results

Environment: local Windows PowerShell and `C:/Python314/python.exe` (no Torch/Warp); remote `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python`. Primary183 transport failure was supplied in the packet; authorized fallback revision/tree were independently checked. No installation, GPU workload, push/reset, broad kill, candidate edit or shared-policy edit was performed. New scratch repro/report are outside the candidate. Test temporary artifacts are under the test harness's temp directories, including remote `/tmp/pytest-of-ang.gao/pytest-123`; tracked trees remain clean.

SSH prefix:

```text
ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101
```

Executed:

1. Initial/final HEAD/porcelain status, immediate/overall diff inspection, design/archive/repro hash checks, parsed catalog comparison and remote source sha256sum. Exact clean candidate throughout.
2. `git diff --check 8cbf864 HEAD`: exit0. `git diff --quiet 8cbf864 HEAD -- optical_worker/lifecycle.py optical_worker/native_setup.py optical_worker/worker.py tools/optical_closed_worker.py tools/optical_closed_worker_supervisor.py`: explicit exit0.
3. `C:/Python314/python.exe -B -m ruff check optical_worker/guarded_lifecycle.py optical_worker/guarded_evidence.py optical_worker/wait_protocol.py tests/support/guarded_peer.py tests/support/guarded_custody_process.py tests/unit/optics/test_guarded_lifecycle.py tests/unit/optics/test_guarded_evidence_fixtures.py tests/unit/optics/test_guarded_custody_process.py`: all checks passed.
4. Original external runner: actual child exit1 at first correct F2 rejection; both F1 outputs checked. Original v2 repro: actual exit1 at correct explicit-unknown context rejection, after explicitly checking both F2 rejections. Original v3 repro: actual exit0, original F3/F4 and four F5 outputs checked. Original v4 repro: actual exit1 at obsolete bug assertion described above. Original v3/v4 hashes match `6b1bf9c692f0e19b0a53a2dd1933199e75a9e72aae2384c940a2948f961a7be3` / `4a4545f896a0773417b5a400d93a6ce861897b91f8f1ed08714e1547a56e06e0`; original scripts were not edited.
5. `C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/c2d1-code-v5-independent-repros.py`: new externally bounded -I -B child, actual exit0; all24 repaired-tail/adjacent-boundary assertions pass. Script SHA-256 `481c264781c4ac0835fe8671b156e6b738a69e2d28afa7c61aef04de414f0526`.
6. Independent remote15-file command below: **769 passed in 171.52s**, no skips, actual command exit0 (tool session90809). Real Go2 fixture enabled. No outer timeout fired. The Warp deprecation warning is neither a failure nor GPU-execution evidence. Author tests had completed and hygiene clearance was received before this independent run began; their outputs are not substituted for this run.

```text
cd /data3/ang.gao/robot_sim_c2d1_code_v5b
ROBOT_SIM_TEST_GO2_MODEL_DIR=/data3/ang.gao/c4_asset_frozen/out/external/mujoco_menagerie/unitree_go2 timeout --signal=TERM --kill-after=15s 420s /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -m pytest tests/unit/optics/test_guarded_lifecycle.py tests/unit/optics/test_guarded_evidence_fixtures.py tests/unit/optics/test_guarded_custody_process.py tests/unit/optics/test_guarded_wait_core.py tests/unit/optics/test_guarded_wait_process.py tests/unit/optics/test_native_bootstrap.py tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/optics/test_sacrificial_sources.py tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_leaf_descriptors.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py tests/unit/benchmarks/test_optical_benchmark_contracts.py -q -s -p no:cacheprovider
```

## Actual exit/reap and process hygiene

Local original runner30372(parent34800), child3396(parent30372), actual child exit1. V2 runner35600(parent36632), child17512(parent35600), exit1. V3 runner2596(parent22148), child20180(parent2596), exit0. V4 runner36000(parent35048), child25216(parent36000), exit1. New v5 runner36020(parent21956), child21832(parent36020), exit0. All10 exact PIDs absent at final Get-Process check. Original15/20/25-second and new30-second child bounds did not fire; no local kill was needed. Peer threads joined and sockets closed. Ruff's short-lived PID was not sampled; command completion was observed.

Remote timeout2500740(parent2500654), pytest2500761(parent2500740), started11:35:39 remote. Eight isolated import-probe children2501007,2501009,2501017,2501024,2501027,2501029,2501031,2501034 each recorded parent2500761 and actual returncode0. Remote command completed exit0, not inferred from absence.

The29 custody-process cases below are `case: supervisor / worker / actual worker returncode`; all supervisors returned0. Tests require the recorded worker parent to equal its exact supervisor, parent liveness, and final actual supervisor process_exited evidence. The supervisor waits/reaps its child; deadline cases are not credited from the test outer timeout.

```text
cpu_success: 2501070 / 2501071 / 0
cpu_cleanup_hang: 2501072 / 2501073 / -9
cpu_exit_hang: 2501089 / 2501090 / -9
normal_success: 2501107 / 2501108 / 0
normal_leaf_raise: 2501109 / 2501110 / 74
inner_abort: 2501114 / 2501119 / 74
foreign_inner_abort: 2501120 / 2501121 / 74
unknown_inner_abort: 2501122 / 2501123 / 74
invalid_permit_inner_abort: 2501125 / 2501126 / 74
cuda_import_failure: 2501127 / 2501128 / 74
warp_import_failure: 2501132 / 2501140 / 74
cuda_import_thread_failure: 2501141 / 2501142 / 74
warp_import_thread_failure: 2501144 / 2501145 / 74
pre_leaf_raise: 2501148 / 2501149 / 74
foreign: 2501150 / 2501151 / 74
unknown: 2501155 / 2501160 / 74
sync_raise: 2501161 / 2501162 / 74
sync_hang: 2501163 / 2501164 / -9
cleanup_raise_before: 2501180 / 2501185 / 74
cleanup_raise_after: 2501186 / 2501187 / 74
normal_no_ack: 2501188 / 2501189 / 74
normal_wrong_ack: 2501194 / 2501199 / 74
normal_ack_eof: 2501203 / 2501204 / 74
normal_stalled_disarm: 2501205 / 2501206 / 74
terminal_no_ack: 2501208 / 2501209 / -9
terminal_wrong_ack: 2501217 / 2501222 / 74
terminal_ack_eof: 2501223 / 2501227 / 74
target_cleanup_hang: 2501231 / 2501234 / -9
target_exit_hang: 2501297 / 2501298 / -9
```

Final exact ps lookup of timeout/pytest, eight cold children and all58 supervisors/workers returned only the header. Final `ps -C python,python3.12,timeout -o pid,ppid,lstart,args` contained only unrelated pre-existing Python626988 and training chain2483327/2483430/2483521, all untouched. Other short-lived broad-suite descendants were not individually sampled; this is a ledger limitation, not a claim that every OS PID was recorded.

Before/after remote `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` showed only unrelated training2483521. Local GPU query contained unrelated desktop applications and none of the ten reviewed runner/child IDs. No GPU work was run or authorized. These are scoped hygiene observations, not a global-idleness proof or a diagnosis of any historical unknown hang.

Read/diagnostic issues: initial default-encoding Chinese text was reread fully with UTF-8, and oversized reads were split. An initial PowerShell per-digest comparison treated source_digests as an object instead of an array and printed SyncRoot; the corrected record-array comparison established the exact single repin above. These did not mutate files or constitute candidate test failures. Author-reported failed runs at earlier/intermediate revisions are preserved in notes, but are neither rerun results at this HEAD nor concealed as passes.

## Exact approval and remaining limitations

Approval covers exact854c2ea's C2d1 CPU guarded lifecycle/controller, actual native-setup exception-boundary composition, ticket-observation extension, descriptive replay/cross-history evidence and fixed CPU supervision tests, under the accepted custody/repair designs. F1–F6 are closed in this bounded implementation. Archive this report unchanged; a separate evidence-only commit may record it without silently broadening its revision or scope.

Still closed: integrated C2d source-frozen guarded runtime adoption and module/session custody before CUDA-capable imports; real Warp/Torch upload and mandatory target/ambient/foreign/pre-return/unknown multi-GPU fault matrices; full setup graph/barrier/warmup/readiness composition; arenas/C2e/C2f; product GPU preparation; a3b/result/baseline authority; R2/host/S; later C4 and C4d/B3; shipping/activation and C4 completion. Existing cleanup/adoption/removal obligations and historical approvals are unchanged.

CPU fixtures establish control flow, CPU-owner retention, evidence consistency and actual subprocess supervision, not CUDA/Warp completion or real allocation-release safety. No source-frozen import guarantee follows from an external import injector. Parsed records/public hashes are not authentication or live authority; no fully rewritten-history detection, hostile-Python sandbox or unsampled global-process-absence claim is made. Later integration must prove its own real resource graph and device behavior and receive fresh independent review.
