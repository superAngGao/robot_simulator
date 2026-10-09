# Render graph native support independent code review v1

Initiative: Q54 / P12.4 / C4 render integration
Stage: independent code review of native-driver support
Author: codex
Version: v1
Date: 2026-10-10
Status: accepted
Related Files: AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md; runtime-integration design-request codex v3; warp-module-preparation design-request codex v2; provisional-render-graph code-review codex v2; render-graph-native-support implementation-note codex v1
Owner Summary: Approve the bounded positive native-driver support at the exact revision below. No substantive finding. This review does not establish native graph acceptance, optical execution, warmup completion or whole-C4 completion.

Reviewer session: /root/render_graph_native_support_v1, fresh independent session; no implementation edits or subagents.
Repository: C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design
Base: 8315b008bf54f294af5bf0c3035d4bc9cc0e009b
Reviewed HEAD: 50143c6a92b2c1d21dd52dae512b43a3cdd3cef9
Actual tree status: clean at initial, intermediate and final checks; HEAD unchanged.
Verdict: **Approve**, exclusively for the support-code scope described below.

## Scope and inspected evidence

Read the collaboration protocol and full review principles, applicable parent setup/readiness/retention/failure/validation requirements, the Warp compilation supplement, implementation note and archived predecessor review. Independently inspected the complete six-file delta: PROGRESS.md, two collaboration records, tests/support/native_runtime_process.py, tests/support/cuda_artifact_runtime_acceptance.py and tests/unit/optics/test_native_render_graph_support.py. There is no production-code change. No new supervisor, process manager, execution registry, renderer or product dispatcher is introduced.

The archived provisional graph v2 report matches the outside-tree original exactly as a Git blob: SHA256 d6ceb5f0e8ce5585f7d437def5c04f874bbb597173dd06b81a360e89870bfaf9. The checked-out file has Windows line endings, explaining its different filesystem hash; its recorded Git content is unchanged. PROGRESS.md accurately keeps real recording/events and optical execution pending and preserves the bounded predecessor verdict.

Approval covers two fixed positive selectors, runtime_render_graph_success and runtime_render_graph_diagnostics, through the existing supervised driver and run_case replay. They compose original source intake, runtime initialization, upload/validation, CUDA factory preparation or Warp compilation/reuse, fixed storage, and the original production graph method before existing terminal disposal. The default historical CASES matrix is unchanged; this is callable support for a separately selected graph campaign, not a claim that a graph campaign ran.

## Findings

No substantive finding. No blocking repair or optional code follow-up is required for this bounded support slice.

The following high-risk areas were checked directly:

- Selector membership reaches the existing guarded_custody_process runtime branch and child parser. The success predicate includes both graph cases. Storage diagnostics are mapped consistently, and the added graph segment increases expected normal completions to 13 for Warp and 9 for CUDA; the existing terminal path still requires one emergency synchronization, successful release and observed exit. Existing error, stderr, deadline, kill and ordinary-atexit rejection remains in force.
- The observer wraps the production SourceSession graph entry and the existing adoption boundary. For command/event adoption it checks an already guardian-held partial-owner list before metadata observation and before delegation to original adoption. It does not create observer-owned native commands or events. Both replaced functions are restored in finally, including exceptions. The original session/controller failure path remains responsible for failed guarded work; this support adds no retry or synchronization on observer failure.
- Warp graph scopes bypass the earlier input-phase ScopedDevice observation wrapper, as compilation/storage already do. Graph observations use native scope snapshots without reinterpreting them as input-validation receipts. The before/after checks surround production's actual graph scopes, while adoption snapshots occur inside the held target/stream scope.
- The earlier compilation observer expects a Warp Module argument and accesses module.name/module.execs. The graph factory actually calls public load_module with module=__name__. Restoring both load_module and compile_module immediately before graph preparation is therefore necessary and correctly placed after original compilation/reuse evidence has been written. Subsequent graph validation still checks the actual retained module set and every command's ModuleExec/kernel module.
- Independently read the local pinned Warp 1.12.0 context.py source, whose SHA256 is eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee, matching the supplement. Its public load_module accepts a string and resolves user_modules; Launch retains module_exec and kernel/device as assumed by the observer. This is static dependency evidence, not native execution evidence. Production prepare uses block_dim=256/max_workers=0, and its guarded recording helper uses record_cmd=True/record_tape=False with the exact device/stream.
- Warp frame inspection reuses production parameter validation and checks actual commands against the retained compiled render executable/module. CUDA resolves the existing adopted factory/context and issued extension, validates each actual returned stage and parameter ABI, then compares captured raygen/shade entries and the renderer's first-hit lookup with the issued module. Inspection of cuda_direct_light confirms first-hit is looked up on the renderer module rather than stored as another captured stage field; the observer follows that real call path.
- Raw replay requires ten ordered frame IDs 1..10, ten distinct bundles, five Warp commands or three CUDA entry observations per frame, exact distinct Warp command custody, completion/diagnostic event custody, correct profile/shadows/pass mode, diagnostic phases, warmup0 and unchanged ambient state. Target snapshots require ordinal1, the held current stream and only primary context1 active. These records describe checked live objects and do not issue runtime authority.
- Replay cross-checks the original worker lifecycle descriptors in the exact graph guard: 1 or 7 event creates and records, one wait as the final descriptor, no allocation and no warmup launch. The production event path actually creates/retains events, records the terminal event last and synchronizes that event under the original guard. This event initialization is not optical completion. Existing full custody-artifact validation precedes the new replay checks.

## Executed verification

Environment: Windows PowerShell, C:/Python314/python.exe, bytecode disabled and pytest cache provider disabled. All test commands ran from the frozen candidate. No dependency installation, native import/GPU campaign or remote command was performed.

1. `git diff --check 8315b008bf54f294af5bf0c3035d4bc9cc0e009b HEAD`: exit 0, initially and finally.
2. `C:/Python314/python.exe -B -m pytest -q -rs -p no:cacheprovider tests/unit/optics/test_native_render_graph_support.py tests/unit/optics/test_native_render_preparation.py tests/unit/optics/test_prepared_fixed_render_inputs.py tests/unit/optics/test_cuda_artifact_runtime.py tests/unit/optics/test_native_scope_observer.py tests/unit/optics/test_native_runtime_driver_process.py`: **173 passed, 13 skipped in 32.55s**, exit 0, tool session 51659.
3. Set command-local `ROBOT_SIM_WARP_CONTEXT_SOURCE=C:/Users/AG/新建文件夹/c4_assets/warp_setup_source/context.py`, then ran `C:/Python314/python.exe -B -m pytest -q -rs -p no:cacheprovider tests/unit/optics/test_native_scope_observer.py`: **33 passed in 2.66s**, exit 0. The fixture itself checks the pinned source hash before extracting exact Device methods. This resolves the nine source-file skips in command 2; it models native calls and is not native scope acceptance.
4. Initial/intermediate/final `git status --short` was empty and `git rev-parse HEAD` returned the exact reviewed revision. Compared the archived report's Git bytes with the original outside-tree artifact as described above.

The four new cases use both lanes and both diagnostic modes through the real session graph entry, guards, recording helper and event helper. Their raw tampering checks reject warmup, executable/entry, event, scope, guard and frame changes, and removal of event-wait descriptors. The broader regression includes partial-return/adoption/event failures and original graph provenance checks. These are fresh executions in this review, not reused author results.

The remaining four skips require actual inherited POSIX supervisor descriptors and cannot execute on this Windows host. Command 2's other nine skips were only the missing default pinned-source path and were subsequently executed in command 3. No skips are counted as passes.

## Limits and gates that remain closed

The integrated Warp fixture replaces factory construction/sealing and parameter validation with doubles while retaining the actual guarded recording helper. CUDA uses production stage constructors with upstream runtime/extension resource doubles. These tests cannot prove actual packed Warp command recording, native ModuleExec reuse, dual-GPU scope, CUDA module execution context, or native event behavior. Static inspection of the pinned source confirms API compatibility assumptions but does not close those runtime obligations.

This is positive bring-up support only. Real native graph recording/event acceptance at the reviewed revision, graph failure/partial-retention/hang cases and actual worker/supervisor process evidence remain mandatory later. The original storage-only remote batch 81866 at 3d6692b was not queried, touched or used as graph validation.

Optical launch, setup-warmup authority, ten observed warmups, R_final/final origin, final factories/graph, final coverage seal/barrier, R1/arena integration, CPU-oracle optical correctness, full C4 validation/refactor and fresh whole review remain outside this approval. Product/sample/IPC/performance/result/baseline/backend-selection gates remain closed. No rendered output or performance claim is made. Existing comparison-stage removal rules remain applicable; this approval does not make the support selectors permanent product interfaces.

## Process hygiene and artifacts

Observed regression PID 42104, parent 43948, creation 2026-10-10 02:50:53 +08:00, with the exact command 2 pytest arguments. Tool session 51659 completed exit 0. The short source-model run completed exit 0 before its OS PID was sampled; there is no claim of exhaustive process attestation. Test temporary fixtures may be written under pytest temporary directories outside the candidate; candidate status remained clean.

Final scoped process check at 2026-10-10T02:52:22.5232066+08:00 found no python.exe and no process with PID42104 or parent42104. Every reviewer test handle was terminal. No reviewer GPU worker, supervisor or SSH process was launched, so GPU-process polling was unnecessary for this CPU-only round. No signal, shared-GPU reset, unrelated-process cleanup or remote operation was performed. There is no observed residual test process.

The only reviewer-written artifact is this report at C:/Users/AG/新建文件夹/c4_assets/render_graph_native_support_review_v1.md. Candidate tracked files were not modified. The reviewer session remains active only to return the report to the coordinator.
