# Provisional render graph independent code review v2

Initiative: Q54 / P12.4 / C4 provisional R0 render graph
Stage: independent code re-review
Author: codex
Version: v2
Date: 2026-10-10
Status: accepted
Related Files: AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md; runtime-integration design-request codex v3 sections 5, 5.1, 6, 8, 9; remaining-setup gap-scan codex v1 G2; warp-module-preparation design-request codex v2; provisional-render-graph implementation-notes v1/v2 and code-review v1
Owner Summary: Approve the bounded code construction scope at the exact revision below. Previous blocking F1 is closed: actual CUDA factory/context/module and returned stages are checked against existing retained owners and issued runtime/module authority. No new substantive finding. Native recording/event acceptance remains required and is not granted by this CPU review.

Reviewer session: /root/provisional_render_graph_code_v2, fresh independent review; no implementation edits or subagents.
Repository: C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design
Whole candidate base: 15f64a96224a2ce27dabacf57c9aa6f2f1e288bc
Previous candidate: 275095edb035135855da73e81b2a43f55543a4b1
Reviewed HEAD: 8315b008bf54f294af5bf0c3035d4bc9cc0e009b
Actual tree status: clean at initial inspection, during testing and final verification; HEAD unchanged.
Verdict: **Approve**, exclusively for the code review scope below. This is not native acceptance or whole-C4 approval.

## Scope and evidence basis

Read the project collaboration protocol and full review principles, the requested parent/supplement requirements, both implementation notes and the complete previous review. Inspected the whole candidate diff and all ten changed files, including source catalog/pins, session phase/admission changes, common resource scope, Warp factory/recording changes, CUDA graph composition, events and tests. Followed original CUDA extension issuance, runtime binding validation, guardian retention, CUDA stage construction/ABI validation, Warp module-set and command bundle validation. Author counts and prior approvals were not treated as current evidence.

Approval covers only one single-use provisional R0 graph: truthful R_setup with warmup0, ten parameter/bundle pairs for indices 0..9, retained existing input/output resources and compiled owners, one initialized terminal event, diagnostics off=0/on=3 initialized pairs, and the shared guarded resource-scope mechanics needed for that construction.

Optical launch, observed warmup credit or setup-warmup execution authority, R_final/final origin, final factories/graph, seal/final barrier, R1, full timeline/arena integration, sample/IPC/product activation, performance/results/baselines and whole C4 remain outside this approval and closed. Native acceptance of this graph is a separate outstanding checkpoint. No remote command or GPU work was requested or executed. The unrelated storage-only campaign 81866 at 3d6692b was untouched and supplies no graph evidence.

## Findings and closure of F1

No new substantive finding. The previous blocking F1 is closed for this code scope.

The former failure was that a validated CUDA extension set could coexist with a rebound renderer.module or renderer.context, allowing a graph whose actual stages did not use its declared issued resources. At native_setup.py:118-131, _require_cuda_render_factory now requires the selected factory tuple itself to be already adopted, requires its context and renderer to be already adopted, requires the exact renderer type and its original tuple context, and resolves the module using the existing _require_cuda_extension authority. It compares the actual renderer.module with that exact issued module. It does not adopt an arbitrary replacement as proof or create another registry.

The existing extension validator, inspected beyond the fixture boundary, validates the live issued set, original source session/controller/admission, runtime binding and actual original module/entry objects. The helper checks exact context type, indexed target equality, exact stream object and saved stream handle; context._validate_identity additionally checks the stream's actual device/handle. The production factory constructor retains context, renderer and immutable tuple before graph preparation, so custody is anchored in original construction.

The initial check at native_setup.py:284 runs before frame binding. The final check at :331-338 resolves the same owners again, compares the actual retained factory/context, and validates every returned stage against that context and factory, including its captured raygen/shade entry identities. This closes a temporary foreign module used to construct a stage and subsequently restored, a stage constructed with a different renderer, and context drift during binding. Returned partial stages are held before this check and before any event initialization.

Independently ran the original v1 reproduction unchanged. Its two tests deliberately assert the old unwanted acceptance; on this candidate they now both fail at the new production checks: module replacement raises ValueError at native_setup.py:130, context replacement at :127. Both failures occur through prepare_fixed_render_graph before any event exists. These two intentional repro failures are evidence of rejection, not a passing normal test run. The separate v2 wrapper executes those same original test bodies and verifies expected rejection, no events and terminal quarantine; it passed in the regression below. Original repro SHA256 observed: 7cbf94ef593a3d96346140c9b348274f49a0e03e91653311dfea08a86597fb0e. Neither repro file was edited. This round did not rerun the predecessor checkout; it independently checked the preserved counterexamples against the repaired revision.

The candidate's ten added cases also passed: changed module/context/renderer/tuple, stream object, saved handle, device, foreign captured entry, foreign returned-stage renderer, and context change during binding. Initial replacements reject before binding; returned-stage inconsistencies reject before events. Checks include original error identity, retained owners, quarantine, restoration and zero emergency sync where asserted. CUDA graph fixtures still double upstream module intake and native resources; inspecting the production authority closes the static dataflow question, while real-resource acceptance remains pending.

## Other high-risk invariants checked

- Common _native_resource_scope retains the original single-use context registry and exact original guard ticket. Per-operation observation makes caught native failures sticky; escaped operations and cross-guard reuse reject. The pre-adopted partial-owner list keeps returned resources before metadata/adoption. Existing allocator regressions, including caught allocation/metadata/adoption and device/stream enter/exit failures, passed after extraction.
- Warp preparation retains its returned factory before public module load. Recording holds kernel/operand roots before record_cmd entry initialization and each returned command before inspection/adoption or another leaf. Legacy and guarded callers share the same implementation.
- Warp compiled-set validation occurs before each frame recording batch and after it, with final session validation after graph construction. Actual recorded command.module_exec and command.kernel.module must match the retained render executable/module. Existing packed-command descriptor and scalar validation remains in the production bundle path; no copied renderer math was introduced.
- Canonical sequence equality and original packed scene identity are checked. Scalar values are decoded from frozen little-endian binary32 records; frame metadata uses indices 0..9. Warp BVH metadata views borrow original arrays; CUDA frame views use the existing builder/stage ABI. Profile/shadows/diagnostics are checked without adding execution authority.
- Events are retained immediately after return and before metadata/adoption; foreign event scope fails before record/wait. Diagnostics-off creates no pairs. Terminal initialization is recorded last and its event-specific guarded wait covers all preceding initialization on the held stream. This supplies no optical-completion or warmup credit.
- Entry is single-use; failure uses the existing controller/session failure path and original guard. Partial graph/event owners remain held; no second synchronization, retry authority or successful final readiness is manufactured.
- Source closure adds only relevant literal phases; source policy changes update exact release pins without introducing broad loader/native exceptions. Actual current source/catalog bytes were exercised through fresh LF release fixtures and isolated subprocesses.

## Executed verification

Environment: Windows PowerShell, C:/Python314/python.exe, -B, pytest cache provider disabled. All commands ran from the reviewed repository. No GPU import/execution acceptance is inferred from native-return doubles.

1. `git diff --check 15f64a96224a2ce27dabacf57c9aa6f2f1e288bc HEAD`: exit 0, no whitespace errors; checked initially and finally.
2. `C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_native_render_preparation.py tests/unit/optics/test_native_output_allocation.py tests/unit/optics/test_prepared_fixed_render_inputs.py tests/unit/optics/test_warp_render_command_contract.py tests/unit/optics/test_warp_render_buffer_contract.py tests/unit/optics/test_render_parameters.py tests/unit/optics/test_cuda_artifact_runtime.py tests/unit/optics/test_warp_module_preparation.py C:/Users/AG/新建文件夹/c4_assets/test_provisional_graph_repro_rejected_v2.py`: **422 passed in 41.12s**, exit 0; tool session 15772.
3. `C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider C:/Users/AG/新建文件夹/c4_assets/test_provisional_graph_review_repro_v1.py`: **2 failed in 1.94s**, exit 1, intentionally retained old acceptance assertions rejected at the new production checks as detailed above. No other failure occurred in that invocation.
4. `C:/Python314/python.exe -B -m pytest -q -rs -p no:cacheprovider tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_native_bootstrap.py`: **281 passed, 13 skipped in 161.05s**, exit 0; tool session 30332. This is a new execution at the reviewed revision, not the predecessor's similarly numbered run.
5. Initial/intermediate/final `git status --short`: empty. `git rev-parse HEAD`: exact reviewed HEAD throughout.

Skipped cases: two real Go2 asset cases because the asset path is not configured; two POSIX release-byte checks; two actual unpackaged POSIX source-entry checks; seven generated-stdlib cases requiring pinned CPython 3.12 POSIX. Skips are not passes. Cold fixtures package current source bytes as LF, verify their hashes against the current catalog, and create temporary fixture commits; those fixture commit identities are not candidate revisions. Tests write release/negative-control/bytecode fixtures under pytest temporary directories outside the candidate. The exploratory root-level REVIEW_PRINCIPLES.md read failed because its actual location is collab/REVIEW_PRINCIPLES.md; the correct full file was subsequently read. Truncated broad document output was followed with targeted section reads.

## Remaining validation and limitations

Native acceptance must still establish actual Warp packed recording/entry initialization and exact retained ModuleExec reuse, actual CUDA issued-module/context identity, CUDA/Warp event creation/record/wait, target1/ambient0 restoration, and applicable failure/retention behavior under the existing supervisor at the reviewed code revision. The integrated Warp graph fixture replaces its factory; separate tests exercise the original prepare/seal methods and command contracts, but CPU doubles cannot prove the combined native path. Actual POSIX/CPython 3.12 and real Go2 cases remain required in their intended environment. No rendered image, numerical correctness, performance or ten completed warmups is claimed.

No optional code follow-up is needed to close this review. The above native work is a required separate acceptance checkpoint, not waived by this code-only approval. Any subsequent code/test/workflow changes need review of their delta; unchanged recording of this report follows the shared evidence-only exception.

## Process hygiene and artifacts

Observed first regression process PID 37580, parent 36276, creation 2026-10-10 02:40:49 +08:00; its tool session completed exit 0. Observed source/bootstrap regression PID 40840, parent 46800, creation 02:41:36 +08:00; its tool session completed exit 0. Sampled child PIDs 40380, 34572, 40468 and 38552 with parent 40840 and matching isolated bootstrap command lines during that run. The short original-repro process completed with expected exit 1 before its OS PID was sampled. Not every short-lived test child/utility was individually sampled; this is a bounded launch ledger, not exhaustive OS attestation.

At 2026-10-10T02:44:23+08:00, final Win32_Process queries returned no python.exe and no process with the sampled reviewer PIDs or parent PIDs above. All reviewer exec test handles were terminal. The source/bootstrap suite includes intentional parent-alive stall tests; its existing _run helper kills and communicates with its own sacrificial child on timeout and verifies it has exited. No manual signal, shared-GPU reset, unrelated-process kill or remote cleanup was performed. No reviewer GPU worker/supervisor/SSH process was launched; GPU-process polling was unnecessary for this CPU-only round, and the unrelated storage campaign was not queried or touched.

The only reviewer-written artifact is this outside-tree report at C:/Users/AG/新建文件夹/c4_assets/provisional_render_graph_code_review_v2.md. Candidate tracked files were not changed. The reviewer session remains active only to return this report to the coordinator.
