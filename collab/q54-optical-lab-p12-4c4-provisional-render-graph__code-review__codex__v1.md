# Provisional render graph independent code review v1

Initiative: Q54 / P12.4 / C4 provisional R0 render graph
Stage: independent code review
Author: codex
Version: v1
Date: 2026-10-10
Status: in_review
Related Files: AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md; runtime-integration design-request codex v3 sections 5, 5.1, 6, 8, 9; remaining-setup gap-scan codex v1 G2; warp-module-preparation design-request codex v2
Owner Summary: Request changes. The guarded recording/event implementation and CPU regressions are substantial, but the CUDA graph does not bind its actual renderer module and execution context back to the issued extensions and runtime stream. Two independent counterexamples construct a successful graph with a rebound renderer. Native recording/event acceptance remains outstanding.

Reviewer session: /root/provisional_render_graph_code_v1, fresh independent session; no implementation conversation inherited and no candidate edits made.
Repository: C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design
Base: 15f64a96224a2ce27dabacf57c9aa6f2f1e288bc
Reviewed HEAD: 275095edb035135855da73e81b2a43f55543a4b1
Actual working tree: clean at initial inspection and after verification; HEAD unchanged.
Verdict: **Request changes**.

## Scope

Inspected all eight changed files: native_setup.py, native_bootstrap.py, native_source_policy.py, native source catalog v15.json, warp_native_render.py, the two changed unit test files, and the provisional-render-graph implementation note. Read the shared collaboration protocol and review principles and the relevant accepted parent/supplement requirements. Followed existing CUDA stage construction, extension issuance/validation, source audit, Warp module validation, readiness, guarded lifecycle and ownership flows.

This review covers only construction of one provisional R0 graph, R_setup with warmup0, ten frame parameter/bundle records, one initialized terminal event and diagnostic off=0/on=3 pairs, and extraction of shared guarded resource-scope mechanics. It does not approve optical execution, warmup completion, final readiness/origin, final factories/graph, seal/barrier, samples, IPC, R1, product activation or performance. The previous storage-only native campaign at 3d6692b was not touched or used as evidence for this delta.

## Finding F1 — P1, blocking: CUDA graph can retain a different renderer module/context from its validated compiled/runtime owners

Locations: optical_worker/native_setup.py:267-269, 303-314; optical_worker/native_bootstrap.py:3391 and 3405.

Trigger: after CUDA factory preparation, replace the mutable renderer.module or renderer.context on session.cuda_factories[3], then call prepare_fixed_render_graph. The replacement can preserve the expected callable names and tensor device metadata, or use a different stream on the same device. No alteration of the registered extension set is necessary.

Observed behavior: the helper validates session.cuda_extensions(), independently takes session.cuda_factories[3], and constructs ten actual CudaDirectLightRenderStage objects. It never compares factory.module to the issued robot_sim_cuda_direct_light_v8 module, or factory.context/stream to the session's validated runtime binding. _hold_runtime_owner then adopts the selected factory rather than proving its original provenance. The final extension-set identity check proves only that the registered set remains unchanged, not that the stages use it.

Why existing checks miss it: CudaDirectLightNativeRenderer is mutable. CudaDirectLightRenderStage validates its renderer's current module and compares the stage's captured entries with that same current module; binding passes its own renderer.context back as the expected context. Those are internal consistency checks, not comparisons with the source session. validate_render_parameters checks scalar/profile/diagnostic ABI, not module or stream identity. Source/external audits check their original admitted module records and do not bind this renderer object to those records. The graph tests use one stable factory and do not attempt module/context replacement.

Independent reproduction: C:/Users/AG/新建文件夹/c4_assets/test_provisional_graph_review_repro_v1.py. This imports the candidate's existing CPU fixtures and runs the actual production session graph entry and CUDA renderer/stage constructors. In one case it replaces renderer.module with an object exposing new callables for the same entries. In the other it replaces renderer.context with a same-device context double holding a different stream. Both return fixed_render_graph_ready with ten bundles and the unchanged compiled_modules owner. The module case checks that the returned bundles actually captured the foreign raygen/shade entries. The context case checks that their renderer stream differs from the session stream. Output: ACCEPTED_REBOUND_RENDERER module; ACCEPTED_REBOUND_RENDERER context; 2 passed in 1.65s. A passing reproduction means the unwanted acceptance occurred.

The test doubles native resources and upstream intake just as the existing integrated graph test does; it is not a real-CUDA execution test. Static inspection of the actual extension validator and source audits independently confirms that those upstream checks validate the issued originals but contain no missing renderer-to-owner comparison. The finding concerns object/dataflow provenance in the closed trusted process, not a hostile-Python security claim.

Consequence: the returned graph's compiled_modules field and initialized events describe the original admitted set/session stream while its real stage objects can refer to different executable entries or a different context/stream. That violates this checkpoint's exact compiled/runtime reuse claim. No optical work is executed in this checkpoint, but this is already an invalid provisional graph for the subsequent guarded warmup stage.

Repair direction: before binding the first CUDA frame, validate the actual factory and context against the original retained factory owners and exact runtime binding; compare its module with the module returned by the existing issued extension-set validator and validate the held device/stream/stream-handle identity. Reuse the existing authority and custody paths; a second authority registry is unnecessary. Ensure the checks protect the actual returned stages as well as the descriptive compiled owner. Retain ordinary legacy constructors without treating them as captured-session provenance.

Required proof after repair: negative tests through prepare_fixed_render_graph for replaced module, replaced renderer/factory, foreign/rebound context and same-device different stream; reject before any frame bundle/event initialization; retain the original error and partial owners with the existing failure disposition. Positive CUDA preparation must still use the actual original issued module/context. Native acceptance must subsequently verify these identities with real resources.

## Verification and findings from high-risk inspection

- The extracted _native_resource_scope shares the original single-use context registry and original guard-ticket check. Per-operation observation makes caught native failures sticky; active-scope checks reject escaped recording/allocation functions. Existing allocator tests continued to pass.
- Returned Warp factory custody is established before the public module-load call. Each record_cmd operand tuple and command return is held before metadata/adoption or the next leaf. Original factory seal uses the common helper rather than a separate recording implementation.
- Warp module validation precedes each frame recording batch and follows it. Command module/module_exec identity is compared with the retained render module record. Existing renderer descriptors/math were preserved.
- Frame records decode the canonical little-endian binary32 scalars and retain indices 0..9. Warp BVH views share the original arrays while updating frame metadata. CUDA uses the existing renderer/stage constructors and actual scalar ABI evidence. Diagnostics allocation mismatch and canonical source mismatch reject inside the setup guard.
- Event creation retains every returned event before metadata/adoption. Diagnostic pairs are absent when disabled; terminal initialization is recorded last, followed by an event-specific guarded wait covering preceding initialization on the held stream. This is not represented as optical completion or warmup credit.
- The new graph phase is single-use. Failures within its segment use the original guard/quarantine machinery; no new execution authority, sample path or normal/emergency retry was added.
- No additional substantive finding is reported. Native resource behavior remains unproven by these CPU doubles.

Environment: Windows PowerShell; C:/Python314/python.exe -B; pytest cache provider disabled. Independent commands/results:

1. git diff --check 15f64a96224a2ce27dabacf57c9aa6f2f1e288bc 275095edb035135855da73e81b2a43f55543a4b1: exit 0, no whitespace errors.
2. C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_native_render_preparation.py tests/unit/optics/test_native_output_allocation.py tests/unit/optics/test_prepared_fixed_render_inputs.py tests/unit/optics/test_warp_render_command_contract.py tests/unit/optics/test_warp_render_buffer_contract.py tests/unit/optics/test_render_parameters.py tests/unit/optics/test_cuda_artifact_runtime.py tests/unit/optics/test_warp_module_preparation.py: **410 passed in 33.43s**, exit 0, tool process session 62983.
3. C:/Python314/python.exe -B -m pytest -q -s -p no:cacheprovider C:/Users/AG/新建文件夹/c4_assets/test_provisional_graph_review_repro_v1.py: **2 passed in 1.65s**, exit 0, both counterexamples observed.
4. C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py: **206 passed, 4 skipped in 43.51s**, exit 0, tool process session 41901. Skip reason details were not emitted with -q; these are not counted as successful tests. The suite contains host-specific POSIX, real Go2 asset and symlink exclusions.
5. Initial and final git status --short: empty. Initial/final git rev-parse HEAD: exact reviewed HEAD above.

Author counts were not used as independent verification. The parent's separate cold bootstrap run was not executed or claimed by this reviewer; its result should be recorded separately. No tests or reports were written into the candidate tree.

## Remaining validation and process hygiene

The integrated Warp factory is doubled in the CPU suite; original factory prepare/seal custody methods are separately exercised, but that does not establish real packed Warp recording. Actual Warp entry initialization, exact ModuleExec reuse during recording, native CUDA/Warp event creation/record/wait, target1/ambient0 restoration and failure behavior need supervised native acceptance at the repaired revision. None was requested or run here. No GPU result, performance result, ten-warmup observation or final-readiness claim is made.

All three reviewer pytest invocations completed with observed exit 0. Source-policy/source pytest was observed as PID 44176, parent 5736, with its exact command line and creation timestamp during execution; final scoped Win32_Process query found no python.exe matching any of the reviewer's three command lines. The earlier two short test invocations were tracked by tool completion but their OS PIDs were not captured before exit; this limits the OS launch-ledger detail. No reviewer GPU worker/supervisor/SSH process was launched and no GPU-process check was needed for this CPU-only round. No process was killed, signalled, reset or otherwise interfered with. The unrelated storage campaign remained untouched. The reviewer session itself remained active to deliver this report.

Outside-tree artifacts: this report and test_provisional_graph_review_repro_v1.py in C:/Users/AG/新建文件夹/c4_assets. Preserve the original reproductions for fresh independent re-review. Fix F1, freeze the revised candidate, then re-review; approval of a repaired CPU construction scope would still not substitute for the required native acceptance or open the excluded gates.
